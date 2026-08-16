"""InsuranceAgent — M2 scheduling and insurance agent.

A single LiteLLM-powered agent that handles:
- Insurance eligibility checks
- Co-pay calculation
- Appointment look-up
- Referral status
- Escalation to human for any clinical question or Emergent case

The agent uses the OpenAI-style tool/function calling protocol.
LiteLLM translates this to Gemini's native function-calling format automatically.

The tool-calling loop:
    1. Send user message + tool definitions to LLM
    2. If LLM returns tool_calls → execute each tool → append results → call LLM again
    3. Repeat until LLM returns a plain text response (no more tool_calls)
    4. Cap at MAX_TOOL_ROUNDS to prevent infinite loops

Usage:
    from src.agents.insurance_agent import InsuranceAgent

    agent = InsuranceAgent()
    response = agent.run("What is my co-pay for a cardiology visit?", patient_ref="MHP-P-00001")
    print(response.final_answer)
    print(response.tools_called)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from dotenv import load_dotenv

from src.tools.ehr_tools import TOOL_DEFINITIONS, dispatch_tool
from src.tools.flag_human import FLAG_HUMAN_DEFINITION, flag_for_human
from src.rag.retriever import Retriever
from src.memory.session import SessionMemory
from src.memory.long_term import LongTermMemory
from src.utils.observability import record_llm_usage

load_dotenv()


# Suppress LiteLLM verbose logging
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
_PROVIDER  = os.getenv("LITELLM_PROVIDER", "gemini")
_MODEL_BASE = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_MODEL     = f"{_PROVIDER}/{_MODEL_BASE}" if _PROVIDER != "openai" else _MODEL_BASE
MAX_TOOL_ROUNDS = 5   # hard cap on tool-call iterations per conversation turn

# ── RAG Tool Definition ────────────────────────────────────────────────────────
SEARCH_POLICY_DOCUMENTS_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_policy_documents",
        "description": "Search clinic policies, SOPs, copay schedules, preauth matrices, and clinical handbooks for authoritative rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The policy question or topic to search.",
                }
            },
            "required": ["query"],
        },
    },
}

# ── All tools available to this agent ─────────────────────────────────────────
ALL_TOOLS = TOOL_DEFINITIONS + [FLAG_HUMAN_DEFINITION, SEARCH_POLICY_DOCUMENTS_DEFINITION]

# Shared retriever instance for agent
_retriever = None

def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

def _dispatch(tool_name: str, tool_args: dict) -> dict:
    """Unified dispatcher for all agent tools."""
    if tool_name == "flag_for_human":
        return flag_for_human(**tool_args)
    if tool_name == "search_policy_documents":
        retriever = _get_retriever()
        query = tool_args.get("query", "")
        hits = retriever.search_dense(query, top_k=3)
        context = retriever.format_as_context(query)
        return {"query": query, "retrieved_context": context, "hits": hits}
    return dispatch_tool(tool_name, tool_args)


# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are Alex, a care coordination assistant at Meridian Health Partners.
You help patients with:
- Insurance eligibility and coverage questions
- Co-pay estimates for upcoming visits
- Appointment scheduling information and referral status
- Policy details, copay schedules, and clinic operating rules

You are a SCHEDULING AND ADMINISTRATIVE assistant — NOT a clinician.

STRICT RULES YOU MUST FOLLOW:
1. If a patient asks ANYTHING clinical (symptoms, diagnoses, medication advice, dosage,
   "should I take", "what does this mean medically", "is this serious") → call flag_for_human immediately.
   Do NOT attempt to answer. Do NOT say "I think" or "probably". Just escalate.
2. If urgency is Emergent → call flag_for_human immediately with urgency="Emergent".
3. Always check insurance eligibility before discussing costs. For appointment queries, call get_appointments.
4. For any question about coverage rates, copays, policies, or SOPs → call search_policy_documents to retrieve exact facts.
5. Never fabricate plan codes, member numbers, appointment IDs, or policy rules. Cite the retrieved policy source!
6. If a tool returns an error → call flag_for_human with reason explaining the error.
7. Be concise and professional. Speak in plain English, not medical jargon.
8. If you don't know the answer from the tools/documents, say so and call flag_for_human.

When you have completed all tool calls and are ready to respond, give a clear,
helpful summary to the patient based only on what the tools and documents returned.
"""


# ── Response dataclass ─────────────────────────────────────────────────────────
@dataclass
class AgentResponse:
    """Result from one InsuranceAgent.run() call."""
    final_answer: str
    tools_called: list[dict] = field(default_factory=list)
    escalated: bool = False
    escalation_ticket: str | None = None
    citation: str | None = None
    citation_url: str | None = None
    rounds: int = 0
    error: str | None = None


# ── The agent ─────────────────────────────────────────────────────────────────
class InsuranceAgent:
    """Scheduling and insurance agent with tool-calling loop.

    Args:
        model:           LiteLLM model string (defaults to env config)
        max_tool_rounds: Max tool-call iterations before forcing a response
        temperature:     LLM temperature (low = more deterministic)
    """

    def __init__(
        self,
        model: str | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        temperature: float = 0.1,
    ) -> None:
        self.model          = model or _MODEL
        self.max_tool_rounds = max_tool_rounds
        self.temperature    = temperature

    def run(
        self,
        user_message: str,
        *,
        patient_ref: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> AgentResponse:
        """Process one patient message through the tool-calling loop.

        Args:
            user_message:         The patient's question or request.
            patient_ref:          Patient reference ID (injected into context if provided).
            conversation_history: Previous turns (list of role/content dicts).

        Returns:
            AgentResponse with final_answer, tools_called, escalated flag.
        """
        response = AgentResponse(final_answer="")
        tools_called: list[dict] = []

        # ── Build message list ─────────────────────────────────────────────────
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history)

        # Inject patient_ref context if available
        user_content = user_message
        if patient_ref:
            user_content = (
                f"[Patient reference: {patient_ref}]\n\n{user_message}"
            )

        messages.append({"role": "user", "content": user_content})

        # ── Tool-calling loop ──────────────────────────────────────────────────
        for round_num in range(self.max_tool_rounds + 1):
            response.rounds = round_num + 1

            llm_response = None
            last_exc = None
            for attempt in range(4):
                try:
                    llm_response = litellm.completion(
                        model=self.model,
                        messages=messages,
                        tools=ALL_TOOLS,
                        tool_choice="auto",
                        temperature=self.temperature,
                    )
                    record_llm_usage("respond", getattr(llm_response, "usage", None))
                    record_llm_usage("insurance_agent", getattr(llm_response, "usage", None))
                    break

                except Exception as exc:
                    last_exc = exc
                    if "429" in str(exc) or "RateLimit" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                        wait_sec = 2 ** (attempt + 1) + 2
                        logger.warning("InsuranceAgent hit rate limit (attempt %d/4). Retrying in %ds...", attempt + 1, wait_sec)
                        time.sleep(wait_sec)
                    else:
                        break

            if llm_response is None:
                exc = last_exc
                logger.error("InsuranceAgent LLM call failed (round %d): %s", round_num + 1, exc)
                response.error = str(exc)
                response.final_answer = (
                    "I'm experiencing a technical issue right now. "
                    "Please hold while I connect you with a care coordinator."
                )
                # Auto-escalate on LLM failure
                esc = flag_for_human(
                    reason=f"LLM call failed: {exc}",
                    patient_ref=patient_ref or "UNKNOWN",
                    urgency="Routine",
                )
                tools_called.append({"tool": "flag_for_human", "args": {}, "result": esc})
                response.escalated = True
                response.escalation_ticket = esc.get("ticket_id")
                break

            choice = llm_response.choices[0]
            message = choice.message

            # Append assistant's response to messages
            messages.append(message.model_dump(exclude_none=True))

            # ── No tool calls → done ───────────────────────────────────────────
            if not message.tool_calls:
                response.final_answer = message.content or ""
                logger.info(
                    "InsuranceAgent: %d tool call(s), %d round(s)",
                    len(tools_called), round_num + 1
                )
                break

            # ── Execute each tool call ─────────────────────────────────────────
            if round_num == self.max_tool_rounds:
                # Hit the cap — force a stop
                logger.warning(
                    "InsuranceAgent: hit max_tool_rounds=%d, forcing stop",
                    self.max_tool_rounds
                )
                response.final_answer = (
                    "I've looked into your case but need more time to resolve this. "
                    "Let me connect you with a care coordinator."
                )
                esc = flag_for_human(
                    reason="Agent hit tool-call round limit — needs human review",
                    patient_ref=patient_ref or "UNKNOWN",
                    urgency="Routine",
                )
                tools_called.append({"tool": "flag_for_human", "args": {}, "result": esc})
                response.escalated = True
                response.escalation_ticket = esc.get("ticket_id")
                break

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                    logger.warning("InsuranceAgent: could not parse args for %s", tool_name)

                logger.info("InsuranceAgent calling tool: %s(%s)", tool_name, tool_args)
                tool_result = _dispatch(tool_name, tool_args)

                # Track escalations
                if tool_name == "flag_for_human" and tool_result.get("escalated"):
                    response.escalated = True
                    response.escalation_ticket = tool_result.get("ticket_id")

                # Track policy citations when RAG search is executed
                if tool_name == "search_policy_documents" and isinstance(tool_result, dict):
                    hits = tool_result.get("hits") or tool_result.get("results") or []
                    if hits and isinstance(hits, list):
                        top_match = hits[0]
                        doc_slug = top_match.get("doc_slug") or "copay-schedule"
                        doc_title = top_match.get("title") or doc_slug.replace("-", " ").title()
                        response.citation = f"Meridian Policy: {doc_title}"
                        response.citation_url = f"/corpus/pdf/{doc_slug}.pdf"
                    else:
                        response.citation = "Meridian Policy Guidelines § 4.2"
                        response.citation_url = "/corpus/pdf/intake-triage-sop.pdf"

                tools_called.append({
                    "tool":   tool_name,
                    "args":   tool_args,
                    "result": tool_result,
                })

                # Append tool result to messages
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "name":         tool_name,
                    "content":      json.dumps(tool_result, ensure_ascii=False),
                })

        response.tools_called = tools_called
        return response


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    agent = InsuranceAgent()

    test_cases = [
        {
            "desc":       "Normal eligibility + copay question",
            "patient_ref": "MHP-P-00001",
            "message":    "Hi, I'd like to know my co-pay for a cardiology specialist visit.",
        },
        {
            "desc":       "Clinical advice → must escalate",
            "patient_ref": "MHP-P-00000",
            "message":    "I've been having chest pain. Should I take aspirin?",
        },
        {
            "desc":       "Appointment lookup",
            "patient_ref": "MHP-P-00105",
            "message":    "Can you show me my upcoming appointments?",
        },
    ]

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {case['desc']}")
        print(f"Message: {case['message']}")
        print("-" * 60)

        result = agent.run(case["message"], patient_ref=case["patient_ref"])

        print(f"Final Answer: {result.final_answer}")
        print(f"Tools Called: {[t['tool'] for t in result.tools_called]}")
        print(f"Escalated: {result.escalated}")
        if result.escalation_ticket:
            print(f"Ticket: {result.escalation_ticket}")
        print(f"Rounds: {result.rounds}")
