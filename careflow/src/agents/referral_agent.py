"""ReferralTrackerAgent — Specialty referral & scheduling specialist for CareFlow (M6).

Manages referral tracking, referral status lookups, and new referral generation
using FastMCP EHR tools, with integrated working (Session) and long-term (Qdrant) memory.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from dotenv import load_dotenv

from src.mcp.ehr_server import (
    create_referral,
    get_appointments,
    get_referral_status,
)
from src.memory.long_term import LongTermMemory
from src.memory.session import SessionMemory
from src.tools.flag_human import FLAG_HUMAN_DEFINITION, flag_for_human
from src.utils.observability import record_llm_usage

load_dotenv()


litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── Shared Memory Instances ────────────────────────────────────────────────────
# Lazily initialised — avoids loading Qdrant/sentence-transformers at import time.
_session_memory: SessionMemory | None = None
_long_term_memory: LongTermMemory | None = None


def _get_session_memory() -> SessionMemory:
    global _session_memory
    if _session_memory is None:
        _session_memory = SessionMemory()
    return _session_memory


def _get_long_term_memory() -> LongTermMemory:
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
    return _long_term_memory

_PROVIDER = os.getenv("LITELLM_PROVIDER", "gemini")
_MODEL_BASE = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_MODEL = f"{_PROVIDER}/{_MODEL_BASE}" if _PROVIDER != "openai" else _MODEL_BASE
MAX_TOOL_ROUNDS = 5


# ── Tool Definitions ───────────────────────────────────────────────────────────
GET_REFERRAL_STATUS_DEF = {
    "type": "function",
    "function": {
        "name": "get_referral_status",
        "description": "Check active specialty referrals for a patient reference ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_ref": {
                    "type": "string",
                    "description": "The patient reference ID (e.g. MHP-P-10021)."
                }
            },
            "required": ["patient_ref"],
        },
    },
}

CREATE_REFERRAL_DEF = {
    "type": "function",
    "function": {
        "name": "create_referral",
        "description": "Create a new outpatient specialty referral record in the EHR system.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_ref": {"type": "string", "description": "Patient reference ID."},
                "specialty": {"type": "string", "description": "Specialty line (e.g. Cardiology, Orthopaedics)."},
                "urgency": {"type": "string", "description": "Urgency tier: Routine, Priority, Urgent, Emergent."},
                "notes": {"type": "string", "description": "Clinical context or reasoning for referral."},
            },
            "required": ["patient_ref", "specialty"],
        },
    },
}

GET_APPOINTMENTS_DEF = {
    "type": "function",
    "function": {
        "name": "get_appointments",
        "description": "Retrieve past and upcoming appointments for a patient reference ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_ref": {"type": "string", "description": "Patient reference ID."}
            },
            "required": ["patient_ref"],
        },
    },
}

REFERRAL_TOOLS = [
    GET_REFERRAL_STATUS_DEF,
    CREATE_REFERRAL_DEF,
    GET_APPOINTMENTS_DEF,
    FLAG_HUMAN_DEFINITION,
]


def _dispatch_referral_tool(tool_name: str, tool_args: dict) -> dict[str, Any]:
    """Dispatch tool calls for ReferralTrackerAgent."""
    if tool_name == "get_referral_status":
        return get_referral_status(**tool_args)
    if tool_name == "create_referral":
        return create_referral(**tool_args)
    if tool_name == "get_appointments":
        return get_appointments(**tool_args)
    if tool_name == "flag_for_human":
        return flag_for_human(**tool_args)
    return {"error": f"Unknown tool {tool_name}"}


# ── System Prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are Jordan, the Referral Tracking and Coordination Specialist at Meridian Health Partners.
You help patients and staff with:
- Checking status of active specialty referrals
- Creating new referral requests for specialty consultations
- Looking up scheduled appointments for follow-up care

STRICT RULES YOU MUST FOLLOW:
1. If a patient asks ANYTHING clinical (symptoms, medication advice, diagnosis, treatment options) -> call flag_for_human immediately.
2. Always check existing referrals or appointments before creating a duplicate.
3. Be professional, concise, and empathetic.
4. Never fabricate referral IDs, doctor names, or status codes. Use only what the tools return!
5. If a tool returns an error, explain politely or call flag_for_human.
"""


# ── Response Dataclass ─────────────────────────────────────────────────────────
@dataclass
class ReferralAgentResponse:
    """Result from ReferralTrackerAgent.run()."""
    final_answer: str
    tools_called: list[dict] = field(default_factory=list)
    escalated: bool = False
    escalation_ticket: str | None = None
    rounds: int = 0
    error: str | None = None


# ── ReferralTrackerAgent ──────────────────────────────────────────────────────
class ReferralTrackerAgent:
    """Specialty referral tracking and coordination agent."""

    def __init__(
        self,
        model: str | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        temperature: float = 0.1,
    ) -> None:
        self.model = model or _MODEL
        self.max_tool_rounds = max_tool_rounds
        self.temperature = temperature

    def run(
        self,
        user_message: str,
        *,
        patient_ref: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> ReferralAgentResponse:
        """Process a referral coordination query through the tool-calling loop.

        Args:
            user_message:         The patient's referral/scheduling question.
            patient_ref:          Patient reference ID for context and memory recall.
            conversation_history: Previous turns (list of role/content dicts).

        Returns:
            ReferralAgentResponse with final_answer, tools_called, escalated flag.
        """
        response = ReferralAgentResponse(final_answer="")
        tools_called: list[dict] = []

        # ── Session Memory: recover working conversation history ───────────────
        session_mem = _get_session_memory()
        session_history = (
            conversation_history
            or (session_mem.get_context_messages(patient_ref) if patient_ref else [])
        )

        # ── Long-Term Memory: recall past patient referral context ─────────────
        ltm_context = ""
        if patient_ref:
            try:
                ltm = _get_long_term_memory()
                ltm_context = ltm.recall_as_context(
                    patient_ref,
                    query=user_message,
                    top_k=3,
                )
                if ltm_context:
                    logger.info(
                        "[%s] ReferralTrackerAgent: injected %d chars of long-term memory",
                        patient_ref, len(ltm_context),
                    )
            except Exception as exc:
                logger.warning(
                    "[%s] ReferralTrackerAgent: long-term memory recall failed: %s",
                    patient_ref, exc,
                )

        # ── Build message list ─────────────────────────────────────────────────
        # Inject long-term memory as a system message immediately after the main
        # system prompt, so the agent is aware of past patient interactions.
        messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if ltm_context:
            messages.append({"role": "system", "content": ltm_context})

        if session_history:
            messages.extend(session_history)

        # Inject patient reference into the user turn for tool grounding
        user_content = user_message
        if patient_ref:
            user_content = f"[Patient Reference: {patient_ref}]\n\n{user_message}"

        messages.append({"role": "user", "content": user_content})

        # Record user turn into session memory
        if patient_ref:
            session_mem.add_turn(patient_ref, "user", user_message)

        for round_num in range(self.max_tool_rounds + 1):
            response.rounds = round_num + 1

            llm_response = None
            last_exc = None

            # Retry loop for transient rate limits
            for attempt in range(4):
                try:
                    llm_response = litellm.completion(
                        model=self.model,
                        messages=messages,
                        tools=REFERRAL_TOOLS,
                        tool_choice="auto",
                        temperature=self.temperature,
                    )
                    record_llm_usage("referral_agent", getattr(llm_response, "usage", None))
                    break

                except Exception as exc:
                    last_exc = exc
                    if "429" in str(exc) or "RateLimit" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                        wait_sec = 2 ** (attempt + 1) + 2
                        logger.warning("ReferralTrackerAgent hit rate limit (attempt %d/4). Retrying in %ds...", attempt + 1, wait_sec)
                        time.sleep(wait_sec)
                    else:
                        break

            if llm_response is None:
                logger.error("ReferralTrackerAgent LLM call failed: %s", last_exc)
                response.error = str(last_exc)
                response.final_answer = "I'm experiencing a temporary issue. Connecting you to a care coordinator."
                esc = flag_for_human(
                    reason=f"ReferralTrackerAgent LLM error: {last_exc}",
                    patient_ref=patient_ref or "UNKNOWN",
                    urgency="Routine",
                )
                tools_called.append({"tool": "flag_for_human", "args": {}, "result": esc})
                response.escalated = True
                response.escalation_ticket = esc.get("ticket_id")
                break

            choice = llm_response.choices[0]
            message = choice.message

            # Check if LLM decided to call a tool
            if hasattr(message, "tool_calls") and message.tool_calls:
                messages.append(message.model_dump())

                for tool_call in message.tool_calls:
                    fn = tool_call.function
                    tool_name = fn.name
                    import json
                    try:
                        args = json.loads(fn.arguments) if isinstance(fn.arguments, str) else fn.arguments
                    except Exception:
                        args = {}

                    logger.info("[%s] ReferralTrackerAgent tool call: %s(%s)", patient_ref, tool_name, args)
                    tool_result = _dispatch_referral_tool(tool_name, args)
                    tools_called.append({"tool": tool_name, "args": args, "result": tool_result})

                    if tool_name == "flag_for_human":
                        response.escalated = True
                        response.escalation_ticket = tool_result.get("ticket_id")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    })
            else:
                # LLM finished with final answer
                response.final_answer = message.content or ""
                break

        response.tools_called = tools_called

        # ── Post-run: persist session turn & save long-term summary ───────────
        if patient_ref and response.final_answer:
            # Record assistant turn in working memory
            session_mem.add_turn(patient_ref, "assistant", response.final_answer)

            # Persist a concise session summary to long-term Qdrant memory
            tools_summary = ", ".join(
                t["tool"] for t in tools_called
                if t["tool"] != "flag_for_human"
            )
            summary = (
                f"Referral query: '{user_message[:120]}'. "
                f"Tools used: [{tools_summary}]. "
                f"Escalated: {response.escalated}."
            )
            try:
                ltm = _get_long_term_memory()
                ltm.save_session_summary(
                    patient_ref,
                    summary,
                    extra_metadata={
                        "agent": "referral_tracker",
                        "escalated": response.escalated,
                        "tools": tools_summary,
                    },
                )
                logger.info(
                    "[%s] ReferralTrackerAgent: saved session summary to long-term memory",
                    patient_ref,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] ReferralTrackerAgent: long-term memory save failed: %s",
                    patient_ref, exc,
                )

        return response
