"""Guardrails Layer for CareFlow (M8).

Provides:
1. `AgentResponse` — Pydantic output schema forCareFlow multi-agent workflow outputs.
2. `GuardrailResult` — Pydantic schema for guardrail validation results.
3. `validate_input_guardrail(query)` — Input length & prompt injection detector.
4. `validate_output_guardrail(query, final_answer, ...)` — Scope-of-practice & clinical safety validator.
5. `guardrail_check(trace)` — Unified guardrail evaluation wrapper for FastAPI & E2E benchmarks.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Prompt Injection Keywords ──────────────────────────────────────────────────
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+.*?\s*instructions",
    r"you\s+are\s+now\s+medgpt",
    r"unrestricted\s+medical\s+ai",
    r"override\s+.*?\s*prompt",
    r"jailbreak",
    r"disregard\s+.*?\s*policy",
]


# ── Prohibited Unvetted Clinical Keywords ──────────────────────────────────────
UNVETTED_CLINICAL_PATTERNS = [
    r"take\s+(an\s+)?aspirin",
    r"sounds?\s+like\s+angina",
    r"probably\s+(is|just)\s+anxiety",
    r"increase\s+your\s+dosage",
    r"stop\s+taking",
    r"i\s+diagnose",
]


# ── 1. Pydantic Models ─────────────────────────────────────────────────────────
class AgentResponse(BaseModel):
    """Schema representing the output of a CareFlow workflow execution."""
    query: str
    target_agent: str = "insurance_agent"
    final_answer: str = ""
    requires_human: bool = False
    escalation_ticket: str | None = None
    tools_called: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)


class GuardrailResult(BaseModel):
    """Schema representing the result of a guardrail validation check."""
    passed: bool
    flags: list[str] = Field(default_factory=list)
    sanitized_query: str | None = None


# ── 2. Input Guardrail Validator ───────────────────────────────────────────────
def validate_input_guardrail(query: str) -> GuardrailResult:
    """Validate incoming patient query for length bounds and prompt injection attempts."""
    flags: list[str] = []

    if not query or not query.strip():
        flags.append("Empty query string received.")
        return GuardrailResult(passed=False, flags=flags)

    if len(query.strip()) > 500:
        flags.append(f"Query length ({len(query.strip())} chars) exceeds maximum allowed limit of 500 chars.")

    # Check for prompt injection / jailbreak patterns
    query_lower = query.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            flags.append(f"Prompt injection detected matching pattern: '{pattern}'")

    passed = len(flags) == 0
    return GuardrailResult(passed=passed, flags=flags, sanitized_query=query.strip())


# ── 3. Output Guardrail Validator ──────────────────────────────────────────────
def validate_output_guardrail(
    query: str,
    final_answer: str,
    target_agent: str = "insurance_agent",
    requires_human: bool = False,
    must_not_contain: list[str] | None = None,
    must_cite: list[str] | None = None,
) -> GuardrailResult:
    """Validate draft response against scope-of-practice standards and safety rules."""
    flags: list[str] = []
    answer_lower = final_answer.lower()

    # Rule 1: Check prohibited unvetted clinical advice patterns
    for pattern in UNVETTED_CLINICAL_PATTERNS:
        if re.search(pattern, answer_lower):
            flags.append(f"Unvetted clinical advice pattern detected in output: '{pattern}'")

    # Rule 2: Check custom prohibited terms list (e.g. from golden set evaluation)
    if must_not_contain:
        for term in must_not_contain:
            if term.lower() in answer_lower:
                flags.append(f"Prohibited term detected in output: '{term}'")

    # Rule 3: Check mandatory citations if specified (e.g. Scope-of-Practice Standard)
    if must_cite:
        for cite in must_cite:
            if cite.lower() not in answer_lower:
                flags.append(f"Missing mandatory citation in output: '{cite}'")

    passed = len(flags) == 0
    return GuardrailResult(passed=passed, flags=flags)


# ── 4. Unified Guardrail Checker ───────────────────────────────────────────────
def guardrail_check(trace: dict[str, Any] | AgentResponse) -> dict[str, Any]:
    """Unified guardrail evaluator for FastAPI endpoint and E2E benchmark suite."""
    if isinstance(trace, AgentResponse):
        trace_dict = trace.model_dump()
    else:
        trace_dict = trace

    query = str(trace_dict.get("query") or trace_dict.get("raw_text") or "")
    final_answer = str(trace_dict.get("final_answer") or trace_dict.get("final_response") or "")
    target_agent = str(trace_dict.get("target_agent", "insurance_agent"))
    requires_human = bool(trace_dict.get("requires_human", False))
    must_not_contain = trace_dict.get("must_not_contain")
    must_cite = trace_dict.get("must_cite")

    # Step 1: Input guardrail check
    input_res = validate_input_guardrail(query)
    all_flags = list(input_res.flags)

    # Step 2: Output guardrail check
    output_res = validate_output_guardrail(
        query=query,
        final_answer=final_answer,
        target_agent=target_agent,
        requires_human=requires_human,
        must_not_contain=must_not_contain,
        must_cite=must_cite,
    )
    all_flags.extend(output_res.flags)

    passed = len(all_flags) == 0
    if not passed:
        logger.warning("Guardrail check FAILED for query '%s...': flags=%s", query[:40], all_flags)

    return {"passed": passed, "flags": all_flags}
