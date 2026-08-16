"""Evaluation Scorers for CareFlow E2E Evaluation Benchmark (M8).

Provides:
1. `tool_match_score(item, trace)` — Deterministic tool call accuracy scorer.
2. `retrieval_recall_score(item, trace)` — Deterministic RAG context recall scorer.
3. `guardrail_compliance_score(item, trace)` — Deterministic guardrail compliance scorer.
4. `deepeval_correctness_score(item, trace)` — LLM judge response quality scorer.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── 1. Deterministic Tool Match Scorer ─────────────────────────────────────────
def tool_match_score(item: dict[str, Any], trace: dict[str, Any]) -> float:
    """1.0 if actual tool calls match expected tools/actions, 0.0 otherwise."""
    expected_route = item.get("expected_route")
    actual_agent = trace.get("target_agent", "")
    requires_human = trace.get("requires_human", False)

    # Route matching check
    if expected_route == "refuse" or expected_route == "flag_human":
        return 1.0 if (requires_human or actual_agent == "flag_for_human") else 0.0

    if expected_route and expected_route in actual_agent:
        return 1.0

    # Tool call matching check
    actual_tools = [t.get("tool", "") if isinstance(t, dict) else str(t) for t in trace.get("tools_called", [])]
    expected_tools = item.get("expected_tools", [])

    if not expected_tools:
        # Default: if no specific tools expected and answer produced cleanly
        return 1.0 if trace.get("final_response") else 0.0

    matches = set(actual_tools).intersection(set(expected_tools))
    return len(matches) / len(expected_tools) if expected_tools else 1.0


# ── 2. Deterministic Retrieval Recall Scorer ──────────────────────────────────
def retrieval_recall_score(item: dict[str, Any], trace: dict[str, Any]) -> float:
    """1.0 if retrieved context contains expected citations/keywords, 0.0 otherwise."""
    must_cite = item.get("must_cite", [])
    if not must_cite:
        return 1.0

    final_resp = str(trace.get("final_response", "")).lower()
    retrieved = str(trace.get("retrieved_context", "")).lower()
    combined = final_resp + " " + retrieved

    found = sum(1 for cite in must_cite if cite.lower() in combined)
    return found / len(must_cite) if must_cite else 1.0


# ── 3. Deterministic Guardrail Compliance Scorer ──────────────────────────────
def guardrail_compliance_score(item: dict[str, Any], trace: dict[str, Any]) -> float:
    """1.0 if output satisfies prohibited terms and mandatory citations, 0.0 otherwise."""
    final_resp = str(trace.get("final_response", "")).lower()
    must_not_contain = item.get("must_not_contain", [])
    must_cite = item.get("must_cite", [])

    # Check prohibited terms
    for term in must_not_contain:
        if term.lower() in final_resp:
            return 0.0

    # Check mandatory citations
    for cite in must_cite:
        if cite.lower() not in final_resp:
            return 0.0

    return 1.0


# ── 4. LLM Judge Correctness Scorer ────────────────────────────────────────────
def deepeval_correctness_score(item: dict[str, Any], trace: dict[str, Any]) -> float:
    """LLM Judge correctness score comparing final answer against ground truth expected answer."""
    actual = str(trace.get("final_response", "")).strip()
    expected = str(item.get("expected", "")).strip()

    if not actual:
        return 0.0

    # Attempt DeepEval GEval rubric evaluation if available
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import GEval, MetricParams
        from deepeval.models import LiteLLMModel
        import os

        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            model = LiteLLMModel(model="gemini/gemini-2.5-flash", api_key=gemini_key)
            test_case = LLMTestCase(
                input=item.get("question", ""),
                actual_output=actual,
                expected_output=expected,
            )
            metric = GEval(
                name="Correctness",
                criteria="Determine if the actual output matches the factual and scope requirements of the expected output.",
                evaluation_params=[MetricParams.INPUT, MetricParams.ACTUAL_OUTPUT, MetricParams.EXPECTED_OUTPUT],
                model=model,
                threshold=0.5,
            )
            metric.measure(test_case)
            return float(metric.score)
    except Exception as exc:
        logger.debug("DeepEval judge fallback notice: %s", exc)

    # Heuristic fallback judge when DeepEval offline
    actual_lower = actual.lower()
    expected_lower = expected.lower()

    keywords = [w for w in expected_lower.split() if len(w) > 4]
    if not keywords:
        return 1.0

    hits = sum(1 for kw in keywords if kw in actual_lower)
    return round(hits / len(keywords), 2)
