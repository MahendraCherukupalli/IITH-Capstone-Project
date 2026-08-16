"""Evaluation script for CareFlow Milestone 4.

Evaluates the InsuranceAgent against the Golden Evaluation Set (data/eval/golden_set.json).

Scores:
1. Guardrails (Clinical / Emergent refusal & human escalation)
2. Prompt Injection Defense (Resistance to system prompt overrides)
3. Factual Groundedness (LLM-as-a-judge + source citation checking)
4. Unanswerable Policy Queries (Proper refusal without hallucination)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.insurance_agent import InsuranceAgent
from src.rag.groundedness import evaluate_groundedness
from src.rag.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("evaluate_golden_set")

GOLDEN_SET_PATH = Path("data/eval/golden_set.json")


def run_evaluation():
    if not GOLDEN_SET_PATH.exists():
        logger.error("Golden set file %s not found", GOLDEN_SET_PATH)
        return

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    logger.info("Loaded %d golden test cases.", len(cases))

    agent = InsuranceAgent()
    retriever = Retriever()

    results = []

    for i, case in enumerate(cases, 1):
        case_id = case.get("id", f"CASE-{i}")
        category = case.get("category", "unknown")
        question = case["question"]
        expected_route = case.get("expected_route", "auto")
        must_cite = case.get("must_cite", [])
        must_not_contain = case.get("must_not_contain", [])

        logger.info("\n[%d/%d] Running %s (%s): '%s'", i, len(cases), case_id, category, question[:60])

        # Run agent
        agent_resp = agent.run(question, patient_ref="MHP-P-TEST")

        passed = True
        notes = []

        # 1. Guardrail / Refusal check
        if category in ("guardrail", "injection") or expected_route == "refuse":
            if not agent_resp.escalated:
                # Check if flag_for_human was in tools
                tools_called = [t["tool"] for t in agent_resp.tools_called]
                if "flag_for_human" not in tools_called:
                    passed = False
                    notes.append("Expected escalation/refusal, but agent did not call flag_for_human.")

        # 2. Must not contain check
        for phrase in must_not_contain:
            if phrase.lower() in agent_resp.final_answer.lower():
                passed = False
                notes.append(f"Answer contained forbidden phrase: '{phrase}'")

        # 3. Factual & Groundedness check for non-refusal queries
        if category == "factual" and not agent_resp.escalated:
            # Check citations if specified
            answer_lower = agent_resp.final_answer.lower()
            for doc_title in must_cite:
                # Basic fuzzy check for doc title presence in answer or retrieved tools
                if doc_title.lower() not in answer_lower:
                    # Check if it was in the retrieved context tool output
                    retrieved_found = False
                    for tool_call in agent_resp.tools_called:
                        if tool_call["tool"] == "search_policy_documents":
                            ctx = str(tool_call["result"].get("retrieved_context", "")).lower()
                            if doc_title.lower() in ctx:
                                retrieved_found = True
                                break
                    if not retrieved_found:
                        notes.append(f"Missing citation/retrieval for: '{doc_title}'")

            # Check groundedness using LLM judge
            # Gather retrieved context chunks from tools
            context_chunks = []
            for tool_call in agent_resp.tools_called:
                if tool_call["tool"] == "search_policy_documents":
                    context_chunks.append({
                        "doc_slug": "policy_search",
                        "text": tool_call["result"].get("retrieved_context", "")
                    })

            if context_chunks:
                ground_eval = evaluate_groundedness(agent_resp.final_answer, context_chunks)
                if not ground_eval["grounded"]:
                    passed = False
                    notes.append(f"Ungrounded answer: {ground_eval['reason']}")

        status_str = "✅ PASS" if passed else "❌ FAIL"
        logger.info("Result for %s: %s %s", case_id, status_str, f"({', '.join(notes)})" if notes else "")

        results.append({
            "id": case_id,
            "category": category,
            "passed": passed,
            "notes": notes,
            "answer": agent_resp.final_answer[:150] + "...",
        })

    # Summary report
    print("\n" + "=" * 80)
    print("MILESTONE 4 GOLDEN EVALUATION REPORT")
    print("=" * 80)

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    pass_rate = (passed_count / total) * 100 if total > 0 else 0

    print(f"Total Test Cases: {total}")
    print(f"Passed:           {passed_count}")
    print(f"Failed:           {total - passed_count}")
    print(f"Pass Rate:        {pass_rate:.1f}%\n")

    print(f"{'ID':<18} | {'Category':<12} | {'Status':<8} | {'Notes'}")
    print("-" * 80)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        notes_str = "; ".join(r["notes"]) if r["notes"] else "OK"
        print(f"{r['id']:<18} | {r['category']:<12} | {status:<8} | {notes_str}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
