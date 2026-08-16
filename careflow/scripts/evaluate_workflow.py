"""Evaluation script for Milestone 5: LangGraph Workflow & Human-in-the-Loop.

Runs the compiled CareFlow workflow against the first 20 intake records in
data/intake/records.jsonl and verifies routing accuracy against ground_truth.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.workflow.graph import create_careflow_graph
from src.workflow.state import create_initial_state

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("evaluate_workflow")

RECORDS_PATH = Path("data/intake/records.jsonl")


def evaluate_workflow(limit: int = 20):
    if not RECORDS_PATH.exists():
        logger.error("Records file %s does not exist", RECORDS_PATH)
        return

    records = []
    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= limit:
                    break

    logger.info("Loaded %d records for Milestone 5 workflow evaluation.", len(records))

    memory = MemorySaver()
    graph = create_careflow_graph(checkpointer=memory)

    results = []

    for i, rec in enumerate(records, 1):
        record_id = rec.get("record_id", f"REC-{i:05d}")
        raw_text = rec["raw_text"]
        patient_ref = rec.get("patient_ref")
        gt = rec.get("ground_truth", {})

        expected_clinical = gt.get("seeks_clinical_advice", False)
        expected_urgency = gt.get("urgency", "Routine")
        expected_requires_human = expected_clinical or (expected_urgency == "Emergent")

        thread_id = f"eval_thread_{record_id}"
        config = {"configurable": {"thread_id": thread_id}}

        logger.info("\n[%d/%d] Testing %s (gt_urgency=%s, gt_clinical=%s)",
                    i, len(records), record_id, expected_urgency, expected_clinical)

        state = create_initial_state(raw_text, record_id=record_id, patient_ref=patient_ref)

        # 1. Initial graph execution
        output = graph.invoke(state, config=config)
        state_snapshot = graph.get_state(config)

        interrupted = bool(state_snapshot.next and "human_checkpoint" in state_snapshot.next)

        # 2. Handle interrupt if paused
        if interrupted:
            logger.info("[%s] ⏸️ Workflow PAUSED at human_checkpoint", record_id)

            # Simulate human care coordinator decision:
            # If clinical advice requested -> Reject (route to escalation refusal)
            # If administrative emergent/priority -> Approve with notes
            approve = not expected_clinical
            notes = "Escalating to clinical care team" if not approve else "Approved for priority handling"

            resume_cmd = Command(resume={"approved": approve, "notes": notes})
            output = graph.invoke(resume_cmd, config=config)

        actual_requires_human = output.get("requires_human", False)
        final_resp = output.get("final_response", "")

        # Check routing match
        routing_correct = (interrupted == expected_requires_human) or (actual_requires_human == expected_requires_human)

        notes = []
        if not routing_correct:
            notes.append(f"Routing mismatch: got requires_human={actual_requires_human}, expected {expected_requires_human}")

        if not final_resp:
            routing_correct = False
            notes.append("Empty final response")

        results.append({
            "record_id": record_id,
            "channel": rec.get("channel"),
            "gt_urgency": expected_urgency,
            "gt_clinical": expected_clinical,
            "expected_hitl": expected_requires_human,
            "actual_hitl": interrupted or actual_requires_human,
            "routing_correct": routing_correct,
            "notes": "; ".join(notes) if notes else "OK",
            "snippet": final_resp[:100] + "..." if final_resp else "",
        })

    # Summary Report
    print("\n" + "=" * 85)
    print("MILESTONE 5 WORKFLOW EVALUATION REPORT (20 INTAKE RECORDS)")
    print("=" * 85)

    total = len(results)
    correct = sum(1 for r in results if r["routing_correct"])
    hitl_count = sum(1 for r in results if r["actual_hitl"])
    auto_count = total - hitl_count
    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"Total Intake Records Evaluated: {total}")
    print(f"Automated Path Records:       {auto_count} (Processed with 0 Human Interrupts)")
    print(f"Human Interrupt Path Records: {hitl_count} (Paused at human_checkpoint_node)")
    print(f"Routing Accuracy:             {accuracy:.1f}%\n")

    print(f"{'Record ID':<15} | {'Channel':<15} | {'Urgency':<10} | {'Expected HITL':<13} | {'Actual HITL':<11} | {'Status'}")
    print("-" * 85)
    for r in results:
        status = "✅ PASS" if r["routing_correct"] else "❌ FAIL"
        exp_str = "YES" if r["expected_hitl"] else "NO"
        act_str = "YES" if r["actual_hitl"] else "NO"
        print(f"{r['record_id']:<15} | {r['channel']:<15} | {r['gt_urgency']:<10} | {exp_str:<13} | {act_str:<11} | {status}")
    print("=" * 85)


if __name__ == "__main__":
    evaluate_workflow()
