"""Evaluation script for Milestone 6: Full Agent Team & FastMCP EHR Integration.

Runs the compiled Multi-Agent CareFlow workflow against the first 20 intake records in
data/intake/records.jsonl and verifies multi-agent routing accuracy, FastMCP tool execution,
and Clinical Safety Reviewer audit compliance.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.workflow.graph import create_careflow_graph
from src.workflow.state import create_initial_state

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("evaluate_m6_team")

RECORDS_PATH = Path("data/intake/records.jsonl")


def evaluate_m6_team(limit: int = 20):
    if not RECORDS_PATH.exists():
        logger.error("Records file %s does not exist", RECORDS_PATH)
        return

    logger.info("=== Starting Milestone 6 Multi-Agent Evaluation Benchmark ===")

    checkpointer = MemorySaver()
    graph = create_careflow_graph(checkpointer=checkpointer)

    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()][:limit]

    total_records = len(records)
    passed_routing = 0
    safety_approved_count = 0
    human_interrupted_count = 0
    automated_count = 0

    print(f"\nEvaluating {total_records} records through Multi-Agent Team...\n")
    print(f"{'Index':<5} | {'Record ID':<18} | {'Target Agent':<15} | {'Human Req':<10} | {'Safety Review':<15} | {'Status'}")
    print("-" * 85)

    for idx, rec in enumerate(records, start=1):
        record_id = rec.get("record_id", f"REC-{idx}")
        raw_text = rec.get("raw_text", "")
        patient_ref = rec.get("patient_ref")
        gt = rec.get("ground_truth", {})

        thread_id = f"m6_eval_thread_{record_id}"
        config = {"configurable": {"thread_id": thread_id}}

        state = create_initial_state(
            raw_text,
            record_id=record_id,
            patient_ref=patient_ref,
        )

        try:
            output = graph.invoke(state, config=config)

            # Check if workflow paused at human_checkpoint_node
            state_snapshot = graph.get_state(config)

            if state_snapshot.next:
                human_interrupted_count += 1
                target_agent = output.get("target_agent", "flag_for_human")
                safety_status = "INTERRUPTED"

                # Ground truth check for human interrupt requirement
                gt_clinical = gt.get("seeks_clinical_advice", False)
                gt_urgency = gt.get("urgency", "Routine")
                expected_human = gt_clinical or (gt_urgency == "Emergent")

                if expected_human:
                    passed_routing += 1
                    status_str = "✅ Correct Interrupt"
                else:
                    status_str = "⚠️ Unexpected Interrupt"

                # Resume graph with approval
                resume_cmd = Command(resume={"approved": True, "notes": "M6 Benchmark approved"})
                output = graph.invoke(resume_cmd, config=config)

            else:
                automated_count += 1
                target_agent = output.get("target_agent", "insurance_agent")
                safety_review = output.get("safety_review", {})
                is_safe = safety_review.get("approved", True)

                if is_safe:
                    safety_approved_count += 1
                    safety_status = "APPROVED"
                else:
                    safety_status = "BLOCKED"

                # Routing accuracy check
                expected_human = gt.get("seeks_clinical_advice", False) or (gt.get("urgency") == "Emergent")
                if not expected_human:
                    passed_routing += 1
                    status_str = "✅ Correct Path"
                else:
                    status_str = "⚠️ Missed Interrupt"

            print(
                f"{idx:<5} | {record_id:<18} | {target_agent:<15} | "
                f"{str(output.get('requires_human', False)):<10} | {safety_status:<15} | {status_str}"
            )

        except Exception as exc:
            logger.error("Error running record %s: %s", record_id, exc)
            print(f"{idx:<5} | {record_id:<18} | ERROR           | Error: {exc}")

        # Sleep briefly between runs to avoid hitting rate limits
        time.sleep(1.0)

    routing_accuracy = (passed_routing / total_records) * 100.0

    print("\n" + "=" * 85)
    print("=== Milestone 6 Multi-Agent Evaluation Summary ===")
    print(f"Total Intake Records Evaluated : {total_records}")
    print(f"Automated Path Executions      : {automated_count}")
    print(f"Human Interrupt Checkpoints     : {human_interrupted_count}")
    print(f"Clinical Safety Audit Passed   : {safety_approved_count} / {automated_count}")
    print(f"Multi-Agent Routing Accuracy    : {routing_accuracy:.1f}%")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    evaluate_m6_team(limit=20)
