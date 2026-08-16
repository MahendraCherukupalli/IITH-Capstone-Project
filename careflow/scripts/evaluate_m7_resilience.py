"""Evaluation Script for Milestone 7: Observability, Telemetry & Resilience.

Runs patient intake messages through the CareFlow workflow, collects telemetry events
via src/utils/observability.py, and outputs the per-agent call count, total time,
average latency, and success rate dashboard.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.checkpoint.memory import MemorySaver

from src.utils.observability import RUN_EVENTS, build_dashboard, clear_events
from src.workflow.graph import create_careflow_graph
from src.workflow.state import create_initial_state

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("evaluate_m7_resilience")

SAMPLE_PROMPTS = [
    {"id": "M7-EVAL-01", "text": "What is my specialist copay under Meridian Gold?", "patient_ref": "MHP-P-00001"},
    {"id": "M7-EVAL-02", "text": "Need to check status of my orthopaedics referral", "patient_ref": "MHP-P-00002"},
    {"id": "M7-EVAL-03", "text": "I have chest tightness and left arm pain. Should I take aspirin?", "patient_ref": "MHP-P-00003"},
]


def run_m7_resilience_evaluation():
    logger.info("=== Starting Milestone 7 Observability & Telemetry Benchmark ===")

    clear_events()
    checkpointer = MemorySaver()
    graph = create_careflow_graph(checkpointer=checkpointer)

    from unittest.mock import MagicMock, patch
    from src.agents.insurance_agent import AgentResponse
    from src.agents.intake_agent import IntakeTriageResult
    from src.agents.safety_reviewer import SafetyReviewResult

    mock_triage = IntakeTriageResult(
        record_id="M7-EVAL-01",
        patient_ref="MHP-P-00001",
        category="insurance",
        target_agent="insurance_agent",
        urgency="Routine",
        specialty="General",
        seeks_clinical_advice=False,
        requires_human=False,
        parsed_request={"urgency": "Routine", "specialty": "General", "seeks_clinical_advice": False},
    )
    mock_resp = AgentResponse(
        final_answer="Specialist copay under Meridian Gold is $25 per visit.",
        tools_called=[{"tool": "get_plan_details", "args": {"plan_code": "MERIDIAN-GOLD"}}],
        escalated=False,
    )
    mock_safety = SafetyReviewResult(approved=True, reason="Verified administrative response.")

    with patch("src.workflow.nodes._get_intake_agent") as mock_get_intake, \
         patch("src.workflow.nodes._get_insurance_agent") as mock_get_ins, \
         patch("src.workflow.nodes._get_safety_reviewer") as mock_get_safe:

        mock_intake = MagicMock()
        mock_intake.triage.return_value = mock_triage
        mock_get_intake.return_value = mock_intake

        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_resp
        mock_get_ins.return_value = mock_agent

        mock_reviewer = MagicMock()
        mock_reviewer.review.return_value = mock_safety
        mock_get_safe.return_value = mock_reviewer

        for i, case in enumerate(SAMPLE_PROMPTS, start=1):
            record_id = case["id"]
            raw_text = case["text"]
            patient_ref = case.get("patient_ref")

            logger.info("\n[%d/%d] Running Telemetry Evaluation for %s: '%s'", i, len(SAMPLE_PROMPTS), record_id, raw_text[:50])

            state = create_initial_state(raw_text, record_id=record_id, patient_ref=patient_ref)
            config = {"configurable": {"thread_id": f"m7_thread_{record_id}"}}

            try:
                output = graph.invoke(state, config=config)
                logger.info("[%s] Result generated: %s...", record_id, str(output.get("final_response", ""))[:60])
            except Exception as exc:
                logger.error("[%s] Workflow failed: %s", record_id, exc)

    # Generate telemetry dashboard
    df = build_dashboard(RUN_EVENTS)

    print("\n" + "=" * 80)
    print("=== MILESTONE 7 TELEMETRY & OBSERVABILITY DASHBOARD ===")
    print("=" * 80)
    print(f"Total Telemetry Events Logged: {len(RUN_EVENTS)}")
    print("-" * 80)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No telemetry events logged.")
    print("=" * 80 + "\n")

    # Flush events live to Langfuse Cloud
    try:
        from langfuse import Langfuse
        lf = Langfuse()
        if lf.auth_check():
            print("Flushing telemetry events live to your Langfuse Cloud dashboard...")
            lf.flush()
            print("✅ Telemetry traces successfully flushed to https://us.cloud.langfuse.com!")
    except Exception as exc:
        logger.warning("Langfuse flush notice: %s", exc)


if __name__ == "__main__":
    run_m7_resilience_evaluation()

