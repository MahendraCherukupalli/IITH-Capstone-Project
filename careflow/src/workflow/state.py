"""State schema for CareFlow LangGraph workflow (M5).

Defines CareFlowState, a TypedDict representing the patient journey
and data accumulated across graph nodes.
"""

from __future__ import annotations

from typing import Any, TypedDict


class CareFlowState(TypedDict, total=False):
    """Typed state dictionary passed between nodes in the CareFlow graph.

    Fields:
        record_id:         Unique identifier for the intake record (e.g. CAREFLOW-00001).
        raw_text:          Raw intake message from patient (form, transcript, email, SMS).
        patient_ref:       Patient reference ID (e.g. MHP-P-10021).
        patient_request:   Parsed PatientRequest dict from M1 Pydantic parser.
        eligibility:       Insurance eligibility check result from M2 EHR tool.
        referral_decision: Dict containing specialty, urgency, and clinical advice flag.
        requires_human:    Boolean flag indicating whether human approval is required.
        human_approved:    Boolean indicating human approval/rejection decision on resume.
        human_notes:       Optional notes provided by the human reviewer on resume.
        messages:          List of message dictionaries representing conversation history.
        final_response:    Final response text returned to the patient.
    """

    record_id: str
    raw_text: str
    patient_ref: str | None
    patient_request: dict[str, Any] | None
    eligibility: dict[str, Any] | None
    referral_decision: dict[str, Any] | None
    target_agent: str | None
    safety_review: dict[str, Any] | None
    requires_human: bool
    human_approved: bool | None
    human_notes: str | None
    messages: list[dict[str, Any]]
    final_response: str | None
    citation: str | None
    citation_url: str | None
    execution_trace: list[dict[str, Any]]


def create_initial_state(
    raw_text: str,
    *,
    record_id: str = "CAREFLOW-INIT",
    patient_ref: str | None = None,
) -> CareFlowState:
    """Helper function to create a clean initial CareFlowState."""
    return CareFlowState(
        record_id=record_id,
        raw_text=raw_text,
        patient_ref=patient_ref,
        patient_request=None,
        eligibility=None,
        referral_decision=None,
        requires_human=False,
        human_approved=None,
        human_notes=None,
        messages=[{"role": "user", "content": raw_text}],
        final_response=None,
        citation=None,
        citation_url=None,
        execution_trace=[],
    )
