"""Graph nodes for CareFlow LangGraph workflow (M5).

Implements nodes for:
1. intake_parser_node         (M1 Pydantic parser)
2. eligibility_check_node     (M2 EHR tools)
3. referral_router_node       (Urgency & Clinical Routing)
4. human_checkpoint_node     (LangGraph interrupt)
5. respond_node               (InsuranceAgent execution)
6. escalation_refusal_node   (Coordinator rejection output)
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from src.agents.insurance_agent import InsuranceAgent
from src.agents.intake_agent import IntakeAgent
from src.agents.referral_agent import ReferralTrackerAgent
from src.agents.safety_reviewer import ClinicalSafetyReviewer
from src.models.patient_request import parse_intake_record
from src.tools.ehr_tools import check_insurance_eligibility
from src.utils.observability import traced, traced_generation
from src.workflow.state import CareFlowState


logger = logging.getLogger(__name__)


# ── Shared Agent Instances ─────────────────────────────────────────────────────
_insurance_agent = None
_intake_agent = None
_referral_agent = None
_safety_reviewer = None


def _get_insurance_agent() -> InsuranceAgent:
    global _insurance_agent
    if _insurance_agent is None:
        _insurance_agent = InsuranceAgent()
    return _insurance_agent

def _get_intake_agent() -> IntakeAgent:
    global _intake_agent
    if _intake_agent is None:
        _intake_agent = IntakeAgent()
    return _intake_agent

def _get_referral_agent() -> ReferralTrackerAgent:
    global _referral_agent
    if _referral_agent is None:
        _referral_agent = ReferralTrackerAgent()
    return _referral_agent

def _get_safety_reviewer() -> ClinicalSafetyReviewer:
    global _safety_reviewer
    if _safety_reviewer is None:
        _safety_reviewer = ClinicalSafetyReviewer()
    return _safety_reviewer


# ── 1. Intake Parser Node ──────────────────────────────────────────────────────
@traced("intake_parser")
def intake_parser_node(state: CareFlowState) -> dict[str, Any]:
    """Parse raw patient intake text and run IntakeAgent primary triage."""
    raw_text = state.get("raw_text", "")
    record_id = state.get("record_id", "CAREFLOW-INIT")
    patient_ref = state.get("patient_ref")
    logger.info("[%s] Node: intake_parser_node (IntakeAgent)", record_id)

    intake_agent = _get_intake_agent()
    triage_res = intake_agent.triage(raw_text, record_id=record_id, patient_ref=patient_ref)

    return {
        "patient_request": triage_res.parsed_request,
        "target_agent": triage_res.target_agent,
        "requires_human": triage_res.requires_human,
        "referral_decision": {
            "specialty": triage_res.specialty,
            "urgency": triage_res.urgency,
            "seeks_clinical_advice": triage_res.seeks_clinical_advice,
            "requires_human": triage_res.requires_human,
            "category": triage_res.category,
        },
    }


# ── 2. Eligibility Check Node ──────────────────────────────────────────────────
@traced("eligibility_check")
def eligibility_check_node(state: CareFlowState) -> dict[str, Any]:
    """Check insurance eligibility for patient_ref if available."""
    patient_ref = state.get("patient_ref")
    logger.info("[%s] Node: eligibility_check_node (patient_ref=%s)", state.get("record_id"), patient_ref)

    if not patient_ref:
        return {"eligibility": {"status": "skipped", "reason": "No patient_ref provided"}}

    try:
        res = check_insurance_eligibility(patient_ref)
        return {"eligibility": res}
    except Exception as exc:
        logger.error("Eligibility check failed for %s: %s", patient_ref, exc)
        return {"eligibility": {"status": "error", "error": str(exc)}}


# ── 3. Referral Router Node ────────────────────────────────────────────────────
@traced("referral_router")
def referral_router_node(state: CareFlowState) -> dict[str, Any]:
    """Evaluate request to determine routing, specialty, and human escalation requirement."""
    req = state.get("patient_request") or {}
    logger.info("[%s] Node: referral_router_node", state.get("record_id"))

    urgency = req.get("urgency", "Routine")
    specialty = req.get("specialty", "General")
    seeks_clinical_advice = req.get("seeks_clinical_advice", False)

    # Human checkpoint trigger logic:
    # 1. Any clinical advice query ("should I take", symptoms, dosage)
    # 2. Any Emergent case
    requires_human = seeks_clinical_advice or (urgency == "Emergent")

    decision = {
        "specialty": specialty,
        "urgency": urgency,
        "seeks_clinical_advice": seeks_clinical_advice,
        "requires_human": requires_human,
    }

    logger.info("Referral Decision: requires_human=%s (urgency=%s, clinical=%s)",
                requires_human, urgency, seeks_clinical_advice)

    return {
        "referral_decision": decision,
        "requires_human": requires_human,
    }


# ── 4. Human Checkpoint Node ───────────────────────────────────────────────────
@traced("human_checkpoint")
def human_checkpoint_node(state: CareFlowState) -> dict[str, Any]:
    """Pause graph using interrupt() to request human care coordinator approval."""
    record_id = state.get("record_id", "UNKNOWN")
    patient_ref = state.get("patient_ref", "UNKNOWN")
    decision = state.get("referral_decision") or {}

    logger.info("[%s] Node: human_checkpoint_node — PAUSING WORKFLOW", record_id)

    # Interrupt graph and yield context payload to the caller
    resume_signal = interrupt({
        "action": "human_review_required",
        "record_id": record_id,
        "patient_ref": patient_ref,
        "urgency": decision.get("urgency"),
        "specialty": decision.get("specialty"),
        "reason": "Clinical advice requested" if decision.get("seeks_clinical_advice") else "Emergent case urgency",
        "raw_text": state.get("raw_text"),
    })

    # Upon resume, parse the resume_signal payload
    logger.info("[%s] Graph RESUMED with signal: %s", record_id, resume_signal)

    approved = False
    notes = ""

    if isinstance(resume_signal, dict):
        approved = bool(resume_signal.get("approved", False))
        notes = str(resume_signal.get("notes", ""))
    elif isinstance(resume_signal, bool):
        approved = resume_signal

    return {
        "human_approved": approved,
        "human_notes": notes,
    }


# ── 5. Respond Node (Insurance Agent) ─────────────────────────────────────────
@traced_generation("respond")
def respond_node(state: CareFlowState) -> dict[str, Any]:
    """Run InsuranceAgent to generate insurance/policy patient response."""
    raw_text = state.get("raw_text", "")
    patient_ref = state.get("patient_ref")
    record_id = state.get("record_id")

    logger.info("[%s] Node: respond_node (InsuranceAgent)", record_id)

    agent = _get_insurance_agent()
    agent_resp = agent.run(raw_text, patient_ref=patient_ref)

    final_text = agent_resp.final_answer
    if state.get("human_notes"):
        final_text += f"\n\n[Care Coordinator Note: {state['human_notes']}]"

    return {
        "final_response": final_text,
        "citation": getattr(agent_resp, "citation", None),
        "citation_url": getattr(agent_resp, "citation_url", None),
    }


# ── 6. Referral Agent Node ─────────────────────────────────────────────────────
@traced_generation("referral_agent")
def referral_agent_node(state: CareFlowState) -> dict[str, Any]:
    """Run ReferralTrackerAgent to handle referral tracking/paperwork requests."""
    raw_text = state.get("raw_text", "")
    patient_ref = state.get("patient_ref")
    record_id = state.get("record_id")

    logger.info("[%s] Node: referral_agent_node (ReferralTrackerAgent)", record_id)

    agent = _get_referral_agent()
    agent_resp = agent.run(raw_text, patient_ref=patient_ref)

    final_text = agent_resp.final_answer
    if state.get("human_notes"):
        final_text += f"\n\n[Care Coordinator Note: {state['human_notes']}]"

    return {
        "final_response": final_text,
        "citation": getattr(agent_resp, "citation", None),
        "citation_url": getattr(agent_resp, "citation_url", None),
    }


# ── 7. Clinical Safety Reviewer Node ───────────────────────────────────────────
@traced_generation("safety_reviewer")
def safety_reviewer_node(state: CareFlowState) -> dict[str, Any]:

    """Audit proposed response using ClinicalSafetyReviewer."""
    record_id = state.get("record_id")
    raw_text = state.get("raw_text", "")
    final_response = state.get("final_response", "")

    logger.info("[%s] Node: safety_reviewer_node", record_id)

    reviewer = _get_safety_reviewer()
    review_res = reviewer.review(raw_text, final_response)

    review_dict = {
        "approved": review_res.approved,
        "reason": review_res.reason,
        "violation_category": review_res.violation_category,
    }

    # If safety reviewer blocks the response -> require human checkpoint
    requires_human = state.get("requires_human", False) or (not review_res.approved)

    return {
        "safety_review": review_dict,
        "requires_human": requires_human,
    }


# ── 8. Escalation Refusal Node ─────────────────────────────────────────────────
@traced("escalation_refusal")
def escalation_refusal_node(state: CareFlowState) -> dict[str, Any]:
    """Generate refusal / ticket escalation response when human coordinator rejects request."""
    record_id = state.get("record_id")
    logger.info("[%s] Node: escalation_refusal_node (Coordinator Rejected)", record_id)

    notes = state.get("human_notes", "Reviewed and routed to clinical operations.")
    response_text = (
        "Your request has been reviewed by a Meridian Care Coordinator and escalated to a licensed clinician. "
        f"Reason/Note: {notes}. A team member will contact you directly."
    )
    return {"final_response": response_text}

