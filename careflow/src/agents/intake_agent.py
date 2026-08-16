"""IntakeAgent — Primary triage agent for CareFlow (M6).

Parses raw patient intake messages across all channels (web, phone, email, SMS),
classifies intent, urgency, and specialty, and determines the appropriate
downstream agent or human escalation route.

Usage:
    from src.agents.intake_agent import IntakeAgent, IntakeTriageResult

    agent = IntakeAgent()
    result = agent.triage("Need to book follow up for my knee arthroscopy", patient_ref="MHP-P-10021")
    print(result.target_agent)  # "referral_agent" or "insurance_agent"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.models.patient_request import parse_intake_record

logger = logging.getLogger(__name__)


@dataclass
class IntakeTriageResult:
    """Result of IntakeAgent triage evaluation."""
    record_id: str
    patient_ref: str | None
    category: str
    target_agent: str
    urgency: str
    specialty: str
    seeks_clinical_advice: bool
    requires_human: bool
    missing_fields: list[str] = field(default_factory=list)
    parsed_request: dict[str, Any] = field(default_factory=dict)


class IntakeAgent:
    """Primary triage agent for classifying and routing intake messages."""

    def triage(
        self,
        raw_text: str,
        *,
        record_id: str = "CAREFLOW-INTAKE",
        patient_ref: str | None = None,
        channel: str = "web_form",
    ) -> IntakeTriageResult:
        """Triage an incoming patient message."""
        logger.info("[%s] IntakeAgent triaging message...", record_id)

        if not raw_text or not raw_text.strip():
            # Edge case: empty message
            return IntakeTriageResult(
                record_id=record_id,
                patient_ref=patient_ref,
                category="unknown",
                target_agent="flag_for_human",
                urgency="Routine",
                specialty="General",
                seeks_clinical_advice=False,
                requires_human=True,
                missing_fields=["raw_text"],
            )

        record_dict = {
            "record_id": record_id,
            "channel": channel,
            "raw_text": raw_text,
            "patient_ref": patient_ref,
        }

        parse_res = parse_intake_record(record_dict)
        req = parse_res.patient_request if (parse_res.success and parse_res.patient_request) else None

        if req:
            urgency = req.urgency
            specialty = req.specialty or "General"
            seeks_clinical = req.seeks_clinical_advice
            missing = req.missing_fields
        else:
            urgency = "Routine"
            specialty = "General"
            seeks_clinical = False
            missing = []

        # ── Routing Decision Matrix ────────────────────────────────────────────
        text_lower = raw_text.lower()

        # 1. Clinical emergency or clinical advice request -> flag_for_human
        if seeks_clinical or urgency == "Emergent":
            category = "clinical_advice" if seeks_clinical else "emergency"
            target_agent = "flag_for_human"
            requires_human = True

        # 2. Referral paperwork / tracking / appointment booking
        elif any(k in text_lower for k in ("referral", "referred", "book", "appointment", "follow up", "follow-up", "schedule")):
            category = "referral"
            target_agent = "referral_agent"
            requires_human = False

        # 3. Insurance eligibility / copay / coverage details
        elif any(k in text_lower for k in ("insurance", "copay", "co-pay", "coverage", "deductible", "cost", "claim")):
            category = "insurance"
            target_agent = "insurance_agent"
            requires_human = False

        # 4. Default specialty routing
        else:
            category = "insurance"
            target_agent = "insurance_agent"
            requires_human = False

        logger.info(
            "[%s] Triage result: category=%s, target_agent=%s, urgency=%s, requires_human=%s",
            record_id, category, target_agent, urgency, requires_human
        )

        return IntakeTriageResult(
            record_id=record_id,
            patient_ref=patient_ref or (req.patient_ref if req else None),
            category=category,
            target_agent=target_agent,
            urgency=urgency,
            specialty=specialty,
            seeks_clinical_advice=seeks_clinical,
            requires_human=requires_human,
            missing_fields=missing,
            parsed_request=req.model_dump() if req else {},
        )
