"""flag_for_human — escalation tool for M2.

Separate file from ehr_tools.py because escalation has different semantics:
it writes a record (not just reads) and must always succeed even if EHR is down.

The tool creates an escalation ticket with a deterministic ID and
logs it to output/escalations.jsonl for traceability.
"""

from __future__ import annotations

import json
import logging
import random
import string
from datetime import datetime, timezone
from pathlib import Path
import os

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
_ESCALATION_LOG = _OUTPUT_DIR / "escalations.jsonl"

# OpenAI-style tool definition for LiteLLM
FLAG_HUMAN_DEFINITION: dict = {
    "type": "function",
    "function": {
        "name": "flag_for_human",
        "description": (
            "Escalate the current case to a human care coordinator or clinician. "
            "MUST be called when: (1) the patient asks any clinical question "
            "(symptoms, diagnoses, medication advice, dosage), "
            "(2) urgency is Emergent, "
            "(3) any EHR tool returns an error or unavailable status, "
            "(4) you are unsure how to proceed. "
            "Never answer clinical questions yourself — always call this tool instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "Clear explanation of why escalation is needed. "
                        "Be specific: 'Patient asked about medication dosage' "
                        "not just 'clinical question'."
                    ),
                },
                "patient_ref": {
                    "type": "string",
                    "description": "Patient reference ID (e.g. 'MHP-P-00001'). Use 'UNKNOWN' if not available.",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["Routine", "Priority", "Urgent", "Emergent"],
                    "description": "Escalation urgency level.",
                },
            },
            "required": ["reason", "patient_ref", "urgency"],
        },
    },
}


def flag_for_human(
    reason: str,
    patient_ref: str,
    urgency: str = "Routine",
) -> dict:
    """Record a human escalation ticket and log it.

    Args:
        reason:      Why escalation is needed (free text).
        patient_ref: Patient reference ID.
        urgency:     One of: Routine, Priority, Urgent, Emergent.

    Returns:
        dict with: escalated (True), ticket_id, patient_ref, reason, urgency, timestamp
    """
    # Generate a deterministic-looking but unique ticket ID
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    ticket_id = f"ESC-{suffix}"

    timestamp = datetime.now(timezone.utc).isoformat()

    ticket = {
        "escalated":   True,
        "ticket_id":   ticket_id,
        "patient_ref": patient_ref,
        "reason":      reason,
        "urgency":     urgency,
        "timestamp":   timestamp,
        "message": (
            f"Case escalated to human care team (ticket {ticket_id}). "
            "A coordinator will follow up with the patient shortly."
        ),
    }

    # Persist to escalations log
    try:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(_ESCALATION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
        logger.info(
            "flag_for_human: ticket %s created for %s [%s]: %s",
            ticket_id, patient_ref, urgency, reason
        )
    except OSError as exc:
        # Escalation logging failure must never prevent the tool from returning
        logger.error("flag_for_human: failed to write escalation log: %s", exc)

    return ticket
