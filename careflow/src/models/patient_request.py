"""PatientRequest — M1 structured intake parser.

Parses raw patient messages (phone transcripts, web forms, emails, SMS) into
a validated Pydantic object using a provider-agnostic LiteLLM client.

Usage:
    from src.models.patient_request import PatientRequest, parse_intake_record

    record = {"raw_text": "Hi I need a cardiology appointment...", ...}
    result = parse_intake_record(record)
    print(result.patient_request)   # validated PatientRequest or None
    print(result.parse_errors)      # list of error strings from repair attempts
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from src.llm.client import complete_json

logger = logging.getLogger(__name__)

# ── Allowed specialties (must match corpus doc slugs) ─────────────────────────
SPECIALTIES = [
    "Cardiology",
    "Orthopaedics",
    "Dermatology",
    "Endocrinology",
    "Gastroenterology",
    "Pulmonology",
    "Neurology",
    "ENT",
]

# ── Pydantic model ─────────────────────────────────────────────────────────────

class PatientRequest(BaseModel):
    """Structured representation of one patient intake message.

    Fields mirror the `ground_truth` block in records.jsonl so we can measure
    parse accuracy against it during M1 testing.
    """

    record_id: str = Field(description="Original record ID from intake/records.jsonl")
    channel: Literal["phone_transcript", "web_form", "email", "sms"] = Field(
        description="How the message arrived."
    )
    raw_text: str = Field(description="The original unmodified patient message.")
    patient_ref: Optional[str] = Field(
        default=None, description="Patient reference ID (e.g. MHP-P-10021) if present."
    )
    insurance_id: Optional[str] = Field(
        default=None,
        description="Insurance plan code and/or member number as stated by patient. Null if not provided.",
    )
    urgency: Literal["Routine", "Priority", "Urgent", "Emergent"] = Field(
        description=(
            "Triage urgency level. "
            "Routine: standard scheduling. "
            "Priority: should be seen within 2-3 days. "
            "Urgent: needs same-day or next-day attention. "
            "Emergent: immediate risk to life or limb."
        )
    )
    specialty: Optional[str] = Field(
        default=None,
        description=(
            f"Target specialty. Must be one of: {SPECIALTIES}. "
            "Null if unclear from the message."
        ),
    )
    seeks_clinical_advice: bool = Field(
        description=(
            "True if the patient is asking for symptom interpretation, "
            "diagnosis, medication guidance, dosage advice, or any question "
            "phrased as 'should I'. False if they only want to book an appointment."
        )
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Field names that a downstream parser would find absent. "
            "Common values: 'insurance_id_stated', 'specialty', 'patient_ref'."
        ),
    )
    parsed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this record was parsed.",
    )


# ── Parse result wrapper ───────────────────────────────────────────────────────

class ParseResult(BaseModel):
    """Outcome of parsing one intake record."""

    record_id: str
    patient_request: Optional[PatientRequest] = None
    parse_errors: list[str] = Field(default_factory=list)
    attempts: int = 0
    success: bool = False


# ── Extraction prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a medical intake triage assistant for Meridian Health Partners,
a multi-specialty outpatient clinic. Your job is to extract structured information
from patient messages. You do NOT provide any clinical advice or diagnoses.
Respond only with the JSON object requested."""

_EXTRACTION_PROMPT_TEMPLATE = """Extract structured intake information from the patient message below.

Patient message details:
- record_id: {record_id}
- channel: {channel}
- patient_ref: {patient_ref}
- insurance_id_stated: {insurance_id_stated}

Raw message:
\"\"\"{raw_text}\"\"\"

Return a JSON object with EXACTLY these fields:
{{
  "record_id": "{record_id}",
  "channel": "{channel}",
  "raw_text": "<copy the raw message exactly>",
  "patient_ref": "<patient ref or null>",
  "insurance_id": "<insurance plan code and member number as stated, or null if not given>",
  "urgency": "<one of: Routine, Priority, Urgent, Emergent>",
  "specialty": "<one of: {specialties}, or null if unclear>",
  "seeks_clinical_advice": <true if patient asks for symptom interpretation/diagnosis/medication advice/dosage/"should I" questions, else false>,
  "missing_fields": ["<list field names that are absent, e.g. insurance_id_stated, specialty>"]
}}

Rules:
- urgency=Emergent: immediate risk to life (chest pain + arm numbness, severe bleeding, loss of consciousness)
- urgency=Urgent: same-day or next-day need (severe pain, worsening symptoms, high blood sugar readings)
- urgency=Priority: 2-3 day window (persistent symptoms, follow-up after procedure)
- urgency=Routine: standard scheduling, no time pressure
- seeks_clinical_advice=true: any question about symptoms, medications, dosages, diagnoses, "should I take/stop/change"
- specialty must be exactly one of the allowed values or null
- missing_fields: list any of [insurance_id_stated, specialty, patient_ref] that are absent or null

Return ONLY the JSON object, no markdown fences, no explanation."""


def _build_prompt(record: dict) -> str:
    return _EXTRACTION_PROMPT_TEMPLATE.format(
        record_id=record.get("record_id", "UNKNOWN"),
        channel=record.get("channel", ""),
        patient_ref=record.get("patient_ref", "null"),
        insurance_id_stated=record.get("insurance_id_stated") or "null",
        raw_text=record.get("raw_text", ""),
        specialties=", ".join(SPECIALTIES),
    )


# ── Parser with repair loop ────────────────────────────────────────────────────

def parse_intake_record(record: dict, max_retries: int = 2) -> ParseResult:
    """Parse one intake record dict into a validated PatientRequest.

    Follows the repair-loop pattern from the M1 lab:
    1. Extract raw JSON from LLM
    2. Validate against PatientRequest schema
    3. On failure, send the error back and retry up to max_retries times

    Args:
        record:      One record dict from intake/records.jsonl
        max_retries: Number of repair attempts after initial failure

    Returns:
        ParseResult with .patient_request set on success, .parse_errors on failure
    """
    record_id = record.get("record_id", "UNKNOWN")
    result = ParseResult(record_id=record_id)
    prompt = _build_prompt(record)
    raw_json: dict | None = None

    for attempt in range(max_retries + 1):
        result.attempts = attempt + 1
        try:
            if attempt == 0:
                # Initial extraction
                raw_json = complete_json(prompt, system=_SYSTEM_PROMPT, temperature=0.1)
            else:
                # Repair: send validation error back to model
                last_error = result.parse_errors[-1]
                repair_prompt = (
                    f"The following JSON failed Pydantic validation with error:\n{last_error}\n\n"
                    f"Original patient message:\n\"\"\"{record.get('raw_text', '')}\"\"\"\n\n"
                    f"Previous JSON attempt:\n{json.dumps(raw_json, indent=2)}\n\n"
                    f"Return ONLY corrected JSON with exactly these fields: "
                    f"record_id, channel, raw_text, patient_ref, insurance_id, "
                    f"urgency, specialty, seeks_clinical_advice, missing_fields.\n"
                    f"urgency must be one of: Routine, Priority, Urgent, Emergent\n"
                    f"specialty must be one of: {', '.join(SPECIALTIES)} or null\n"
                    f"Return ONLY valid JSON, no markdown, no commentary."
                )
                raw_json = complete_json(repair_prompt, system=_SYSTEM_PROMPT, temperature=0.0)

            # Ensure record_id and channel are correct (model may drift on repair)
            if isinstance(raw_json, dict):
                raw_json["record_id"] = record_id
                raw_json["channel"] = record.get("channel", raw_json.get("channel", "web_form"))
                raw_json["raw_text"] = record.get("raw_text", raw_json.get("raw_text", ""))

            # Validate
            patient_req = PatientRequest(**raw_json)
            result.patient_request = patient_req
            result.success = True
            return result

        except ValidationError as exc:
            error_str = str(exc)
            result.parse_errors.append(f"Attempt {attempt + 1} ValidationError: {error_str}")
            logger.warning(
                "[%s] Attempt %d/%d — Pydantic validation failed: %s",
                record_id, attempt + 1, max_retries + 1, error_str[:200]
            )

        except Exception as exc:
            error_str = str(exc)
            result.parse_errors.append(f"Attempt {attempt + 1} Error: {error_str}")
            logger.warning(
                "[%s] Attempt %d/%d — Unexpected error: %s",
                record_id, attempt + 1, max_retries + 1, error_str[:200]
            )

    logger.error("[%s] Failed after %d attempts.", record_id, max_retries + 1)
    return result
