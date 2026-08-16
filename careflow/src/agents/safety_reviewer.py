"""ClinicalSafetyReviewer — Post-processing safety critic for CareFlow (M6).

Inspects draft responses before delivery to patients. Blocks any response containing
unvetted clinical advice, diagnostic interpretations, or medication recommendations,
citing Clinical Escalation & Scope-of-Practice Standards.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

import litellm
from dotenv import load_dotenv

from src.utils.observability import record_llm_usage

load_dotenv()


litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_PROVIDER = os.getenv("LITELLM_PROVIDER", "gemini")
_MODEL_BASE = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_MODEL = f"{_PROVIDER}/{_MODEL_BASE}" if _PROVIDER != "openai" else _MODEL_BASE


# ── Response Dataclass ─────────────────────────────────────────────────────────
@dataclass
class SafetyReviewResult:
    """Result from ClinicalSafetyReviewer review."""
    approved: bool
    reason: str
    violation_category: str | None = None
    raw_response: dict | None = None


# ── System Prompt ──────────────────────────────────────────────────────────────
_SAFETY_SYSTEM_PROMPT = """You are Dr. Elara Vance, the Chief Clinical Safety Reviewer at Meridian Health Partners.
Your job is to audit draft messages written by administrative AI assistants BEFORE they are sent to patients.

YOU MUST STRICTLY ENFORCE THE CLINICAL ESCALATION AND SCOPE-OF-PRACTICE STANDARD:
1. NO DIAGNOSTIC INTERPRETATION: Assistants must NEVER interpret symptoms or suggest diagnoses (e.g., "sounds like angina", "might be shingles", "likely eczema").
2. NO MEDICATION GUIDANCE: Assistants must NEVER advise on taking, changing, or stopping medications or dosages (e.g., "take an aspirin", "increase your Lantus", "stop eating dairy").
3. NO CLINICAL TRIAGE: Assistants cannot judge medical seriousness — clinical queries MUST be escalated to human staff.
4. EMERGENCY MANDATE: Emergent symptoms (chest tightness, severe dyspnea, sudden neurological deficit) MUST be directed to 911 / Emergency Room.

RETURN ONLY VALID JSON WITH THIS EXACT STRUCTURE:
{
    "approved": true | false,
    "reason": "Detailed clinical reasoning for approval or rejection",
    "violation_category": "diagnostic_interpretation" | "medication_guidance" | "unvetted_advice" | null
}
"""


# ── ClinicalSafetyReviewer Class ───────────────────────────────────────────────
class ClinicalSafetyReviewer:
    """Post-processing safety reviewer (no tools, evaluator model)."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or _MODEL

    def review(
        self,
        patient_message: str,
        proposed_response: str,
    ) -> SafetyReviewResult:
        """Audit a proposed draft response against clinical safety guardrails."""
        logger.info("ClinicalSafetyReviewer auditing response...")

        user_prompt = (
            f"Original Patient Message:\n\"\"\"{patient_message}\"\"\"\n\n"
            f"Proposed Assistant Response:\n\"\"\"{proposed_response}\"\"\"\n\n"
            "Audit this response against Meridian Health Partners Clinical Scope-of-Practice Standards."
        )

        resp = None
        last_exc = None

        # Retry loop for rate limits
        for attempt in range(5):
            try:
                resp = litellm.completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SAFETY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                record_llm_usage("safety_reviewer", getattr(resp, "usage", None))
                break

            except Exception as exc:
                last_exc = exc
                if "429" in str(exc) or "RateLimit" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    wait_sec = 2 ** (attempt + 1) + 4
                    logger.warning("ClinicalSafetyReviewer hit rate limit (attempt %d/5). Retrying in %ds...", attempt + 1, wait_sec)
                    time.sleep(wait_sec)
                else:
                    break

        if resp is None:
            logger.error("ClinicalSafetyReviewer call failed: %s", last_exc)
            # Fail closed: block response if audit service fails
            return SafetyReviewResult(
                approved=False,
                reason=f"Safety review service error: {last_exc}",
                violation_category="service_error",
            )

        try:
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            approved = bool(data.get("approved", False))
            reason = str(data.get("reason", "No reasoning provided"))
            violation = data.get("violation_category")

            logger.info("Safety Review Result: approved=%s, category=%s", approved, violation)

            return SafetyReviewResult(
                approved=approved,
                reason=reason,
                violation_category=violation,
                raw_response=data,
            )
        except Exception as exc:
            logger.error("Failed to parse safety review JSON: %s", exc)
            return SafetyReviewResult(
                approved=False,
                reason=f"Failed to parse safety audit JSON: {exc}",
                violation_category="parse_error",
            )
