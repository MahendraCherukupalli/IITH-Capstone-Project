"""Mock EHR tool functions for M2.

These four functions simulate calls to a real EHR / insurance system.
They read from the local mock_api JSON files (no real database, no network).

Tools:
    check_insurance_eligibility(patient_ref)  → eligibility + plan info
    get_appointments(patient_ref)             → list of scheduled appointments
    calculate_copay(plan_code, visit_type)    → copay amount + deductible info
    get_referrals(patient_ref)                → open/scheduled referrals

All functions return plain dicts so they work as LiteLLM tool results.
They are also registered in TOOL_DEFINITIONS (OpenAI-style JSON Schema format)
so the agent can pass them directly to LiteLLM's `tools=` parameter.
"""

from __future__ import annotations

import json
import logging
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Data paths ─────────────────────────────────────────────────────────────────
_DATA_DIR = Path(os.getenv("DATA_DIR", "./data")) / "mock_api"


@lru_cache(maxsize=1)
def _load_patients() -> list[dict]:
    with open(_DATA_DIR / "patients.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_plans() -> list[dict]:
    with open(_DATA_DIR / "plans.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_appointments() -> list[dict]:
    with open(_DATA_DIR / "appointments.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_referrals() -> list[dict]:
    with open(_DATA_DIR / "referrals.json", encoding="utf-8") as f:
        return json.load(f)


# ── Copay rules for visit types not explicitly in the plan table ───────────────
# The plan table has specialist_copay_usd and telehealth_copay_usd.
# For imaging and lab we apply a simple multiplier per plan tier.
_IMAGING_MULTIPLIER = 2.5   # imaging copay = specialist_copay * multiplier
_LAB_MULTIPLIER     = 0.8   # lab copay = specialist_copay * multiplier

VALID_VISIT_TYPES = ("specialist", "telehealth", "imaging", "lab")


# ── Tool 1: check_insurance_eligibility ───────────────────────────────────────
def check_insurance_eligibility(patient_ref: str) -> dict[str, Any]:
    """Look up a patient's insurance eligibility by patient_ref.

    Args:
        patient_ref: Patient reference ID (e.g. "MHP-P-00001")

    Returns:
        dict with keys:
            plan_code, plan_name, member_number, eligibility_status,
            deductible_met_usd, annual_deductible_usd, out_of_pocket_max_usd
        or an error dict if patient not found.
    """
    patients = _load_patients()
    plans    = _load_plans()

    # Find patient
    patient = next(
        (p for p in patients if p["patient_ref"] == patient_ref), None
    )
    if patient is None:
        logger.warning("check_insurance_eligibility: patient %s not found", patient_ref)
        return {
            "error": "patient_not_found",
            "message": f"No patient found with patient_ref '{patient_ref}'. "
                       "Please verify the patient reference ID.",
        }

    # Find matching plan
    plan = next(
        (pl for pl in plans if pl["plan_code"] == patient["plan_code"]), None
    )
    if plan is None:
        logger.error(
            "check_insurance_eligibility: plan %s missing for patient %s",
            patient["plan_code"], patient_ref
        )
        return {
            "error": "plan_not_found",
            "message": f"Patient {patient_ref} has plan code '{patient['plan_code']}' "
                       "which is not in the plans table.",
        }

    deductible_met  = patient["deductible_met_usd"]
    annual_ded      = plan["annual_deductible_usd"]
    deductible_remaining = max(0, annual_ded - deductible_met)

    result = {
        "patient_ref":            patient_ref,
        "plan_code":              patient["plan_code"],
        "plan_name":              plan["plan_name"],
        "member_number":          patient["member_number"],
        "eligibility_status":     patient["eligibility_status"],   # active | pending_verification | lapsed
        "deductible_met_usd":     deductible_met,
        "annual_deductible_usd":  annual_ded,
        "deductible_remaining_usd": deductible_remaining,
        "out_of_pocket_max_usd":  plan["out_of_pocket_max_usd"],
        "coinsurance_rate":       plan["coinsurance_rate"],
    }

    logger.info(
        "check_insurance_eligibility: %s → %s (%s)",
        patient_ref, patient["plan_code"], patient["eligibility_status"]
    )
    return result


# ── Tool 2: get_appointments ──────────────────────────────────────────────────
def get_appointments(patient_ref: str) -> dict[str, Any]:
    """Look up all appointments for a patient.

    Args:
        patient_ref: Patient reference ID (e.g. "MHP-P-00001")

    Returns:
        dict with key "appointments" containing a list of appointment dicts,
        each with: appointment_id, specialty, scheduled_for, status, duration_minutes.
        Returns empty list if patient has no appointments.
    """
    appointments = _load_appointments()

    patient_appts = [
        {
            "appointment_id":   a["appointment_id"],
            "specialty":        a["specialty"],
            "scheduled_for":    a["scheduled_for"],
            "status":           a["status"],       # scheduled | completed | cancelled | no_show
            "duration_minutes": a["duration_minutes"],
            "provider_id":      a["provider_id"],
        }
        for a in appointments
        if a["patient_ref"] == patient_ref
    ]

    # Sort: upcoming scheduled first, then by date
    patient_appts.sort(key=lambda x: (x["status"] != "scheduled", x["scheduled_for"]))

    logger.info(
        "get_appointments: %s → %d appointment(s) found",
        patient_ref, len(patient_appts)
    )
    return {
        "patient_ref":  patient_ref,
        "appointments": patient_appts,
        "count":        len(patient_appts),
    }


# ── Tool 3: calculate_copay ───────────────────────────────────────────────────
def calculate_copay(plan_code: str, visit_type: str) -> dict[str, Any]:
    """Calculate the expected co-pay for a visit type under a given plan.

    Args:
        plan_code:  Insurance plan code (e.g. "MERIDIAN-GOLD")
        visit_type: One of: "specialist", "telehealth", "imaging", "lab"

    Returns:
        dict with: copay_usd, visit_type, plan_name, notes
        or error dict if plan_code or visit_type is invalid.
    """
    visit_type = visit_type.lower().strip()
    if visit_type not in VALID_VISIT_TYPES:
        return {
            "error": "invalid_visit_type",
            "message": (
                f"'{visit_type}' is not a recognised visit type. "
                f"Valid types: {', '.join(VALID_VISIT_TYPES)}"
            ),
        }

    plans = _load_plans()
    plan  = next((p for p in plans if p["plan_code"] == plan_code), None)

    if plan is None:
        # Try case-insensitive match
        plan = next(
            (p for p in plans if p["plan_code"].upper() == plan_code.upper()), None
        )

    if plan is None:
        return {
            "error": "plan_not_found",
            "message": (
                f"Plan code '{plan_code}' not found. "
                f"Valid plan codes: {', '.join(p['plan_code'] for p in plans)}"
            ),
        }

    specialist_copay = plan["specialist_copay_usd"]
    telehealth_copay = plan["telehealth_copay_usd"]

    copay_map = {
        "specialist": specialist_copay,
        "telehealth": telehealth_copay,
        "imaging":    round(specialist_copay * _IMAGING_MULTIPLIER),
        "lab":        round(specialist_copay * _LAB_MULTIPLIER),
    }

    copay_usd = copay_map[visit_type]

    notes_map = {
        "specialist": "Standard specialist co-pay applies after deductible met.",
        "telehealth": "Telehealth co-pay applies. No deductible required.",
        "imaging":    "Imaging co-pay estimate. Pre-authorisation may be required.",
        "lab":        "Standard lab co-pay. No referral required for routine labs.",
    }

    result = {
        "plan_code":    plan["plan_code"],
        "plan_name":    plan["plan_name"],
        "visit_type":   visit_type,
        "copay_usd":    copay_usd,
        "coinsurance_rate": plan["coinsurance_rate"],
        "annual_deductible_usd": plan["annual_deductible_usd"],
        "notes":        notes_map[visit_type],
    }
    logger.info(
        "calculate_copay: %s / %s → $%s",
        plan_code, visit_type, copay_usd
    )
    return result


# ── Tool 4: get_referrals ─────────────────────────────────────────────────────
def get_referrals(patient_ref: str) -> dict[str, Any]:
    """Look up all referrals for a patient.

    Args:
        patient_ref: Patient reference ID (e.g. "MHP-P-00001")

    Returns:
        dict with key "referrals" containing a list of referral dicts.
        Each referral has: referral_id, to_specialty, urgency, status,
        raised_on, preauth_required.
    """
    referrals = _load_referrals()

    patient_refs = [
        {
            "referral_id":      r["referral_id"],
            "from_specialty":   r["from_specialty"],
            "to_specialty":     r["to_specialty"],
            "urgency":          r["urgency"],
            "status":           r["status"],        # open | scheduled | accepted | rejected | closed
            "raised_on":        r["raised_on"],
            "preauth_required": r["preauth_required"],
        }
        for r in referrals
        if r["patient_ref"] == patient_ref
    ]

    # Open/scheduled referrals first
    patient_refs.sort(key=lambda x: x["status"] not in ("open", "scheduled"))

    logger.info(
        "get_referrals: %s → %d referral(s) found",
        patient_ref, len(patient_refs)
    )
    return {
        "patient_ref": patient_ref,
        "referrals":   patient_refs,
        "count":       len(patient_refs),
    }


# ── Tool registry (OpenAI-style function definitions for LiteLLM) ─────────────
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_insurance_eligibility",
            "description": (
                "Look up a patient's insurance eligibility, plan details, "
                "and deductible status using their patient reference ID. "
                "Call this before discussing co-pays or coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_ref": {
                        "type": "string",
                        "description": "Patient reference ID, e.g. 'MHP-P-00001'",
                    }
                },
                "required": ["patient_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_appointments",
            "description": (
                "Retrieve all appointments (past and upcoming) for a patient. "
                "Use this to check if the patient already has a scheduled visit "
                "or to review their appointment history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_ref": {
                        "type": "string",
                        "description": "Patient reference ID, e.g. 'MHP-P-00001'",
                    }
                },
                "required": ["patient_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_copay",
            "description": (
                "Calculate the expected co-pay amount for a specific visit type "
                "under a given insurance plan. Use this to answer patient questions "
                "about their expected out-of-pocket cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_code": {
                        "type": "string",
                        "description": (
                            "Insurance plan code. "
                            "One of: MERIDIAN-GOLD, MERIDIAN-SILVER, MERIDIAN-BRONZE, CIVIC-BASE"
                        ),
                    },
                    "visit_type": {
                        "type": "string",
                        "enum": ["specialist", "telehealth", "imaging", "lab"],
                        "description": "Type of visit to calculate co-pay for.",
                    },
                },
                "required": ["plan_code", "visit_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_referrals",
            "description": (
                "Retrieve all referrals for a patient — open, scheduled, and historical. "
                "Use this to check referral status or whether a pre-authorisation is required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_ref": {
                        "type": "string",
                        "description": "Patient reference ID, e.g. 'MHP-P-00001'",
                    }
                },
                "required": ["patient_ref"],
            },
        },
    },
]


# ── Tool dispatcher — maps tool name → function ───────────────────────────────
TOOL_MAP: dict[str, Any] = {
    "check_insurance_eligibility": check_insurance_eligibility,
    "get_appointments":            get_appointments,
    "calculate_copay":             calculate_copay,
    "get_referrals":               get_referrals,
}


def dispatch_tool(tool_name: str, tool_args: dict) -> dict:
    """Call the named tool with the given arguments.

    Args:
        tool_name: Name matching one of the keys in TOOL_MAP
        tool_args: Arguments dict to pass to the tool

    Returns:
        Tool result dict, or error dict if tool not found.
    """
    if tool_name not in TOOL_MAP:
        logger.error("dispatch_tool: unknown tool '%s'", tool_name)
        return {"error": "unknown_tool", "message": f"Tool '{tool_name}' is not registered."}
    try:
        result = TOOL_MAP[tool_name](**tool_args)
        return result
    except TypeError as exc:
        logger.error("dispatch_tool: bad args for '%s': %s", tool_name, exc)
        return {"error": "bad_arguments", "message": str(exc)}


def get_patient_profile(patient_ref: str) -> dict[str, Any]:
    """Retrieve full EHR patient profile for frontend UI consumption.

    Queries patients.json, plans.json, and referrals.json.
    """
    patients = _load_patients()
    plans = _load_plans()
    referrals = _load_referrals()

    clean_ref = patient_ref.strip().upper()
    patient = next((p for p in patients if p["patient_ref"].upper() == clean_ref), None)
    if patient is None:
        return {
            "error": "patient_not_found",
            "message": f"Patient '{patient_ref}' not found in Meridian EHR database."
        }

    plan = next((pl for pl in plans if pl["plan_code"] == patient["plan_code"]), None)
    active_refs = [
        r for r in referrals
        if r["patient_ref"].upper() == clean_ref and r.get("status") in ("open", "approved", "scheduled")
    ]

    given = patient.get("given_name", "Patient")
    family = patient.get("family_name", "")
    full_name = f"{given} {family}".strip()
    avatar = (given[0] + (family[0] if family else "")).upper()

    copay_str = "$25 / visit"
    if plan:
        spec_copay = plan.get("specialist_copay_usd")
        if spec_copay is not None:
            copay_str = f"${spec_copay} / visit"
        elif plan.get("coinsurance_rate"):
            copay_str = f"{int(plan['coinsurance_rate'] * 100)}% Coinsurance"

    deductible_total = plan.get("deductible_usd", 500.0) if plan else 500.0
    deductible_met = float(patient.get("deductible_met_usd", 0.0))
    pct_met = int(min(100, max(0, (deductible_met / max(1, deductible_total)) * 100)))

    # Dynamic age calculation from birth_date
    birth_date = patient.get("birth_date", "")
    age = 34
    if birth_date and len(birth_date) >= 4 and birth_date[:4].isdigit():
        age = max(18, 2026 - int(birth_date[:4]))

    ref_title = "None (All clear)"
    ref_status_text = "No active referrals"
    if active_refs:
        top_ref = active_refs[0]
        spec = top_ref.get("to_specialty", "Specialist")
        doctor = top_ref.get("provider", "Dr. Aris Thorne")
        ref_title = f"{spec} ({doctor})"
        st = (top_ref.get("status") or "APPROVED").upper()
        facility = top_ref.get("facility", "Metro Health Cardiology")
        ref_status_text = f"{st} • {facility}"

    return {
        "ref": patient["patient_ref"],
        "name": full_name,
        "avatar": avatar,
        "gender": (patient.get("gender") or "Female").capitalize(),
        "age": age,
        "plan": patient.get("plan_code", "UNKNOWN"),
        "plan_name": plan.get("plan_name", patient.get("plan_code")) if plan else patient.get("plan_code"),
        "status": (patient.get("eligibility_status") or "active").upper(),
        "copay": copay_str,
        "referrals": f"{len(active_refs)} Active",
        "member_number": patient.get("member_number", "N/A"),
        "deductible_text": f"${deductible_met:,.2f} / ${deductible_total:,.2f} ({pct_met}% Met)",
        "deductible_pct": pct_met,
        "bp": patient.get("vitals_bp", "118/76 mmHg"),
        "allergies": patient.get("allergies", "Penicillin (Moderate)"),
        "diagnosis": patient.get("diagnosis", "Mild Asthma, Mild Hypertension"),
        "lipid": patient.get("lipid_panel", "NORMAL (HDL 58 / LDL 92)"),
        "hba1c": patient.get("hba1c", "5.4% (Normal)"),
        "referral_title": ref_title,
        "referral_status": ref_status_text,
    }


def search_patients(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Search patient database records by patient_ref, given_name, or family_name."""
    if not query or not query.strip():
        return []

    q = query.strip().upper()
    patients = _load_patients()
    matches = []

    for p in patients:
        ref = p.get("patient_ref", "").upper()
        given = p.get("given_name", "").upper()
        family = p.get("family_name", "").upper()
        full_name = f"{given} {family}"

        if q in ref or q in given or q in family or q in full_name:
            matches.append({
                "ref": p["patient_ref"],
                "name": f"{p.get('given_name', '')} {p.get('family_name', '')}".strip(),
                "plan": p.get("plan_code", "UNKNOWN"),
                "status": (p.get("eligibility_status") or "active").upper(),
            })
            if len(matches) >= limit:
                break

    return matches
