"""FastMCP EHR & Scheduling Server (M6).

Exposes mock EHR data tables (patients, plans, appointments, referrals) over
the Model Context Protocol (MCP) using FastMCP.

Usage:
    from src.mcp.ehr_server import (
        get_patient,
        get_plan_details,
        get_appointments,
        get_referral_status,
        create_referral,
        mcp_server
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Initialize FastMCP Server
mcp_server = FastMCP("CareFlow EHR & Scheduling Server")

DATA_DIR = Path("data/mock_api")


def _load_json_data(filename: str) -> list[dict[str, Any]]:
    """Helper to load JSON tables from data/mock_api/."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        logger.warning("Mock EHR file %s not found", file_path)
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# In-memory storage tables
_patients_db: list[dict[str, Any]] = _load_json_data("patients.json")
_plans_db: list[dict[str, Any]] = _load_json_data("plans.json")
_appointments_db: list[dict[str, Any]] = _load_json_data("appointments.json")
_referrals_db: list[dict[str, Any]] = _load_json_data("referrals.json")


# ── MCP Tools & Resources ──────────────────────────────────────────────────────

@mcp_server.tool()
def get_patient(patient_ref: str) -> dict[str, Any]:
    """Retrieve patient demographic and plan information by patient reference ID."""
    if not patient_ref:
        return {"error": "patient_ref is required"}
    
    for p in _patients_db:
        if p.get("patient_ref") == patient_ref:
            return p
    return {"error": f"Patient reference {patient_ref} not found", "patient_ref": patient_ref}


@mcp_server.tool()
def get_plan_details(plan_code: str) -> dict[str, Any]:
    """Retrieve insurance plan cost-share rules, deductible, and OOP max by plan code."""
    if not plan_code:
        return {"error": "plan_code is required"}

    for plan in _plans_db:
        if plan.get("plan_code") == plan_code:
            return plan
    return {"error": f"Plan code {plan_code} not found", "plan_code": plan_code}


@mcp_server.tool()
def get_appointments(patient_ref: str) -> dict[str, Any]:
    """Retrieve past and upcoming appointments for a patient reference ID."""
    if not patient_ref:
        return {"error": "patient_ref is required", "appointments": []}

    matched = [a for a in _appointments_db if a.get("patient_ref") == patient_ref]
    return {
        "patient_ref": patient_ref,
        "count": len(matched),
        "appointments": matched,
    }


@mcp_server.tool()
def get_referral_status(patient_ref: str) -> dict[str, Any]:
    """Retrieve active specialty referrals for a patient reference ID."""
    if not patient_ref:
        return {"error": "patient_ref is required", "referrals": []}

    matched = [r for r in _referrals_db if r.get("patient_ref") == patient_ref]
    return {
        "patient_ref": patient_ref,
        "count": len(matched),
        "referrals": matched,
    }


@mcp_server.tool()
def create_referral(
    patient_ref: str,
    specialty: str,
    urgency: str = "Routine",
    notes: str = "",
) -> dict[str, Any]:
    """Create a new outpatient specialty referral record in the EHR system."""
    if not patient_ref:
        return {"error": "patient_ref is required", "created": False}
    if not specialty:
        return {"error": "specialty is required", "created": False}

    import random
    referral_id = f"REF-{random.randint(10000, 99999)}"

    new_ref = {
        "referral_id": referral_id,
        "patient_ref": patient_ref,
        "specialty": specialty,
        "urgency": urgency,
        "status": "pending_approval",
        "notes": notes,
    }

    _referrals_db.append(new_ref)
    logger.info("Created new referral %s for patient %s (%s)", referral_id, patient_ref, specialty)

    return {
        "created": True,
        "referral_id": referral_id,
        "patient_ref": patient_ref,
        "specialty": specialty,
        "status": "pending_approval",
        "urgency": urgency,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"FastMCP EHR Server initialized with {len(_patients_db)} patients, {len(_plans_db)} plans.")
