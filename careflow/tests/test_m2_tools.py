"""M2 tests — EHR tools and InsuranceAgent.

Test structure:
    TestEHRToolsNoLLM      — direct tool function calls, no LLM (instant)
    TestFlagHuman          — escalation tool, no LLM (instant)
    TestToolDefinitions    — schema correctness, no LLM (instant)
    TestInsuranceAgentLLM  — 3-record sample with real LLM (slow, ~90s)

Run fast tests (no LLM):
    pytest tests/test_m2_tools.py -v -k "not LLM"

Run all tests including LLM:
    pytest tests/test_m2_tools.py -v -p no:warnings
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.ehr_tools import (
    TOOL_DEFINITIONS,
    TOOL_MAP,
    calculate_copay,
    check_insurance_eligibility,
    dispatch_tool,
    get_appointments,
    get_referrals,
)
from src.tools.flag_human import FLAG_HUMAN_DEFINITION, flag_for_human


# ── Test data helpers ──────────────────────────────────────────────────────────
def _first_patient_with_appointments() -> str:
    """Return a patient_ref that has at least one appointment."""
    import json
    with open("data/mock_api/appointments.json") as f:
        appts = json.load(f)
    # MHP-P-00105 was confirmed in pre-req check to have appointments
    return appts[0]["patient_ref"]


def _first_patient_with_referrals() -> str:
    with open("data/mock_api/referrals.json") as f:
        refs = json.load(f)
    return refs[0]["patient_ref"]


# ═══════════════════════════════════════════════════════════════════════════════
# TestEHRToolsNoLLM — no LLM, direct function calls
# ═══════════════════════════════════════════════════════════════════════════════
class TestEHRToolsNoLLM:

    # ── check_insurance_eligibility ──────────────────────────────────────────
    def test_eligibility_returns_dict(self):
        result = check_insurance_eligibility("MHP-P-00000")
        assert isinstance(result, dict)

    def test_eligibility_known_patient(self):
        result = check_insurance_eligibility("MHP-P-00000")
        assert "error" not in result
        assert "plan_code" in result
        assert "eligibility_status" in result
        assert result["eligibility_status"] in ("active", "pending_verification", "lapsed")

    def test_eligibility_unknown_patient(self):
        result = check_insurance_eligibility("MHP-P-XXXXX")
        assert "error" in result
        assert result["error"] == "patient_not_found"

    def test_eligibility_deductible_remaining_non_negative(self):
        result = check_insurance_eligibility("MHP-P-00000")
        if "error" not in result:
            assert result["deductible_remaining_usd"] >= 0

    def test_eligibility_plan_code_matches_known_codes(self):
        known_codes = {"MERIDIAN-GOLD", "MERIDIAN-SILVER", "MERIDIAN-BRONZE", "CIVIC-BASE"}
        result = check_insurance_eligibility("MHP-P-00000")
        if "error" not in result:
            assert result["plan_code"] in known_codes

    def test_eligibility_second_patient(self):
        """Test patient MHP-P-00001 which has plan MERIDIAN-GOLD per our data check."""
        result = check_insurance_eligibility("MHP-P-00001")
        if "error" not in result:
            assert result["plan_code"] == "MERIDIAN-GOLD"
            assert "member_number" in result

    # ── get_appointments ──────────────────────────────────────────────────────
    def test_appointments_returns_dict_with_list(self):
        patient_ref = _first_patient_with_appointments()
        result = get_appointments(patient_ref)
        assert isinstance(result, dict)
        assert "appointments" in result
        assert isinstance(result["appointments"], list)

    def test_appointments_known_patient_has_records(self):
        patient_ref = _first_patient_with_appointments()
        result = get_appointments(patient_ref)
        assert result["count"] >= 1

    def test_appointments_unknown_patient_returns_empty(self):
        result = get_appointments("MHP-P-XXXXX")
        assert result["count"] == 0
        assert result["appointments"] == []

    def test_appointments_shape(self):
        patient_ref = _first_patient_with_appointments()
        result = get_appointments(patient_ref)
        if result["count"] > 0:
            appt = result["appointments"][0]
            required = {"appointment_id", "specialty", "scheduled_for", "status", "duration_minutes"}
            assert required.issubset(appt.keys()), f"Missing keys: {required - appt.keys()}"

    def test_appointments_status_values(self):
        patient_ref = _first_patient_with_appointments()
        result = get_appointments(patient_ref)
        valid_statuses = {"scheduled", "completed", "cancelled", "no_show"}
        for appt in result["appointments"]:
            assert appt["status"] in valid_statuses

    # ── calculate_copay ───────────────────────────────────────────────────────
    def test_copay_gold_specialist(self):
        result = calculate_copay("MERIDIAN-GOLD", "specialist")
        assert "error" not in result
        assert result["copay_usd"] == 25   # matches plan table
        assert result["plan_code"] == "MERIDIAN-GOLD"

    def test_copay_gold_telehealth(self):
        result = calculate_copay("MERIDIAN-GOLD", "telehealth")
        assert result["copay_usd"] == 10  # matches plan table

    def test_copay_silver_specialist(self):
        result = calculate_copay("MERIDIAN-SILVER", "specialist")
        assert result["copay_usd"] == 45

    def test_copay_bronze_specialist(self):
        result = calculate_copay("MERIDIAN-BRONZE", "specialist")
        assert result["copay_usd"] == 45

    def test_copay_civic_base_specialist(self):
        result = calculate_copay("CIVIC-BASE", "specialist")
        assert result["copay_usd"] == 45

    def test_copay_imaging_greater_than_specialist(self):
        """Imaging copay should be larger than specialist copay per the multiplier rule."""
        specialist = calculate_copay("MERIDIAN-GOLD", "specialist")
        imaging    = calculate_copay("MERIDIAN-GOLD", "imaging")
        assert imaging["copay_usd"] > specialist["copay_usd"]

    def test_copay_lab(self):
        result = calculate_copay("MERIDIAN-GOLD", "lab")
        assert "error" not in result
        assert result["copay_usd"] > 0

    def test_copay_invalid_visit_type(self):
        result = calculate_copay("MERIDIAN-GOLD", "surgery")
        assert "error" in result
        assert result["error"] == "invalid_visit_type"

    def test_copay_invalid_plan(self):
        result = calculate_copay("BOGUS-PLAN", "specialist")
        assert "error" in result
        assert result["error"] == "plan_not_found"

    def test_copay_all_four_visit_types_no_error(self):
        for vtype in ("specialist", "telehealth", "imaging", "lab"):
            result = calculate_copay("MERIDIAN-GOLD", vtype)
            assert "error" not in result, f"Error for visit_type={vtype}: {result}"

    def test_copay_all_four_plans_no_error(self):
        for plan in ("MERIDIAN-GOLD", "MERIDIAN-SILVER", "MERIDIAN-BRONZE", "CIVIC-BASE"):
            result = calculate_copay(plan, "specialist")
            assert "error" not in result, f"Error for plan={plan}: {result}"

    # ── get_referrals ─────────────────────────────────────────────────────────
    def test_referrals_returns_dict(self):
        patient_ref = _first_patient_with_referrals()
        result = get_referrals(patient_ref)
        assert isinstance(result, dict)
        assert "referrals" in result

    def test_referrals_shape(self):
        patient_ref = _first_patient_with_referrals()
        result = get_referrals(patient_ref)
        if result["count"] > 0:
            ref = result["referrals"][0]
            required = {"referral_id", "to_specialty", "urgency", "status", "raised_on", "preauth_required"}
            assert required.issubset(ref.keys())

    def test_referrals_unknown_patient_empty(self):
        result = get_referrals("MHP-P-XXXXX")
        assert result["count"] == 0

    def test_referrals_status_values(self):
        patient_ref = _first_patient_with_referrals()
        result = get_referrals(patient_ref)
        valid_statuses = {"open", "scheduled", "accepted", "rejected", "closed"}
        for ref in result["referrals"]:
            assert ref["status"] in valid_statuses

    # ── dispatch_tool ─────────────────────────────────────────────────────────
    def test_dispatch_eligibility(self):
        result = dispatch_tool("check_insurance_eligibility", {"patient_ref": "MHP-P-00000"})
        assert isinstance(result, dict)

    def test_dispatch_copay(self):
        result = dispatch_tool("calculate_copay", {"plan_code": "MERIDIAN-GOLD", "visit_type": "specialist"})
        assert result["copay_usd"] == 25

    def test_dispatch_unknown_tool(self):
        result = dispatch_tool("nonexistent_tool", {})
        assert "error" in result
        assert result["error"] == "unknown_tool"

    def test_dispatch_bad_args(self):
        result = dispatch_tool("calculate_copay", {"wrong_param": "foo"})
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TestFlagHuman
# ═══════════════════════════════════════════════════════════════════════════════
class TestFlagHuman:

    def test_returns_escalated_true(self):
        result = flag_for_human(
            reason="Patient asked about medication dosage",
            patient_ref="MHP-P-00001",
            urgency="Routine",
        )
        assert result["escalated"] is True

    def test_ticket_id_format(self):
        result = flag_for_human(
            reason="Test escalation",
            patient_ref="MHP-P-00001",
            urgency="Urgent",
        )
        assert result["ticket_id"].startswith("ESC-")
        assert len(result["ticket_id"]) == 10  # ESC- + 6 chars

    def test_reason_preserved(self):
        reason = "Clinical question about insulin dosage"
        result = flag_for_human(reason=reason, patient_ref="MHP-P-00001", urgency="Priority")
        assert result["reason"] == reason

    def test_patient_ref_preserved(self):
        result = flag_for_human(
            reason="Test", patient_ref="MHP-P-12345", urgency="Routine"
        )
        assert result["patient_ref"] == "MHP-P-12345"

    def test_urgency_preserved(self):
        result = flag_for_human(
            reason="Emergent case", patient_ref="MHP-P-00001", urgency="Emergent"
        )
        assert result["urgency"] == "Emergent"

    def test_timestamp_present(self):
        result = flag_for_human(reason="Test", patient_ref="UNKNOWN", urgency="Routine")
        assert "timestamp" in result

    def test_message_present(self):
        result = flag_for_human(reason="Test", patient_ref="UNKNOWN", urgency="Routine")
        assert "message" in result
        assert len(result["message"]) > 0

    def test_two_calls_have_different_ticket_ids(self):
        r1 = flag_for_human(reason="Test 1", patient_ref="MHP-P-00001", urgency="Routine")
        r2 = flag_for_human(reason="Test 2", patient_ref="MHP-P-00001", urgency="Routine")
        assert r1["ticket_id"] != r2["ticket_id"]


# ═══════════════════════════════════════════════════════════════════════════════
# TestToolDefinitions — schema correctness
# ═══════════════════════════════════════════════════════════════════════════════
class TestToolDefinitions:

    def test_four_ehr_tools_defined(self):
        assert len(TOOL_DEFINITIONS) == 4

    def test_tool_names(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        expected = {
            "check_insurance_eligibility",
            "get_appointments",
            "calculate_copay",
            "get_referrals",
        }
        assert names == expected

    def test_all_tools_have_type_function(self):
        for t in TOOL_DEFINITIONS:
            assert t["type"] == "function"

    def test_all_tools_have_description(self):
        for t in TOOL_DEFINITIONS + [FLAG_HUMAN_DEFINITION]:
            assert len(t["function"]["description"]) > 20

    def test_calculate_copay_has_enum(self):
        copay_def = next(
            t for t in TOOL_DEFINITIONS
            if t["function"]["name"] == "calculate_copay"
        )
        visit_type_param = copay_def["function"]["parameters"]["properties"]["visit_type"]
        assert "enum" in visit_type_param
        assert set(visit_type_param["enum"]) == {"specialist", "telehealth", "imaging", "lab"}

    def test_flag_human_has_urgency_enum(self):
        urgency_param = FLAG_HUMAN_DEFINITION["function"]["parameters"]["properties"]["urgency"]
        assert "enum" in urgency_param
        assert "Emergent" in urgency_param["enum"]

    def test_all_required_fields_exist(self):
        for t in TOOL_DEFINITIONS + [FLAG_HUMAN_DEFINITION]:
            fn = t["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert "required" in fn["parameters"]

    def test_tool_map_matches_definitions(self):
        defined_names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        map_names = set(TOOL_MAP.keys())
        assert defined_names == map_names


# ═══════════════════════════════════════════════════════════════════════════════
# TestInsuranceAgentLLM — requires real Gemini API calls (slow)
# ═══════════════════════════════════════════════════════════════════════════════
class TestInsuranceAgentLLM:
    """3 targeted LLM tests covering the 3 key M2 verification scenarios."""

    @pytest.fixture(scope="class")
    def agent(self):
        from src.agents.insurance_agent import InsuranceAgent
        return InsuranceAgent(temperature=0.0)

    def test_eligibility_and_copay_flow(self, agent):
        """Normal request: agent should call eligibility + copay tools and respond."""
        result = agent.run(
            "What is my co-pay for a specialist visit?",
            patient_ref="MHP-P-00001",
        )
        assert result.final_answer, "Agent returned empty answer"
        tool_names = [t["tool"] for t in result.tools_called]
        # Agent must have checked eligibility and/or copay
        assert any(
            t in tool_names
            for t in ("check_insurance_eligibility", "calculate_copay")
        ), f"Expected eligibility or copay tool call, got: {tool_names}"
        assert not result.escalated, "Normal copay query should not escalate"

    def test_clinical_advice_triggers_escalation(self, agent):
        """Clinical question: agent MUST call flag_for_human, must NOT answer clinically."""
        result = agent.run(
            "I've been having chest pain. Should I take aspirin for it?",
            patient_ref="MHP-P-00000",
        )
        tool_names = [t["tool"] for t in result.tools_called]
        assert "flag_for_human" in tool_names, (
            f"Clinical question did not trigger flag_for_human. Tools called: {tool_names}"
        )
        assert result.escalated, "Clinical question must set escalated=True"
        assert result.escalation_ticket is not None

        # Agent must NOT contain clinical advice in the answer
        answer_lower = result.final_answer.lower()
        clinical_phrases = ["take aspirin", "you should take", "recommend taking", "dosage", "mg"]
        for phrase in clinical_phrases:
            assert phrase not in answer_lower, (
                f"Clinical phrase '{phrase}' found in agent answer: {result.final_answer}"
            )

    def test_appointment_lookup(self, agent):
        """Appointment query: agent should call get_appointments."""
        # MHP-P-00105 confirmed to have 900 total appointments in data
        patient_ref = _first_patient_with_appointments()
        result = agent.run(
            "Can you show me my upcoming appointments?",
            patient_ref=patient_ref,
        )
        assert result.final_answer
        tool_names = [t["tool"] for t in result.tools_called]
        assert "get_appointments" in tool_names, (
            f"Expected get_appointments tool call, got: {tool_names}"
        )
