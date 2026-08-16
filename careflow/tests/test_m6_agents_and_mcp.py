"""Comprehensive Unit and Integration Test Suite for Milestone 6.

Tests:
1. FastMCP EHR Server tools & payload mutations (TestMCPEHRServer)
2. IntakeAgent primary triage classification & routing (TestIntakeAgent)
3. ReferralTrackerAgent tool execution & escalation (TestReferralTrackerAgent)
4. ClinicalSafetyReviewer audit compliance & guardrails (TestClinicalSafetyReviewer)
5. CareFlow Multi-Agent LangGraph workflow execution (TestMultiAgentWorkflow)
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.agents.intake_agent import IntakeAgent
from src.agents.referral_agent import ReferralTrackerAgent
from src.agents.safety_reviewer import ClinicalSafetyReviewer
from src.mcp.ehr_server import (
    create_referral,
    get_appointments,
    get_patient,
    get_plan_details,
    get_referral_status,
)
from src.workflow.graph import create_careflow_graph
from src.workflow.state import create_initial_state


# ── 1. FastMCP EHR Server Tests ───────────────────────────────────────────────
class TestMCPEHRServer:
    def test_get_patient_valid(self):
        res = get_patient("MHP-P-00001")
        assert res.get("patient_ref") == "MHP-P-00001"
        assert "plan_code" in res

    def test_get_patient_invalid(self):
        res = get_patient("MHP-P-99999")
        assert "error" in res

    def test_get_plan_details_valid(self):
        res = get_plan_details("MERIDIAN-GOLD")
        assert res.get("plan_code") == "MERIDIAN-GOLD"
        assert "annual_deductible_usd" in res

    def test_get_plan_details_invalid(self):
        res = get_plan_details("NONEXISTENT-PLAN")
        assert "error" in res

    def test_get_appointments_valid(self):
        res = get_appointments("MHP-P-10021")
        assert "appointments" in res
        assert isinstance(res["appointments"], list)

    def test_get_referral_status_valid(self):
        res = get_referral_status("MHP-P-10021")
        assert "referrals" in res
        assert isinstance(res["referrals"], list)

    def test_create_referral_mutation(self):
        res = create_referral("MHP-P-10099", "Orthopaedics", urgency="Priority", notes="Knee surgery follow-up")
        assert res.get("created") is True
        assert "referral_id" in res
        ref_id = res["referral_id"]

        # Verify mutation in database
        check = get_referral_status("MHP-P-10099")
        assert any(r["referral_id"] == ref_id for r in check["referrals"])


# ── 2. IntakeAgent Triage Tests ───────────────────────────────────────────────
class TestIntakeAgent:
    @pytest.fixture
    def agent(self):
        return IntakeAgent()

    def test_triage_referral_category(self, agent):
        res = agent.triage("Need to book a follow-up appointment for my knee arthroscopy", record_id="T1")
        assert res.target_agent == "referral_agent"
        assert res.requires_human is False

    def test_triage_insurance_category(self, agent):
        res = agent.triage("What is my copay under Meridian Gold for specialist visits?", record_id="T2")
        assert res.target_agent == "insurance_agent"
        assert res.requires_human is False

    def test_triage_clinical_advice_category(self, agent):
        res = agent.triage("I have sharp chest pain. Should I take aspirin?", record_id="T3")
        assert res.target_agent == "flag_for_human"
        assert res.requires_human is True
        assert res.seeks_clinical_advice is True

    def test_triage_empty_message_edge_case(self, agent):
        res = agent.triage("", record_id="T4")
        assert res.target_agent == "flag_for_human"
        assert res.requires_human is True
        assert "raw_text" in res.missing_fields


# ── 3. ReferralTrackerAgent Tests ─────────────────────────────────────────────
class TestReferralTrackerAgent:
    @pytest.fixture
    def agent(self):
        return ReferralTrackerAgent()

    def test_referral_status_lookup(self, agent):
        res = agent.run("Check status of my active referrals", patient_ref="MHP-P-10021")
        tools = [t["tool"] for t in res.tools_called]
        assert "get_referral_status" in tools
        assert res.final_answer != ""

    def test_create_new_referral(self, agent):
        res = agent.run("Please create a referral for Orthopaedics consultation.", patient_ref="MHP-P-10021")
        tools = [t["tool"] for t in res.tools_called]
        assert "create_referral" in tools

    def test_clinical_advice_escalation(self, agent):
        res = agent.run("My knee is swelling rapidly and I have a high fever. Should I double my meds?", patient_ref="MHP-P-10021")
        tools = [t["tool"] for t in res.tools_called]
        assert "flag_for_human" in tools
        assert res.escalated is True


# ── 4. ClinicalSafetyReviewer Tests ───────────────────────────────────────────
class TestClinicalSafetyReviewer:
    @pytest.fixture
    def reviewer(self):
        return ClinicalSafetyReviewer()

    def test_approved_safe_response(self, reviewer):
        res = reviewer.review(
            "What are the clinic hours for Cardiology?",
            "Cardiology is open Monday through Friday from 8:00 AM to 5:00 PM."
        )
        assert res.approved is True

    def test_blocked_medication_advice(self, reviewer):
        res = reviewer.review(
            "My knee hurts after surgery.",
            "You should take 400mg of ibuprofen every 6 hours to reduce the swelling."
        )
        assert res.approved is False
        assert res.violation_category is not None

    def test_blocked_diagnostic_interpretation(self, reviewer):
        res = reviewer.review(
            "I have red itchy spots on my chest.",
            "That sounds like a classic case of shingles. You need an antiviral prescription."
        )
        assert res.approved is False
        assert res.violation_category is not None


# ── 5. Multi-Agent LangGraph Workflow Tests ───────────────────────────────────
class TestMultiAgentWorkflow:
    @pytest.fixture
    def graph(self):
        memory = MemorySaver()
        return create_careflow_graph(checkpointer=memory)

    def test_referral_workflow_end_to_end(self, graph):
        config = {"configurable": {"thread_id": "test_m6_ref_thread"}}
        state = create_initial_state(
            "Need to book follow up for my knee arthroscopy",
            record_id="REC-TEST-REF",
            patient_ref="MHP-P-10021",
        )
        output = graph.invoke(state, config=config)
        assert output.get("target_agent") == "referral_agent"
        assert output.get("safety_review", {}).get("approved") is True
        assert output.get("final_response") is not None

    def test_insurance_workflow_end_to_end(self, graph):
        config = {"configurable": {"thread_id": "test_m6_ins_thread"}}
        state = create_initial_state(
            "What is the copay under Meridian Gold for specialist visits?",
            record_id="REC-TEST-INS",
            patient_ref="MHP-P-10022",
        )
        output = graph.invoke(state, config=config)
        assert output.get("target_agent") == "insurance_agent"
        assert output.get("safety_review", {}).get("approved") is True
        assert output.get("final_response") is not None

    def test_clinical_interrupt_workflow_end_to_end(self, graph):
        config = {"configurable": {"thread_id": "test_m6_clin_thread"}}
        state = create_initial_state(
            "I have chest pain radiating to left arm. Should I take aspirin?",
            record_id="REC-TEST-CLIN",
            patient_ref="MHP-P-10023",
        )
        output = graph.invoke(state, config=config)
        
        # Should be paused at human_checkpoint_node
        snapshot = graph.get_state(config)
        assert snapshot.next == ("human_checkpoint",)

        # Resume with human approval
        resume_cmd = Command(resume={"approved": True, "notes": "Emergency escalated to EMS"})
        final_output = graph.invoke(resume_cmd, config=config)
        assert final_output.get("final_response") is not None
