"""Unit and Integration Test Suite for Milestone 5: LangGraph Workflow & HITL.

Tests:
1. CareFlowState schema and helper functions (TestCareFlowState)
2. Individual workflow nodes in isolation (TestWorkflowNodes)
3. Graph routing logic and conditional edge functions (TestGraphRouting)
4. End-to-end stateful LangGraph workflow execution & HITL interrupts (TestCareFlowWorkflow)
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.workflow.graph import (
    create_careflow_graph,
    route_after_human,
    route_after_referral,
    route_after_safety,
)
from src.workflow.nodes import (
    eligibility_check_node,
    escalation_refusal_node,
    intake_parser_node,
    referral_router_node,
)
from src.workflow.state import CareFlowState, create_initial_state


# ── 1. CareFlowState Schema Tests ──────────────────────────────────────────────
class TestCareFlowState:
    def test_create_initial_state_defaults(self):
        state = create_initial_state("Hello, I need help with my insurance")
        assert state["record_id"] == "CAREFLOW-INIT"
        assert state["raw_text"] == "Hello, I need help with my insurance"
        assert state["patient_ref"] is None
        assert state["requires_human"] is False
        assert state["human_approved"] is None
        assert len(state["messages"]) == 1
        assert state["messages"][0]["role"] == "user"

    def test_create_initial_state_with_patient_ref(self):
        state = create_initial_state(
            "Check my copay",
            record_id="REC-100",
            patient_ref="MHP-P-00001",
        )
        assert state["record_id"] == "REC-100"
        assert state["patient_ref"] == "MHP-P-00001"


# ── 2. Individual Workflow Nodes Tests ─────────────────────────────────────────
class TestWorkflowNodes:
    def test_intake_parser_node(self):
        state = create_initial_state("What is my copay for specialist visit?", record_id="T-INTAKE")
        res = intake_parser_node(state)
        assert "patient_request" in res
        assert "target_agent" in res
        assert "referral_decision" in res
        assert res["target_agent"] in ("insurance_agent", "referral_agent", "flag_for_human")

    def test_eligibility_check_node_no_patient_ref(self):
        state = create_initial_state("Check copay without patient ID", record_id="T-ELIG-1")
        res = eligibility_check_node(state)
        assert res["eligibility"]["status"] == "skipped"

    def test_eligibility_check_node_valid_patient(self):
        state = create_initial_state("Check copay", record_id="T-ELIG-2", patient_ref="MHP-P-00001")
        res = eligibility_check_node(state)
        assert "eligibility" in res
        assert res["eligibility"].get("patient_ref") == "MHP-P-00001"

    def test_referral_router_node_routine(self):
        state = create_initial_state("Routine appointment check")
        state["patient_request"] = {
            "urgency": "Routine",
            "specialty": "General",
            "seeks_clinical_advice": False,
        }
        res = referral_router_node(state)
        assert res["requires_human"] is False

    def test_referral_router_node_emergent_triggers_human(self):
        state = create_initial_state("Severe trauma")
        state["patient_request"] = {
            "urgency": "Emergent",
            "specialty": "General",
            "seeks_clinical_advice": False,
        }
        res = referral_router_node(state)
        assert res["requires_human"] is True

    def test_referral_router_node_clinical_advice_triggers_human(self):
        state = create_initial_state("Chest pain meds")
        state["patient_request"] = {
            "urgency": "Routine",
            "specialty": "Cardiology",
            "seeks_clinical_advice": True,
        }
        res = referral_router_node(state)
        assert res["requires_human"] is True

    def test_escalation_refusal_node(self):
        state = create_initial_state("Request clinical advice", record_id="T-REFUSE")
        state["human_notes"] = "Clinician review required for safety."
        res = escalation_refusal_node(state)
        assert "final_response" in res
        assert "Clinician review required for safety." in res["final_response"]


# ── 3. Conditional Routing Edge Tests ──────────────────────────────────────────
class TestGraphRouting:
    def test_route_after_referral_human(self):
        state = CareFlowState(requires_human=True)
        assert route_after_referral(state) == "human_checkpoint"

    def test_route_after_referral_agent(self):
        state = CareFlowState(requires_human=False, target_agent="referral_agent")
        assert route_after_referral(state) == "referral_agent"

    def test_route_after_referral_insurance(self):
        state = CareFlowState(requires_human=False, target_agent="insurance_agent")
        assert route_after_referral(state) == "respond"

    def test_route_after_safety_approved(self):
        state = CareFlowState(safety_review={"approved": True}, requires_human=False)
        assert route_after_safety(state) == "end"

    def test_route_after_safety_blocked(self):
        state = CareFlowState(safety_review={"approved": False}, requires_human=False)
        assert route_after_safety(state) == "human_checkpoint"

    def test_route_after_human_approved(self):
        state = CareFlowState(human_approved=True)
        assert route_after_human(state) == "respond"

    def test_route_after_human_rejected(self):
        state = CareFlowState(human_approved=False)
        assert route_after_human(state) == "escalation_refusal"


# ── 4. End-to-End CareFlow Workflow Tests ─────────────────────────────────────
class TestCareFlowWorkflow:
    @pytest.fixture
    def graph(self):
        return create_careflow_graph(checkpointer=MemorySaver())

    def test_automated_workflow_path(self, graph):
        config = {"configurable": {"thread_id": "test_m5_auto_thread"}}
        state = create_initial_state(
            "What is my specialist copay under Meridian Gold?",
            record_id="M5-AUTO-01",
            patient_ref="MHP-P-00001",
        )
        output = graph.invoke(state, config=config)
        assert output.get("final_response") is not None
        assert output.get("requires_human") is False

    def test_hitl_interrupt_and_approve_flow(self, graph):
        config = {"configurable": {"thread_id": "test_m5_hitl_approve"}}
        state = create_initial_state(
            "I have chest tightness and dizziness. Should I take aspirin?",
            record_id="M5-HITL-01",
            patient_ref="MHP-P-00002",
        )
        # 1. Initial invoke pauses at human_checkpoint
        output = graph.invoke(state, config=config)
        snapshot = graph.get_state(config)
        assert snapshot.next == ("human_checkpoint",)

        # 2. Resume with approval
        resume_cmd = Command(resume={"approved": True, "notes": "Approved for priority care coordinator review"})
        final_output = graph.invoke(resume_cmd, config=config)
        assert final_output.get("human_approved") is True
        assert final_output.get("final_response") is not None

    def test_hitl_interrupt_and_reject_flow(self, graph):
        config = {"configurable": {"thread_id": "test_m5_hitl_reject"}}
        state = create_initial_state(
            "Can you tell me how to double my prescription dosage?",
            record_id="M5-HITL-02",
            patient_ref="MHP-P-00003",
        )
        # 1. Initial invoke pauses at human_checkpoint
        graph.invoke(state, config=config)
        snapshot = graph.get_state(config)
        assert snapshot.next == ("human_checkpoint",)

        # 2. Resume with rejection
        resume_cmd = Command(resume={"approved": False, "notes": "Medication adjustment requests must be handled via clinic call."})
        final_output = graph.invoke(resume_cmd, config=config)
        assert final_output.get("human_approved") is False
        assert "Medication adjustment requests must be handled via clinic call." in final_output.get("final_response", "")
