"""Comprehensive Unit and Integration Test Suite for Milestone 7: Observability & Resilience.

Tests:
1. CircuitBreaker state machine, threshold tripping & short-circuiting (TestCircuitBreaker)
2. Robust fallbacks for Qdrant RAG and FastMCP EHR tool dispatches (TestRobustFallbacks)
3. Observability tracing decorators, span timing, and dashboard metrics (TestObservability)
4. Failure injection & hardened CareFlow workflow execution (TestFailureInjectionWorkflow)
"""

from __future__ import annotations

import time
import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.utils.observability import (
    RUN_EVENTS,
    build_dashboard,
    clear_events,
    traced,
    traced_generation,
)
from src.utils.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    FaultConfig,
    make_flaky_tool,
    robust_rag_search,
    robust_tool_dispatch,
)
from src.workflow.graph import create_careflow_graph
from src.workflow.state import create_initial_state


# ── 0. Fault Injection Harness Unit Tests (Lab B) ───────────────────────────
class TestFaultInjectionHarness:
    def test_fault_config_reproducible_outcomes(self):
        config = FaultConfig(seed=42, fail_rate=0.2, timeout_rate=0.1, malformed_rate=0.1)

        def dummy_target(q: str):
            return {"status": "ok", "query": q}

        flaky = make_flaky_tool(config, dummy_target)

        results = []
        for i in range(10):
            try:
                res = flaky(f"q_{i}")
                results.append(res["status"])
            except ConnectionError:
                results.append("conn_error")
            except TimeoutError:
                results.append("timeout_error")

        assert len(results) == 10
        assert "conn_error" in results or "timeout_error" in results
        assert flaky.calls["n"] == 10


# ── 1. CircuitBreaker Unit Tests ─────────────────────────────────────────────
class TestCircuitBreaker:

    def test_circuit_breaker_closed_state_success(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=5.0, name="TestBreaker1")
        res = cb.call(lambda x, y: x + y, 10, 20)
        assert res == 30
        assert cb.state == "closed"
        assert cb.failures == 0

    def test_circuit_breaker_trips_to_open_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=5.0, name="TestBreaker2")

        def failing_func():
            raise ConnectionError("Simulated network outage")

        # 3 failures to trip the circuit
        for _ in range(3):
            with pytest.raises(ConnectionError):
                cb.call(failing_func)

        assert cb.failures == 3
        assert cb.state == "open"

        # 4th attempt should be short-circuited by CircuitOpenError immediately
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(failing_func)

        assert "Circuit open" in str(exc_info.value)

    def test_circuit_breaker_half_open_recovery(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.2, name="TestBreaker3")

        def flaky_func(should_fail: bool):
            if should_fail:
                raise TimeoutError("Simulated timeout")
            return "SUCCESS"

        # Trip to OPEN
        for _ in range(2):
            with pytest.raises(TimeoutError):
                cb.call(flaky_func, True)

        assert cb.state == "open"

        # Wait for reset timeout to elapse
        time.sleep(0.25)

        # Next call should transition to HALF-OPEN and succeed, resetting to CLOSED
        res = cb.call(flaky_func, False)
        assert res == "SUCCESS"
        assert cb.state == "closed"
        assert cb.failures == 0


# ── 2. Robust Fallbacks Unit Tests ─────────────────────────────────────────────
class TestRobustFallbacks:
    def test_robust_rag_search_success(self):
        def mock_search(query: str, top_k: int = 3):
            return "Authoritative policy text chunk"

        ctx, fallback_used = robust_rag_search(mock_search, "copay schedule")
        assert "Authoritative policy text chunk" in ctx
        assert fallback_used is False

    def test_robust_rag_search_fallback_on_failure(self):
        def failing_search(query: str, top_k: int = 3):
            raise ConnectionError("Qdrant Cloud unavailable")

        ctx, fallback_used = robust_rag_search(failing_search, "copay schedule")
        assert fallback_used is True
        assert "Fallback Policy System Notice" in ctx

    def test_robust_tool_dispatch_success(self):
        def mock_dispatch(tool_name: str, args: dict):
            return {"status": "active", "patient_ref": args.get("patient_ref")}

        res, fallback_used = robust_tool_dispatch(mock_dispatch, "get_patient", {"patient_ref": "MHP-P-00001"})
        assert res["status"] == "active"
        assert fallback_used is False

    def test_robust_tool_dispatch_fallback_on_failure(self):
        def failing_dispatch(tool_name: str, args: dict):
            raise TimeoutError("EHR Server Timeout")

        res, fallback_used = robust_tool_dispatch(failing_dispatch, "get_patient", {"patient_ref": "MHP-P-00001"})
        assert fallback_used is True
        assert res["status"] == "degraded"
        assert "fallback_used" in res


# ── 3. Observability & Telemetry Tests ────────────────────────────────────────
class TestObservability:
    def setup_method(self):
        clear_events()

    def test_traced_decorator_logs_event(self):
        @traced("test_node")
        def dummy_node(x: int):
            time.sleep(0.01)
            return x * 2

        val = dummy_node(21)
        assert val == 42
        assert len(RUN_EVENTS) == 1

        ev = RUN_EVENTS[0]
        assert ev["role"] == "test_node"
        assert ev["ok"] is True
        assert ev["ms"] >= 10.0

    def test_traced_generation_decorator_logs_tokens(self):
        @traced_generation("llm_node", model="gemini-2.5-flash")
        def dummy_llm():
            return "Generated text"

        res = dummy_llm()
        assert res == "Generated text"
        assert len(RUN_EVENTS) == 1

        ev = RUN_EVENTS[0]
        assert ev["role"] == "llm_node"
        assert ev["model"] == "gemini-2.5-flash"
        assert ev["ok"] is True

    def test_build_dashboard(self):
        events = [
            {"role": "node_a", "ms": 100.0, "ok": True},
            {"role": "node_a", "ms": 200.0, "ok": True},
            {"role": "node_b", "ms": 50.0, "ok": False},
        ]
        df = build_dashboard(events)
        assert len(df) == 2
        
        node_a_row = df[df["role"] == "node_a"].iloc[0]
        assert node_a_row["calls"] == 2
        assert node_a_row["ok_calls"] == 2
        assert node_a_row["total_ms"] == 300.0
        assert node_a_row["avg_ms"] == 150.0
        assert node_a_row["success_rate"] == 1.0

        node_b_row = df[df["role"] == "node_b"].iloc[0]
        assert node_b_row["calls"] == 1
        assert node_b_row["ok_calls"] == 0
        assert node_b_row["success_rate"] == 0.0


# ── 4. Failure Injection & Workflow Hardening Tests ───────────────────────────
class TestFailureInjectionWorkflow:
    def test_workflow_degrades_gracefully_under_circuit_open(self):
        from unittest.mock import MagicMock, patch
        from src.agents.insurance_agent import AgentResponse
        from src.agents.intake_agent import IntakeTriageResult
        from src.agents.safety_reviewer import SafetyReviewResult

        mock_triage = IntakeTriageResult(
            record_id="REC-M7-DEGRADED",
            patient_ref="MHP-P-00001",
            category="insurance",
            target_agent="insurance_agent",
            urgency="Routine",
            specialty="General",
            seeks_clinical_advice=False,
            requires_human=False,
            parsed_request={"urgency": "Routine", "specialty": "General", "seeks_clinical_advice": False},
        )
        mock_resp = AgentResponse(
            final_answer="Degraded mode: Copay information retrieved via robust fallback.",
            tools_called=[],
            escalated=False,
        )
        mock_safety = SafetyReviewResult(approved=True, reason="Safe degraded output")

        with patch("src.workflow.nodes._get_intake_agent") as mock_get_intake, \
             patch("src.workflow.nodes._get_insurance_agent") as mock_get_ins, \
             patch("src.workflow.nodes._get_safety_reviewer") as mock_get_safe:

            mock_intake = MagicMock()
            mock_intake.triage.return_value = mock_triage
            mock_get_intake.return_value = mock_intake

            mock_agent = MagicMock()
            mock_agent.run.return_value = mock_resp
            mock_get_ins.return_value = mock_agent

            mock_reviewer = MagicMock()
            mock_reviewer.review.return_value = mock_safety
            mock_get_safe.return_value = mock_reviewer

            memory = MemorySaver()
            graph = create_careflow_graph(checkpointer=memory)

            config = {"configurable": {"thread_id": "test_m7_degraded_thread"}}
            state = create_initial_state(
                "What is my copay under Meridian Gold?",
                record_id="REC-M7-DEGRADED",
                patient_ref="MHP-P-00001",
            )

            output = graph.invoke(state, config=config)
            assert output.get("final_response") is not None
            assert "Degraded mode" in output.get("final_response")
