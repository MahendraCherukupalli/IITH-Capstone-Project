"""LangGraph workflow definition for CareFlow (M5).

Orchestrates the patient journey through a stateful graph:
START -> intake_parser -> eligibility_check -> referral_router -> (conditional)
                                                                 ├── human_checkpoint (interrupt) ── (conditional)
                                                                 │                                  ├── respond -> END
                                                                 │                                  └── escalation_refusal -> END
                                                                 └── respond -> END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.workflow.nodes import (
    eligibility_check_node,
    escalation_refusal_node,
    human_checkpoint_node,
    intake_parser_node,
    referral_agent_node,
    referral_router_node,
    respond_node,
    safety_reviewer_node,
)
from src.workflow.state import CareFlowState, create_initial_state

logger = logging.getLogger(__name__)


# ── Conditional Routing Functions ──────────────────────────────────────────────
def route_after_referral(state: CareFlowState) -> str:
    """Route to human_checkpoint, referral_agent, or respond (insurance_agent)."""
    if state.get("requires_human", False):
        logger.info("Routing -> human_checkpoint (requires_human=True)")
        return "human_checkpoint"

    target = state.get("target_agent", "insurance_agent")
    if target == "referral_agent":
        logger.info("Routing -> referral_agent (Specialty Referral Path)")
        return "referral_agent"

    logger.info("Routing -> respond (Insurance Policy Path)")
    return "respond"


def route_after_safety(state: CareFlowState) -> str:
    """Route to human_checkpoint if safety reviewer blocked, else to END."""
    review = state.get("safety_review") or {}
    if not review.get("approved", True) or state.get("requires_human", False):
        logger.info("Routing -> human_checkpoint (Safety Reviewer Blocked Response)")
        return "human_checkpoint"
    logger.info("Routing -> END (Safety Review Passed)")
    return "end"


def route_after_human(state: CareFlowState) -> str:
    """Route to respond if human_approved is True, else to escalation_refusal."""
    if state.get("human_approved", False):
        logger.info("Routing -> respond (Human Approved)")
        return "respond"
    logger.info("Routing -> escalation_refusal (Human Rejected)")
    return "escalation_refusal"


# ── Graph Builder ─────────────────────────────────────────────────────────────
def create_careflow_graph(checkpointer: Any | None = None) -> Any:
    """Construct and compile the CareFlow LangGraph workflow with Multi-Agent team.

    Args:
        checkpointer: LangGraph checkpointer instance (defaults to MemorySaver()).
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(CareFlowState)

    # 1. Add Nodes
    builder.add_node("intake_parser", intake_parser_node)
    builder.add_node("eligibility_check", eligibility_check_node)
    builder.add_node("referral_router", referral_router_node)
    builder.add_node("referral_agent", referral_agent_node)
    builder.add_node("respond", respond_node)
    builder.add_node("safety_reviewer", safety_reviewer_node)
    builder.add_node("human_checkpoint", human_checkpoint_node)
    builder.add_node("escalation_refusal", escalation_refusal_node)

    # 2. Add Fixed Intake Edges
    builder.add_edge(START, "intake_parser")
    builder.add_edge("intake_parser", "eligibility_check")
    builder.add_edge("eligibility_check", "referral_router")

    # 3. Add Conditional Triage Routing Edge
    builder.add_conditional_edges(
        "referral_router",
        route_after_referral,
        {
            "human_checkpoint": "human_checkpoint",
            "referral_agent": "referral_agent",
            "respond": "respond",
        },
    )

    # 4. Agent Responses Route to Safety Reviewer
    builder.add_edge("respond", "safety_reviewer")
    builder.add_edge("referral_agent", "safety_reviewer")

    # 5. Conditional Post-Processing Safety Review Edge
    builder.add_conditional_edges(
        "safety_reviewer",
        route_after_safety,
        {
            "human_checkpoint": "human_checkpoint",
            "end": END,
        },
    )

    # 6. Human Checkpoint Resume Routing
    builder.add_conditional_edges(
        "human_checkpoint",
        route_after_human,
        {
            "respond": "respond",
            "escalation_refusal": "escalation_refusal",
        },
    )

    # 7. Terminal Edges
    builder.add_edge("escalation_refusal", END)

    app = builder.compile(checkpointer=checkpointer)
    logger.info("CareFlow Multi-Agent LangGraph workflow compiled successfully.")
    return app


# ── Singleton Compiled Graph Instance ──────────────────────────────────────────
_default_memory = MemorySaver()
compiled_careflow_graph = create_careflow_graph(checkpointer=_default_memory)
