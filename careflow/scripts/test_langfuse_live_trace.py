"""Send a live verified multi-span trace to Langfuse Cloud at https://us.cloud.langfuse.com."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langfuse import Langfuse, observe

# Load environment variables from .env
load_dotenv()


@observe(name="intake_parser")
def parse_intake_node(raw_text: str) -> dict:
    time.sleep(0.05)
    return {
        "category": "insurance",
        "target_agent": "insurance_agent",
        "urgency": "Routine",
        "seeks_clinical_advice": False,
    }


@observe(name="eligibility_check")
def check_eligibility_node(patient_ref: str) -> dict:
    time.sleep(0.03)
    return {
        "patient_ref": patient_ref,
        "plan_code": "MERIDIAN-GOLD",
        "status": "active",
    }


@observe(name="referral_router")
def referral_router_node(request: dict) -> dict:
    time.sleep(0.02)
    return {"requires_human": False, "target_agent": "insurance_agent"}


@observe(name="insurance_agent")
def run_insurance_agent_node(patient_ref: str, question: str) -> str:
    time.sleep(0.1)
    return "Under Meridian Gold, specialist office visits require a $25 copay per visit."


@observe(name="safety_reviewer")
def audit_safety_node(prompt: str, response: str) -> dict:
    time.sleep(0.04)
    return {"approved": True, "reason": "Verified administrative response.", "violation_category": None}


@observe(name="CareFlow Multi-Agent Patient Journey")
def execute_careflow_patient_journey(raw_text: str, patient_ref: str) -> str:
    print(f"  [1/5] Running intake_parser for '{raw_text[:40]}...'")
    parsed = parse_intake_node(raw_text)

    print(f"  [2/5] Running eligibility_check for {patient_ref}...")
    eligibility = check_eligibility_node(patient_ref)

    print("  [3/5] Running referral_router...")
    routing = referral_router_node(parsed)

    print(f"  [4/5] Running insurance_agent for plan {eligibility['plan_code']}...")
    agent_output = run_insurance_agent_node(patient_ref, raw_text)

    print("  [5/5] Running safety_reviewer audit...")
    safety = audit_safety_node(raw_text, agent_output)

    return agent_output


if __name__ == "__main__":
    print("=" * 75)
    print("CAREFLOW LANGFUSE CLOUD LIVE TRACE SENDER")
    print("=" * 75)

    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    pub_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sec_key = os.getenv("LANGFUSE_SECRET_KEY", "")

    print(f"Target Langfuse Cloud Host : {host}")
    print(f"Public Key                 : {pub_key[:16]}...")
    print(f"Secret Key                 : {sec_key[:16]}...")

    lf = Langfuse()
    print("\nAuthenticating with Langfuse Cloud...")
    auth_ok = lf.auth_check()

    if not auth_ok:
        print("❌ Authentication FAILED! Check your public and secret keys in .env")
        sys.exit(1)

    print("✅ Authentication SUCCESSFUL!\n")

    print("Executing traced multi-agent patient journey...")
    output = execute_careflow_patient_journey(
        raw_text="What is my copay under Meridian Gold for specialist visits?",
        patient_ref="MHP-P-10021",
    )

    print(f"\nFinal Response: '{output}'")

    print("\nFlushing events to Langfuse Cloud...")
    lf.flush()
    print("=" * 75)
    print("🎉 SUCCESS! Live trace uploaded to Langfuse Cloud!")
    print(f"Open your dashboard at: {host}")
    print("Navigate to: Traces -> 'CareFlow Multi-Agent Patient Journey'")
    print("=" * 75)
