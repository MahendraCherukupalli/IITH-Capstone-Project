"""Script to test live connection to Langfuse Cloud and send a verified test trace."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()


def send_test_trace():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    print(f"Connecting to Langfuse Cloud at: {host}")
    print(f"Public Key: {public_key[:12]}...")

    # Initialize Langfuse SDK
    langfuse = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )

    # 1. Create a root trace
    trace = langfuse.trace(
        name="CareFlow Intake Pipeline Test",
        user_id="MHP-P-10021",
        metadata={
            "environment": "development",
            "milestone": "M7",
            "channel": "web_form",
        },
        tags=["CareFlow", "M7-Test"],
    )

    print(f"Created Root Trace ID: {trace.id}")

    # 2. Add child span for IntakeParser
    span_intake = trace.span(
        name="intake_parser",
        input={"raw_text": "What is my copay under Meridian Gold for specialist visits?"},
    )
    time.sleep(0.05)
    span_intake.end(output={"category": "insurance", "target_agent": "insurance_agent"})

    # 3. Add child span for InsuranceAgent (with generation sub-span)
    span_agent = trace.span(
        name="insurance_agent",
        input={"patient_ref": "MHP-P-10021", "plan_code": "MERIDIAN-GOLD"},
    )

    gen = span_agent.generation(
        name="litellm_completion",
        model="gemini-2.5-flash",
        input=[{"role": "user", "content": "What is my copay under Meridian Gold for specialist visits?"}],
        output="Under Meridian Gold, specialist office visits have a $25 copay.",
        usage={
            "input": 120,
            "output": 18,
            "total": 138,
        },
    )
    gen.end()
    span_agent.end(output={"final_response": "Under Meridian Gold, specialist office visits have a $25 copay."})

    # 4. Add child span for ClinicalSafetyReviewer
    span_safety = trace.span(
        name="safety_reviewer",
        input={"proposed_response": "Under Meridian Gold, specialist office visits have a $25 copay."},
    )
    time.sleep(0.02)
    span_safety.end(output={"approved": True, "violation_category": None})

    # Update trace final output
    trace.update(
        output={"final_response": "Under Meridian Gold, specialist office visits have a $25 copay."}
    )

    # Flush events synchronously to Langfuse Cloud
    print("Flushing trace events to Langfuse Cloud...")
    langfuse.flush()

    print("\n✅ Trace successfully sent to Langfuse Cloud!")
    print(f"View your trace live at: {host}/project/traces/{trace.id}")


if __name__ == "__main__":
    send_test_trace()
