"""Groundedness evaluator for M4 RAG.

Checks if the LLM's response contains fabricated coverage claims
that are not supported by the retrieved chunks.

Uses the LLM itself as a judge.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_groundedness(response: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate if the response is fully grounded in the retrieved chunks.

    Returns:
        dict with:
        - grounded: bool (True if all claims are supported)
        - reason: str (Explanation from the LLM judge)
    """
    import litellm
    from dotenv import load_dotenv

    load_dotenv()
    
    # We only care about factual claims in the text
    context = "\n\n".join(
        f"--- Chunk from {c.get('doc_slug', 'unknown')} ---\n{c.get('text', '')}"
        for c in retrieved_chunks
    )

    system_prompt = (
        "You are an expert fact-checker evaluating an AI agent's response to a patient. "
        "Your task is to determine if ALL factual claims in the agent's response "
        "are strictly supported by the provided Context.\n\n"
        "Rules:\n"
        "1. If the agent makes a factual claim (e.g. a dollar amount, a policy rule, "
        "a specific service name) that is NOT in the context, it is UNGROUNDED.\n"
        "2. If the agent simply says 'I don't know' or 'I cannot find that', it is GROUNDED.\n"
        "3. If the agent asks clarifying questions or gives general pleasantries, ignore them.\n\n"
        "Output ONLY a JSON object with two keys:\n"
        '- "grounded": true or false\n'
        '- "reason": a short explanation of why'
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Agent Response:\n{response}\n\n"
        "Is the agent response fully grounded in the context?"
    )

    provider = os.getenv("LITELLM_PROVIDER", "gemini")
    model_base = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    model = f"{provider}/{model_base}" if provider != "openai" else model_base

    import time
    resp = None
    last_exc = None
    for attempt in range(4):
        try:
            resp = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            break
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) or "RateLimit" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                wait_sec = 2 ** (attempt + 1) + 2
                logger.warning("Groundedness eval hit rate limit (attempt %d/4). Retrying in %ds...", attempt + 1, wait_sec)
                time.sleep(wait_sec)
            else:
                break

    if resp is None:
        logger.error("Failed to run groundedness eval: %s", last_exc)
        return {"grounded": False, "reason": f"Eval failed: {last_exc}"}

    try:
        import json
        result = json.loads(resp.choices[0].message.content)
        return {
            "grounded": bool(result.get("grounded", False)),
            "reason": str(result.get("reason", "No reason provided")),
        }
    except Exception as exc:
        logger.error("Failed to parse groundedness eval response: %s", exc)
        return {"grounded": False, "reason": f"Eval parse failed: {exc}"}
