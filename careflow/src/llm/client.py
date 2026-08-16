"""LiteLLM wrapper — provider-agnostic LLM client.

A single `complete()` function that works across Gemini, OpenAI, Groq, Ollama
and any other provider LiteLLM supports. The only thing that changes per
provider is the model string (e.g. "gemini/gemini-2.5-flash", "groq/llama-3.1-8b-instant").

Usage:
    from src.llm.client import complete, complete_json

    reply = complete("Summarise this message in one sentence.", system="You are a triage assistant.")
    data  = complete_json("Extract fields from: ...", system="...")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import litellm
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

# ── Silence LiteLLM verbose logging ───────────────────────────────────────────
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── Provider config ────────────────────────────────────────────────────────────
# Reads from .env.  Default provider is Gemini.
PROVIDER = os.getenv("LITELLM_PROVIDER", "gemini")

_MODEL_MAP = {
    "gemini":    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "openai":    os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "groq":      os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    "ollama":    os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
}

# LiteLLM model strings are prefixed with provider name: "gemini/...", "groq/..."
def _model_string() -> str:
    base = _MODEL_MAP.get(PROVIDER, _MODEL_MAP["gemini"])
    # LiteLLM requires prefix only for non-openai providers
    if PROVIDER == "openai":
        return base
    return f"{PROVIDER}/{base}"


# ── JSON fence stripper ────────────────────────────────────────────────────────
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

def _extract_json_text(text: str) -> str:
    """Strip markdown fences from model output if present."""
    text = text.strip()
    fenced = _JSON_FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    return text


# ── Core completion function ───────────────────────────────────────────────────
def complete(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> str:
    """Send a prompt to the configured LLM and return the reply text.

    Retries up to `max_retries` times with exponential backoff on transient
    failures (rate limits, timeouts).  Raises RuntimeError after all retries
    are exhausted.

    Args:
        prompt:      The user message.
        system:      Optional system message (persona / constraints).
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens:  Maximum tokens in the response.
        max_retries: How many times to retry on transient failure.

    Returns:
        The model's reply as a plain string.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    model = _model_string()
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reply = resp.choices[0].message.content or ""
            if reply.strip():
                return reply
            # Empty reply — treat as transient and retry
            last_error = RuntimeError("LLM returned an empty response.")
        except Exception as exc:
            last_error = exc
            logger.warning(
                "LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, exc
            )

        # Exponential backoff: 2s, 4s, 8s …
        wait = 2 ** (attempt + 1)
        logger.info("Retrying in %ds …", wait)
        time.sleep(wait)

    raise RuntimeError(
        f"LLM failed after {max_retries} attempts. Last error: {last_error}"
    )


# ── JSON completion function ───────────────────────────────────────────────────
def complete_json(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> Any:
    """Like `complete()` but parses and returns the response as JSON.

    Appends a JSON instruction to the system prompt so the model always returns
    raw JSON.  On parse failure, sends the error back and asks the model to fix
    its output (repair loop).

    Args:
        prompt:      The user message.
        system:      Optional system message.
        temperature: Lower is better for structured extraction.
        max_retries: Repair attempts on top of the initial call.

    Returns:
        Parsed Python object (dict or list).

    Raises:
        RuntimeError: If JSON cannot be parsed after all repair attempts.
    """
    json_instruction = (
        "Respond with raw JSON only. "
        "No markdown fences, no prose before or after, no commentary. "
        "The response must be valid JSON."
    )
    full_system = f"{system}\n\n{json_instruction}" if system else json_instruction

    raw = complete(prompt, system=full_system, temperature=temperature)

    for attempt in range(max_retries):
        try:
            cleaned = _extract_json_text(raw)
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Could not parse JSON after {max_retries} repair attempts.\n"
                    f"Last raw output:\n{raw[:500]}"
                ) from exc

            logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, exc)
            repair_prompt = (
                f"The following was supposed to be valid JSON but failed to parse.\n\n"
                f"Error: {exc}\n\n"
                f"Broken output:\n{raw[:2000]}\n\n"
                f"Return ONLY the corrected JSON and nothing else."
            )
            raw = complete(repair_prompt, system=json_instruction, temperature=0.0)

    # Should never reach here
    raise RuntimeError("complete_json: exhausted all repair attempts.")


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Model:", _model_string())
    reply = complete("Say hello in exactly three words.")
    print("Reply:", reply)
    data = complete_json(
        'Return a JSON object with keys "status" and "message". '
        'Set status to "ok" and message to "hello from careflow".'
    )
    print("JSON:", data)
