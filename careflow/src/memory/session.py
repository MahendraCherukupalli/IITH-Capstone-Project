"""Session memory for M3.

Implements in-process working memory per patient (keyed by patient_ref).

Design (from the lab):
  - Stores the last N conversation turns verbatim (working memory).
  - When token count exceeds TOKEN_BUDGET, OLD turns are compressed into a
    rolling summary via the LLM (summarize_if_needed).
  - The rolling summary is prepended as a system message so context is never lost.
  - get_context_messages() returns what to inject into every LLM call:
      [{"role": "system", "content": "<summary>"}]  (if summary exists)
      + last RECENT_KEEP turns verbatim

Usage:
    mem = SessionMemory()
    mem.add_turn("MHP-P-00001", "user", "What is my co-pay?")
    mem.add_turn("MHP-P-00001", "assistant", "Your co-pay is $25.")
    history = mem.get_context_messages("MHP-P-00001")
    mem.clear("MHP-P-00001")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
RECENT_KEEP  = 6      # turns kept verbatim (last N user+assistant messages)
TOKEN_BUDGET = 800    # summarise once the buffer exceeds this many tokens
MAX_SESSIONS = 500    # max concurrent patient sessions in memory


def _token_count(text: str) -> int:
    """Approximate token count. Uses tiktoken if available, else 4-char heuristic."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _total_tokens(turns: list[dict]) -> int:
    return sum(_token_count(t.get("content", "")) for t in turns)


# ── Per-session state ──────────────────────────────────────────────────────────
@dataclass
class _Session:
    patient_ref: str
    turns: list[dict] = field(default_factory=list)
    rolling_summary: str = ""
    total_turns: int = 0   # cumulative count, never decreases


# ── Session memory store ───────────────────────────────────────────────────────
class SessionMemory:
    """In-process session memory for all active patients.

    Thread-safety: not thread-safe by default (single-process use).
    For multi-threaded use, wrap with a lock.
    """

    def __init__(
        self,
        recent_keep: int = RECENT_KEEP,
        token_budget: int = TOKEN_BUDGET,
    ) -> None:
        self._sessions: dict[str, _Session] = {}
        self.recent_keep   = recent_keep
        self.token_budget  = token_budget

    # ── Public API ─────────────────────────────────────────────────────────────
    def add_turn(self, patient_ref: str, role: str, content: str) -> None:
        """Add one conversation turn for a patient.

        Args:
            patient_ref: Patient reference ID.
            role:        "user" | "assistant" | "tool"
            content:     Message text.
        """
        session = self._get_or_create(patient_ref)
        session.turns.append({"role": role, "content": content})
        session.total_turns += 1

        # Compress if over budget
        if _total_tokens(session.turns) > self.token_budget:
            self._compress(patient_ref)

        logger.debug(
            "SessionMemory.add_turn: %s [%s] turn %d (%d tokens in buffer)",
            patient_ref, role, session.total_turns,
            _total_tokens(session.turns),
        )

    def get_context_messages(self, patient_ref: str) -> list[dict]:
        """Return messages to inject into the LLM call for this patient.

        Returns:
            List of role/content dicts:
            - If a rolling summary exists: a system message with the summary
            - Plus the last RECENT_KEEP turns verbatim
        """
        session = self._sessions.get(patient_ref)
        if session is None:
            return []

        messages: list[dict] = []

        if session.rolling_summary:
            messages.append({
                "role": "system",
                "content": (
                    f"[Session memory — earlier in this conversation]\n"
                    f"{session.rolling_summary}"
                ),
            })

        # Return only last RECENT_KEEP turns verbatim
        messages.extend(session.turns[-self.recent_keep:])
        return messages

    def get_summary(self, patient_ref: str) -> str:
        """Return the current rolling summary for a patient (empty string if none)."""
        session = self._sessions.get(patient_ref)
        return session.rolling_summary if session else ""

    def turn_count(self, patient_ref: str) -> int:
        """Total turns ever added in this session (not just buffered ones)."""
        session = self._sessions.get(patient_ref)
        return session.total_turns if session else 0

    def token_usage(self, patient_ref: str) -> int:
        """Token count of currently buffered turns."""
        session = self._sessions.get(patient_ref)
        return _total_tokens(session.turns) if session else 0

    def clear(self, patient_ref: str) -> None:
        """Clear all session state for a patient (call at end of session)."""
        if patient_ref in self._sessions:
            del self._sessions[patient_ref]
            logger.info("SessionMemory.clear: cleared session for %s", patient_ref)

    def active_sessions(self) -> list[str]:
        """List of patient_refs with active sessions."""
        return list(self._sessions.keys())

    # ── Internal ───────────────────────────────────────────────────────────────
    def _get_or_create(self, patient_ref: str) -> _Session:
        if patient_ref not in self._sessions:
            self._sessions[patient_ref] = _Session(patient_ref=patient_ref)
        return self._sessions[patient_ref]

    def _compress(self, patient_ref: str) -> None:
        """Compress old turns into rolling summary, keep last RECENT_KEEP verbatim."""
        session = self._sessions[patient_ref]
        if len(session.turns) <= self.recent_keep:
            return  # nothing old enough to compress

        old_turns  = session.turns[:-self.recent_keep]
        keep_turns = session.turns[-self.recent_keep:]

        logger.info(
            "SessionMemory: compressing %d old turns for %s",
            len(old_turns), patient_ref,
        )

        new_summary = _summarize_turns(old_turns, session.rolling_summary)
        session.rolling_summary = new_summary
        session.turns = keep_turns

        logger.info(
            "SessionMemory: compressed → %d chars summary, %d turns kept",
            len(new_summary), len(keep_turns),
        )


# ── LLM summarizer (mirrors Lab B) ────────────────────────────────────────────
def _summarize_turns(turns: list[dict], prev_summary: str = "") -> str:
    """Compress a list of old turns into a rolling summary via LLM.

    Merges with prev_summary so earlier facts are never dropped.
    Falls back to a simple concatenation if LLM call fails.
    """
    import litellm
    from dotenv import load_dotenv
    load_dotenv()

    provider  = os.getenv("LITELLM_PROVIDER", "gemini")
    model_base = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    model = f"{provider}/{model_base}" if provider != "openai" else model_base

    convo = "\n".join(
        f"{t['role']}: {t['content']}" for t in turns
        if t.get("content")
    )
    system = (
        "You maintain a running summary of a patient care coordination conversation. "
        "Update the existing summary with the new turns, preserving concrete facts "
        "(patient IDs, plan codes, appointment dates, co-pay amounts, referral IDs). "
        "Return ONLY the updated summary, under 120 words."
    )
    user = (
        f"Existing summary:\n{prev_summary or '(none)'}\n\n"
        f"New turns:\n{convo}\n\nUpdated summary:"
    )

    try:
        resp = litellm.completion(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("_summarize_turns: LLM call failed: %s", exc)
        # Fallback: simple truncated concatenation
        fallback = (prev_summary + "\n" + convo).strip()
        return fallback[:600]  # keep within reason
