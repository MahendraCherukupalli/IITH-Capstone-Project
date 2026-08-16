"""Observability, Telemetry, and Tracing Controls for CareFlow (M7).

Provides:
1. `RUN_EVENTS` — In-memory telemetry log of execution events.
2. `traced(role)` — Decorator for measuring node/function latency, status, and spans.
3. `traced_generation(role, model)` — Decorator for tracking LLM token usage and latency.
4. `build_dashboard(events)` — Generates a per-agent cost, token, and latency pandas DataFrame.
"""

from __future__ import annotations

import functools
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── 1. Telemetry Collector ──────────────────────────────────────────────────
RUN_EVENTS: list[dict[str, Any]] = []
LAST_USAGE: dict[str, dict[str, int]] = {}


def clear_events() -> None:
    """Clear telemetry event collector."""
    global RUN_EVENTS, LAST_USAGE
    RUN_EVENTS.clear()
    LAST_USAGE.clear()


def record_llm_usage(role: str, usage_obj: Any) -> None:
    """Extract and record token usage metrics from an LLM response into LAST_USAGE[role]."""
    if usage_obj is None:
        return
    if isinstance(usage_obj, dict):
        inp = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or 0
        out = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or 0
        tot = usage_obj.get("total_tokens") or (inp + out)
    else:
        inp = getattr(usage_obj, "prompt_tokens", 0) or getattr(usage_obj, "input_tokens", 0)
        out = getattr(usage_obj, "completion_tokens", 0) or getattr(usage_obj, "output_tokens", 0)
        tot = getattr(usage_obj, "total_tokens", 0) or (inp + out)

    LAST_USAGE[role] = {
        "input_tokens": int(inp),
        "output_tokens": int(out),
        "total_tokens": int(tot),
    }



_langfuse_client: Any = None
_langfuse_checked: bool = False


def _get_langfuse() -> Any:
    """Lazy initializer for Langfuse client connection."""
    global _langfuse_client, _langfuse_checked
    if not _langfuse_checked:
        try:
            from langfuse import Langfuse
            lf = Langfuse()
            if lf.auth_check():
                _langfuse_client = lf
        except Exception:
            _langfuse_client = None
        _langfuse_checked = True
    return _langfuse_client


# ── 2. Span Tracing Decorator ────────────────────────────────────────────────
def traced(role: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to measure execution latency (ms) and status (ok: bool) for graph nodes."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            t0 = time.perf_counter()
            ok = True
            lf = _get_langfuse()
            try:
                if lf:
                    with lf.start_as_current_observation(as_type="span", name=f"node:{role}"):
                        result = fn(*args, **kwargs)
                        return result
                else:
                    return fn(*args, **kwargs)
            except Exception as exc:
                ok = False
                logger.error("Traced node '%s' raised exception: %s", role, exc)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                event = {
                    "role": role,
                    "ms": elapsed_ms,
                    "ok": ok,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                RUN_EVENTS.append(event)
                logger.info("Trace span [%s]: %.2fms | ok=%s", role, elapsed_ms, ok)

        return wrapper

    return decorator


# ── 3. Generation Tracing Decorator ──────────────────────────────────────────
def traced_generation(role: str, model: str = "gemini-2.5-flash") -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to trace LLM generations, capturing token usage and latency."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            t0 = time.perf_counter()
            ok = True
            lf = _get_langfuse()
            try:
                if lf:
                    with lf.start_as_current_observation(as_type="generation", name=f"agent:{role}", model=model) as gen:
                        result = fn(*args, **kwargs)
                        usage = LAST_USAGE.get(role, {})
                        if usage:
                            gen.update(usage_details={
                                "input": usage.get("input_tokens", 0),
                                "output": usage.get("output_tokens", 0),
                                "total": usage.get("total_tokens", 0),
                            })
                        return result
                else:
                    return fn(*args, **kwargs)
            except Exception as exc:
                ok = False
                raise
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                usage = LAST_USAGE.get(role, {})
                event = {
                    "role": role,
                    "model": model,
                    "ms": elapsed_ms,
                    "ok": ok,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                RUN_EVENTS.append(event)
                logger.info(
                    "Generation span [%s] (%s): %.2fms | tokens: in=%d, out=%d, total=%d | ok=%s",
                    role, model, elapsed_ms,
                    event["input_tokens"], event["output_tokens"], event["total_tokens"],
                    ok
                )

        return wrapper

    return decorator



# ── 4. Telemetry Dashboard Summary ───────────────────────────────────────────
def build_dashboard(events: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Build a pandas summary dashboard from collected telemetry events."""
    target_events = events if events is not None else RUN_EVENTS

    if not target_events:
        return pd.DataFrame(columns=["role", "calls", "ok_calls", "total_ms", "avg_ms", "success_rate"])

    df = pd.DataFrame(target_events)

    summary = []
    for role, group in df.groupby("role"):
        calls = len(group)
        ok_calls = int(group["ok"].sum())
        total_ms = float(group["ms"].sum())
        avg_ms = total_ms / calls if calls > 0 else 0.0
        success_rate = ok_calls / calls if calls > 0 else 0.0

        row = {
            "role": role,
            "calls": calls,
            "ok_calls": ok_calls,
            "total_ms": round(total_ms, 2),
            "avg_ms": round(avg_ms, 2),
            "success_rate": round(success_rate, 4),
        }
        if "total_tokens" in group.columns:
            row["input_tokens"] = int(group["input_tokens"].sum()) if "input_tokens" in group.columns else 0
            row["output_tokens"] = int(group["output_tokens"].sum()) if "output_tokens" in group.columns else 0
            row["total_tokens"] = int(group["total_tokens"].sum())
        summary.append(row)

    return pd.DataFrame(summary).sort_values("calls", ascending=False).reset_index(drop=True)

