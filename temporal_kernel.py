"""Deterministic temporal state for online-agent experiments.

The temporal kernel is a deliberately boring baseline. It does not learn time.
It receives timestamps from the runtime and derives relative temporal facts that
would otherwise have to be reconstructed from text.

Important claim boundary:
- this is systems bookkeeping, not a consciousness mechanism;
- no freshness/staleness decision is inferred unless an external policy supplies
  a threshold;
- for benchmark use, deriving elapsed time is allowed; deriving the target label
  from human preferences is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


def parse_time(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected non-empty ISO timestamp, got {value!r}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_utc(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def human_duration(seconds: float) -> str:
    """Stable, arithmetic-free rendering of a non-negative duration."""
    seconds = max(0.0, float(seconds))
    whole = int(round(seconds))
    days, rem = divmod(whole, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def select_variant_time(value: Any, level: int = 0) -> str:
    """Select one benchmark time variant while accepting ordinary scalar times."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty time-variant list")
        if level < 0 or level >= len(value):
            raise IndexError(f"time level {level} outside {len(value)} variants")
        chosen = value[level]
        if not isinstance(chosen, str):
            raise TypeError("time variants must be strings")
        return chosen
    raise TypeError(f"unsupported time value: {type(value).__name__}")


@dataclass(frozen=True)
class MessageTime:
    index: int
    role: str
    timestamp: datetime
    age_seconds: float
    name: Optional[str] = None


@dataclass(frozen=True)
class TemporalKernelState:
    now: datetime
    conversation_age_seconds: float
    messages: tuple[MessageTime, ...]

    def by_role(self, role: str) -> tuple[MessageTime, ...]:
        return tuple(m for m in self.messages if m.role == role)

    def most_recent(self, role: str) -> Optional[MessageTime]:
        matches = self.by_role(role)
        return matches[-1] if matches else None

    @property
    def last_tool_age_seconds(self) -> Optional[float]:
        item = self.most_recent("tool")
        return None if item is None else item.age_seconds

    def render(self, include_all_messages: bool = True) -> str:
        """Render derived state without making a task decision.

        The rendering intentionally supplies elapsed durations but never words
        such as 'fresh', 'stale', 'reuse', or 'call tool'.
        """
        lines = [
            "TEMPORAL RUNTIME STATE (authoritative clock-derived values):",
            f"- current decision time: {iso_utc(self.now)}",
            f"- elapsed since conversation start: {human_duration(self.conversation_age_seconds)} "
            f"({self.conversation_age_seconds:.3f} s)",
        ]
        if self.last_tool_age_seconds is None:
            lines.append("- no previous tool observation is present")
        else:
            age = self.last_tool_age_seconds
            lines.append(
                f"- elapsed since most recent tool observation: {human_duration(age)} ({age:.3f} s)"
            )

        if include_all_messages:
            lines.append("- age of each prior message at this decision:")
            for m in self.messages:
                extra = f" name={m.name}" if m.name else ""
                lines.append(
                    f"  - message[{m.index}] role={m.role}{extra}: "
                    f"{human_duration(m.age_seconds)} ago ({m.age_seconds:.3f} s)"
                )

        lines.append(
            "These are timing facts only. No freshness threshold or action recommendation is supplied."
        )
        return "\n".join(lines)


def derive_temporal_state(history: Iterable[dict[str, Any]], level: int = 0) -> TemporalKernelState:
    """Derive relative ages using the final history message as the decision `now`.

    `level` selects among list-valued timestamps such as TicToc's three elapsed-
    time variants. All scalar timestamps are left unchanged.
    """
    history = list(history)
    if not history:
        raise ValueError("history must not be empty")

    selected: list[tuple[int, dict[str, Any], datetime]] = []
    for i, msg in enumerate(history):
        if "time" not in msg:
            raise KeyError(f"history message {i} has no 'time'")
        timestamp = parse_time(select_variant_time(msg["time"], level))
        selected.append((i, msg, timestamp))

    now = selected[-1][2]
    start = selected[0][2]
    messages: list[MessageTime] = []
    for i, msg, timestamp in selected:
        age = (now - timestamp).total_seconds()
        # Clock anomalies should remain visible rather than silently becoming age 0.
        if age < -1e-6:
            raise ValueError(f"message {i} occurs after decision time by {-age:.3f}s")
        messages.append(
            MessageTime(
                index=i,
                role=str(msg.get("role", "unknown")),
                timestamp=timestamp,
                age_seconds=max(age, 0.0),
                name=msg.get("name"),
            )
        )

    return TemporalKernelState(
        now=now,
        conversation_age_seconds=max((now - start).total_seconds(), 0.0),
        messages=tuple(messages),
    )


def is_stale(age_seconds: float, max_age_seconds: float) -> bool:
    """Optional explicit policy helper, kept separate from temporal state.

    Experiments may use this only when the task externally specifies a validity
    threshold. It must NOT be used to reverse-engineer human preference labels.
    """
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds must be non-negative")
    return float(age_seconds) > float(max_age_seconds)
