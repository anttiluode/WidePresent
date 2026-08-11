"""Smoke/demo for temporal_validity.py.

Run:
    python experiments/temporal_validity_runtime_demo.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporal_validity import (
    CurrentContext,
    DecisionUtilities,
    Evidence,
    EventDistanceTTL,
    ExponentialAgePlane,
    TemporalValidityRuntime,
    UntilChange,
    WorldTimeTTL,
)


def main() -> None:
    t = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    utilities = DecisionUtilities(valid_reuse=1.0, stale_reuse=-1.5, refresh=0.55)
    assert abs(utilities.reuse_probability_threshold - 0.82) < 1e-12

    runtime = TemporalValidityRuntime(
        {
            "weather": WorldTimeTTL(8.0),
            "discourse": EventDistanceTTL(8),
            "reservation": UntilChange(),
            "mixed": ExponentialAgePlane(per_second=0.04, per_event=0.04),
        },
        utilities=utilities,
    )

    cases = []

    # Same arrival age, different valid/world age: provenance matters.
    now = CurrentContext(time=t, structural_index=20, source_versions={"reservation": 1})
    for name, world_age in (("weather_7.5s", 7.5), ("weather_8.5s", 8.5)):
        evidence = Evidence(
            source="weather",
            key="Northport",
            value="clear",
            valid_time=t - timedelta(seconds=world_age),
            known_time=t - timedelta(seconds=4.0),
            known_structural_index=14,
        )
        cases.append((name, runtime.evaluate(evidence, now)))

    # Same wall-time age, event distance crosses the discourse boundary.
    for name, event_age in (("discourse_8turns", 8), ("discourse_9turns", 9)):
        evidence = Evidence(
            source="discourse",
            key="draft",
            value="Project Cedar",
            valid_time=t - timedelta(seconds=30.0),
            known_time=t - timedelta(seconds=4.0),
            known_structural_index=20 - event_age,
        )
        cases.append((name, runtime.evaluate(evidence, now)))

    # Very old reservation remains valid until version changes.
    reservation = Evidence(
        source="reservation",
        key="R-1001",
        value="CONFIRMED",
        valid_time=t - timedelta(hours=3),
        known_time=t - timedelta(hours=3),
        known_structural_index=0,
        source_version=1,
    )
    cases.append(("reservation_old_same_version", runtime.evaluate(reservation, now)))
    changed = CurrentContext(time=t, structural_index=20, source_versions={"reservation": 2})
    cases.append(("reservation_changed", runtime.evaluate(reservation, changed)))

    # Generic mixed-yoking probabilistic state.
    mixed = Evidence(
        source="mixed",
        key="m",
        value=1,
        valid_time=t - timedelta(seconds=5),
        known_time=t - timedelta(seconds=5),
        known_structural_index=15,
    )
    cases.append(("mixed_age_plane", runtime.evaluate(mixed, now)))

    expected = {
        "weather_7.5s": "reuse",
        "weather_8.5s": "refresh",
        "discourse_8turns": "reuse",
        "discourse_9turns": "refresh",
        "reservation_old_same_version": "reuse",
        "reservation_changed": "refresh",
    }
    for name, decision in cases:
        if name in expected:
            assert decision.action == expected[name], (name, decision)
        c = decision.coordinates
        print(
            f"{name:31s} action={decision.action:7s} "
            f"p={decision.p_valid:.3f} "
            f"world={c.world_age_seconds:7.2f}s "
            f"known={c.knowledge_age_seconds:7.2f}s "
            f"events={c.event_age:3d} "
            f"invalidated={str(c.invalidated):5s} "
            f"model={decision.model}"
        )

    print()
    print(f"reuse threshold = {utilities.reuse_probability_threshold:.3f}")
    print("all deterministic semantic checks passed")


if __name__ == "__main__":
    main()
