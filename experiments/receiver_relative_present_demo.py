"""Local demo: one global now, several receiver-relative causal frontiers.

No LLM/API is required.

Run:
    python experiments/receiver_relative_present_demo.py

The demo sends one event down three fixed-latency paths. At the same objective
clock tick, the fast receiver has integrated the event while the slower receivers
still have it in flight.

This is deliberately a known-answer construction, analogous to the calibrated
PerceptionLab wave-field demo. It demonstrates the bookkeeping object; it does
not claim a novel distributed-systems mechanism.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from receiver_present import ReceiverPresent


def main() -> None:
    present = ReceiverPresent(
        {
            ("sensor", "reflex"): 2,
            ("sensor", "planner"): 8,
            ("sensor", "logger"): 15,
        }
    )

    # Move to world tick 10, then emit one event.
    present.advance(10)
    event_id = present.emit("sensor", {"kind": "danger", "level": 0.9})

    # Four ticks later, everyone shares the same global now and the event has the
    # same objective world age (4 ticks), but access/maturity differs by path.
    present.advance(4)

    print("receiver-relative present demo")
    print(f"event_id={event_id} global_now={present.now_tick}")
    print(f"sensor frontier width={present.frontier_width('sensor')} ticks")
    print()

    for receiver in present.receivers:
        print(present.snapshot(receiver).render())
        print()

    reflex = present.snapshot("reflex")
    planner = present.snapshot("planner")
    logger = present.snapshot("logger")

    assert reflex.latest_arrived("sensor") is not None
    assert planner.latest_arrived("sensor") is None
    assert logger.latest_arrived("sensor") is None
    assert planner.in_flight and logger.in_flight

    # Same global now, same event world age, different receiver maturity.
    reflex_item = reflex.latest_arrived("sensor")
    planner_item = planner.in_flight[0]
    assert reflex_item is not None
    assert reflex_item.world_age(present.now_tick) == planner_item.world_age(present.now_tick) == 4
    assert reflex_item.path_progress(present.now_tick) == 1.0
    assert 0.0 < planner_item.path_progress(present.now_tick) < 1.0

    deadline_tick = 15
    print("deadline example")
    print(f"- action deadline: {deadline_tick}")
    print(
        "- reflex can act from the arrived event before deadline: "
        f"{reflex_item.arrival_tick <= deadline_tick}"
    )
    print(
        "- planner receives the same event before deadline: "
        f"{planner_item.arrival_tick <= deadline_tick}"
    )
    print()
    print("Interpretation:")
    print("  objective event age is global; it did not split into three ages.")
    print("  what split is receiver access / causal maturity.")
    print("  each source->receiver path therefore has a retarded-time frontier:")
    print("      frontier(source, receiver, now) = now - path_delay")
    print("  a single global temporal matrix can hide this distinction in async systems.")


if __name__ == "__main__":
    main()
