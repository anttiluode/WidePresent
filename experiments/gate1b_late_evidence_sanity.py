"""Gate 1B sanity check: absence is not the same as evidence of absence.

Events happen on world-time ticks but arrive after a bounded random delay.
At processing/knowledge time `k`, we query a recent world-time slot.

Naive policy:
    if no event for that slot has arrived yet -> declare EMPTY

Watermark policy:
    declare EMPTY only when the slot is older than the guaranteed delay bound;
    otherwise return INCOMPLETE / ABSTAIN.

This is established stream-processing logic, not a WidePresent advantage.
The purpose of the sanity test is to make the epistemic distinction executable
before we put a neural model on top of it.
"""
from __future__ import annotations

import argparse
import numpy as np


def run(seed: int = 0, ticks: int = 20_000, event_p: float = 0.20, max_delay: int = 6):
    rng = np.random.default_rng(seed)
    truth = rng.random(ticks) < event_p
    delays = rng.integers(0, max_delay + 1, size=ticks)

    arrivals: list[list[int]] = [[] for _ in range(ticks + max_delay + 1)]
    for world_tick, event in enumerate(truth):
        if event:
            arrivals[world_tick + int(delays[world_tick])].append(world_tick)

    arrived_world_ticks: set[int] = set()

    naive_empty_calls = 0
    naive_false_empty = 0
    watermark_empty_calls = 0
    watermark_false_empty = 0
    watermark_abstain = 0
    no_arrival_queries = 0

    for known_tick in range(ticks):
        for world_tick in arrivals[known_tick]:
            arrived_world_ticks.add(world_tick)

        # Query a recent world slot. Some queried events are still in flight.
        lag = int(rng.integers(0, 2 * max_delay + 1))
        world_tick = max(0, known_tick - lag)

        if world_tick in arrived_world_ticks:
            continue

        no_arrival_queries += 1

        # Arrival-order thinking: no record yet means empty.
        naive_empty_calls += 1
        if truth[world_tick]:
            naive_false_empty += 1

        # With a known hard delay bound, this watermark is safe.
        watermark = known_tick - max_delay
        if world_tick <= watermark:
            watermark_empty_calls += 1
            if truth[world_tick]:
                watermark_false_empty += 1
        else:
            watermark_abstain += 1

    return {
        "naive_false_empty_rate": naive_false_empty / max(1, naive_empty_calls),
        "watermark_false_empty_rate": watermark_false_empty / max(1, watermark_empty_calls),
        "watermark_coverage": watermark_empty_calls / max(1, no_arrival_queries),
        "watermark_abstain_rate": watermark_abstain / max(1, no_arrival_queries),
        "naive_false_empty_count": naive_false_empty,
        "no_arrival_queries": no_arrival_queries,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ticks", type=int, default=20_000)
    p.add_argument("--event-p", type=float, default=0.20)
    p.add_argument("--max-delay", type=int, default=6)
    a = p.parse_args()

    result = run(a.seed, a.ticks, a.event_p, a.max_delay)
    print("Gate 1B sanity — late evidence and interval completeness")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:29s} {value:.5f}")
        else:
            print(f"{key:29s} {value}")

    print("\nInterpretation:")
    print("- no arrival yet is not evidence that no event occurred")
    print("- a valid watermark can trade coverage for zero false-empty claims")
    print("- timestamp replay / stream-processing logic can do the same; this is only a prerequisite")


if __name__ == "__main__":
    main()
