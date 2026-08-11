"""Attack the sleep/offline branch with boring temporal baselines.

This experiment deliberately tries to explain away the benefit seen in
`offline_quiet_probe.py`.

Question
--------
If delayed evidence from old and current world states is mixed at decision time,
which matters more?

1. passively going quiet;
2. explicit temporal provenance;
3. a separate offline mode;
4. simply asking the current world a fresh diagnostic question while remaining
   online.

This benchmark is event-based rather than wave-based so that world/valid time and
arrival/knowledge time are exact, not inferred from an FFT.

Environment
-----------
There are three possible hidden states. One change occurs late in an episode. Sensors
emit noisy observations of the state that was true when the observation was generated.
Observation delivery has a long-tailed random delay, so old-state evidence can arrive
near the final decision at t=100.

The task is to classify the CURRENT state at t=100.

Representations / modes
-----------------------

normal:order
    Latest 24 arrived labels, no timestamps.

normal:arrival
    Label counts binned by ARRIVAL age only.

normal:valid
    Label counts binned by WORLD/VALID age only.

normal:bitemp
    Label counts jointly binned by world age and delivery delay.

quiet:arrival
    External observation generation stops at t=92. No probe is added.

quiet_probe:arrival
    Same quiet interval, plus three controlled noisy current-state probes at
    t=95,96,97.

online_probe:arrival
    External observations continue, and the exact same three controlled probes are
    added. If this matches or beats quiet_probe, the useful ingredient is active
    sensing rather than a special offline state.

online_probe:order
    Same online probing, but the readout does not even receive explicit timestamps.

The active probe is intentionally simple and strong. It is a control, not a proposed
WidePresent mechanism. If a boring current-state query solves the ambiguity, the
sleep-inspired branch should be demoted.

Run
---
    python experiments/provenance_vs_offline_attack.py
    python experiments/provenance_vs_offline_attack.py --examples 3000 --seeds 5

The default parameters reproduce the five-seed exploratory attack documented in
`docs/OFFLINE_ATTACK_RESULTS.md`.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_STATES = 3
DECISION_TIME = 100.0

ARRIVAL_BINS = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.1])
WORLD_BINS_2D = np.array([0.0, 5.0, 15.0, 35.0, 100.1])
DELAY_BINS_2D = np.array([0.0, 1.0, 4.0, 12.0, 100.1])


def draw_other_state(rng: np.random.Generator, state: int) -> int:
    candidate = int(rng.integers(N_STATES - 1))
    if candidate >= state:
        candidate += 1
    return candidate


def noisy_label(
    rng: np.random.Generator,
    true_state: int,
    accuracy: float,
) -> int:
    if rng.random() < accuracy:
        return true_state
    return draw_other_state(rng, true_state)


def sample_episode(
    rng: np.random.Generator,
    *,
    quiet: bool,
    active_probe: bool,
) -> tuple[list[tuple[int, float, float, int]], int]:
    """Return arrived events and current-state label.

    Event tuple:
        (observed_label, world_generation_time, arrival_time, source_id)

    source_id 0 = external sensor
    source_id 1 = controlled diagnostic probe
    """
    old_state = int(rng.integers(N_STATES))
    new_state = draw_other_state(rng, old_state)
    switch_time = rng.uniform(82.0, 94.0)

    generation_end = 92.0 if quiet else DECISION_TIME
    event_rate = 0.28

    generation_times: list[float] = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / event_rate)
        if t > generation_end:
            break
        generation_times.append(t)

    events: list[tuple[int, float, float, int]] = []

    for world_time in generation_times:
        state_then = old_state if world_time < switch_time else new_state
        value = noisy_label(rng, state_then, accuracy=0.74)

        # Long-tailed latency makes old-world evidence arrive late enough to be
        # confused with evidence about the current state.
        if rng.random() < 0.55:
            delay = rng.exponential(2.5)
        else:
            delay = rng.exponential(18.0)

        arrival_time = world_time + delay
        if arrival_time <= DECISION_TIME:
            events.append((value, world_time, arrival_time, 0))

    if active_probe:
        # Same active query in both quiet_probe and online_probe. The probe is not
        # magical: each answer is still noisy. Its advantage is temporal freshness.
        for world_time in (95.0, 96.0, 97.0):
            value = noisy_label(rng, new_state, accuracy=0.88)
            arrival_time = world_time + rng.uniform(0.05, 0.35)
            if arrival_time <= DECISION_TIME:
                events.append((value, world_time, arrival_time, 1))

    events.sort(key=lambda event: event[2])
    return events, new_state


def order_feature(
    events: list[tuple[int, float, float, int]],
    n_events: int = 24,
) -> np.ndarray:
    """Arrival-order baseline with no clock values."""
    out = np.zeros((n_events, N_STATES), dtype=np.float64)
    tail = events[-n_events:]
    offset = n_events - len(tail)
    for index, event in enumerate(tail):
        out[offset + index, event[0]] = 1.0
    return out.ravel()


def age_histogram(
    events: list[tuple[int, float, float, int]],
    *,
    clock: str,
) -> np.ndarray:
    """Counts by either arrival age or world/valid age."""
    counts = np.zeros((len(ARRIVAL_BINS) - 1, N_STATES), dtype=np.float64)

    for value, world_time, arrival_time, _source in events:
        if clock == "arrival":
            age = DECISION_TIME - arrival_time
        elif clock == "valid":
            age = DECISION_TIME - world_time
        else:
            raise ValueError(f"unknown clock: {clock}")

        bucket = int(np.searchsorted(ARRIVAL_BINS, age, side="right") - 1)
        if 0 <= bucket < len(ARRIVAL_BINS) - 1:
            counts[bucket, value] += 1.0

    # Compress count dynamic range so the linear classifier is not dominated by
    # episodes that simply happened to emit more observations.
    return np.sqrt(counts.ravel())


def bitemporal_histogram(
    events: list[tuple[int, float, float, int]],
) -> np.ndarray:
    """Joint world-age x delivery-delay x observed-label feature."""
    counts = np.zeros((4, 4, N_STATES), dtype=np.float64)

    for value, world_time, arrival_time, _source in events:
        world_age = DECISION_TIME - world_time
        delivery_delay = arrival_time - world_time

        world_bucket = int(
            np.searchsorted(WORLD_BINS_2D, world_age, side="right") - 1
        )
        delay_bucket = int(
            np.searchsorted(DELAY_BINS_2D, delivery_delay, side="right") - 1
        )

        if 0 <= world_bucket < 4 and 0 <= delay_bucket < 4:
            counts[world_bucket, delay_bucket, value] += 1.0

    return np.sqrt(counts.ravel())


def condition_flags(condition: str) -> tuple[bool, bool]:
    if condition == "normal":
        return False, False
    if condition == "quiet":
        return True, False
    if condition == "quiet_probe":
        return True, True
    if condition == "online_probe":
        return False, True
    raise ValueError(f"unknown condition: {condition}")


def feature_for(
    events: list[tuple[int, float, float, int]],
    representation: str,
) -> np.ndarray:
    if representation == "order":
        return order_feature(events)
    if representation == "arrival":
        return age_histogram(events, clock="arrival")
    if representation == "valid":
        return age_histogram(events, clock="valid")
    if representation == "bitemp":
        return bitemporal_histogram(events)
    raise ValueError(f"unknown representation: {representation}")


def make_dataset(
    *,
    seed: int,
    examples: int,
    condition: str,
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    quiet, active_probe = condition_flags(condition)

    X: list[np.ndarray] = []
    y: list[int] = []

    for _ in range(examples):
        events, current_state = sample_episode(
            rng,
            quiet=quiet,
            active_probe=active_probe,
        )
        X.append(feature_for(events, representation))
        y.append(current_state)

    return np.asarray(X), np.asarray(y)


def score_condition(
    *,
    seed: int,
    examples: int,
    condition: str,
    representation: str,
) -> float:
    X, y = make_dataset(
        seed=seed,
        examples=examples,
        condition=condition,
        representation=representation,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=seed,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5),
    )
    model.fit(X_train, y_train)
    return float(accuracy_score(y_test, model.predict(X_test)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    tests = (
        ("normal", "order"),
        ("normal", "arrival"),
        ("normal", "valid"),
        ("normal", "bitemp"),
        ("quiet", "arrival"),
        ("quiet_probe", "arrival"),
        ("online_probe", "arrival"),
        ("online_probe", "order"),
    )

    scores: dict[tuple[str, str], list[float]] = {test: [] for test in tests}

    print("provenance vs offline attack")
    print("chance accuracy = 0.333")
    print()

    for seed in range(args.seeds):
        row: list[str] = []
        for condition, representation in tests:
            value = score_condition(
                seed=seed,
                examples=args.examples,
                condition=condition,
                representation=representation,
            )
            scores[(condition, representation)].append(value)
            row.append(f"{condition}:{representation}={value:.3f}")
        print(f"seed {seed}: " + "  ".join(row))

    print("\nmeans")
    for condition, representation in tests:
        values = np.asarray(scores[(condition, representation)])
        print(
            f"{condition + ':' + representation:>24s}: "
            f"{values.mean():.3f} +/- {values.std(ddof=0):.3f}"
        )

    print("\nAttack logic:")
    print("- valid > arrival means explicit world-time provenance helps;")
    print("- bitemp <= valid means the second clock is not needed on this task;")
    print("- quiet < normal means silence itself is not beneficial;")
    print("- online_probe >= quiet_probe kills the need for a separate offline mode here.")


if __name__ == "__main__":
    main()
