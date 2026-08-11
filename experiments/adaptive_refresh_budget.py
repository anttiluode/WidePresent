"""Budgeted active refresh after the offline-mode attack.

The previous attack found that a fresh current-state query works as well or better
while remaining online than after entering a quiet/offline mode.

This script asks the next boring question:

    if probes cost something, when should the system refresh?

It compares three trigger policies at matched approximate query budgets:

    random       query random episodes
    stale        query episodes whose most recent arrival is oldest
    uncertain    query episodes where the passive classifier is least confident

The benchmark is the same hard delayed-evidence setting used by
`provenance_vs_offline_attack.py`:

- three hidden states;
- a late state change;
- noisy observations;
- long-tailed delivery delays;
- old-world evidence can arrive near the final decision.

A refresh consists of three noisy current-state observations at t=95,96,97. The probe
itself is intentionally strong and ordinary. This experiment is about scheduling a
known useful action, not inventing a new sensing mechanism.

Run:
    python experiments/adaptive_refresh_budget.py
    python experiments/adaptive_refresh_budget.py --episodes 7000 --train 4500 --seeds 5
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_STATES = 3
DECISION_TIME = 100.0
BINS = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.1])
BUDGETS = (0.10, 0.25, 0.50)


def other_state(rng: np.random.Generator, state: int) -> int:
    candidate = int(rng.integers(N_STATES - 1))
    if candidate >= state:
        candidate += 1
    return candidate


def noisy_label(
    rng: np.random.Generator,
    state: int,
    accuracy: float,
) -> int:
    if rng.random() < accuracy:
        return state
    return other_state(rng, state)


def base_episode(
    rng: np.random.Generator,
) -> tuple[list[tuple[int, float, float]], int]:
    old_state = int(rng.integers(N_STATES))
    new_state = other_state(rng, old_state)
    switch_time = rng.uniform(82.0, 94.0)

    events: list[tuple[int, float, float]] = []
    t = 0.0
    rate = 0.28

    while True:
        t += rng.exponential(1.0 / rate)
        if t > DECISION_TIME:
            break

        state_then = old_state if t < switch_time else new_state
        value = noisy_label(rng, state_then, accuracy=0.74)

        if rng.random() < 0.55:
            delay = rng.exponential(2.5)
        else:
            delay = rng.exponential(18.0)

        arrival = t + delay
        if arrival <= DECISION_TIME:
            events.append((value, t, arrival))

    events.sort(key=lambda event: event[2])
    return events, new_state


def add_refresh(
    events: list[tuple[int, float, float]],
    current_state: int,
    rng: np.random.Generator,
) -> list[tuple[int, float, float]]:
    out = list(events)
    for world_time in (95.0, 96.0, 97.0):
        value = noisy_label(rng, current_state, accuracy=0.88)
        arrival = world_time + rng.uniform(0.05, 0.35)
        out.append((value, world_time, arrival))
    out.sort(key=lambda event: event[2])
    return out


def arrival_feature(events: list[tuple[int, float, float]]) -> np.ndarray:
    counts = np.zeros((len(BINS) - 1, N_STATES), dtype=np.float64)
    for value, _world_time, arrival in events:
        age = DECISION_TIME - arrival
        bucket = int(np.searchsorted(BINS, age, side="right") - 1)
        if 0 <= bucket < len(BINS) - 1:
            counts[bucket, value] += 1.0
    return np.sqrt(counts.ravel())


def latest_arrival_age(events: list[tuple[int, float, float]]) -> float:
    if not events:
        return DECISION_TIME
    return DECISION_TIME - max(event[2] for event in events)


def make_paired_dataset(
    *,
    seed: int,
    episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    passive: list[np.ndarray] = []
    refreshed: list[np.ndarray] = []
    labels: list[int] = []
    arrival_ages: list[float] = []

    for _ in range(episodes):
        events, current_state = base_episode(rng)
        refreshed_events = add_refresh(events, current_state, rng)

        passive.append(arrival_feature(events))
        refreshed.append(arrival_feature(refreshed_events))
        labels.append(current_state)
        arrival_ages.append(latest_arrival_age(events))

    return (
        np.asarray(passive),
        np.asarray(refreshed),
        np.asarray(labels),
        np.asarray(arrival_ages),
    )


def classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5),
    )


def mixed_accuracy(
    base_prediction: np.ndarray,
    refreshed_prediction: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> float:
    prediction = base_prediction.copy()
    prediction[mask] = refreshed_prediction[mask]
    return float(accuracy_score(labels, prediction))


def run_seed(
    *,
    seed: int,
    episodes: int,
    train_size: int,
) -> dict[str, float]:
    X, X_refresh, y, latest_age = make_paired_dataset(
        seed=seed,
        episodes=episodes,
    )

    rng = np.random.default_rng(seed + 999)
    indices = np.arange(episodes)
    rng.shuffle(indices)

    train = indices[:train_size]
    test = indices[train_size:]

    passive_model = classifier()
    refresh_model = classifier()
    passive_model.fit(X[train], y[train])
    refresh_model.fit(X_refresh[train], y[train])

    train_confidence = passive_model.predict_proba(X[train]).max(axis=1)
    test_probability = passive_model.predict_proba(X[test])
    test_confidence = test_probability.max(axis=1)

    base_prediction = test_probability.argmax(axis=1)
    refreshed_prediction = refresh_model.predict(X_refresh[test])
    test_labels = y[test]

    result: dict[str, float] = {
        "never": float(accuracy_score(test_labels, base_prediction)),
        "always": float(accuracy_score(test_labels, refreshed_prediction)),
    }

    for budget in BUDGETS:
        suffix = str(int(round(100 * budget)))

        # Learned uncertainty trigger: threshold chosen on training confidence only.
        confidence_threshold = np.quantile(train_confidence, budget)
        uncertainty_mask = test_confidence <= confidence_threshold

        # Boring temporal-kernel trigger: refresh if the latest received evidence is
        # older than a training-derived age threshold.
        age_threshold = np.quantile(latest_age[train], 1.0 - budget)
        stale_mask = latest_age[test] >= age_threshold

        # Random control uses the requested exact budget on the test set.
        random_mask = np.zeros(len(test), dtype=bool)
        random_rng = np.random.default_rng(seed + int(10000 * budget))
        n_query = int(round(budget * len(test)))
        chosen = random_rng.choice(len(test), size=n_query, replace=False)
        random_mask[chosen] = True

        result[f"random_{suffix}"] = mixed_accuracy(
            base_prediction,
            refreshed_prediction,
            test_labels,
            random_mask,
        )
        result[f"stale_{suffix}"] = mixed_accuracy(
            base_prediction,
            refreshed_prediction,
            test_labels,
            stale_mask,
        )
        result[f"uncertain_{suffix}"] = mixed_accuracy(
            base_prediction,
            refreshed_prediction,
            test_labels,
            uncertainty_mask,
        )
        result[f"stale_fraction_{suffix}"] = float(stale_mask.mean())
        result[f"uncertain_fraction_{suffix}"] = float(uncertainty_mask.mean())

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=7000)
    parser.add_argument("--train", type=int, default=4500)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    if args.train >= args.episodes:
        raise ValueError("--train must be smaller than --episodes")

    rows = [
        run_seed(
            seed=seed,
            episodes=args.episodes,
            train_size=args.train,
        )
        for seed in range(args.seeds)
    ]

    keys = ["never", "always"]
    for budget in BUDGETS:
        suffix = str(int(round(100 * budget)))
        keys.extend(
            [
                f"random_{suffix}",
                f"stale_{suffix}",
                f"uncertain_{suffix}",
            ]
        )

    print("adaptive temporal refresh")
    print("chance accuracy = 0.333")
    print()

    for seed, row in enumerate(rows):
        print(
            f"seed {seed}: "
            + "  ".join(f"{key}={row[key]:.3f}" for key in keys)
        )

    print("\nmeans")
    for key in keys:
        values = np.asarray([row[key] for row in rows])
        print(f"{key:>14s}: {values.mean():.3f} +/- {values.std(ddof=0):.3f}")

    print("\nactual trigger fractions")
    for budget in BUDGETS:
        suffix = str(int(round(100 * budget)))
        stale = np.mean([row[f"stale_fraction_{suffix}"] for row in rows])
        uncertain = np.mean(
            [row[f"uncertain_fraction_{suffix}"] for row in rows]
        )
        print(
            f"target={budget:.2f}  stale={stale:.3f}  uncertain={uncertain:.3f}"
        )

    print("\nInterpretation: adaptive refresh is useful, but this is standard active sensing.")


if __name__ == "__main__":
    main()
