"""A single mixed-yoking kernel as a robust hedge.

The preceding attacks showed that a bank over physical age (dt) and event distance (dn)
can preserve both invariants.  This script asks how much of that bank is actually
needed on the symmetric two-target toy.

Use one fading state per content channel:

    m = sum_i x_i * exp(-a * dt_i - b * dn_i)

Parameterize the orientation by g in [0,1] while keeping the nominal effective decay
q fixed:

    a = g * q / c0
    b = (1-g) * q

where c0=0.5 s/event and q=0.2/event.

Thus:

    g=0   pure structure/event yoking
    g=1   pure physical-time yoking
    0<g<1 mixed yoking

Two independent channels are used simultaneously:

    channel 0 target is physical-time-yoked
    channel 1 target is structure-yoked

The script trains only two logistic readout heads and evaluates compressed/stretched
rate shift.  It reports joint accuracy and the minimum of the two head accuracies.

Run:
    python experiments/mixed_yoking_hedge.py
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_EVENTS = 32
NOMINAL_GAP = 0.5
TIME_TAU = 2.5
EVENT_TAU = 5.0
Q_NOMINAL = 1.0 / EVENT_TAU  # 0.2 /event == NOMINAL_GAP / TIME_TAU
SQRT3 = math.sqrt(3.0)
DISTANCE = np.arange(N_EVENTS - 1, -1, -1, dtype=np.float64)[None, :]


def make_data(
    n: int,
    *,
    seed: int,
    gap: float,
    interval_jitter: float = 0.03,
    target_noise: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gaps = gap * rng.uniform(
        1.0 - interval_jitter,
        1.0 + interval_jitter,
        size=(n, N_EVENTS - 1),
    )

    content = rng.normal(size=(n, N_EVENTS, 2)).astype(np.float64)
    ages = np.zeros((n, N_EVENTS), dtype=np.float64)
    ages[:, :-1] = np.cumsum(gaps[:, ::-1], axis=1)[:, ::-1]

    time_weights = np.exp(-ages / TIME_TAU)
    event_weights = np.exp(-DISTANCE / EVENT_TAU)

    time_score = (content[:, :, 0] * time_weights).sum(axis=1)
    time_score /= np.sqrt((time_weights**2).sum(axis=1))

    event_score = (content[:, :, 1] * event_weights).sum(axis=1)
    event_score /= np.sqrt((event_weights**2).sum())

    time_score += rng.normal(0.0, target_noise, size=n)
    event_score += rng.normal(0.0, target_noise, size=n)

    labels = np.stack((time_score > 0.0, event_score > 0.0), axis=1).astype(np.int64)
    return content, ages, labels


def features(content: np.ndarray, ages: np.ndarray, g: float) -> np.ndarray:
    a = g * Q_NOMINAL / NOMINAL_GAP
    b = (1.0 - g) * Q_NOMINAL
    weights = np.exp(-a * ages - b * DISTANCE)
    return np.column_stack(
        [
            (content[:, :, 0] * weights).sum(axis=1),
            (content[:, :, 1] * weights).sum(axis=1),
        ]
    )


def train_heads(g: float, seed: int, n: int) -> list:
    content, ages, labels = make_data(n, seed=seed, gap=NOMINAL_GAP)
    X = features(content, ages, g)
    heads = []
    for column in range(2):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.5),
        )
        model.fit(X, labels[:, column])
        heads.append(model)
    return heads


def evaluate(heads: list, g: float, seed: int, gap: float, n: int) -> tuple[float, float, float]:
    content, ages, labels = make_data(n, seed=seed, gap=gap)
    X = features(content, ages, g)
    prediction = np.column_stack([head.predict(X) for head in heads])
    time_accuracy = float(np.mean(prediction[:, 0] == labels[:, 0]))
    structure_accuracy = float(np.mean(prediction[:, 1] == labels[:, 1]))
    joint = float(np.mean(np.all(prediction == labels, axis=1)))
    return time_accuracy, structure_accuracy, joint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--test", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=21)
    args = parser.parse_args()

    g_values = np.linspace(0.0, 1.0, args.steps)
    rows = []

    for g in g_values:
        seed_rows = []
        for seed in range(args.seeds):
            heads = train_heads(g, seed=21000 + seed, n=args.train)
            compressed = evaluate(
                heads,
                g,
                seed=22000 + seed,
                gap=NOMINAL_GAP / SQRT3,
                n=args.test,
            )
            stretched = evaluate(
                heads,
                g,
                seed=23000 + seed,
                gap=NOMINAL_GAP * SQRT3,
                n=args.test,
            )
            seed_rows.append(
                tuple((compressed[i] + stretched[i]) / 2.0 for i in range(3))
            )
        values = np.asarray(seed_rows)
        mean = values.mean(axis=0)
        rows.append((g, mean[0], mean[1], mean[2], min(mean[0], mean[1])))

    best_joint = max(rows, key=lambda row: row[3])
    best_worst = max(rows, key=lambda row: row[4])

    print("mixed-yoking one-kernel hedge")
    print("g=0 structure-yoked, g=1 physical-time-yoked")
    print("scores average compressed and stretched tests")
    print()
    print("   g     time    structure   joint   min_head")
    for index, row in enumerate(rows):
        if index % max(1, (len(rows) - 1) // 10) == 0 or index == len(rows) - 1:
            print(
                f" {row[0]:5.2f}   {row[1]:6.3f}    {row[2]:6.3f}   "
                f"{row[3]:6.3f}    {row[4]:6.3f}"
            )

    print()
    print(
        f"best joint: g={best_joint[0]:.3f}  joint={best_joint[3]:.3f}  "
        f"time={best_joint[1]:.3f} structure={best_joint[2]:.3f}"
    )
    print(
        f"best worst-head: g={best_worst[0]:.3f}  min_head={best_worst[4]:.3f}  "
        f"time={best_worst[1]:.3f} structure={best_worst[2]:.3f}"
    )

    print()
    print("Interpretation:")
    print("  on the symmetric toy, a mid-orientation kernel is a robust compromise")
    print("  this does not replace a multi-kernel basis on tasks requiring sharper or multiple horizons")
    print("  the useful coordinate is (physical age, event age), not a named two-module architecture")


if __name__ == "__main__":
    main()
