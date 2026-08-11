"""Offline-mode toy: continuous drive vs quiet vs controlled self-probing.

This experiment is inspired by the idea that an input-quiet interval can support a
*different kind of processing* than ordinary online sensing. It is not a model of
biological sleep.

Setup
-----
A damped spring graph has one observable/actuated boundary node. The hidden topology
switches at t=9 s. Three current topologies are possible: path, fork, loop. All share
the same local boundary stem.

We compare four post-switch modes:

    continuous       external nuisance pulses continue
    quiet            incoming drive stops at t=11 s
    self_probe       quiet, then a standardized internal probe at t=13.5 s
    downscale_probe  same, but mixed fast state is forcibly damped before probing

The task is to classify the CURRENT hidden topology from a boundary-only trace over
13.5--18 s. Features use a fixed physical-frequency grid plus six coarse ring-down
energy bins, so all modes have the same dimensionality.

The deliberately stronger damping condition is included as a negative control: if
"sleep" were merely aggressive erasure, it should help. In exploratory scratch runs
(seed 0..2, 150 examples/seed), it instead hurt relative to ordinary self-probing.

Run:
    python experiments/offline_quiet_probe.py
    python experiments/offline_quiet_probe.py --examples 300 --seeds 5

Interpretation discipline
-------------------------
This toy can establish only that reducing uncontrolled input may improve active
system identification in a dynamical medium, and that controlled internal probes can
use the quiet interval. That is standard systems intuition. It does NOT establish a
sleep mechanism, replay mechanism, consciousness claim, or biological correspondence.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KINDS = ("path", "fork", "loop")
NODES = 16
DT = 0.02
DURATION = 18.0
STEPS = int(DURATION / DT)
TIMES = np.arange(STEPS) * DT


def graph_edges(kind: str) -> list[tuple[int, int]]:
    """Three hidden interiors with an identical boundary stem 0-1-2-3."""
    edges: list[tuple[int, int]] = [(0, 1), (1, 2), (2, 3)]

    if kind == "path":
        edges.extend((i, i + 1) for i in range(3, NODES - 1))
        return edges

    if kind == "fork":
        left = list(range(4, 10))
        right = list(range(10, 16))
        edges.append((3, left[0]))
        edges.extend(zip(left[:-1], left[1:]))
        edges.append((3, right[0]))
        edges.extend(zip(right[:-1], right[1:]))
        return edges

    if kind == "loop":
        edges.append((3, 4))
        edges.extend((i, i + 1) for i in range(4, NODES - 1))
        edges.append((NODES - 1, 4))
        return edges

    raise ValueError(f"unknown topology: {kind}")


def stiffness(kind: str, rng: np.random.Generator) -> np.ndarray:
    edges = graph_edges(kind)
    weights = rng.uniform(0.90, 1.10, len(edges))
    K = np.zeros((NODES, NODES), dtype=np.float64)
    for (i, j), w in zip(edges, weights):
        K[i, i] += w
        K[j, j] += w
        K[i, j] -= w
        K[j, i] -= w
    return K + 0.05 * np.eye(NODES)


def pulse(center: float, amp: float = 4.0, width: float = 0.18, freq: float = 1.1) -> np.ndarray:
    z = TIMES - center
    return amp * np.exp(-(z / width) ** 2) * np.sin(2.0 * np.pi * freq * z)


COMMON_WAKE_DRIVE = pulse(2.5) + pulse(5.5) + pulse(8.0)


def simulate(
    mode: str,
    old_kind: str,
    new_kind: str,
    rng: np.random.Generator,
    switch_time: float = 9.0,
    quiet_start: float = 11.0,
    probe_time: float = 13.5,
) -> np.ndarray:
    old_K = stiffness(old_kind, rng)
    new_K = stiffness(new_kind, rng)

    u = np.zeros(NODES, dtype=np.float64)
    v = np.zeros(NODES, dtype=np.float64)
    boundary = np.zeros(STEPS, dtype=np.float64)

    base_damping = rng.uniform(0.06, 0.11)
    drive = COMMON_WAKE_DRIVE.copy() * rng.uniform(0.85, 1.15)

    if mode == "continuous":
        centers = np.array([11.2, 12.5, 13.8, 15.0, 16.2, 17.2])
        centers += rng.uniform(-0.2, 0.2, len(centers))
        for center in centers:
            drive += pulse(
                center,
                amp=rng.uniform(2.5, 5.0),
                width=0.16,
                freq=rng.uniform(0.75, 1.55),
            )
        drive += rng.normal(0.0, 0.06, STEPS)

    elif mode in {"quiet", "self_probe", "downscale_probe"}:
        drive[TIMES >= quiet_start] = 0.0
        if mode in {"self_probe", "downscale_probe"}:
            drive += pulse(probe_time, amp=4.0, width=0.18, freq=1.1)
    else:
        raise ValueError(f"unknown mode: {mode}")

    for step, t in enumerate(TIMES):
        K = old_K if t < switch_time else new_K
        damping = base_damping

        # Negative-control hypothesis: aggressively drain mixed fast state before
        # probing. In exploratory runs this reduced, rather than improved, accuracy.
        if mode == "downscale_probe" and quiet_start <= t < probe_time:
            damping = 0.35

        acceleration = -(K @ u) - damping * v
        acceleration[0] += drive[step]
        v += DT * acceleration
        u += DT * v
        boundary[step] = u[0] + rng.normal(0.0, 0.008)

    return boundary


def features(
    boundary: np.ndarray,
    start: float = 13.5,
    end: float = 18.0,
    n_freq: int = 48,
) -> np.ndarray:
    segment = boundary[int(start / DT) : int(end / DT)].copy()
    segment -= np.mean(segment)

    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    frequencies = np.fft.rfftfreq(len(segment), DT)
    target = np.linspace(0.1, 2.8, n_freq)
    spectral = np.log1p(np.interp(target, frequencies, spectrum))

    # Coarse temporal envelope preserves when ring-down energy arrives while keeping
    # dimensionality fixed across all conditions.
    chunks = np.array_split(segment, 6)
    envelope = np.log1p([1000.0 * np.mean(chunk**2) for chunk in chunks])
    return np.concatenate([spectral, envelope])


def make_dataset(mode: str, seed: int, examples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X: list[np.ndarray] = []
    y: list[int] = []

    for _ in range(examples):
        new_kind = KINDS[int(rng.integers(len(KINDS)))]
        old_kind = KINDS[int(rng.integers(len(KINDS)))]
        trace = simulate(mode, old_kind, new_kind, rng)
        X.append(features(trace))
        y.append(KINDS.index(new_kind))

    return np.asarray(X), np.asarray(y)


def score_mode(mode: str, seed: int, examples: int) -> float:
    X, y = make_dataset(mode, seed, examples)
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
    parser.add_argument("--examples", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    modes = ("continuous", "quiet", "self_probe", "downscale_probe")
    all_scores: dict[str, list[float]] = {mode: [] for mode in modes}

    print("offline-mode boundary identification")
    print("chance accuracy = 0.333")
    print()

    for seed in range(args.seeds):
        row = []
        for mode in modes:
            score = score_mode(mode, seed, args.examples)
            all_scores[mode].append(score)
            row.append(f"{mode}={score:.3f}")
        print(f"seed {seed}: " + "  ".join(row))

    print("\nmeans")
    for mode in modes:
        values = np.asarray(all_scores[mode])
        print(f"{mode:>15s}: {values.mean():.3f} +/- {values.std(ddof=0):.3f}")

    print("\nInterpretation: quiet/self-probe gains are system-identification gains, not sleep claims.")


if __name__ == "__main__":
    main()
