"""Control: is a too-wide present bad, or did the representation erase provenance?

A hidden spring-graph topology changes at t=9 s. One probe is launched under the old
operator at t=5 s and another under the current operator at t=11 s. At t=18 s we
classify the CURRENT topology using a boundary-only history.

All feature families have exactly 64 dimensions:

    global_psd : one 64-bin spectrum over the entire temporal aperture
    tf_4x16    : four temporal bins x sixteen frequency bins
    tf_8x8     : eight temporal bins x eight frequency bins

The control asks whether preserving coarse arrival-time provenance lets a wide history
remain useful even when it contains obsolete echoes.

This is intentionally exploratory. Time-frequency tilings trade temporal against
spectral resolution, so no tiling is privileged a priori.

Run:
    python experiments/boundary_provenance_control.py
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
    raise ValueError(kind)


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


def pulse(center: float, amp: float, width: float, freq: float) -> np.ndarray:
    z = TIMES - center
    return amp * np.exp(-(z / width) ** 2) * np.sin(2.0 * np.pi * freq * z)


def simulate(old_kind: str, new_kind: str, rng: np.random.Generator) -> np.ndarray:
    old_K = stiffness(old_kind, rng)
    new_K = stiffness(new_kind, rng)
    u = np.zeros(NODES, dtype=np.float64)
    v = np.zeros(NODES, dtype=np.float64)
    boundary = np.zeros(STEPS, dtype=np.float64)

    damping = rng.uniform(0.035, 0.070)
    drive = pulse(5.0, rng.uniform(3.5, 4.5), 0.18, rng.uniform(0.95, 1.25))
    drive += pulse(11.0, rng.uniform(3.5, 4.5), 0.18, rng.uniform(0.95, 1.25))
    drive += rng.normal(0.0, 0.015, STEPS)

    for step, t in enumerate(TIMES):
        K = old_K if t < 9.0 else new_K
        acceleration = -(K @ u) - damping * v
        acceleration[0] += drive[step]
        v += DT * acceleration
        u += DT * v
        boundary[step] = u[0] + rng.normal(0.0, 0.006)

    return boundary


def _spectrum(segment: np.ndarray, targets: np.ndarray) -> np.ndarray:
    segment = segment.copy()
    segment -= np.mean(segment)
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    frequencies = np.fft.rfftfreq(len(segment), DT)
    return np.log1p(np.interp(targets, frequencies, spectrum))


def global_psd(boundary: np.ndarray, width: float) -> np.ndarray:
    segment = boundary[-int(width / DT) :]
    return _spectrum(segment, np.linspace(0.1, 2.8, 64))


def time_frequency(boundary: np.ndarray, width: float, time_bins: int, freq_bins: int) -> np.ndarray:
    if time_bins * freq_bins != 64:
        raise ValueError("feature budget must remain 64")
    segment = boundary[-int(width / DT) :]
    targets = np.linspace(0.1, 2.8, freq_bins)
    parts = np.array_split(segment, time_bins)
    return np.concatenate([_spectrum(part, targets) for part in parts])


def extract(boundary: np.ndarray, width: float, family: str) -> np.ndarray:
    if family == "global_psd":
        return global_psd(boundary, width)
    if family == "tf_4x16":
        return time_frequency(boundary, width, 4, 16)
    if family == "tf_8x8":
        return time_frequency(boundary, width, 8, 8)
    raise ValueError(family)


def score(seed: int, width: float, family: str, examples: int) -> float:
    rng = np.random.default_rng(seed)
    X: list[np.ndarray] = []
    y: list[int] = []

    for _ in range(examples):
        old_kind = KINDS[int(rng.integers(3))]
        new_kind = KINDS[int(rng.integers(3))]
        boundary = simulate(old_kind, new_kind, rng)
        X.append(extract(boundary, width, family))
        y.append(KINDS.index(new_kind))

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_array,
        y_array,
        test_size=0.35,
        random_state=seed,
        stratify=y_array,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    model.fit(X_train, y_train)
    return float(accuracy_score(y_test, model.predict(X_test)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=180)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--widths",
        type=float,
        nargs="+",
        default=[6.0, 8.0, 10.0, 12.0, 14.0],
    )
    args = parser.parse_args()

    families = ("global_psd", "tf_4x16", "tf_8x8")
    print("boundary provenance control; chance=0.333; all feature budgets=64")
    for seed in range(args.seeds):
        print(f"\nseed {seed}")
        for width in args.widths:
            values = [score(seed, width, family, args.examples) for family in families]
            print(
                f"width={width:>4.1f}s  "
                + "  ".join(f"{family}={value:.3f}" for family, value in zip(families, values))
            )


if __name__ == "__main__":
    main()
