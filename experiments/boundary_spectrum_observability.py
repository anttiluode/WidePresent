"""Boundary time-width as an observability experiment.

This experiment is deliberately modest. It asks whether a temporally wide trace
at a thin boundary can reveal hidden structure that an instantaneous boundary
sample cannot.

Three 24-node spring graphs share the SAME four-edge stem from boundary node 0.
Only the remote interior topology differs: path, fork, or loop. Edge stiffness,
damping, drive amplitude, noise, and the decision time are randomized each run.

We drive the boundary with an impulse and observe ONLY node 0. For each sample we
compare:

    instantaneous  : y(now), dy/dt(now)
    wide spectrum  : normalized FFT magnitude of y(t) over a window ending at now

The point is not that FFT is special. Boundary spectral inverse problems and graph
wave recovery are established prior art. The question for WidePresent is narrower:
can temporal width itself be treated as an observability resource?

Run:
    python experiments/boundary_spectrum_observability.py

Optional:
    python experiments/boundary_spectrum_observability.py --seeds 0 1 2 3 4

Expected qualitative behavior:
    - instantaneous features remain near chance for 3 classes;
    - wider boundary windows become progressively more informative;
    - long windows recover a strong topology fingerprint from the same thin sensor.

This is a toy. It is not evidence for consciousness, holography, black-hole
physics, Connes geometry, or a new inverse-problem theorem.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


KINDS = ("path", "fork", "loop")


@dataclass
class Dataset:
    traces: list[np.ndarray]
    nows: np.ndarray
    labels: np.ndarray
    dt: float


def graph_edges(kind: str, n: int = 24) -> list[tuple[int, int]]:
    """Graphs share boundary stem 0-1-2-3-4; only the hidden interior differs."""
    if n != 24:
        raise ValueError("This registered toy currently assumes n=24")

    edges: list[tuple[int, int]] = [(0, 1), (1, 2), (2, 3), (3, 4)]

    if kind == "path":
        edges.extend((i, i + 1) for i in range(4, n - 1))
        return edges

    if kind == "fork":
        left = list(range(5, 15))
        right = list(range(15, 24))
        edges.append((4, left[0]))
        edges.extend(zip(left[:-1], left[1:]))
        edges.append((4, right[0]))
        edges.extend(zip(right[:-1], right[1:]))
        return edges

    if kind == "loop":
        edges.append((4, 5))
        edges.extend((i, i + 1) for i in range(5, n - 1))
        edges.append((n - 1, 5))
        return edges

    raise ValueError(f"unknown graph kind: {kind}")


def graph_laplacian(
    n: int,
    edges: list[tuple[int, int]],
    weights: np.ndarray,
) -> np.ndarray:
    L = np.zeros((n, n), dtype=np.float64)
    for (i, j), w in zip(edges, weights):
        L[i, i] += w
        L[j, j] += w
        L[i, j] -= w
        L[j, i] -= w
    return L


def boundary_trace(
    kind: str,
    rng: np.random.Generator,
    *,
    n: int = 24,
    dt: float = 0.02,
    duration: float = 30.0,
) -> np.ndarray:
    """Analytic modal solution of a uniformly damped graph wave equation.

    x'' + gamma x' + K x = 0

    Initial displacement is zero and node 0 receives an impulse-like initial
    velocity. We observe only x_0(t).
    """
    edges = graph_edges(kind, n=n)
    weights = rng.uniform(0.93, 1.07, len(edges))

    # Small on-site stiffness removes the zero-frequency rigid mode.
    K = graph_laplacian(n, edges, weights) + 0.08 * np.eye(n)
    eigenvalues, eigenvectors = np.linalg.eigh(K)

    gamma = rng.uniform(0.025, 0.05)
    damped_omega = np.sqrt(np.maximum(eigenvalues - (gamma**2) / 4.0, 1e-8))

    t = np.arange(0.0, duration, dt)
    impulse_scale = rng.uniform(0.8, 1.2)

    # For initial velocity v(0)=impulse_scale*e_0, the boundary trace is the
    # modal sum below. Boundary participation is phi_k(0)^2.
    coeff = (eigenvectors[0, :] ** 2) * impulse_scale
    y = np.exp(-gamma * t / 2.0) * np.sum(
        (coeff / damped_omega)[:, None]
        * np.sin(damped_omega[:, None] * t[None, :]),
        axis=0,
    )

    noise_scale = 0.003 * max(float(np.std(y)), 1e-8)
    y += rng.normal(0.0, noise_scale, len(y))
    return y


def make_dataset(
    seed: int,
    *,
    samples_per_class: int = 180,
    dt: float = 0.02,
    duration: float = 30.0,
    max_window: float = 18.0,
) -> Dataset:
    rng = np.random.default_rng(seed)
    traces: list[np.ndarray] = []
    labels: list[int] = []
    nows: list[int] = []

    max_window_steps = int(round(max_window / dt))
    min_now_steps = max(max_window_steps, int(round(18.0 / dt)))

    for label, kind in enumerate(KINDS):
        for _ in range(samples_per_class):
            y = boundary_trace(kind, rng, dt=dt, duration=duration)
            now = int(rng.integers(min_now_steps, len(y) + 1))
            traces.append(y)
            labels.append(label)
            nows.append(now)

    return Dataset(
        traces=traces,
        nows=np.asarray(nows, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        dt=dt,
    )


def instantaneous_features(data: Dataset) -> np.ndarray:
    features = []
    for y, now in zip(data.traces, data.nows):
        value = y[now - 1]
        velocity = (y[now - 1] - y[now - 2]) / data.dt
        features.append([value, velocity])
    return np.asarray(features, dtype=np.float64)


def spectral_features(
    data: Dataset,
    window_seconds: float,
    *,
    feature_bins: int = 79,
) -> np.ndarray:
    window_steps = int(round(window_seconds / data.dt))
    if window_steps < 8:
        raise ValueError("window too short")

    features = []
    for y, now in zip(data.traces, data.nows):
        segment = y[now - window_steps : now]
        rms = np.sqrt(np.mean(segment**2)) + 1e-9
        z = segment / rms

        # The spectrum is only a convenient diagnostic basis. A learned temporal
        # encoder, LMU/HiPPO state, scattering transform, etc. could replace it.
        magnitude = np.abs(np.fft.rfft(z * np.hanning(len(z))))[1:]
        out = np.zeros(feature_bins, dtype=np.float64)
        take = min(feature_bins, len(magnitude))
        out[:take] = magnitude[:take]
        out /= np.linalg.norm(out) + 1e-9
        features.append(out)

    return np.asarray(features, dtype=np.float64)


def score_features(
    X: np.ndarray,
    y: np.ndarray,
    *,
    split_seed: int = 42,
) -> float:
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.35,
        random_state=split_seed,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    model.fit(X[train_idx], y[train_idx])
    return float(accuracy_score(y[test_idx], model.predict(X[test_idx])))


def run_seed(
    seed: int,
    windows: list[float],
    samples_per_class: int,
) -> dict[str, float]:
    data = make_dataset(
        seed,
        samples_per_class=samples_per_class,
        max_window=max(windows),
    )

    scores: dict[str, float] = {}
    scores["instant"] = score_features(
        instantaneous_features(data), data.labels
    )

    for window in windows:
        key = f"spectrum_{window:g}s"
        scores[key] = score_features(
            spectral_features(data, window), data.labels
        )

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--windows",
        nargs="+",
        type=float,
        default=[2.0, 4.0, 6.0, 10.0, 14.0, 18.0],
    )
    parser.add_argument("--samples-per-class", type=int, default=180)
    args = parser.parse_args()

    windows = sorted(float(w) for w in args.windows)
    all_scores: list[dict[str, float]] = []

    print("boundary-spectrum observability")
    print("3 hidden graph classes; chance = 0.333")
    print("-----------------------------------------")

    for seed in args.seeds:
        scores = run_seed(seed, windows, args.samples_per_class)
        all_scores.append(scores)
        formatted = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
        print(f"seed={seed}: {formatted}")

    keys = list(all_scores[0])
    print("\nmeans")
    print("-----")
    for key in keys:
        values = np.asarray([row[key] for row in all_scores])
        print(f"{key:>14s}: {values.mean():.3f} +/- {values.std(ddof=0):.3f}")

    # These are intentionally loose sanity conditions, not registered performance
    # thresholds. They catch broken data generation or feature extraction.
    instant_mean = np.mean([row["instant"] for row in all_scores])
    widest_key = f"spectrum_{max(windows):g}s"
    widest_mean = np.mean([row[widest_key] for row in all_scores])

    assert instant_mean < 0.50, "instantaneous boundary state became too informative"
    assert widest_mean > 0.70, "wide boundary spectrum should reveal remote topology"

    print("\nSanity checks passed.")


if __name__ == "__main__":
    main()
