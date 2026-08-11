"""When a wide boundary history becomes too old to describe the current world.

This is the companion to boundary_spectrum_observability.py.

The first boundary experiment uses a stationary hidden graph and therefore rewards
longer and longer observation windows. Real environments change. Here the hidden
operator switches from one topology to another at t=18 s.

Protocol:
    - old and new graphs share the same local boundary stem;
    - an old probe is launched before the switch;
    - the graph topology changes at t=18 s;
    - a new probe is launched shortly after the switch;
    - at t=30 s we classify the CURRENT topology from a boundary-only spectrum;
    - window width is swept from 2 s to 20 s.

A good current-state representation should be wide enough to capture the response to
the new probe but not so wide that obsolete old-topology echoes dominate it.

The experiment is intentionally a toy. It does not establish a universal human
'present duration'. It asks whether an optimal temporal aperture can emerge from a
specific dynamical observability/staleness tradeoff.

Run:
    python experiments/boundary_width_staleness.py
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


def graph_edges(kind: str, n: int = 24) -> list[tuple[int, int]]:
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

    raise ValueError(f"unknown kind: {kind}")


def stiffness(kind: str, rng: np.random.Generator, n: int = 24) -> np.ndarray:
    edges = graph_edges(kind, n=n)
    weights = rng.uniform(0.95, 1.05, len(edges))
    K = np.zeros((n, n), dtype=np.float64)
    for (i, j), w in zip(edges, weights):
        K[i, i] += w
        K[j, j] += w
        K[i, j] -= w
        K[j, i] -= w
    return K + 0.08 * np.eye(n)


def simulate_switch(
    old_kind: str,
    new_kind: str,
    rng: np.random.Generator,
    *,
    n: int = 24,
    dt: float = 0.02,
    duration: float = 30.0,
    switch_time: float = 18.0,
) -> np.ndarray:
    """Piecewise local spring dynamics with one pre- and one post-switch probe."""
    K_old = stiffness(old_kind, rng, n=n)
    K_new = stiffness(new_kind, rng, n=n)
    gamma = rng.uniform(0.03, 0.05)

    x = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)

    pre_probe_time = 7.0 + rng.uniform(-0.2, 0.2)
    post_probe_time = switch_time + 0.5 + rng.uniform(-0.1, 0.1)
    pre_probe_amp = rng.uniform(0.8, 1.2) * rng.choice((-1.0, 1.0))
    post_probe_amp = rng.uniform(0.8, 1.2) * rng.choice((-1.0, 1.0))

    pre_fired = False
    post_fired = False
    boundary = []

    for step in range(int(round(duration / dt))):
        t = step * dt

        if not pre_fired and t >= pre_probe_time:
            v[0] += pre_probe_amp
            pre_fired = True

        if not post_fired and t >= post_probe_time:
            v[0] += post_probe_amp
            post_fired = True

        K = K_old if t < switch_time else K_new
        acceleration = -(K @ x) - gamma * v

        # Semi-implicit Euler. The step is deliberately small for this toy.
        v += dt * acceleration
        x += dt * v
        boundary.append(float(x[0]))

    y = np.asarray(boundary, dtype=np.float64)
    y += rng.normal(0.0, 0.002 * max(float(np.std(y)), 1e-8), len(y))
    return y


def fixed_frequency_spectrum(
    segment: np.ndarray,
    dt: float,
    *,
    bins: int = 64,
    max_hz: float = 4.0,
) -> np.ndarray:
    """Compress every temporal window to the same physical-frequency grid."""
    rms = np.sqrt(np.mean(segment**2)) + 1e-9
    z = segment / rms

    magnitude = np.abs(np.fft.rfft(z * np.hanning(len(z))))
    frequencies = np.fft.rfftfreq(len(z), dt)

    target = np.linspace(0.1, max_hz, bins)
    feature = np.interp(target, frequencies, magnitude, left=0.0, right=0.0)
    feature /= np.linalg.norm(feature) + 1e-9
    return feature


def run_seed(
    seed: int,
    windows: list[float],
    *,
    samples_per_class: int = 150,
    dt: float = 0.02,
) -> dict[float, float]:
    rng = np.random.default_rng(seed)

    traces: list[np.ndarray] = []
    labels: list[int] = []

    for new_label, new_kind in enumerate(KINDS):
        possible_old = [kind for kind in KINDS if kind != new_kind]
        for _ in range(samples_per_class):
            old_kind = str(rng.choice(possible_old))
            traces.append(simulate_switch(old_kind, new_kind, rng, dt=dt))
            labels.append(new_label)

    y = np.asarray(labels, dtype=np.int64)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.35,
        random_state=42,
        stratify=y,
    )

    scores: dict[float, float] = {}
    for window in windows:
        steps = int(round(window / dt))
        X = np.asarray(
            [fixed_frequency_spectrum(trace[-steps:], dt) for trace in traces]
        )

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2500),
        )
        model.fit(X[train_idx], y[train_idx])
        scores[window] = float(
            accuracy_score(y[test_idx], model.predict(X[test_idx]))
        )

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--windows",
        nargs="+",
        type=float,
        default=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0],
    )
    parser.add_argument("--samples-per-class", type=int, default=150)
    args = parser.parse_args()

    windows = sorted(float(w) for w in args.windows)
    rows = []

    print("boundary present-width / staleness tradeoff")
    print("current hidden graph classification; chance = 0.333")
    print("--------------------------------------------------")

    for seed in args.seeds:
        scores = run_seed(
            seed,
            windows,
            samples_per_class=args.samples_per_class,
        )
        rows.append(scores)
        text = "  ".join(f"{w:g}s={scores[w]:.3f}" for w in windows)
        print(f"seed={seed}: {text}")

    means = {
        w: float(np.mean([row[w] for row in rows]))
        for w in windows
    }

    print("\nmean accuracy by boundary-history width")
    print("---------------------------------------")
    for w in windows:
        values = np.asarray([row[w] for row in rows])
        print(f"{w:>5g}s: {values.mean():.3f} +/- {values.std(ddof=0):.3f}")

    best_window = max(windows, key=lambda w: means[w])
    widest = max(windows)
    print(f"\nbest mean window: {best_window:g}s ({means[best_window]:.3f})")
    print(f"widest window:    {widest:g}s ({means[widest]:.3f})")

    # Loose mechanical checks only. They are not scientific thresholds.
    assert means[best_window] > 0.65, "new-topology probe should be decodable"
    assert means[widest] < means[best_window] - 0.15, (
        "obsolete pre-switch history should hurt the widest window"
    )

    print("\nSanity checks passed.")


if __name__ == "__main__":
    main()
