"""Stationary substrate, moving flow, local gate, slow modulator.

This is a mechanical sanity demo for docs/LAKE_SUBSTRATE_FLOW.md.

It deliberately does *not* claim that a spring graph is a brain, that gating is
Transformer attention, or that the slow field is memory.  It only makes four
objects explicit:

    topology/substrate   what interactions are physically possible
    fast wave state      what is moving now
    transient gate       which legal route is favored now
    slow modulator       a toy "oil" field spreading over the same topology

Run:
    python experiments/lake_flow_gate_demo.py

Expected qualitative behavior:
    - ungated symmetric Y graph splits wave energy approximately 50/50;
    - a left gate routes more integrated energy left;
    - a right gate mirrors that result;
    - a local slow field diffuses to additional nodes over time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class YGraph:
    n: int
    edges: list[tuple[int, int]]
    left: list[int]
    right: list[int]
    left_gate_edge: int
    right_gate_edge: int


def build_y_graph(branch_len: int = 10) -> YGraph:
    """Create source -> junction -> two symmetric chains."""
    if branch_len < 2:
        raise ValueError("branch_len must be >= 2")

    n = 2 + 2 * branch_len
    edges: list[tuple[int, int]] = [(0, 1)]

    left = list(range(2, 2 + branch_len))
    right = list(range(2 + branch_len, 2 + 2 * branch_len))

    left_gate_edge = len(edges)
    edges.append((1, left[0]))
    edges.extend(zip(left[:-1], left[1:]))

    right_gate_edge = len(edges)
    edges.append((1, right[0]))
    edges.extend(zip(right[:-1], right[1:]))

    return YGraph(
        n=n,
        edges=edges,
        left=left,
        right=right,
        left_gate_edge=left_gate_edge,
        right_gate_edge=right_gate_edge,
    )


def graph_laplacian(
    n: int,
    edges: list[tuple[int, int]],
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted graph Laplacian used as a spring/stiffness operator."""
    L = np.zeros((n, n), dtype=np.float64)
    for (i, j), w in zip(edges, weights):
        L[i, i] += w
        L[j, j] += w
        L[i, j] -= w
        L[j, i] -= w
    return L


def branch_weights(graph: YGraph, gate: str) -> np.ndarray:
    """Keep topology fixed; only transiently change two legal couplings."""
    weights = np.ones(len(graph.edges), dtype=np.float64)

    if gate == "none":
        return weights
    if gate == "left":
        weights[graph.left_gate_edge] = 2.0
        weights[graph.right_gate_edge] = 0.25
        return weights
    if gate == "right":
        weights[graph.left_gate_edge] = 0.25
        weights[graph.right_gate_edge] = 2.0
        return weights

    raise ValueError(f"unknown gate: {gate}")


@dataclass
class SimulationResult:
    left_energy: float
    right_energy: float
    oil: np.ndarray

    @property
    def left_right_ratio(self) -> float:
        return self.left_energy / max(self.right_energy, 1e-12)


def simulate(
    *,
    gate: str = "none",
    with_oil: bool = False,
    branch_len: int = 10,
    duration: float = 20.0,
    dt: float = 0.002,
    damping: float = 0.08,
    oil_diffusion: float = 0.12,
    oil_decay: float = 0.003,
) -> SimulationResult:
    """Integrate a damped graph wave equation with an optional slow field.

    Fast field:
        u'' = -K u - gamma u' + drive

    Slow toy modulator:
        m' = -D L m - lambda m

    If enabled, the modulator reduces local damping.  It does not change graph
    topology and is not treated as learned memory.
    """
    graph = build_y_graph(branch_len)

    base_weights = np.ones(len(graph.edges), dtype=np.float64)
    base_L = graph_laplacian(graph.n, graph.edges, base_weights)

    effective_weights = branch_weights(graph, gate)
    K = graph_laplacian(graph.n, graph.edges, effective_weights)

    u = np.zeros(graph.n, dtype=np.float64)
    v = np.zeros(graph.n, dtype=np.float64)

    oil = np.zeros(graph.n, dtype=np.float64)
    oil[0] = 1.0

    left_energy = 0.0
    right_energy = 0.0

    steps = int(duration / dt)
    for step in range(steps):
        t = step * dt

        drive = np.zeros(graph.n, dtype=np.float64)
        if t < 3.0:
            envelope = np.exp(-((t - 1.2) / 0.45) ** 2)
            drive[0] = 5.0 * envelope * np.sin(2.0 * np.pi * t)

        gamma = np.full(graph.n, damping, dtype=np.float64)
        if with_oil:
            oil += dt * (-oil_diffusion * (base_L @ oil) - oil_decay * oil)
            oil = np.maximum(oil, 0.0)
            gamma = damping / (1.0 + 3.0 * oil)

        acceleration = -(K @ u) - gamma * v + drive

        # Semi-implicit Euler is sufficient for this qualitative sanity demo.
        v += dt * acceleration
        u += dt * v

        # Ignore the initial source pulse when measuring branch propagation.
        if t > 2.0:
            left = np.asarray(graph.left)
            right = np.asarray(graph.right)
            left_energy += float(np.mean(u[left] ** 2 + v[left] ** 2)) * dt
            right_energy += float(np.mean(u[right] ** 2 + v[right] ** 2)) * dt

    return SimulationResult(
        left_energy=left_energy,
        right_energy=right_energy,
        oil=oil,
    )


def run_demo(oil_duration: float = 50.0) -> None:
    none = simulate(gate="none")
    left = simulate(gate="left")
    right = simulate(gate="right")
    oil = simulate(gate="none", with_oil=True, duration=oil_duration)

    print("stationary-substrate Y-junction")
    print("--------------------------------")
    for name, result in (("none", none), ("left", left), ("right", right)):
        print(
            f"gate={name:>5s}  "
            f"left_E={result.left_energy:.6f}  "
            f"right_E={result.right_energy:.6f}  "
            f"L/R={result.left_right_ratio:.3f}"
        )

    covered = int(np.count_nonzero(oil.oil > 0.01))
    print()
    print(
        f"slow-field spread after {oil_duration:.1f}s: "
        f"{covered}/{oil.oil.size} nodes above 0.01"
    )
    print(
        f"slow-field mass={oil.oil.sum():.6f}, "
        f"min={oil.oil.min():.6f}, max={oil.oil.max():.6f}"
    )

    # Mechanical sanity conditions, intentionally loose.
    assert abs(none.left_right_ratio - 1.0) < 0.05, (
        "ungated symmetric graph should split approximately equally"
    )
    assert left.left_right_ratio > 1.5, "left gate should favor left branch"
    assert right.left_right_ratio < (1.0 / 1.5), "right gate should favor right branch"
    assert covered > 1, "local slow field should spread beyond its injection node"

    print("\nSanity checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oil-duration",
        type=float,
        default=50.0,
        help="seconds used for the deliberately slow diffusion demonstration",
    )
    args = parser.parse_args()
    run_demo(oil_duration=args.oil_duration)


if __name__ == "__main__":
    main()
