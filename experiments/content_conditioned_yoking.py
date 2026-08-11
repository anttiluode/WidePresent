"""Content-dependent yoking in the (physical age, event age) plane.

Two event types require different forgetting coordinates:

    type 0: pure physical-time decay, a=0.4/s, b=0
    type 1: pure event-distance decay, a=0, b=0.2/event

At the nominal 0.5 s/event rate, the two kernels are identical because both have
nominal effective decay q=0.2/event.  A one-rate training set therefore cannot reveal
that the event types require different yoking.

Compare:

    universal
        one shared (a,b) kernel for both event types

    conditioned
        separate (a_type,b_type) kernels selected by event type

As rate diversity grows, the universal model is forced into a compromise while the
conditioned model can recover the two true axes.

Run:
    python experiments/content_conditioned_yoking.py
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch.nn import functional as F

N_EVENTS = 32
NOMINAL_GAP = 0.5
A_TIME = 0.4
B_STRUCTURE = 0.2
DISTANCE = torch.arange(N_EVENTS - 1, -1, -1, dtype=torch.float64)


def inv_softplus(value: float) -> float:
    return math.log(math.expm1(value)) if value > 1e-8 else -20.0


def fit(
    rates: np.ndarray,
    *,
    conditioned: bool,
    steps: int,
    lr: float,
) -> tuple[float, list[float]]:
    rate_tensor = torch.as_tensor(rates, dtype=torch.float64)
    ages = rate_tensor[:, None] * DISTANCE[None, :]

    target_time = torch.exp(-A_TIME * ages)
    target_structure = torch.exp(-B_STRUCTURE * DISTANCE[None, :]).expand(len(rates), -1)

    if conditioned:
        initial = (0.10, 0.15, 0.10, 0.15)
    else:
        initial = (0.10, 0.15)

    raw = [
        torch.tensor(inv_softplus(value), dtype=torch.float64, requires_grad=True)
        for value in initial
    ]
    optimizer = torch.optim.Adam(raw, lr=lr)

    best_loss = float("inf")
    best_params: list[float] = []

    for _ in range(steps):
        values = [F.softplus(parameter) for parameter in raw]
        if conditioned:
            a_time, b_time, a_structure, b_structure = values
        else:
            a_time, b_time = values
            a_structure, b_structure = a_time, b_time

        predicted_time = torch.exp(
            -a_time * ages - b_time * DISTANCE[None, :]
        )
        predicted_structure = torch.exp(
            -a_structure * ages - b_structure * DISTANCE[None, :]
        )

        loss = torch.mean((predicted_time - target_time) ** 2)
        loss = loss + torch.mean((predicted_structure - target_structure) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_params = [float(parameter.detach()) for parameter in values]

    return best_loss, best_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()

    rate_sets = {
        "fixed": np.asarray([0.5], dtype=np.float64),
        "narrow": np.linspace(0.45, 0.55, 7, dtype=np.float64),
        "wide": np.linspace(0.25, 0.90, 9, dtype=np.float64),
    }

    print("content-conditioned yoking")
    print("type 0 truth: a=0.4, b=0")
    print("type 1 truth: a=0, b=0.2")
    print()

    for name, rates in rate_sets.items():
        universal_loss, universal = fit(
            rates,
            conditioned=False,
            steps=args.steps,
            lr=args.lr,
        )
        conditioned_loss, conditioned = fit(
            rates,
            conditioned=True,
            steps=args.steps,
            lr=args.lr,
        )

        print(f"training rates: {name}")
        print(
            f"  universal:   loss={universal_loss:.3e}  "
            f"a={universal[0]:.4f} b={universal[1]:.4f}"
        )
        print(
            f"  conditioned: loss={conditioned_loss:.3e}  "
            f"time(a,b)=({conditioned[0]:.4f},{conditioned[1]:.4f})  "
            f"structure(a,b)=({conditioned[2]:.4f},{conditioned[3]:.4f})"
        )
        print()

    print("Interpretation:")
    print("  fixed-rate data cannot reveal type-specific yoking because the kernels coincide")
    print("  rate diversity exposes the conflict: one shared temporal metric becomes insufficient")
    print("  content-conditioned yoking can then recover different orientations in the age plane")
    print("  this is close to existing content-dependent timescale models and is not a novelty claim")


if __name__ == "__main__":
    main()
