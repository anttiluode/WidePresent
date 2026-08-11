"""Time-vs-structure yoking is unidentifiable at a single fixed event rate.

This experiment extracts the mathematical core of the structure-yoking branch.

A generic fading kernel is allowed to depend on both physical age (seconds) and
structural/event distance:

    k(dt, dn) = exp(-a * dt - b * dn)

At a fixed event spacing c, dt = c * dn, so

    k = exp(-(a*c + b) * dn).

Only the combination a*c+b is observable.  Infinite (a,b) pairs fit the nominal-rate
training data exactly but make different predictions after rate shift.

Once training contains at least two distinct rates, [c, 1] has rank 2 and the
physical-time versus event-distance decomposition becomes identifiable for this model.

The target families are deliberately matched at the nominal rate c=0.5 s/event:

    time-yoked      a*=1/2.5 = 0.4 /s, b*=0
    structure-yoked a*=0, b*=1/5 = 0.2 /event

At c=0.5 both give the same effective decay per event: 0.2.

Run:
    python experiments/time_structure_identifiability.py

This is an identifiability demonstration, not a novelty claim.  Continuous-time/event
RNN literature already studies timestamp-driven recurrent dynamics; the point here is
to state exactly why a one-rate training set cannot reveal what the memory horizon is
yoked to.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch.nn import functional as F

N_EVENTS = 32
NOMINAL_GAP = 0.5
TIME_TAU = 2.5
EVENT_TAU = 5.0
TRUE_A_TIME = 1.0 / TIME_TAU
TRUE_B_STRUCTURE = 1.0 / EVENT_TAU
TARGET_EFFECTIVE_RATE = TRUE_A_TIME * NOMINAL_GAP  # == TRUE_B_STRUCTURE == 0.2
SQRT3 = math.sqrt(3.0)

DISTANCE = torch.arange(N_EVENTS - 1, -1, -1, dtype=torch.float64)


def inverse_softplus(value: float) -> float:
    if value <= 1e-8:
        return -20.0
    return math.log(math.expm1(value))


def target_kernel(target: str, rates: np.ndarray) -> torch.Tensor:
    rate_tensor = torch.as_tensor(rates, dtype=torch.float64)
    ages = rate_tensor[:, None] * DISTANCE[None, :]
    if target == "time":
        return torch.exp(-TRUE_A_TIME * ages)
    if target == "structure":
        return torch.exp(-TRUE_B_STRUCTURE * DISTANCE[None, :]).expand(len(rates), -1)
    raise ValueError(target)


def fit_ab(
    target: str,
    rates: np.ndarray,
    init_a: float,
    init_b: float,
    *,
    steps: int,
    lr: float,
) -> tuple[float, float, float]:
    """Fit non-negative a,b directly against the target temporal kernel."""
    raw_a = torch.tensor(inverse_softplus(init_a), dtype=torch.float64, requires_grad=True)
    raw_b = torch.tensor(inverse_softplus(init_b), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([raw_a, raw_b], lr=lr)

    rate_tensor = torch.as_tensor(rates, dtype=torch.float64)
    ages = rate_tensor[:, None] * DISTANCE[None, :]
    target_values = target_kernel(target, rates)

    best_loss = float("inf")
    best_a = best_b = float("nan")

    for _ in range(steps):
        a = F.softplus(raw_a)
        b = F.softplus(raw_b)
        predicted = torch.exp(-a * ages - b * DISTANCE[None, :])
        loss = torch.mean((predicted - target_values) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_a = float(a.detach())
            best_b = float(b.detach())

    return best_a, best_b, best_loss


def normalized_kernel_mse(
    a: float,
    b: float,
    *,
    target: str,
    rate: float,
) -> float:
    ages = rate * DISTANCE
    predicted = torch.exp(-a * ages - b * DISTANCE)
    if target == "time":
        truth = torch.exp(-TRUE_A_TIME * ages)
    elif target == "structure":
        truth = torch.exp(-TRUE_B_STRUCTURE * DISTANCE)
    else:
        raise ValueError(target)

    numerator = torch.mean((predicted - truth) ** 2)
    denominator = torch.mean(truth**2)
    return float((numerator / denominator).detach())


def exact_single_rate_valley_check() -> None:
    """Assert that many decompositions on a*c+b=0.2 are exactly equivalent."""
    candidates = [
        (0.00, 0.20),
        (0.08, 0.16),
        (0.20, 0.10),
        (0.34, 0.03),
        (0.40, 0.00),
    ]
    reference = torch.exp(-TARGET_EFFECTIVE_RATE * DISTANCE)
    for a, b in candidates:
        assert abs(a * NOMINAL_GAP + b - TARGET_EFFECTIVE_RATE) < 1e-12
        kernel = torch.exp(-(a * NOMINAL_GAP + b) * DISTANCE)
        assert torch.allclose(kernel, reference, atol=1e-12, rtol=1e-12)


def design_rank(rates: np.ndarray) -> tuple[int, np.ndarray]:
    design = np.column_stack([rates, np.ones_like(rates)])
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    return rank, singular


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()

    exact_single_rate_valley_check()

    regimes = {
        "fixed": np.asarray([NOMINAL_GAP], dtype=np.float64),
        "narrow": np.linspace(0.45, 0.55, 7, dtype=np.float64),
        "wide": np.linspace(0.25, 0.90, 9, dtype=np.float64),
    }
    initializations = [
        (0.03, 0.185),
        (0.08, 0.160),
        (0.20, 0.100),
        (0.34, 0.030),
        (0.40, 0.005),
    ]

    print("time-vs-structure yoking identifiability")
    print(f"nominal gap = {NOMINAL_GAP:.3f} s/event")
    print(f"matched nominal effective rate = {TARGET_EFFECTIVE_RATE:.3f} /event")
    print()

    for name, rates in regimes.items():
        rank, singular = design_rank(rates)
        print(
            f"{name:>6s} rate design: rank={rank}  "
            f"singular_values={np.array2string(singular, precision=4)}"
        )
    print()

    fixed_solutions: list[tuple[float, float]] = []

    for target in ("time", "structure"):
        print(f"TARGET: {target}")
        for regime, rates in regimes.items():
            print(f"  training rates: {regime}")
            for index, (init_a, init_b) in enumerate(initializations):
                a, b, loss = fit_ab(
                    target,
                    rates,
                    init_a,
                    init_b,
                    steps=args.steps,
                    lr=args.lr,
                )
                effective = a * NOMINAL_GAP + b
                print(
                    f"    init {index}: a={a:.4f}  b={b:.4f}  "
                    f"a*c+b={effective:.4f}  loss={loss:.2e}"
                )
                if target == "time" and regime == "fixed":
                    fixed_solutions.append((a, b))
        print()

    print("FIXED-RATE SOLUTIONS: identical nominal fit, different OOD behavior")
    print("normalized kernel MSE; lower is better")
    compressed = NOMINAL_GAP / SQRT3
    stretched = NOMINAL_GAP * SQRT3
    print(
        " idx      a      b    a*c+b    "
        "time_comp time_stretch struct_comp struct_stretch"
    )
    for index, (a, b) in enumerate(fixed_solutions):
        tc = normalized_kernel_mse(a, b, target="time", rate=compressed)
        ts = normalized_kernel_mse(a, b, target="time", rate=stretched)
        sc = normalized_kernel_mse(a, b, target="structure", rate=compressed)
        ss = normalized_kernel_mse(a, b, target="structure", rate=stretched)
        print(
            f" {index:>2d}   {a:6.3f} {b:6.3f}   {a*NOMINAL_GAP+b:6.3f}      "
            f"{tc:8.3f}     {ts:8.3f}      {sc:8.3f}       {ss:8.3f}"
        )

    print()
    print("Interpretation:")
    print("  fixed rate -> rank 1: only a*c+b is identifiable")
    print("  multiple rates -> rank 2: seconds-vs-events decomposition can be learned")
    print("  a hard clock is therefore an inductive constraint, not evidence supplied by one-rate data")


if __name__ == "__main__":
    main()
