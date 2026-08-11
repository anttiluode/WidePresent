"""How much rate diversity is needed to identify time-vs-structure yoking?

The companion experiment `time_structure_identifiability.py` shows the exact rank
failure at one event rate.  This script studies conditioning once the rank is formally
restored.

For the effective exponent

    q_j = a * c_j + b + noise,

the design matrix has rows [c_j, 1].  Its Gram determinant is

    det(X.T @ X) = n**2 * Var(c).

So practical information about the decomposition collapses continuously as experienced
rate variance goes to zero.

The script also demonstrates the fixed-rate null direction.  If training occurs only
at rate c, then theta=(a,b) can move along v=(1,-c) without changing the nominal
prediction.  At a new rate c', however, that invisible change alters the effective
exponent by delta * (c' - c).

Run:
    python experiments/rate_diversity_uncertainty.py
"""

from __future__ import annotations

import argparse
import math

import numpy as np

NOMINAL = 0.5
TRUE_TIME = np.asarray([0.4, 0.0], dtype=np.float64)
TRUE_STRUCTURE = np.asarray([0.0, 0.2], dtype=np.float64)
SQRT3 = math.sqrt(3.0)


def design_stats(rates: np.ndarray) -> tuple[int, float, float]:
    X = np.column_stack([rates, np.ones_like(rates)])
    singular = np.linalg.svd(X, compute_uv=False)
    rank = int(np.linalg.matrix_rank(X))
    condition = float("inf") if singular[-1] < 1e-12 else float(singular[0] / singular[-1])
    gram_det = float(np.linalg.det(X.T @ X))
    expected_det = float(len(rates) ** 2 * np.var(rates))
    assert np.isclose(gram_det, expected_det, rtol=1e-9, atol=1e-9)
    return rank, condition, gram_det


def monte_carlo(
    *,
    half_width: float,
    target: np.ndarray,
    n_rates: int,
    sigma: float,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    errors: list[float] = []
    conditions: list[float] = []

    for _ in range(repeats):
        if half_width == 0.0:
            rates = np.full(n_rates, NOMINAL, dtype=np.float64)
        else:
            rates = rng.uniform(
                NOMINAL - half_width,
                NOMINAL + half_width,
                size=n_rates,
            )

        X = np.column_stack([rates, np.ones_like(rates)])
        observation = X @ target + rng.normal(0.0, sigma, size=n_rates)
        estimate = np.linalg.pinv(X) @ observation
        errors.append(float(np.linalg.norm(estimate - target)))

        singular = np.linalg.svd(X, compute_uv=False)
        conditions.append(
            float("inf") if singular[-1] < 1e-12 else float(singular[0] / singular[-1])
        )

    return (
        float(np.median(errors)),
        float(np.mean(errors)),
        float(np.median(conditions)),
    )


def nullspace_demo() -> None:
    print("fixed-rate null-space disagreement")
    print("all solutions satisfy 0.5*a+b=0.2 and are identical at the nominal rate")
    compressed = NOMINAL / SQRT3
    stretched = NOMINAL * SQRT3
    print("   a      b    q_nom   q_comp  q_stretch")
    for a in np.linspace(0.0, 0.4, 5):
        b = 0.2 - NOMINAL * a
        q_nom = a * NOMINAL + b
        q_comp = a * compressed + b
        q_stretch = a * stretched + b
        print(f" {a:5.2f}  {b:5.3f}   {q_nom:5.3f}   {q_comp:6.3f}    {q_stretch:6.3f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()

    widths = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)

    print("rate-diversity conditioning")
    print(f"n rate observations = {args.rates}")
    print(f"effective-rate noise sigma = {args.sigma}")
    print()
    print(" halfwidth   rank*   median_cond   median_err_time   median_err_structure")

    for index, width in enumerate(widths):
        if width == 0.0:
            example_rates = np.full(args.rates, NOMINAL)
        else:
            example_rates = np.linspace(NOMINAL - width, NOMINAL + width, args.rates)
        rank, _condition, _det = design_stats(example_rates)

        time_med, _time_mean, cond = monte_carlo(
            half_width=width,
            target=TRUE_TIME,
            n_rates=args.rates,
            sigma=args.sigma,
            repeats=args.repeats,
            seed=100 + index,
        )
        struct_med, _struct_mean, _cond2 = monte_carlo(
            half_width=width,
            target=TRUE_STRUCTURE,
            n_rates=args.rates,
            sigma=args.sigma,
            repeats=args.repeats,
            seed=500 + index,
        )
        cond_text = "inf" if not np.isfinite(cond) else f"{cond:10.1f}"
        print(
            f" {width:8.3f}     {rank:d}     {cond_text:>10s}"
            f"        {time_med:10.3f}          {struct_med:10.3f}"
        )

    print()
    print("* rank is computed on an evenly spaced example design of the same width")
    print("  Monte Carlo condition/error values use randomly sampled rates within the band")
    print()
    nullspace_demo()

    print("Interpretation:")
    print("  det(X^T X) = n^2 Var(rate), so yoking information vanishes with rate variance")
    print("  exact IID agreement does not imply OOD agreement: the hidden null direction is exposed by rate shift")
    print("  hard time/structure constraints are priors that can help when the data are underidentified")


if __name__ == "__main__":
    main()
