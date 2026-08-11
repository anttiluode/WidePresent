"""Attack hard temporal-semantic selection with model uncertainty.

Question
--------
When audited experience does not yet identify whether a source becomes stale in
wall time, event distance, or only after an explicit change, should the runtime:

1. commit to the maximum-posterior semantic model (MAP / hard selector),
2. average validity across plausible semantic models (Bayesian model average),
3. use a conservative worst-case validity across plausible models, or
4. know the true semantic axis (oracle-axis reference)?

The environment matches the operational event-agent validity attack:

* weather validity fades in world time;
* discourse validity fades in intervening event count;
* state validity persists until explicit invalidation.

Training labels are noisy realized validity outcomes. Narrow training has almost
one second/event and leaves weather/discourse semantics weakly identified. Wide
training varies episode tempo strongly and identifies the true coordinate.

Run documented evaluation:
    python experiments/semantic_uncertainty_attack.py
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporal_validity import TemporalCoordinates
from temporal_validity_learning import AuditObservation, fit_semantic_posterior


KINDS = ("weather", "discourse", "state")
REWARD_VALID_REUSE = 1.0
REWARD_STALE_REUSE = -1.5
REWARD_REFRESH = 0.55
REUSE_THRESHOLD = (
    (REWARD_REFRESH - REWARD_STALE_REUSE)
    / (REWARD_VALID_REUSE - REWARD_STALE_REUSE)
)

HAZARD = -math.log(REUSE_THRESHOLD) / 8.0
MAX_EVENTS = 20

NARROW_TRAIN = (0.95, 1.05)
WIDE_TRAIN = (0.40, 1.60)
REGIMES = {
    "iid": (0.95, 1.05, 1.5),
    "dense": (0.25, 0.45, 1.5),
    "sparse": (1.80, 2.40, 1.5),
    "long_delay": (0.95, 1.05, 7.0),
    "dense_long": (0.25, 0.45, 7.0),
}
OOD_REGIMES = ("dense", "sparse", "dense_long")


@dataclass
class Batch:
    kind: np.ndarray
    valid_age: np.ndarray
    arrival_age: np.ndarray
    event_age: np.ndarray
    invalidation: np.ndarray
    observed_valid: np.ndarray
    true_p_valid: np.ndarray


def generate_batch(
    *,
    seed: int,
    n: int,
    gap_lo: float,
    gap_hi: float,
    delay_scale: float,
) -> Batch:
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, len(KINDS), size=n)
    event_age = rng.integers(0, MAX_EVENTS + 1, size=n)

    # Episode-level tempo is the crucial manipulation. Small local jitter remains,
    # but a whole episode is fast or slow rather than independent gaps averaging
    # back to one common rate.
    episode_gap = rng.uniform(gap_lo, gap_hi, size=n)
    jitter = rng.uniform(0.90, 1.10, size=(n, MAX_EVENTS))
    gaps = episode_gap[:, None] * jitter
    mask = np.arange(MAX_EVENTS)[None, :] < event_age[:, None]
    valid_age = (gaps * mask).sum(axis=1)

    delay = rng.exponential(delay_scale, size=n)
    delay = np.minimum(delay, valid_age * 0.95)
    delay = np.where(valid_age > 0.0, delay, 0.0)
    arrival_age = np.maximum(valid_age - delay, 0.0)

    p_invalidation = 1.0 - (1.0 - 0.06) ** event_age
    invalidation = (
        (rng.random(n) < p_invalidation) & (kind == 2)
    )

    true_p = np.where(
        kind == 0,
        np.exp(-HAZARD * valid_age),
        np.where(
            kind == 1,
            np.exp(-HAZARD * event_age),
            1.0 - invalidation.astype(np.float64),
        ),
    )
    observed_valid = (rng.random(n) < true_p).astype(np.int64)

    return Batch(
        kind=kind,
        valid_age=valid_age,
        arrival_age=arrival_age,
        event_age=event_age,
        invalidation=invalidation,
        observed_valid=observed_valid,
        true_p_valid=true_p,
    )


def fit_posteriors(batch: Batch):
    posteriors = {}
    for source_index, source in enumerate(KINDS):
        mask = batch.kind == source_index
        audits = [
            AuditObservation(
                world_age_seconds=float(world_age),
                event_age=int(event_age),
                invalidated=bool(invalidated),
                still_valid=bool(valid),
            )
            for world_age, event_age, invalidated, valid in zip(
                batch.valid_age[mask],
                batch.event_age[mask],
                batch.invalidation[mask],
                batch.observed_valid[mask],
            )
        ]
        posteriors[source] = fit_semantic_posterior(audits)
    return posteriors


def posterior_probability(
    batch: Batch,
    posteriors,
    *,
    strategy: str,
    plausible_weight: float = 0.10,
) -> np.ndarray:
    out = np.empty(len(batch.kind), dtype=np.float64)
    for i in range(len(out)):
        source = KINDS[int(batch.kind[i])]
        age = TemporalCoordinates(
            world_age_seconds=float(batch.valid_age[i]),
            knowledge_age_seconds=float(batch.arrival_age[i]),
            event_age=int(batch.event_age[i]),
            invalidated=bool(batch.invalidation[i]),
        )
        out[i] = posteriors[source].probability(
            age,
            strategy=strategy,
            plausible_weight=plausible_weight,
        )
    return out


def oracle_axis_probability(batch: Batch, posteriors) -> np.ndarray:
    """Correct semantic class, nuisance hazard rate still learned from audits."""
    out = np.empty(len(batch.kind), dtype=np.float64)
    true_semantic = {
        "weather": "world_hazard",
        "discourse": "event_hazard",
        "state": "until_change",
    }
    for i in range(len(out)):
        source = KINDS[int(batch.kind[i])]
        age = TemporalCoordinates(
            world_age_seconds=float(batch.valid_age[i]),
            knowledge_age_seconds=float(batch.arrival_age[i]),
            event_age=int(batch.event_age[i]),
            invalidated=bool(batch.invalidation[i]),
        )
        out[i] = posteriors[source].axis_predictive(true_semantic[source], age)
    return out


def metrics(batch: Batch, probability: np.ndarray) -> tuple[float, float, float, float]:
    reuse = probability >= REUSE_THRESHOLD
    oracle_reuse = batch.true_p_valid >= REUSE_THRESHOLD

    agreement = float(np.mean(reuse == oracle_reuse))
    reuse_utility = (
        batch.true_p_valid * REWARD_VALID_REUSE
        + (1.0 - batch.true_p_valid) * REWARD_STALE_REUSE
    )
    utility = float(np.mean(np.where(reuse, reuse_utility, REWARD_REFRESH)))
    bad_reuse = float(np.mean(reuse & ~oracle_reuse))
    refresh_rate = float(np.mean(~reuse))
    return agreement, utility, bad_reuse, refresh_rate


def run_condition(
    *,
    train_range: tuple[float, float],
    seeds: int,
    train_n: int,
    test_n: int,
    plausible_weight: float,
):
    rows = []
    for seed in range(seeds):
        train = generate_batch(
            seed=1000 + seed,
            n=train_n,
            gap_lo=train_range[0],
            gap_hi=train_range[1],
            delay_scale=1.5,
        )
        posteriors = fit_posteriors(train)

        row = {
            "seed": seed,
            "posterior": {
                source: posteriors[source].summary()
                for source in KINDS
            },
        }

        for regime_index, (regime, (lo, hi, delay)) in enumerate(REGIMES.items()):
            test = generate_batch(
                seed=5000 + 100 * seed + regime_index,
                n=test_n,
                gap_lo=lo,
                gap_hi=hi,
                delay_scale=delay,
            )
            row[(regime, "map")] = metrics(
                test,
                posterior_probability(test, posteriors, strategy="map"),
            )
            row[(regime, "average")] = metrics(
                test,
                posterior_probability(test, posteriors, strategy="average"),
            )
            row[(regime, "robust")] = metrics(
                test,
                posterior_probability(
                    test,
                    posteriors,
                    strategy="robust",
                    plausible_weight=plausible_weight,
                ),
            )
            row[(regime, "oracle_axis")] = metrics(
                test,
                oracle_axis_probability(test, posteriors),
            )

        rows.append(row)
    return rows


def print_condition(name: str, rows) -> None:
    strategies = ("map", "average", "robust", "oracle_axis")
    print(f"\n{name.upper()} TRAINING")
    print("semantic posterior weights by seed")
    for row in rows:
        parts = []
        for source in KINDS:
            w = row["posterior"][source]["model_weights"]
            parts.append(
                f"{source}:world={w['world_hazard']:.3f},"
                f"event={w['event_hazard']:.3f},"
                f"change={w['until_change']:.3f}"
            )
        print(f"  seed {row['seed']}: " + " | ".join(parts))

    for regime in REGIMES:
        print(f"\n{regime}")
        for strategy in strategies:
            values = np.asarray([row[(regime, strategy)] for row in rows])
            mean = values.mean(axis=0)
            print(
                f"  {strategy:>11s}: agreement={mean[0]:.4f} "
                f"utility={mean[1]:.4f} bad_reuse={mean[2]:.4f} "
                f"refresh={mean[3]:.4f}"
            )

    print("\nOOD aggregate: dense + sparse + dense_long")
    for strategy in strategies:
        values = np.asarray(
            [
                row[(regime, strategy)]
                for row in rows
                for regime in OOD_REGIMES
            ]
        )
        mean = values.mean(axis=0)
        print(
            f"  {strategy:>11s}: agreement={mean[0]:.4f} "
            f"utility={mean[1]:.4f} bad_reuse={mean[2]:.4f} "
            f"refresh={mean[3]:.4f}"
        )

    # Seed-level aggregate utility is useful for seeing brittle semantic selection.
    print("\nseed-level OOD utility")
    for strategy in ("map", "average", "robust"):
        seed_utility = np.asarray(
            [
                np.mean([row[(regime, strategy)][1] for regime in OOD_REGIMES])
                for row in rows
            ]
        )
        print(
            f"  {strategy:>11s}: mean={seed_utility.mean():.4f} "
            f"std={seed_utility.std(ddof=0):.4f} "
            f"min={seed_utility.min():.4f} max={seed_utility.max():.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--test", type=int, default=3500)
    parser.add_argument(
        "--plausible-weight",
        type=float,
        default=0.10,
        help="minimum semantic posterior mass included in robust worst-case",
    )
    args = parser.parse_args()

    narrow = run_condition(
        train_range=NARROW_TRAIN,
        seeds=args.seeds,
        train_n=args.train,
        test_n=args.test,
        plausible_weight=args.plausible_weight,
    )
    wide = run_condition(
        train_range=WIDE_TRAIN,
        seeds=args.seeds,
        train_n=args.train,
        test_n=args.test,
        plausible_weight=args.plausible_weight,
    )

    print(f"reuse probability threshold = {REUSE_THRESHOLD:.3f}")
    print_condition("narrow", narrow)
    print_condition("wide", wide)


if __name__ == "__main__":
    main()
