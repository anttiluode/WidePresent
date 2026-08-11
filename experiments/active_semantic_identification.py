"""Active temporal-semantic identification under a fixed probe budget.

A source is known to obey one stable hidden validity rule:

* world_hazard, or
* event_hazard.

The runtime begins with audited experience at exactly one second/event.  That
history is deliberately non-identifying because world age == event distance.

The agent then receives the same number of opportunities to deliberately audit a
cached item.  Schedulers:

random
    Pick an opportunity uniformly.

freshness_uncertainty
    Probe near the current reuse/refresh probability threshold.

semantic_disagreement
    Probe where plausible semantic models disagree most about P(valid).

semantic_information_gain
    Maximize expected one-step entropy reduction over semantic class.

After equal probe budgets, evaluate semantic posterior concentration and future
reuse/refresh utility on dense/sparse rate-shift cases.

Run:
    python experiments/active_semantic_identification.py
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporal_validity import TemporalCoordinates
from temporal_validity_active import ProbeOpportunity, choose_probe
from temporal_validity_learning import AuditObservation
from temporal_validity_online import OnlineSemanticAccumulator


REWARD_VALID_REUSE = 1.0
REWARD_STALE_REUSE = -1.5
REWARD_REFRESH = 0.55
REUSE_THRESHOLD = (
    (REWARD_REFRESH - REWARD_STALE_REUSE)
    / (REWARD_VALID_REUSE - REWARD_STALE_REUSE)
)
HAZARD = -math.log(REUSE_THRESHOLD) / 8.0

METHODS = (
    "random",
    "freshness_uncertainty",
    "semantic_disagreement",
    "semantic_information_gain",
)
TRUE_SEMANTICS = ("world_hazard", "event_hazard")
DEFAULT_BUDGETS = (0, 2, 5, 10, 20, 40)


def initial_ambiguous_posterior(seed: int, n: int) -> OnlineSemanticAccumulator:
    """Generate audits for which seconds and event count are exactly collinear."""
    rng = np.random.default_rng(seed)
    learner = OnlineSemanticAccumulator()

    for _ in range(n):
        event_age = int(rng.integers(0, 21))
        world_age = float(event_age)  # exactly 1 second/event
        p_valid = math.exp(-HAZARD * event_age)
        still_valid = bool(rng.random() < p_valid)
        learner.update(
            AuditObservation(
                world_age_seconds=world_age,
                event_age=event_age,
                invalidated=False,
                still_valid=still_valid,
            )
        )
    return learner


def true_probability(semantic: str, world_age: float, event_age: int) -> float:
    if semantic == "world_hazard":
        return math.exp(-HAZARD * world_age)
    if semantic == "event_hazard":
        return math.exp(-HAZARD * event_age)
    raise ValueError(semantic)


def candidate_pool(
    rng: np.random.Generator,
    *,
    size: int,
) -> tuple[list[ProbeOpportunity], np.ndarray]:
    event_age = rng.integers(1, 21, size=size)
    tempo = rng.uniform(0.20, 2.50, size=size)
    world_age = (
        event_age
        * tempo
        * rng.uniform(0.95, 1.05, size=size)
    )
    uniforms = rng.random(size)

    opportunities = [
        ProbeOpportunity(
            coordinates=TemporalCoordinates(
                world_age_seconds=float(world_age[i]),
                knowledge_age_seconds=float(world_age[i]),
                event_age=int(event_age[i]),
                invalidated=False,
            ),
            payload=i,
        )
        for i in range(size)
    ]
    return opportunities, uniforms


def downstream_metrics(
    learner: OnlineSemanticAccumulator,
    *,
    true_semantic: str,
    seed: int,
    n: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    posterior = learner.posterior()

    agreement = []
    utility = []
    bad_reuse = []
    refresh = []

    for _ in range(n):
        event_age = int(rng.integers(0, 21))
        if rng.random() < 0.5:
            tempo = float(rng.uniform(0.25, 0.45))
        else:
            tempo = float(rng.uniform(1.80, 2.40))
        world_age = float(
            event_age
            * tempo
            * rng.uniform(0.95, 1.05)
        )

        age = TemporalCoordinates(
            world_age_seconds=world_age,
            knowledge_age_seconds=world_age,
            event_age=event_age,
            invalidated=False,
        )
        p_pred = posterior.probability(age, strategy="average")
        p_true = true_probability(true_semantic, world_age, event_age)

        reuse = p_pred >= REUSE_THRESHOLD
        oracle_reuse = p_true >= REUSE_THRESHOLD
        reuse_utility = (
            p_true * REWARD_VALID_REUSE
            + (1.0 - p_true) * REWARD_STALE_REUSE
        )

        agreement.append(reuse == oracle_reuse)
        utility.append(reuse_utility if reuse else REWARD_REFRESH)
        bad_reuse.append(reuse and not oracle_reuse)
        refresh.append(not reuse)

    return (
        float(np.mean(agreement)),
        float(np.mean(utility)),
        float(np.mean(bad_reuse)),
        float(np.mean(refresh)),
    )


def run_one(
    *,
    seed: int,
    true_semantic: str,
    method: str,
    max_budget: int,
    budgets: tuple[int, ...],
    initial_audits: int,
    pool_size: int,
    test_n: int,
):
    learner = initial_ambiguous_posterior(10000 + seed, initial_audits)
    opportunity_rng = np.random.default_rng(20000 + seed)
    random_picker = np.random.default_rng(
        30000
        + seed
        + {
            "random": 0,
            "freshness_uncertainty": 1000,
            "semantic_disagreement": 2000,
            "semantic_information_gain": 3000,
        }[method]
    )

    checkpoints = {}

    def record(budget: int) -> None:
        posterior = learner.posterior()
        correct_name = true_semantic
        checkpoints[budget] = {
            "metrics": downstream_metrics(
                learner,
                true_semantic=true_semantic,
                seed=50000 + seed + (0 if true_semantic == "world_hazard" else 1000),
                n=test_n,
            ),
            "true_semantic_weight": posterior.model_weights[correct_name],
            "identified": posterior.model_weights[correct_name] >= 0.90,
        }

    if 0 in budgets:
        record(0)

    for step in range(1, max_budget + 1):
        opportunities, uniforms = candidate_pool(opportunity_rng, size=pool_size)
        posterior = learner.posterior()

        if method == "random":
            selected = opportunities[int(random_picker.integers(len(opportunities)))]
        else:
            selected = choose_probe(
                posterior,
                opportunities,
                method=method,
                decision_threshold=REUSE_THRESHOLD,
            )

        index = int(selected.payload)
        age = selected.coordinates
        p_true = true_probability(
            true_semantic,
            age.world_age_seconds,
            age.event_age,
        )
        still_valid = bool(uniforms[index] < p_true)
        learner.update(
            AuditObservation(
                world_age_seconds=age.world_age_seconds,
                event_age=age.event_age,
                invalidated=False,
                still_valid=still_valid,
            )
        )

        if step in budgets:
            record(step)

    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--initial-audits", type=int, default=200)
    parser.add_argument("--pool", type=int, default=64)
    parser.add_argument("--test", type=int, default=1500)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
    )
    args = parser.parse_args()

    budgets = tuple(sorted(set(args.budgets)))
    if budgets[0] < 0:
        raise ValueError("budgets must be non-negative")
    max_budget = budgets[-1]

    collected = {
        budget: {method: [] for method in METHODS}
        for budget in budgets
    }

    for true_index, true_semantic in enumerate(TRUE_SEMANTICS):
        for seed in range(args.seeds):
            for method in METHODS:
                checkpoints = run_one(
                    seed=seed,
                    true_semantic=true_semantic,
                    method=method,
                    max_budget=max_budget,
                    budgets=budgets,
                    initial_audits=args.initial_audits,
                    pool_size=args.pool,
                    test_n=args.test,
                )
                for budget, record in checkpoints.items():
                    collected[budget][method].append(record)

    print("active semantic identification")
    print(
        f"true sources={TRUE_SEMANTICS} seeds/source={args.seeds} "
        f"initial_confounded_audits={args.initial_audits} pool={args.pool}"
    )

    for budget in budgets:
        print(f"\nprobe budget = {budget}")
        for method in METHODS:
            records = collected[budget][method]
            metrics = np.asarray([r["metrics"] for r in records])
            true_weight = float(np.mean([r["true_semantic_weight"] for r in records]))
            identified = float(np.mean([r["identified"] for r in records]))
            mean = metrics.mean(axis=0)
            print(
                f"  {method:>25s}: agreement={mean[0]:.4f} "
                f"utility={mean[1]:.4f} bad_reuse={mean[2]:.4f} "
                f"refresh={mean[3]:.4f} true_weight={true_weight:.3f} "
                f"identified={identified:.3f}"
            )


if __name__ == "__main__":
    main()
