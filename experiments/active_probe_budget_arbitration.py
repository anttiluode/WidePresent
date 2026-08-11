"""Shared-budget attack on active temporal-semantic identification.

The free-probe experiment in active_semantic_identification.py shows that semantic
information-gain probes identify a source rule quickly. This script removes the
free diagnostic budget.

Every round contains several cached requests, but exactly ONE tool refresh can be
spent. Refreshing a request:

* gives safe immediate refresh utility for that request;
* reveals whether its old cache was valid, producing a semantic audit.

All unselected requests must reuse their cache in this deliberately harsh budget
model. The scheduler therefore trades immediate protection against longer-term
semantic information.

Policies
--------
risk
    Refresh the request with largest predicted immediate utility advantage over
    reuse.

risk_info_tie
    Protect immediate utility first; among requests within 0.05 utility of the
    best risk score, use semantic information gain as a tie-breaker.

info_until_confident
    Maximize semantic information gain until a semantic hypothesis has posterior
    >= 0.90, then switch to immediate risk.

disagreement_until_confident
    Same explore-then-exploit structure using cheap semantic disagreement.

oracle
    Knows the true semantic class only for choosing which current request is most
    valuable to refresh. It still learns from noisy audits and is an upper
    immediate-allocation reference, not a deployable policy.

Run:
    python experiments/active_probe_budget_arbitration.py
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporal_validity import TemporalCoordinates
from temporal_validity_active import (
    ProbeOpportunity,
    semantic_disagreement_score,
    semantic_information_gain_score,
)
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

TRUE_SEMANTICS = ("world_hazard", "event_hazard")
POLICIES = (
    "risk",
    "risk_info_tie",
    "info_until_confident",
    "disagreement_until_confident",
    "oracle",
)


def initial_ambiguous(seed: int, n: int) -> OnlineSemanticAccumulator:
    rng = np.random.default_rng(seed)
    learner = OnlineSemanticAccumulator()
    for _ in range(n):
        event_age = int(rng.integers(0, 21))
        world_age = float(event_age)  # exact 1 second/event confound
        p_valid = math.exp(-HAZARD * event_age)
        learner.update(
            AuditObservation(
                world_age_seconds=world_age,
                event_age=event_age,
                invalidated=False,
                still_valid=bool(rng.random() < p_valid),
            )
        )
    return learner


def true_probability(
    true_semantic: str,
    world_age: np.ndarray,
    event_age: np.ndarray,
) -> np.ndarray:
    age = world_age if true_semantic == "world_hazard" else event_age
    return np.exp(-HAZARD * age)


def candidate_coordinates(
    world_age: np.ndarray,
    event_age: np.ndarray,
) -> list[TemporalCoordinates]:
    return [
        TemporalCoordinates(
            world_age_seconds=float(world_age[i]),
            knowledge_age_seconds=float(world_age[i]),
            event_age=int(event_age[i]),
            invalidated=False,
        )
        for i in range(len(world_age))
    ]


def choose_index(
    learner: OnlineSemanticAccumulator,
    *,
    world_age: np.ndarray,
    event_age: np.ndarray,
    policy: str,
    true_semantic: str,
    tie_margin: float,
) -> tuple[int, bool]:
    posterior = learner.posterior()
    ages = candidate_coordinates(world_age, event_age)

    p_valid = np.asarray(
        [posterior.probability(age, strategy="average") for age in ages],
        dtype=np.float64,
    )
    expected_reuse_utility = (
        p_valid * REWARD_VALID_REUSE
        + (1.0 - p_valid) * REWARD_STALE_REUSE
    )
    immediate_gain = REWARD_REFRESH - expected_reuse_utility

    if policy == "risk":
        return int(np.argmax(immediate_gain)), False

    if policy == "oracle":
        p_true = true_probability(true_semantic, world_age, event_age)
        true_reuse_utility = (
            p_true * REWARD_VALID_REUSE
            + (1.0 - p_true) * REWARD_STALE_REUSE
        )
        return int(np.argmax(REWARD_REFRESH - true_reuse_utility)), False

    max_semantic_weight = max(posterior.model_weights.values())

    if policy == "risk_info_tie":
        best = float(np.max(immediate_gain))
        eligible = np.where(immediate_gain >= best - tie_margin)[0]
        info = np.asarray(
            [semantic_information_gain_score(posterior, ages[i]) for i in eligible]
        )
        return int(eligible[int(np.argmax(info))]), False

    if max_semantic_weight >= 0.90:
        return int(np.argmax(immediate_gain)), False

    if policy == "info_until_confident":
        scores = np.asarray(
            [semantic_information_gain_score(posterior, age) for age in ages]
        )
        return int(np.argmax(scores)), True

    if policy == "disagreement_until_confident":
        scores = np.asarray(
            [semantic_disagreement_score(posterior, age) for age in ages]
        )
        return int(np.argmax(scores)), True

    raise ValueError(policy)


def run_one(
    *,
    seed: int,
    true_semantic: str,
    policy: str,
    rounds: int,
    requests_per_round: int,
    initial_audits: int,
    tie_margin: float,
) -> tuple[float, float, float, float, float]:
    learner = initial_ambiguous(10000 + seed, initial_audits)
    rng = np.random.default_rng(20000 + seed)

    total_utility = 0.0
    bad_reuse = 0
    total_decisions = 0
    exploration_rounds = 0

    for _ in range(rounds):
        event_age = rng.integers(1, 21, size=requests_per_round).astype(np.float64)
        tempo = rng.uniform(0.20, 2.50, size=requests_per_round)
        world_age = (
            event_age
            * tempo
            * rng.uniform(0.95, 1.05, size=requests_per_round)
        )
        audit_uniform = rng.random(requests_per_round)

        selected, exploring = choose_index(
            learner,
            world_age=world_age,
            event_age=event_age,
            policy=policy,
            true_semantic=true_semantic,
            tie_margin=tie_margin,
        )
        exploration_rounds += int(exploring)

        p_true = true_probability(true_semantic, world_age, event_age)

        # Exactly one tool call: selected request is safely refreshed.
        total_utility += REWARD_REFRESH

        # All other requests must reuse under the fixed shared budget.
        reuse_mask = np.ones(requests_per_round, dtype=bool)
        reuse_mask[selected] = False
        total_utility += float(
            np.sum(
                p_true[reuse_mask] * REWARD_VALID_REUSE
                + (1.0 - p_true[reuse_mask]) * REWARD_STALE_REUSE
            )
        )
        bad_reuse += int(np.sum(p_true[reuse_mask] < REUSE_THRESHOLD))
        total_decisions += requests_per_round

        # The refresh audits whether the old selected cache was still valid.
        learner.update(
            AuditObservation(
                world_age_seconds=float(world_age[selected]),
                event_age=int(event_age[selected]),
                invalidated=False,
                still_valid=bool(audit_uniform[selected] < p_true[selected]),
            )
        )

    posterior = learner.posterior()
    correct_weight = posterior.model_weights[true_semantic]

    return (
        total_utility / total_decisions,
        bad_reuse / total_decisions,
        correct_weight,
        float(correct_weight >= 0.90),
        float(exploration_rounds),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--initial-audits", type=int, default=200)
    parser.add_argument("--requests-per-round", type=int, default=6)
    parser.add_argument("--tie-margin", type=float, default=0.05)
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=[20, 40, 80, 160, 320],
    )
    args = parser.parse_args()

    print("shared tool-budget arbitration")
    print(
        f"one refresh per {args.requests_per_round} requests; "
        f"initial_confounded_audits={args.initial_audits}"
    )

    for rounds in args.rounds:
        print(f"\nrounds = {rounds}")
        for policy in POLICIES:
            rows = []
            for true_semantic in TRUE_SEMANTICS:
                for seed in range(args.seeds):
                    rows.append(
                        run_one(
                            seed=seed,
                            true_semantic=true_semantic,
                            policy=policy,
                            rounds=rounds,
                            requests_per_round=args.requests_per_round,
                            initial_audits=args.initial_audits,
                            tie_margin=args.tie_margin,
                        )
                    )
            mean = np.asarray(rows, dtype=np.float64).mean(axis=0)
            print(
                f"  {policy:>30s}: utility={mean[0]:.4f} "
                f"bad_reuse={mean[1]:.4f} true_weight={mean[2]:.4f} "
                f"identified={mean[3]:.3f} explore_rounds={mean[4]:.1f}"
            )


if __name__ == "__main__":
    main()
