"""Online temporal-semantic drift attack.

A single anonymous source has a stable world-time hazard, then changes midway to
an event-distance hazard. Interaction tempo remains rate-diverse throughout, so
both semantics are statistically identifiable when recent evidence is considered.

Strategies
----------
frozen
    Learn from warm-up audits, then never update.

cumulative
    Keep every audit forever. This tests whether old certainty creates semantic
    inertia after a real source-policy change.

rolling
    Bayesian semantic posterior over only the most recent N audits. No switch time
    is supplied.

oracle_reset
    Cumulative learner whose evidence is reset exactly at the true switch. This is
    an unattainable upper reference for adaptation timing, not a proposed policy.

sentinel
    Exploratory change detector. A cumulative primary posterior is reset to the
    recent rolling buffer when the recent MAP semantic disagrees with the primary
    MAP and recent confidence exceeds a threshold. Included to test whether extra
    reset machinery beats the simpler rolling baseline.

All predictions use Bayesian semantic model averaging and the same asymmetric
reuse/refresh utility.

Run documented evaluation:
    python experiments/semantic_drift_attack.py
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


@dataclass(frozen=True)
class StreamItem:
    world_age: float
    event_age: int
    observed_valid: bool
    true_p_valid: float
    true_semantic: str


def generate_stream(
    *,
    seed: int,
    total: int,
    drift_at: int,
) -> list[StreamItem]:
    rng = np.random.default_rng(seed)
    rows: list[StreamItem] = []

    for index in range(total):
        event_age = int(rng.integers(0, 21))
        episode_tempo = float(rng.uniform(0.40, 1.60))
        if event_age:
            world_age = float(
                np.sum(
                    episode_tempo
                    * rng.uniform(0.90, 1.10, size=event_age)
                )
            )
        else:
            world_age = 0.0

        if index < drift_at:
            true_semantic = "world_hazard"
            true_p = math.exp(-HAZARD * world_age)
        else:
            true_semantic = "event_hazard"
            true_p = math.exp(-HAZARD * event_age)

        observed_valid = bool(rng.random() < true_p)
        rows.append(
            StreamItem(
                world_age=world_age,
                event_age=event_age,
                observed_valid=observed_valid,
                true_p_valid=true_p,
                true_semantic=true_semantic,
            )
        )

    return rows


def to_audit(item: StreamItem) -> AuditObservation:
    return AuditObservation(
        world_age_seconds=item.world_age,
        event_age=item.event_age,
        invalidated=False,
        still_valid=item.observed_valid,
    )


def coordinates(item: StreamItem) -> TemporalCoordinates:
    return TemporalCoordinates(
        world_age_seconds=item.world_age,
        knowledge_age_seconds=item.world_age,
        event_age=item.event_age,
        invalidated=False,
    )


def action_metrics(p_valid_pred: float, p_valid_true: float) -> tuple[float, float, float, float]:
    reuse = p_valid_pred >= REUSE_THRESHOLD
    oracle_reuse = p_valid_true >= REUSE_THRESHOLD

    agreement = float(reuse == oracle_reuse)
    reuse_utility = (
        p_valid_true * REWARD_VALID_REUSE
        + (1.0 - p_valid_true) * REWARD_STALE_REUSE
    )
    utility = float(reuse_utility if reuse else REWARD_REFRESH)
    bad_reuse = float(reuse and not oracle_reuse)
    refresh = float(not reuse)
    return agreement, utility, bad_reuse, refresh


def posterior_prediction(accumulator: OnlineSemanticAccumulator, item: StreamItem) -> float:
    return accumulator.posterior().probability(coordinates(item), strategy="average")


def rebuild_from_recent(recent_audits: list[AuditObservation]) -> OnlineSemanticAccumulator:
    accumulator = OnlineSemanticAccumulator()
    for audit in recent_audits:
        accumulator.update(audit)
    return accumulator


def run_seed(
    *,
    seed: int,
    total: int,
    warmup: int,
    drift_at: int,
    rolling_window: int,
    sentinel_confidence: float,
):
    stream = generate_stream(seed=seed, total=total, drift_at=drift_at)

    initial = OnlineSemanticAccumulator()
    rolling = OnlineSemanticAccumulator(window=rolling_window)
    recent_audits: list[AuditObservation] = []
    for item in stream[:warmup]:
        audit = to_audit(item)
        initial.update(audit)
        rolling.update(audit)
        recent_audits.append(audit)
        recent_audits = recent_audits[-rolling_window:]

    frozen = initial.copy_as_cumulative()
    cumulative = initial.copy_as_cumulative()
    oracle_reset = initial.copy_as_cumulative()
    sentinel = initial.copy_as_cumulative()

    results = {
        name: []
        for name in (
            "frozen",
            "cumulative",
            "rolling",
            "oracle_reset",
            "sentinel",
        )
    }
    weights = {name: [] for name in results}
    sentinel_resets: list[int] = []

    for index in range(warmup, total):
        if index == drift_at:
            oracle_reset = OnlineSemanticAccumulator()

        item = stream[index]
        models = {
            "frozen": frozen,
            "cumulative": cumulative,
            "rolling": rolling,
            "oracle_reset": oracle_reset,
            "sentinel": sentinel,
        }

        # Prequential evaluation: predict before revealing this audit outcome.
        for name, model in models.items():
            p_pred = posterior_prediction(model, item)
            results[name].append(
                (
                    index,
                    item.true_semantic,
                    *action_metrics(p_pred, item.true_p_valid),
                )
            )
            posterior = model.posterior()
            weights[name].append(
                (
                    index,
                    posterior.model_weights["world_hazard"],
                    posterior.model_weights["event_hazard"],
                    posterior.model_weights["until_change"],
                )
            )

        audit = to_audit(item)
        cumulative.update(audit)
        rolling.update(audit)
        oracle_reset.update(audit)
        sentinel.update(audit)

        recent_audits.append(audit)
        recent_audits = recent_audits[-rolling_window:]

        # Exploratory sentinel: the rolling learner already *is* the recent-window
        # posterior, so use it as the change monitor rather than refitting a new
        # recent model at every step. Rebuild only when a reset actually fires.
        recent_post = rolling.posterior()
        sentinel_post = sentinel.posterior()
        if (
            recent_post.map_semantic != sentinel_post.map_semantic
            and max(recent_post.model_weights.values()) >= sentinel_confidence
        ):
            sentinel = rebuild_from_recent(recent_audits)
            sentinel_resets.append(index)

    return results, weights, sentinel_resets


def region_metrics(rows, start: int, end: int | None = None) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row[0] >= start and (end is None or row[0] < end)
    ]
    if not selected:
        return np.full(4, np.nan)
    return np.asarray([row[2:] for row in selected], dtype=np.float64).mean(axis=0)


def first_event_confidence(weights, *, drift_at: int, threshold: float = 0.90) -> float:
    for index, _world, event, _change in weights:
        if index >= drift_at and event >= threshold:
            return float(index - drift_at)
    return math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=100)
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--drift-at", type=int, default=1000)
    parser.add_argument("--rolling-window", type=int, default=240)
    parser.add_argument("--sentinel-confidence", type=float, default=0.95)
    args = parser.parse_args()

    if not (0 < args.warmup < args.drift_at < args.total):
        raise ValueError("require 0 < warmup < drift-at < total")

    names = ("frozen", "cumulative", "rolling", "oracle_reset", "sentinel")
    collected = {name: [] for name in names}
    adaptation = {name: [] for name in names}
    sentinel_false_reset = 0
    sentinel_post_delay: list[float] = []

    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        results, weights, resets = run_seed(
            seed=seed,
            total=args.total,
            warmup=args.warmup,
            drift_at=args.drift_at,
            rolling_window=args.rolling_window,
            sentinel_confidence=args.sentinel_confidence,
        )

        for name in names:
            collected[name].append(
                {
                    "pre": region_metrics(
                        results[name], args.warmup, args.drift_at
                    ),
                    "early": region_metrics(
                        results[name], args.drift_at, min(args.total, args.drift_at + 200)
                    ),
                    "post": region_metrics(
                        results[name], args.drift_at, None
                    ),
                    "late": region_metrics(
                        results[name], min(args.total, args.drift_at + 300), None
                    ),
                }
            )
            adaptation[name].append(
                first_event_confidence(
                    weights[name],
                    drift_at=args.drift_at,
                    threshold=0.90,
                )
            )

        sentinel_false_reset += int(any(index < args.drift_at for index in resets))
        after = [index for index in resets if index >= args.drift_at]
        sentinel_post_delay.append(
            float(after[0] - args.drift_at) if after else math.nan
        )

    print("semantic drift: world_hazard -> event_hazard")
    print(
        f"seeds={args.seeds} warmup={args.warmup} drift={args.drift_at} "
        f"total={args.total} rolling_window={args.rolling_window}"
    )
    print()

    for name in names:
        print(name)
        for region in ("pre", "early", "post", "late"):
            values = np.asarray([row[region] for row in collected[name]])
            mean = values.mean(axis=0)
            print(
                f"  {region:>5s}: agreement={mean[0]:.4f} "
                f"utility={mean[1]:.4f} bad_reuse={mean[2]:.4f} "
                f"refresh={mean[3]:.4f}"
            )

        delays = np.asarray(adaptation[name], dtype=np.float64)
        finite = np.isfinite(delays)
        if np.any(finite):
            print(
                f"  P(event_hazard)>0.90: success={finite.mean():.3f} "
                f"mean_delay={delays[finite].mean():.1f} audits"
            )
        else:
            print("  P(event_hazard)>0.90: never reached")
        print()

    sentinel_delay = np.asarray(sentinel_post_delay, dtype=np.float64)
    finite = np.isfinite(sentinel_delay)
    print("sentinel diagnostics")
    print(
        f"  false-reset seeds before drift: {sentinel_false_reset}/{args.seeds}"
    )
    if np.any(finite):
        print(
            f"  post-drift reset success={finite.mean():.3f} "
            f"mean_delay={sentinel_delay[finite].mean():.1f} audits"
        )


if __name__ == "__main__":
    main()
