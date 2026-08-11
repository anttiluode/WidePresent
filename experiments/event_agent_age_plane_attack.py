"""Event-agent attack for WidePresent's age-plane idea.

This leaves the hand-designed exponential classification toys and moves to an
operational agent decision:

    REUSE a cached result, or REFRESH the tool?

The environment mixes three evidence semantics in the same stream:

* weather-like cache validity decays with elapsed WALL TIME;
* discourse/reference validity decays with INTERVENING EVENT COUNT;
* state/reservation facts do not age; they remain valid until an explicit
  invalidation event.

Tool results also have valid/world time and arrival/knowledge time. A result can
arrive recently while already being old in world time.

The learner is not given oracle freshness labels. Training labels are noisy
observations of whether a cached fact was actually still correct. Policies
estimate P(valid), then decide whether reuse is worth the risk relative to a
refresh cost.

The central comparison is between ordinary timestamp/position summaries, a
content-conditioned age-plane linear policy, a generic boosted tree, and a boring
per-source hazard model that explicitly chooses whether its hazard lives in
seconds or event count.

Run a quick smoke test:
    python experiments/event_agent_age_plane_attack.py --seeds 2 --train 1800 --test 1200

Run the documented evaluation:
    python experiments/event_agent_age_plane_attack.py --seeds 5 --train 6000 --test 3500
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


KINDS = ("weather", "discourse", "state")

REWARD_VALID_REUSE = 1.0
REWARD_STALE_REUSE = -1.5
REWARD_REFRESH = 0.55
P_REUSE_THRESHOLD = (
    (REWARD_REFRESH - REWARD_STALE_REUSE)
    / (REWARD_VALID_REUSE - REWARD_STALE_REUSE)
)

# Hazards are chosen so the optimal nominal reuse horizon is roughly 8 seconds
# for weather and 8 events for discourse.
HAZARD = -math.log(P_REUSE_THRESHOLD) / 8.0
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


@dataclass
class Batch:
    raw: np.ndarray
    plane: np.ndarray
    observed_valid: np.ndarray
    true_p_valid: np.ndarray
    kind: np.ndarray


def generate_batch(
    *,
    seed: int,
    n: int,
    gap_lo: float,
    gap_hi: float,
    delay_scale: float,
) -> Batch:
    """Generate independent final decision states from event-agent histories.

    Every feature corresponds to a quantity available from a timestamped event
    stream. For speed the histories are generated in vectorized form.

    raw columns:
        0..2  source-kind one hot
        3     valid/world age in seconds
        4     arrival/knowledge age in seconds
        5     intervening event count
        6     explicit invalidation observed
        7     source-confidence distractor
        8     content-value distractor

    `plane` expands age coordinates by source type so a linear readout can assign
    a different temporal metric to each evidence type.
    """

    rng = np.random.default_rng(seed)
    kind = rng.integers(0, len(KINDS), size=n)
    event_age = rng.integers(0, MAX_EVENTS + 1, size=n)

    # Each episode has its own interaction tempo. Local gaps jitter around that
    # tempo. If individual gaps were independently redrawn from the whole range,
    # long histories would average back toward one common rate and erase the very
    # rate diversity this assay is intended to manipulate.
    episode_gap = rng.uniform(gap_lo, gap_hi, size=n)
    jitter = rng.uniform(0.90, 1.10, size=(n, MAX_EVENTS))
    gaps = episode_gap[:, None] * jitter
    mask = np.arange(MAX_EVENTS)[None, :] < event_age[:, None]
    valid_age = (gaps * mask).sum(axis=1)

    # Asynchronous tool delivery. A cached result may have arrived recently even
    # when its world-valid timestamp is much older.
    delay = rng.exponential(delay_scale, size=n)
    delay = np.minimum(delay, valid_age * 0.95)
    delay = np.where(valid_age > 0.0, delay, 0.0)
    arrival_age = np.maximum(valid_age - delay, 0.0)

    # State/reservation facts are event-invalidated, not age-decayed.
    p_invalidation = 1.0 - (1.0 - 0.06) ** event_age
    invalidation = (
        (rng.random(n) < p_invalidation) & (kind == 2)
    ).astype(np.float64)

    # Ground-truth probability that the cached fact is still correct.
    p_valid = np.where(
        kind == 0,
        np.exp(-HAZARD * valid_age),
        np.where(
            kind == 1,
            np.exp(-HAZARD * event_age),
            1.0 - invalidation,
        ),
    )

    # Learners see outcomes, not p_valid or the oracle policy.
    observed_valid = (rng.random(n) < p_valid).astype(np.int64)

    source_confidence = rng.uniform(0.65, 0.99, size=n)
    content_value = rng.normal(size=n)
    one_hot = np.eye(len(KINDS), dtype=np.float64)[kind]

    raw = np.column_stack(
        [
            one_hot,
            valid_age,
            arrival_age,
            event_age.astype(np.float64),
            invalidation,
            source_confidence,
            content_value,
        ]
    )

    plane_parts: list[np.ndarray] = []
    for source_index in range(len(KINDS)):
        source = (kind == source_index).astype(np.float64)
        plane_parts.extend(
            [
                source,
                source * valid_age,
                source * event_age,
                source * arrival_age,
                source * invalidation,
            ]
        )
    plane = np.column_stack(plane_parts + [source_confidence, content_value])

    return Batch(raw, plane, observed_valid, p_valid, kind)


def realized_reward(reuse: np.ndarray, observed_valid: np.ndarray) -> np.ndarray:
    return np.where(
        reuse,
        np.where(observed_valid == 1, REWARD_VALID_REUSE, REWARD_STALE_REUSE),
        REWARD_REFRESH,
    )


def expected_reward(reuse: np.ndarray, p_valid: np.ndarray) -> np.ndarray:
    reuse_utility = (
        p_valid * REWARD_VALID_REUSE
        + (1.0 - p_valid) * REWARD_STALE_REUSE
    )
    return np.where(reuse, reuse_utility, REWARD_REFRESH)


def choose_threshold(probability: np.ndarray, observed_valid: np.ndarray) -> float:
    """Choose a reuse threshold from held-out training outcomes only."""
    best_utility = -np.inf
    best_threshold = P_REUSE_THRESHOLD
    for threshold in np.linspace(0.05, 0.99, 95):
        utility = realized_reward(
            probability >= threshold,
            observed_valid,
        ).mean()
        if utility > best_utility:
            best_utility = float(utility)
            best_threshold = float(threshold)
    return best_threshold


class LearnedPolicy:
    def __init__(self, model, representation: str, columns, threshold: float):
        self.model = model
        self.representation = representation
        self.columns = columns
        self.threshold = threshold

    def _features(self, batch: Batch) -> np.ndarray:
        if self.representation == "plane":
            return batch.plane
        if self.columns is None:
            return batch.raw
        return batch.raw[:, self.columns]

    def probability(self, batch: Batch) -> np.ndarray:
        return self.model.predict_proba(self._features(batch))[:, 1]

    def action(self, batch: Batch) -> np.ndarray:
        return self.probability(batch) >= self.threshold


class HazardSelector:
    """Boring survival-model attacker.

    For weather/discourse separately, fit an exponential survival law on either
    wall-time age or event-count age and choose the axis with the larger training
    likelihood. State facts use the explicit invalidation bit.
    """

    def __init__(self) -> None:
        self.rules: dict[int, tuple] = {}
        self.threshold = P_REUSE_THRESHOLD

    @staticmethod
    def _fit_hazard(age: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        best_nll = np.inf
        best_lambda = HAZARD
        for lam in np.geomspace(0.001, 0.20, 240):
            p = np.clip(np.exp(-lam * age), 1e-6, 1.0 - 1e-6)
            nll = -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
            if nll < best_nll:
                best_nll = nll
                best_lambda = float(lam)
        return best_nll, best_lambda

    def fit(
        self,
        batch: Batch,
        train_index: np.ndarray,
        valid_index: np.ndarray,
    ) -> "HazardSelector":
        kind = np.argmax(batch.raw[:, :3], axis=1)
        for source_index in range(len(KINDS)):
            if source_index == 2:
                self.rules[source_index] = ("explicit",)
                continue

            source_train = train_index[kind[train_index] == source_index]
            candidates = []
            for column, name in ((3, "seconds"), (5, "events")):
                nll, lam = self._fit_hazard(
                    batch.raw[source_train, column],
                    batch.observed_valid[source_train],
                )
                candidates.append((nll, name, column, lam))
            self.rules[source_index] = min(candidates, key=lambda item: item[0])

        p_valid = self.probability(batch, index=valid_index)
        self.threshold = choose_threshold(
            p_valid,
            batch.observed_valid[valid_index],
        )
        return self

    def probability(
        self,
        batch: Batch,
        index: np.ndarray | None = None,
    ) -> np.ndarray:
        X = batch.raw if index is None else batch.raw[index]
        kind = np.argmax(X[:, :3], axis=1)
        out = np.empty(len(X), dtype=np.float64)
        for source_index in range(len(KINDS)):
            mask = kind == source_index
            if source_index == 2:
                out[mask] = 1.0 - X[mask, 6]
            else:
                _nll, _name, column, lam = self.rules[source_index]
                out[mask] = np.exp(-lam * X[mask, column])
        return out

    def action(self, batch: Batch) -> np.ndarray:
        return self.probability(batch) >= self.threshold

    def summary(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for source_index, source in enumerate(KINDS):
            rule = self.rules[source_index]
            if source_index == 2:
                out[source] = "explicit invalidation"
            else:
                out[source] = {
                    "axis": rule[1],
                    "hazard": rule[3],
                }
        out["reuse_threshold"] = self.threshold
        return out


def train_policies(
    *,
    seed: int,
    n_train: int,
    train_gap: tuple[float, float],
) -> tuple[dict[str, LearnedPolicy], HazardSelector]:
    batch = generate_batch(
        seed=100 + seed,
        n=n_train,
        gap_lo=train_gap[0],
        gap_hi=train_gap[1],
        delay_scale=1.5,
    )

    rng = np.random.default_rng(900 + seed)
    index = np.arange(n_train)
    rng.shuffle(index)
    split = int(0.75 * n_train)
    train_index = index[:split]
    valid_index = index[split:]

    arrival_columns = [0, 1, 2, 4, 6, 7, 8]
    timestamp_columns = [0, 1, 2, 3, 4, 6, 7, 8]
    position_columns = [0, 1, 2, 5, 6, 7, 8]

    specs = (
        ("arrival_logit", "raw", arrival_columns),
        ("timestamp_logit", "raw", timestamp_columns),
        ("position_logit", "raw", position_columns),
        ("raw_both_logit", "raw", None),
        ("age_plane_logit", "plane", None),
    )

    policies: dict[str, LearnedPolicy] = {}
    for name, representation, columns in specs:
        source = batch.plane if representation == "plane" else batch.raw
        X_train = source[train_index] if columns is None else source[train_index][:, columns]
        X_valid = source[valid_index] if columns is None else source[valid_index][:, columns]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1800, C=0.5),
        )
        model.fit(X_train, batch.observed_valid[train_index])
        threshold = choose_threshold(
            model.predict_proba(X_valid)[:, 1],
            batch.observed_valid[valid_index],
        )
        policies[name] = LearnedPolicy(model, representation, columns, threshold)

    tree = HistGradientBoostingClassifier(
        max_iter=120,
        max_depth=4,
        learning_rate=0.06,
        l2_regularization=0.2,
        random_state=seed,
    )
    tree.fit(batch.raw[train_index], batch.observed_valid[train_index])
    tree_threshold = choose_threshold(
        tree.predict_proba(batch.raw[valid_index])[:, 1],
        batch.observed_valid[valid_index],
    )
    policies["gbdt_both"] = LearnedPolicy(tree, "raw", None, tree_threshold)

    hazard = HazardSelector().fit(batch, train_index, valid_index)
    return policies, hazard


def score_action(
    reuse: np.ndarray,
    batch: Batch,
) -> tuple[float, float, float, float]:
    oracle_reuse = batch.true_p_valid >= P_REUSE_THRESHOLD
    action_accuracy = float(np.mean(reuse == oracle_reuse))
    utility = float(expected_reward(reuse, batch.true_p_valid).mean())
    refresh_rate = float(np.mean(~reuse))
    bad_reuse = float(np.mean(reuse & (~oracle_reuse)))
    return action_accuracy, utility, refresh_rate, bad_reuse


def evaluate(
    *,
    train_name: str,
    train_gap: tuple[float, float],
    seeds: int,
    n_train: int,
    n_test: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    hazard_summaries: list[dict[str, object]] = []

    for seed in range(seeds):
        policies, hazard = train_policies(
            seed=seed,
            n_train=n_train,
            train_gap=train_gap,
        )
        hazard_summaries.append({"seed": seed, **hazard.summary()})

        for regime_index, (regime, (lo, hi, delay_scale)) in enumerate(REGIMES.items()):
            test = generate_batch(
                seed=10000 + 100 * seed + regime_index,
                n=n_test,
                gap_lo=lo,
                gap_hi=hi,
                delay_scale=delay_scale,
            )

            oracle = test.true_p_valid >= P_REUSE_THRESHOLD
            acc, utility, refresh, bad = score_action(oracle, test)
            rows.append(
                {
                    "train": train_name,
                    "seed": seed,
                    "regime": regime,
                    "model": "oracle",
                    "action_accuracy": acc,
                    "utility": utility,
                    "refresh_rate": refresh,
                    "bad_reuse": bad,
                }
            )

            hazard_action = hazard.action(test)
            acc, utility, refresh, bad = score_action(hazard_action, test)
            rows.append(
                {
                    "train": train_name,
                    "seed": seed,
                    "regime": regime,
                    "model": "hazard_selector",
                    "action_accuracy": acc,
                    "utility": utility,
                    "refresh_rate": refresh,
                    "bad_reuse": bad,
                }
            )

            for name, policy in policies.items():
                action = policy.action(test)
                acc, utility, refresh, bad = score_action(action, test)
                rows.append(
                    {
                        "train": train_name,
                        "seed": seed,
                        "regime": regime,
                        "model": name,
                        "action_accuracy": acc,
                        "utility": utility,
                        "refresh_rate": refresh,
                        "bad_reuse": bad,
                    }
                )

    return rows, hazard_summaries


def summarize(rows: list[dict[str, object]]) -> None:
    metrics = ("action_accuracy", "utility", "refresh_rate", "bad_reuse")
    train_names = sorted({str(row["train"]) for row in rows})
    models = (
        "arrival_logit",
        "timestamp_logit",
        "position_logit",
        "raw_both_logit",
        "age_plane_logit",
        "gbdt_both",
        "hazard_selector",
        "oracle",
    )

    for train_name in train_names:
        print(f"\n=== training rate: {train_name} ===")
        for regime in REGIMES:
            print(f"\n[{regime}]")
            for model in models:
                selected = [
                    row
                    for row in rows
                    if row["train"] == train_name
                    and row["regime"] == regime
                    and row["model"] == model
                ]
                if not selected:
                    continue
                values = {
                    metric: np.asarray([float(row[metric]) for row in selected])
                    for metric in metrics
                }
                print(
                    f"{model:>18s}  "
                    f"action={values['action_accuracy'].mean():.3f}  "
                    f"utility={values['utility'].mean():.3f}  "
                    f"refresh={values['refresh_rate'].mean():.3f}  "
                    f"bad_reuse={values['bad_reuse'].mean():.3f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--test", type=int, default=3500)
    parser.add_argument(
        "--only",
        choices=("narrow", "wide", "both"),
        default="both",
        help="training-rate diversity condition",
    )
    args = parser.parse_args()

    print("event-agent age-plane attack")
    print(f"reuse threshold from utility = {P_REUSE_THRESHOLD:.3f}")
    print(f"true weather/event hazard = {HAZARD:.5f}")
    print("actions: REUSE cached evidence vs REFRESH tool")

    conditions = []
    if args.only in {"narrow", "both"}:
        conditions.append(("narrow_0.95_1.05", NARROW_TRAIN))
    if args.only in {"wide", "both"}:
        conditions.append(("wide_0.40_1.60", WIDE_TRAIN))

    all_rows: list[dict[str, object]] = []
    all_hazards: list[dict[str, object]] = []
    for name, gap in conditions:
        rows, hazards = evaluate(
            train_name=name,
            train_gap=gap,
            seeds=args.seeds,
            n_train=args.train,
            n_test=args.test,
        )
        all_rows.extend(rows)
        all_hazards.extend({"train": name, **row} for row in hazards)

    summarize(all_rows)

    print("\nHazard-selector choices")
    for row in all_hazards:
        print(row)

    print("\nInterpretation discipline:")
    print("  - high IID score does not establish the correct temporal invariant")
    print("  - recent arrival is not the same as recent valid/world time")
    print("  - age-plane interactions are a representation baseline, not a novelty claim")
    print("  - if rate diversity identifies the axis, ordinary hazard modeling is the primary attacker")


if __name__ == "__main__":
    main()
