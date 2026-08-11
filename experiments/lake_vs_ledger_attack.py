"""Matched attack: local dynamical state versus boring timestamped temporal state.

This experiment is deliberately hostile to the lake/wave branch.

Every representation receives the SAME delivered observations.  Each observation is
(label, valid/world time, arrival/knowledge time).  The task is to identify the
current hidden state at t=100 after a late state change.  Old observations can arrive
late and masquerade as current evidence.

All learned readouts are the same standardized multinomial logistic regression.
The main state budget is 60 real numbers.

Representations
---------------
arrival_hist  : 3 labels x 20 ARRIVAL-age bins = 60
valid_hist    : 3 labels x 20 VALID-age bins   = 60
exp_bank      : 3 labels x 20 exponential valid-time kernels = 60
wave_only     : 30-node local spring ring, displacement+velocity = 60
wave_oil      : 20-node spring ring, displacement+velocity+slow diffusion = 60
oil_only      : only the 20-number slow diffusion field (under-uses the budget)
modal_exp20   : exact non-geometric modal form of oil_only = 20

The wave states are given a generous valid-time compensation: when a delayed event
arrives with a valid timestamp, its contribution is aged as if it had propagated from
that valid time.  Thus the lake is NOT penalized for lacking the timestamp that the
ledger receives.

The graph/damping constants were selected on separate development seeds, then frozen
before the five evaluation seeds reported in docs/LAKE_VS_LEDGER_ATTACK.md.

A matched 25% active-refresh evaluation is also included.  Each representation ranks
test episodes by its own uncertainty; exactly 25% receive the same three fresh noisy
observations.  This tests whether a dynamical state is unusually good at knowing when
its estimate of NOW is unreliable.

Run the documented full evaluation:
    python experiments/lake_vs_ledger_attack.py --train 4000 --test 1800 --seeds 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_STATES = 3
DECISION_TIME = 100.0
N_BINS = 20
REFRESH_BUDGET = 0.25
REGIMES = ("iid", "sparse", "dense", "longdelay")


def other_state(rng: np.random.Generator, state: int) -> int:
    candidate = int(rng.integers(N_STATES - 1))
    return candidate + 1 if candidate >= state else candidate


def noisy_label(rng: np.random.Generator, state: int, accuracy: float) -> int:
    return state if rng.random() < accuracy else other_state(rng, state)


def regime_parameters(regime: str) -> tuple[float, float, float, float]:
    if regime == "iid":
        return 0.28, 0.58, 2.5, 18.0
    if regime == "sparse":
        return 0.12, 0.58, 2.5, 18.0
    if regime == "dense":
        return 0.55, 0.58, 2.5, 18.0
    if regime == "longdelay":
        return 0.28, 0.25, 4.0, 28.0
    raise ValueError(regime)


def base_episode(
    rng: np.random.Generator, regime: str
) -> tuple[list[tuple[int, float, float]], int]:
    old_state = int(rng.integers(N_STATES))
    new_state = other_state(rng, old_state)
    switch_time = rng.uniform(80.0, 94.0)
    rate, p_short, short_scale, long_scale = regime_parameters(regime)

    events: list[tuple[int, float, float]] = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t > DECISION_TIME:
            break
        state_then = old_state if t < switch_time else new_state
        value = noisy_label(rng, state_then, accuracy=0.74)
        scale = short_scale if rng.random() < p_short else long_scale
        arrival = t + rng.exponential(scale)
        if arrival <= DECISION_TIME:
            events.append((value, t, arrival))

    events.sort(key=lambda event: event[2])
    return events, new_state


def add_refresh(
    events: list[tuple[int, float, float]],
    current_state: int,
    rng: np.random.Generator,
) -> list[tuple[int, float, float]]:
    out = list(events)
    for world_time in (96.0, 97.0, 98.0):
        value = noisy_label(rng, current_state, accuracy=0.88)
        arrival = world_time + rng.uniform(0.05, 0.30)
        if arrival <= DECISION_TIME:
            out.append((value, world_time, arrival))
    out.sort(key=lambda event: event[2])
    return out


AGE_BINS = np.linspace(0.0, 100.000001, N_BINS + 1)


def histogram_feature(
    events: list[tuple[int, float, float]], *, use_valid_time: bool
) -> np.ndarray:
    counts = np.zeros((N_STATES, N_BINS), dtype=np.float64)
    for value, valid_time, arrival_time in events:
        source_time = valid_time if use_valid_time else arrival_time
        age = DECISION_TIME - source_time
        bucket = int(np.searchsorted(AGE_BINS, age, side="right") - 1)
        if 0 <= bucket < N_BINS:
            counts[value, bucket] += 1.0
    return np.sqrt(counts).ravel()


# ---------------------------------------------------------------------------
# Local graph substrates.
# ---------------------------------------------------------------------------


def ring_laplacian(n: int, anchor: float) -> np.ndarray:
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        weight = 0.7 + 0.22 * np.sin(0.8 * i + 0.2) + 0.05 * ((i % 4) - 1.5)
        K[i, i] += weight
        K[j, j] += weight
        K[i, j] -= weight
        K[j, i] -= weight
    return K + anchor * np.eye(n)


# Pure-wave substrate: 30 positions + 30 velocities = 60 state numbers.
WAVE_N = 30
WAVE_K = ring_laplacian(WAVE_N, anchor=0.08)
WAVE_LAM, WAVE_Q = np.linalg.eigh(WAVE_K)
WAVE_PORTS = np.zeros((WAVE_N, N_STATES), dtype=np.float64)
WAVE_PORTS[0, 0] = 1.0
WAVE_PORTS[10, 1] = 1.0
WAVE_PORTS[20, 2] = 1.0
WAVE_MODAL_PORTS = WAVE_Q.T @ WAVE_PORTS
WAVE_GAMMA = 0.40  # chosen on separate development data
WAVE_W = np.sqrt(np.maximum(WAVE_LAM - WAVE_GAMMA**2 / 4.0, 1e-6))


def wave_only_feature(events: list[tuple[int, float, float]]) -> np.ndarray:
    if not events:
        return np.zeros(2 * WAVE_N, dtype=np.float64)
    labels = np.fromiter((e[0] for e in events), dtype=int)
    ages = DECISION_TIME - np.fromiter((e[1] for e in events), dtype=float)
    ports = WAVE_MODAL_PORTS[:, labels].T
    decay = np.exp(-WAVE_GAMMA * ages[:, None] / 2.0)
    sin = np.sin(ages[:, None] * WAVE_W[None, :])
    cos = np.cos(ages[:, None] * WAVE_W[None, :])
    uq = (decay * sin / WAVE_W[None, :] * ports).sum(axis=0)
    vq = (
        decay
        * (cos - WAVE_GAMMA * sin / (2.0 * WAVE_W[None, :]))
        * ports
    ).sum(axis=0)
    return np.concatenate([WAVE_Q @ uq, WAVE_Q @ vq])


# Wave+oil substrate: 20 displacement + 20 velocity + 20 slow field = 60.
OIL_N = 20
OIL_K = ring_laplacian(OIL_N, anchor=0.06)
OIL_LAM, OIL_Q = np.linalg.eigh(OIL_K)
OIL_PORTS = np.zeros((OIL_N, N_STATES), dtype=np.float64)
OIL_PORTS[0, 0] = 1.0
OIL_PORTS[7, 1] = 1.0
OIL_PORTS[14, 2] = 1.0
OIL_MODAL_PORTS = OIL_Q.T @ OIL_PORTS
OIL_GAMMA = 0.50
OIL_DIFFUSION = 0.05
OIL_DECAY = 0.02
OIL_W = np.sqrt(np.maximum(OIL_LAM - OIL_GAMMA**2 / 4.0, 1e-6))
OIL_RATES = OIL_DIFFUSION * OIL_LAM + OIL_DECAY


def _oil_modal(events: list[tuple[int, float, float]]) -> np.ndarray:
    if not events:
        return np.zeros(OIL_N, dtype=np.float64)
    labels = np.fromiter((e[0] for e in events), dtype=int)
    ages = DECISION_TIME - np.fromiter((e[1] for e in events), dtype=float)
    ports = OIL_MODAL_PORTS[:, labels].T
    return (np.exp(-ages[:, None] * OIL_RATES[None, :]) * ports).sum(axis=0)


def oil_only_feature(events: list[tuple[int, float, float]]) -> np.ndarray:
    return OIL_Q @ _oil_modal(events)


def modal_exp20_feature(events: list[tuple[int, float, float]]) -> np.ndarray:
    """Exact non-geometric modal coordinates of oil_only_feature."""
    return _oil_modal(events)


def wave_oil_feature(events: list[tuple[int, float, float]]) -> np.ndarray:
    if not events:
        return np.zeros(3 * OIL_N, dtype=np.float64)
    labels = np.fromiter((e[0] for e in events), dtype=int)
    ages = DECISION_TIME - np.fromiter((e[1] for e in events), dtype=float)
    ports = OIL_MODAL_PORTS[:, labels].T

    decay = np.exp(-OIL_GAMMA * ages[:, None] / 2.0)
    sin = np.sin(ages[:, None] * OIL_W[None, :])
    cos = np.cos(ages[:, None] * OIL_W[None, :])
    uq = (decay * sin / OIL_W[None, :] * ports).sum(axis=0)
    vq = (
        decay
        * (cos - OIL_GAMMA * sin / (2.0 * OIL_W[None, :]))
        * ports
    ).sum(axis=0)
    mq = (np.exp(-ages[:, None] * OIL_RATES[None, :]) * ports).sum(axis=0)
    return np.concatenate([OIL_Q @ uq, OIL_Q @ vq, OIL_Q @ mq])


def exp_bank_feature(events: list[tuple[int, float, float]]) -> np.ndarray:
    """Boring 60-number exponential filter bank matched to the oil decay rates."""
    state = np.zeros((N_STATES, OIL_N), dtype=np.float64)
    for value, valid_time, _arrival_time in events:
        age = DECISION_TIME - valid_time
        state[value] += np.exp(-OIL_RATES * age)
    return state.ravel()


FEATURES = {
    "arrival_hist": lambda e: histogram_feature(e, use_valid_time=False),
    "valid_hist": lambda e: histogram_feature(e, use_valid_time=True),
    "exp_bank": exp_bank_feature,
    "wave_only": wave_only_feature,
    "oil_only": oil_only_feature,
    "wave_oil": wave_oil_feature,
    "modal_exp20": modal_exp20_feature,
}


@dataclass
class EpisodeSet:
    passive: list[list[tuple[int, float, float]]]
    refreshed: list[list[tuple[int, float, float]]]
    labels: np.ndarray


def generate_set(seed: int, n: int, regime: str) -> EpisodeSet:
    rng = np.random.default_rng(seed)
    passive = []
    refreshed = []
    labels = []
    for _ in range(n):
        events, current = base_episode(rng, regime)
        passive.append(events)
        refreshed.append(add_refresh(events, current, rng))
        labels.append(current)
    return EpisodeSet(passive, refreshed, np.asarray(labels))


def feature_set(events: list[list[tuple[int, float, float]]]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([function(episode) for episode in events])
        for name, function in FEATURES.items()
    }


def classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5),
    )


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    confidence = probability.max(axis=1)
    correct = (probability.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        high = edges[index + 1] + (1e-12 if index == bins - 1 else 0.0)
        mask = (confidence >= edges[index]) & (confidence < high)
        if np.any(mask):
            total += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def score_condition(
    passive_model,
    refresh_model,
    X: np.ndarray,
    X_refresh: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    probability = passive_model.predict_proba(X)
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    errors = prediction != y

    # Exact matched query budget: rank only by the model's own uncertainty.
    n_query = int(round(REFRESH_BUDGET * len(y)))
    query = np.zeros(len(y), dtype=bool)
    query[np.argsort(confidence)[:n_query]] = True
    refreshed_prediction = refresh_model.predict(X_refresh)
    mixed = prediction.copy()
    mixed[query] = refreshed_prediction[query]

    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "logloss": float(log_loss(y, probability, labels=[0, 1, 2])),
        "ece": ece(y, probability),
        "error_auc": float(roc_auc_score(errors.astype(int), 1.0 - confidence)),
        "refresh25_accuracy": float(accuracy_score(y, mixed)),
        "error_recall25": float(query[errors].mean()) if np.any(errors) else 0.0,
    }


def run_seed(seed: int, train_n: int, test_n: int) -> dict:
    train = generate_set(seed * 1000 + 1, train_n, "iid")
    train_X = feature_set(train.passive)
    train_X_refresh = feature_set(train.refreshed)

    passive_models = {}
    refresh_models = {}
    for name in FEATURES:
        passive_models[name] = classifier().fit(train_X[name], train.labels)
        refresh_models[name] = classifier().fit(train_X_refresh[name], train.labels)

    result = {}
    for regime_index, regime in enumerate(REGIMES):
        test = generate_set(seed * 1000 + 100 + regime_index, test_n, regime)
        X = feature_set(test.passive)
        X_refresh = feature_set(test.refreshed)
        result[regime] = {
            name: score_condition(
                passive_models[name],
                refresh_models[name],
                X[name],
                X_refresh[name],
                test.labels,
            )
            for name in FEATURES
        }
    return result


def print_means(rows: list[dict]) -> None:
    names = tuple(FEATURES)
    for regime in REGIMES:
        print(f"\n{regime}")
        print("representation   acc     refresh25   error-AUC   logloss")
        for name in names:
            acc = np.mean([row[regime][name]["accuracy"] for row in rows])
            refreshed = np.mean(
                [row[regime][name]["refresh25_accuracy"] for row in rows]
            )
            auc = np.mean([row[regime][name]["error_auc"] for row in rows])
            loss = np.mean([row[regime][name]["logloss"] for row in rows])
            print(f"{name:14s} {acc:7.3f}   {refreshed:7.3f}     {auc:7.3f}   {loss:7.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--test", type=int, default=1800)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    # Exact algebraic guard: graph-space oil is only an orthogonal rotation of the
    # 20-number modal exponential state.
    probe = [(0, 12.0, 14.0), (2, 81.0, 83.0), (1, 95.0, 95.2)]
    assert np.allclose(oil_only_feature(probe), OIL_Q @ modal_exp20_feature(probe))

    rows = [run_seed(seed, args.train, args.test) for seed in range(args.seeds)]
    print("lake versus boring temporal state")
    print("chance accuracy = 0.333; exact refresh budget = 0.25")
    print_means(rows)
    print("\nInterpretation: see docs/LAKE_VS_LEDGER_ATTACK.md")


if __name__ == "__main__":
    main()
