"""Gate 0: distinguish event index from elapsed time.

This is not intended to prove WidePresent is better than time-aware models.
It checks the minimum premise: two streams can be identical as event sequences
while differing in objective duration. An event-index-only system cannot solve
that distinction; an explicit timestamp or a fixed-pitch substrate can.

A second adversarial split makes event density a misleading proxy for duration.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@dataclass
class Trial:
    duration: float
    distractors: int
    label: int


def make_trials(n: int, rng: np.random.Generator, split: str) -> list[Trial]:
    out: list[Trial] = []
    for _ in range(n):
        label = int(rng.integers(0, 2))
        duration = rng.uniform(0.35, 0.75) if label == 0 else rng.uniform(1.25, 1.65)

        if split == "train":
            lam = 3.0 if label == 0 else 11.0
        elif split == "iid":
            lam = 3.0 if label == 0 else 11.0
        elif split == "ood":
            lam = 11.0 if label == 0 else 3.0
        else:
            raise ValueError(split)
        distractors = int(rng.poisson(lam=lam))
        out.append(Trial(float(duration), distractors, label))
    return out


def arr(trials: list[Trial], dt: float):
    y = np.array([t.label for t in trials], dtype=int)
    event_x = np.array([[t.distractors] for t in trials], dtype=float)
    time_x = np.array([[t.duration] for t in trials], dtype=float)
    tick_x = np.array([[round(t.duration / dt)] for t in trials], dtype=float)
    return y, event_x, time_x, tick_x


def fit_and_score(xtr, ytr, xte, yte):
    model = LogisticRegression(C=1000.0, solver="lbfgs")
    model.fit(xtr, ytr)
    return accuracy_score(yte, model.predict(xte))


def run(seed: int = 0, n_train: int = 3000, n_test: int = 2000, dt: float = 0.02):
    rng = np.random.default_rng(seed)
    train = make_trials(n_train, rng, "train")
    iid = make_trials(n_test, rng, "iid")
    ood = make_trials(n_test, rng, "ood")

    ytr, e_tr, t_tr, k_tr = arr(train, dt)
    yi, e_i, t_i, k_i = arr(iid, dt)
    yo, e_o, t_o, k_o = arr(ood, dt)

    result = {
        "event_index_iid": fit_and_score(e_tr, ytr, e_i, yi),
        "event_index_ood": fit_and_score(e_tr, ytr, e_o, yo),
        "timestamp_iid": fit_and_score(t_tr, ytr, t_i, yi),
        "timestamp_ood": fit_and_score(t_tr, ytr, t_o, yo),
        "fixed_tick_iid": fit_and_score(k_tr, ytr, k_i, yi),
        "fixed_tick_ood": fit_and_score(k_tr, ytr, k_o, yo),
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.02)
    args = p.parse_args()
    r = run(seed=args.seed, dt=args.dt)
    print("Gate 0 — clock vs event index")
    for k, v in r.items():
        print(f"{k:22s} {v:.4f}")
    print("\nInterpretation:")
    print("- event-index can look excellent IID while reversing under a rate confound")
    print("- explicit timestamps and fixed ticks should both survive")
    print("- therefore Gate 0 does NOT establish a WidePresent advantage over time-aware baselines")


if __name__ == "__main__":
    main()
