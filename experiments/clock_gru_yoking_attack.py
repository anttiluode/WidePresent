"""Harder yoking attack: free GRUs trained at one nominal event rate.

The scalar assay in structure_yoking_clock_assay.py proves that a hard content-blind
clock can enforce absolute-time yoking.  This script asks the harder question:

    does merely giving a learned recurrent network blank clock ticks make its own
    receptive field time-yoked and useful under rate shift?

Training
--------
All models see only 32-event sequences with a 0.5 s inter-event gap.  The target is an
exponentially weighted sum with tau=2.5 s.  At this one rate, that target is exactly
indistinguishable from a structure-yoked target with a 5-event decay scale.

Models
------
event_gru
    Updates only on content observations.

dt_gru
    Updates only on content observations but receives [value, dt].  Since dt is
    constant during training, the network is free to ignore it.

clock_gru
    Runs on a fixed 0.1 s grid and receives [value, content_mask].  Blank ticks occur
    between content events.  Under rate shift the number of blank ticks changes.

Evaluation
----------
The same trained networks are tested on compressed and stretched event spacing.  We
score both:

1. absolute-time-yoked targets;
2. structure/event-yoked targets.

We also estimate each network's integration width by gradient sensitivity of the final
logit to each prior content value.  The width is measured in two ways:

- age containing 80% of total gradient mass;
- gradient-weighted mean age.

For either width W,

    SYI = log(W_stretched / W_compressed) / log(3)

where 0 is time-yoked and 1 is structure-yoked.

This is still a synthetic proof/attack.  It does not establish a biological clock or
novel neural mechanism.

Run:
    python experiments/clock_gru_yoking_attack.py
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


N_EVENTS = 32
NOMINAL_GAP = 0.5
SQRT3 = math.sqrt(3.0)
GAPS = {
    "compressed": NOMINAL_GAP / SQRT3,
    "iid": NOMINAL_GAP,
    "stretched": NOMINAL_GAP * SQRT3,
}
TAU = 2.5
TARGET_NOISE = 0.35
CLOCK_TICK = 0.1


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _out, hidden = self.gru(x)
        return self.head(hidden[-1])


def make_target_data(
    n: int,
    *,
    seed: int,
    gap: float,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.choice((-1.0, 1.0), size=(n, N_EVENTS)).astype(np.float32)

    if target == "time":
        ages = np.arange(N_EVENTS - 1, -1, -1, dtype=np.float64) * gap
        weights = np.exp(-ages / TAU)
    elif target == "structure":
        event_tau = TAU / NOMINAL_GAP
        distance = np.arange(N_EVENTS - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-distance / event_tau)
    else:
        raise ValueError(target)

    score = x @ weights + rng.normal(0.0, TARGET_NOISE, size=n)
    y = (score > 0.0).astype(np.int64)
    return x, y


def event_input(x: np.ndarray, *, gap: float, include_dt: bool) -> torch.Tensor:
    if not include_dt:
        return torch.from_numpy(x[..., None])
    elapsed = np.full_like(x, gap, dtype=np.float32)
    return torch.from_numpy(np.stack((x, elapsed), axis=-1))


def clock_input(x: np.ndarray, *, gap: float) -> tuple[torch.Tensor, int]:
    # This intentionally uses literal fixed clock ticks.  The test gaps are rounded to
    # the closest integer number of 0.1 s ticks; the measured physical width uses that
    # realized clock spacing.
    tick_gap = max(1, int(round(gap / CLOCK_TICK)))
    steps = 1 + (N_EVENTS - 1) * tick_gap
    out = np.zeros((x.shape[0], steps, 2), dtype=np.float32)
    indices = np.arange(N_EVENTS) * tick_gap
    out[:, indices, 0] = x
    out[:, indices, 1] = 1.0
    return torch.from_numpy(out), tick_gap


def prepare_input(kind: str, x: np.ndarray, gap: float) -> tuple[torch.Tensor, float, np.ndarray]:
    if kind == "event":
        tensor = event_input(x, gap=gap, include_dt=False)
        indices = np.arange(N_EVENTS)
        return tensor, gap, indices
    if kind == "dt":
        tensor = event_input(x, gap=gap, include_dt=True)
        indices = np.arange(N_EVENTS)
        return tensor, gap, indices
    if kind == "clock":
        tensor, tick_gap = clock_input(x, gap=gap)
        indices = np.arange(N_EVENTS) * tick_gap
        return tensor, tick_gap * CLOCK_TICK, indices
    raise ValueError(kind)


def train_model(
    kind: str,
    *,
    seed: int,
    hidden: int,
    train_n: int,
    valid_n: int,
    epochs: int,
) -> GRUClassifier:
    torch.manual_seed(seed)
    x, y = make_target_data(
        train_n,
        seed=100 + seed,
        gap=NOMINAL_GAP,
        target="time",
    )
    xv, yv = make_target_data(
        valid_n,
        seed=200 + seed,
        gap=NOMINAL_GAP,
        target="time",
    )

    input_dim = 1 if kind == "event" else 2
    model = GRUClassifier(input_dim, hidden)
    X, _gap, _idx = prepare_input(kind, x, NOMINAL_GAP)
    XV, _gapv, _idxv = prepare_input(kind, xv, NOMINAL_GAP)
    Y = torch.from_numpy(y)
    YV = torch.from_numpy(yv)

    opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    best_state: dict[str, torch.Tensor] | None = None
    best_valid = float("inf")
    batch = 128

    for _epoch in range(epochs):
        order = torch.randperm(train_n)
        model.train()
        for start in range(0, train_n, batch):
            ix = order[start : start + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(X[ix]), Y[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            valid_loss = float(F.cross_entropy(model(XV), YV))
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    return model


def accuracy(
    model: GRUClassifier,
    kind: str,
    *,
    seed: int,
    gap_name: str,
    target: str,
    n: int,
) -> float:
    gap = GAPS[gap_name]
    x, y = make_target_data(n, seed=seed, gap=gap, target=target)
    X, _realized_gap, _indices = prepare_input(kind, x, gap)
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1).numpy()
    return float(np.mean(pred == y))


def sensitivity_width(
    model: GRUClassifier,
    kind: str,
    *,
    gap_name: str,
    seed: int,
    n: int,
    mass_fraction: float = 0.80,
) -> tuple[float, float]:
    gap = GAPS[gap_name]
    x, _y = make_target_data(n, seed=seed, gap=gap, target="time")
    X, realized_gap, content_indices = prepare_input(kind, x, gap)
    X = X.clone().requires_grad_(True)

    logits = model(X)
    margin = (logits[:, 1] - logits[:, 0]).sum()
    grad = torch.autograd.grad(margin, X)[0][:, :, 0].abs().mean(dim=0).detach().numpy()
    content_grad = grad[content_indices]

    ages = np.arange(N_EVENTS - 1, -1, -1, dtype=np.float64) * realized_gap
    order = np.argsort(ages)
    mass = content_grad[order]
    sorted_ages = ages[order]
    cumulative = np.cumsum(mass) / (np.sum(mass) + 1e-12)
    index = int(np.searchsorted(cumulative, mass_fraction))
    width80 = float(sorted_ages[min(index, len(sorted_ages) - 1)])
    mean_age = float(np.sum(content_grad * ages) / (np.sum(content_grad) + 1e-12))
    return width80, mean_age


def structure_yoking_index(compressed: float, stretched: float) -> float:
    return math.log((stretched + 1e-6) / (compressed + 1e-6)) / math.log(3.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train", type=int, default=6500)
    parser.add_argument("--valid", type=int, default=1600)
    parser.add_argument("--test", type=int, default=3000)
    parser.add_argument("--sensitivity", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=12)
    args = parser.parse_args()

    kinds = ("event", "dt", "clock")
    rows: list[dict[tuple[str, str, str], float]] = []

    for seed in range(args.seeds):
        row: dict[tuple[str, str, str], float] = {}
        for kind in kinds:
            model = train_model(
                kind,
                seed=10 + seed,
                hidden=args.hidden,
                train_n=args.train,
                valid_n=args.valid,
                epochs=args.epochs,
            )

            for target in ("time", "structure"):
                for offset, gap_name in enumerate(GAPS):
                    row[(kind, target, gap_name)] = accuracy(
                        model,
                        kind,
                        seed=5000 + 100 * seed + 10 * (target == "structure") + offset,
                        gap_name=gap_name,
                        target=target,
                        n=args.test,
                    )

            c80, cmean = sensitivity_width(
                model,
                kind,
                gap_name="compressed",
                seed=7000 + seed,
                n=args.sensitivity,
            )
            s80, smean = sensitivity_width(
                model,
                kind,
                gap_name="stretched",
                seed=8000 + seed,
                n=args.sensitivity,
            )
            row[(kind, "yoking", "width80")] = structure_yoking_index(c80, s80)
            row[(kind, "yoking", "mean_age")] = structure_yoking_index(cmean, smean)

        rows.append(row)

    print("learned GRU clock-yoking attack")
    print("0 = time-yoked, 1 = structure/event-yoked")
    print()

    for kind in kinds:
        syi80 = np.asarray([row[(kind, "yoking", "width80")] for row in rows])
        syimean = np.asarray([row[(kind, "yoking", "mean_age")] for row in rows])
        print(
            f"{kind:>5s} measured SYI: "
            f"width80={syi80.mean():.3f}+/-{syi80.std(ddof=0):.3f}  "
            f"mean_age={syimean.mean():.3f}+/-{syimean.std(ddof=0):.3f}"
        )
        for target in ("time", "structure"):
            parts = []
            for gap_name in GAPS:
                values = np.asarray(
                    [row[(kind, target, gap_name)] for row in rows]
                )
                parts.append(
                    f"{gap_name}={values.mean():.3f}+/-{values.std(ddof=0):.3f}"
                )
            print(f"      {target:>9s}: " + "  ".join(parts))
        print()

    print("Interpretation:")
    print("  blank clock ticks can alter learned yoking, but do not guarantee useful time-yoking")
    print("  a constant dt feature can be ignored when training never varies dt")
    print("  the hard clock/decay constraint in the scalar assay is stronger than clock exposure")


if __name__ == "__main__":
    main()
