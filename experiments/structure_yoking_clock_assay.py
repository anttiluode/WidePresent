"""Assay: does a content-blind clock force time-yoked integration under rate shift?

This experiment returns to the empirical hook that was skipped while WidePresent
wandered into timestamp/provenance attacks.

The setup is deliberately minimal.  Thirty-two +/-1 observations are integrated by a
single fading state.  At the nominal inter-event gap (0.5 s), two possible target
worlds are EXACTLY THE SAME:

    time-yoked target
        weight evidence by exp(-physical_age / tau)

    structure-yoked target
        weight evidence by exp(-event_distance / kappa)

with kappa = tau / nominal_gap.

Therefore training at the nominal rate cannot distinguish the two hypotheses.
Only rate shift does.

Three equal-state models are trained on the same nominal data:

    event
        h <- alpha * h + x
        One decay per CONTENT EVENT.  Its physical integration width therefore
        stretches/compresses with event spacing.

    dt
        h <- exp(-r * dt) * h + x
        Event-driven, but explicitly supplied elapsed time.

    clock
        h decays on a fixed content-blind 0.1 s clock whether or not content arrives.
        Between content events the exact accumulated decay is alpha_tick**(dt/tick).

The stretched and compressed gaps differ by sqrt(3) around nominal, so total
structure duration differs by a factor of 3, matching the useful convention in the
Norman-Haignere time-vs-structure yoking assay.

For an effective physical integration width W, define

    structure_yoking_index = log(W_stretched / W_compressed) / log(3)

so 0 is time-yoked and 1 is structure-yoked.

This is a proof-of-mechanism toy, not a novelty claim.  The dt model is a strong
boring attacker: if it matches the clock, fixed ticking is not uniquely necessary.

Run:
    python experiments/structure_yoking_clock_assay.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

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
TRUE_TAU = 2.5
CLOCK_TICK = 0.1
TARGET_NOISE = 0.35


class EventLeak(nn.Module):
    """One learned decay per content event: structurally yoked by construction."""

    def __init__(self) -> None:
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.tensor(1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
        del gap
        alpha = torch.sigmoid(self.logit_alpha)
        h = torch.zeros(x.shape[0], device=x.device)
        for j in range(x.shape[1]):
            h = alpha * h + x[:, j]
        score = self.scale * h + self.bias
        return torch.stack((-score, score), dim=1)

    def alpha(self) -> float:
        return float(torch.sigmoid(self.logit_alpha).detach())


class DtLeak(nn.Module):
    """Event-driven state with an explicit elapsed-time decay."""

    def __init__(self) -> None:
        super().__init__()
        self.log_rate = nn.Parameter(torch.tensor(-1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
        rate = F.softplus(self.log_rate)
        alpha = torch.exp(-rate * gap)
        h = torch.zeros(x.shape[0], device=x.device)
        for j in range(x.shape[1]):
            h = alpha * h + x[:, j]
        score = self.scale * h + self.bias
        return torch.stack((-score, score), dim=1)

    def time_constant(self) -> float:
        return float(1.0 / F.softplus(self.log_rate).detach())


class ClockLeak(nn.Module):
    """Decay controlled by a fixed content-blind clock, not content count."""

    def __init__(self, tick: float = CLOCK_TICK) -> None:
        super().__init__()
        self.tick = tick
        self.logit_tick_alpha = nn.Parameter(torch.tensor(2.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
        alpha_tick = torch.sigmoid(self.logit_tick_alpha)
        # Exact accumulation of fixed content-blind tick decay between events.
        alpha_between_events = torch.exp(torch.log(alpha_tick) * gap / self.tick)
        h = torch.zeros(x.shape[0], device=x.device)
        for j in range(x.shape[1]):
            h = alpha_between_events * h + x[:, j]
        score = self.scale * h + self.bias
        return torch.stack((-score, score), dim=1)

    def time_constant(self) -> float:
        alpha = float(torch.sigmoid(self.logit_tick_alpha).detach())
        return -self.tick / math.log(alpha)


@dataclass
class Dataset:
    x: torch.Tensor
    gap: torch.Tensor
    y: torch.Tensor


def make_dataset(
    n: int,
    *,
    seed: int,
    gap: float,
    target: str,
) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.choice((-1.0, 1.0), size=(n, N_EVENTS)).astype(np.float32)

    if target == "time":
        ages = np.arange(N_EVENTS - 1, -1, -1, dtype=np.float64) * gap
        weights = np.exp(-ages / TRUE_TAU)
    elif target == "structure":
        # At nominal gap this is exactly the same weighting as the time target.
        kappa_events = TRUE_TAU / NOMINAL_GAP
        distance = np.arange(N_EVENTS - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-distance / kappa_events)
    else:
        raise ValueError(target)

    score = x @ weights + rng.normal(0.0, TARGET_NOISE, size=n)
    y = (score > 0.0).astype(np.int64)
    gaps = np.full(n, gap, dtype=np.float32)

    return Dataset(
        x=torch.from_numpy(x),
        gap=torch.from_numpy(gaps),
        y=torch.from_numpy(y),
    )


def train_model(
    model: nn.Module,
    *,
    seed: int,
    train_n: int,
    valid_n: int,
    epochs: int,
) -> nn.Module:
    torch.manual_seed(seed)

    # Crucial control: time- and structure-yoked targets are identical at nominal
    # rate, so there is only one training dataset / one target.
    train = make_dataset(
        train_n,
        seed=100 + seed,
        gap=NOMINAL_GAP,
        target="time",
    )
    valid = make_dataset(
        valid_n,
        seed=200 + seed,
        gap=NOMINAL_GAP,
        target="time",
    )

    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    best_state: dict[str, torch.Tensor] | None = None
    best_valid = float("inf")

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(train.x, train.gap), train.y)
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                valid_loss = float(
                    F.cross_entropy(model(valid.x, valid.gap), valid.y)
                )
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
    model: nn.Module,
    *,
    seed: int,
    gap_name: str,
    target: str,
    n: int,
) -> float:
    data = make_dataset(
        n,
        seed=seed,
        gap=GAPS[gap_name],
        target=target,
    )
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.gap).argmax(dim=1)
    return float((pred == data.y).float().mean())


def physical_widths(model: nn.Module) -> tuple[float, float]:
    """Return compressed and stretched physical exponential time constants."""
    if isinstance(model, EventLeak):
        alpha = model.alpha()
        return (
            -GAPS["compressed"] / math.log(alpha),
            -GAPS["stretched"] / math.log(alpha),
        )
    if isinstance(model, DtLeak):
        tau = model.time_constant()
        return tau, tau
    if isinstance(model, ClockLeak):
        tau = model.time_constant()
        return tau, tau
    raise TypeError(type(model))


def yoking_index(model: nn.Module) -> float:
    compressed, stretched = physical_widths(model)
    return math.log(stretched / compressed) / math.log(3.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--train", type=int, default=12000)
    parser.add_argument("--valid", type=int, default=3000)
    parser.add_argument("--test", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=220)
    args = parser.parse_args()

    constructors = {
        "event": EventLeak,
        "dt": DtLeak,
        "clock": ClockLeak,
    }

    rows: list[dict[tuple[str, str, str], float]] = []
    yoking_rows: list[dict[str, float]] = []

    for seed in range(args.seeds):
        row: dict[tuple[str, str, str], float] = {}
        yrow: dict[str, float] = {}

        for name, constructor in constructors.items():
            torch.manual_seed(seed)
            model = train_model(
                constructor(),
                seed=seed,
                train_n=args.train,
                valid_n=args.valid,
                epochs=args.epochs,
            )
            yrow[name] = yoking_index(model)

            for target in ("time", "structure"):
                for offset, gap_name in enumerate(GAPS):
                    row[(name, target, gap_name)] = accuracy(
                        model,
                        seed=1000 + 100 * seed + 10 * (target == "structure") + offset,
                        gap_name=gap_name,
                        target=target,
                        n=args.test,
                    )

        rows.append(row)
        yoking_rows.append(yrow)

    print("clock versus structure-yoking assay")
    print(f"nominal gap = {NOMINAL_GAP:.3f} s")
    print(
        "compressed/stretched structure-duration ratio = "
        f"{GAPS['stretched'] / GAPS['compressed']:.3f}"
    )
    print()

    for target in ("time", "structure"):
        print(f"TARGET = {target}-yoked")
        for name in constructors:
            pieces = []
            for gap_name in GAPS:
                values = np.asarray(
                    [row[(name, target, gap_name)] for row in rows]
                )
                pieces.append(
                    f"{gap_name}={values.mean():.3f}+/-{values.std(ddof=0):.3f}"
                )
            yi = np.asarray([row[name] for row in yoking_rows])
            print(
                f"{name:>6s}: "
                + "  ".join(pieces)
                + f"  structure_yoking_index={yi.mean():.3f}"
            )
        print()

    print("Interpretation:")
    print("  index 0 = time-yoked; index 1 = structure/event-yoked")
    print("  the nominal training problem is identical for both target worlds")
    print("  rate shift reveals the architecture's temporal inductive bias")
    print("  dt is the boring attacker for the content-blind clock")


if __name__ == "__main__":
    main()
