"""Learned attack on a dual-yoked temporal state.

Each episode contains TWO independent content channels simultaneously:

channel 0
    target memory is yoked to physical seconds (tau = 2.5 s)

channel 1
    target memory is yoked to structural/event distance (tau = 5 events)

The model must predict both binary targets from the same episode.  There is no
post-hoc test-time switch between "time task" and "structure task".

Training rates are centered at 0.5 s/event.  Evaluation is performed on compressed
and stretched rates differing by sqrt(3), giving a factor-three duration ratio.

State budget
------------
All principal models expose 16 state numbers at readout:

    event_gru  : 16 learned recurrent coordinates
    dt_gru     : 16 learned recurrent coordinates; dt concatenated to input
    decay_gru  : 16 learned coordinates with physical-time decay before each event
    dual       : 8 event-GRU coordinates + 8 fixed physical-time fading coordinates

The dual model's physical bank contains four log-spaced time constants for each input
channel.  Both the event path and time path see both content channels; only the final
readout decides what to use.

This is a synthetic architecture attack, not a novelty claim.  In particular,
continuous-time recurrent models such as CT-GRU are strong prior-art attackers and
should be tested before promoting the split architecture.

Default run reproduces the compact three-seed endpoint comparison documented in
`docs/DUAL_YOKED_MEMORY_ATTACK.md`:

    python experiments/dual_yoked_memory_attack.py
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
TIME_TAU = 2.5
EVENT_TAU = 5.0
SQRT3 = math.sqrt(3.0)
TEST_GAPS = {
    "compressed": NOMINAL_GAP / SQRT3,
    "iid": NOMINAL_GAP,
    "stretched": NOMINAL_GAP * SQRT3,
}


def make_data(
    n: int,
    *,
    seed: int,
    gap_low: float,
    gap_high: float | None = None,
    interval_jitter: float = 0.03,
    target_noise: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    if gap_high is None:
        gap_high = gap_low

    base_gap = rng.uniform(gap_low, gap_high, size=(n, 1))
    gaps = base_gap * rng.uniform(
        1.0 - interval_jitter,
        1.0 + interval_jitter,
        size=(n, N_EVENTS - 1),
    )

    content = rng.normal(size=(n, N_EVENTS, 2)).astype(np.float32)

    ages = np.zeros((n, N_EVENTS), dtype=np.float32)
    ages[:, :-1] = np.cumsum(gaps[:, ::-1], axis=1)[:, ::-1].astype(np.float32)

    distance = np.arange(N_EVENTS - 1, -1, -1, dtype=np.float32)[None, :]
    time_weights = np.exp(-ages / TIME_TAU)
    event_weights = np.exp(-distance / EVENT_TAU)

    time_score = (content[:, :, 0] * time_weights).sum(axis=1)
    time_score /= np.sqrt((time_weights**2).sum(axis=1))

    event_score = (content[:, :, 1] * event_weights).sum(axis=1)
    event_score /= np.sqrt((event_weights**2).sum())

    time_score += rng.normal(0.0, target_noise, size=n)
    event_score += rng.normal(0.0, target_noise, size=n)

    labels = np.stack((time_score > 0.0, event_score > 0.0), axis=1).astype(np.float32)

    dt = np.zeros((n, N_EVENTS), dtype=np.float32)
    dt[:, 1:] = gaps.astype(np.float32)

    return (
        torch.from_numpy(content),
        torch.from_numpy(dt),
        torch.from_numpy(labels),
    )


class EventGRU(nn.Module):
    def __init__(self, hidden: int = 16, include_dt: bool = False) -> None:
        super().__init__()
        self.include_dt = include_dt
        self.gru = nn.GRU(3 if include_dt else 2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, content: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if self.include_dt:
            model_input = torch.cat([content, dt[:, :, None]], dim=2)
        else:
            model_input = content
        _out, hidden = self.gru(model_input)
        return self.head(hidden[-1])


class DecayGRU(nn.Module):
    """Simple physical-time-decay GRU attacker; not an exact CT-GRU implementation."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(2, hidden)
        self.raw_rate = nn.Parameter(torch.full((hidden,), -2.0) + 0.1 * torch.randn(hidden))
        self.head = nn.Linear(hidden, 2)

    def forward(self, content: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        state = content.new_zeros((content.shape[0], self.hidden))
        rates = F.softplus(self.raw_rate)
        for index in range(content.shape[1]):
            if index > 0:
                state = state * torch.exp(-dt[:, index, None] * rates[None, :])
            state = self.cell(content[:, index], state)
        return self.head(state)

    def rates(self) -> np.ndarray:
        return F.softplus(self.raw_rate).detach().cpu().numpy()


class DualYokedState(nn.Module):
    """8 event-GRU states plus 8 deterministic physical-time fading states."""

    def __init__(self, structural_hidden: int = 8) -> None:
        super().__init__()
        self.structural_hidden = structural_hidden
        self.structural_cell = nn.GRUCell(2, structural_hidden)

        # Four broad physical-time constants for each of the two content channels.
        taus = np.geomspace(0.5, 8.0, 4).astype(np.float32)
        rates = np.repeat(1.0 / taus, 2)
        self.register_buffer("time_rates", torch.from_numpy(rates))

        self.head = nn.Linear(structural_hidden + 8, 2)

    def forward(self, content: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        structural = content.new_zeros((content.shape[0], self.structural_hidden))
        timed = content.new_zeros((content.shape[0], 8))

        for index in range(content.shape[1]):
            if index > 0:
                timed = timed * torch.exp(
                    -dt[:, index, None] * self.time_rates[None, :]
                )

            # Feature order is [ch0, ch1] repeated across the four time constants.
            timed = timed + content[:, index].repeat(1, 4)
            structural = self.structural_cell(content[:, index], structural)

        return self.head(torch.cat([structural, timed], dim=1))


def build_model(kind: str, hidden: int) -> nn.Module:
    if kind == "event":
        return EventGRU(hidden=hidden, include_dt=False)
    if kind == "dt":
        return EventGRU(hidden=hidden, include_dt=True)
    if kind == "decay":
        return DecayGRU(hidden=hidden)
    if kind == "dual":
        if hidden % 2 != 0:
            raise ValueError("dual model requires an even --hidden value")
        return DualYokedState(structural_hidden=hidden // 2)
    raise ValueError(kind)


def train_model(
    kind: str,
    *,
    seed: int,
    rate_half_width: float,
    hidden: int,
    train_n: int,
    valid_n: int,
    epochs: int,
) -> nn.Module:
    torch.manual_seed(seed)

    low = max(0.05, NOMINAL_GAP - rate_half_width)
    high = NOMINAL_GAP + rate_half_width

    content, dt, labels = make_data(
        train_n,
        seed=1000 + seed,
        gap_low=low,
        gap_high=high,
    )
    valid_content, valid_dt, valid_labels = make_data(
        valid_n,
        seed=2000 + seed,
        gap_low=low,
        gap_high=high,
    )

    model = build_model(kind, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    batch_size = 128

    for _epoch in range(epochs):
        order = torch.randperm(train_n)
        model.train()
        for start in range(0, train_n, batch_size):
            indices = order[start : start + batch_size]
            logits = model(content[indices], dt[indices])
            loss = F.binary_cross_entropy_with_logits(logits, labels[indices])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = float(
                F.binary_cross_entropy_with_logits(
                    model(valid_content, valid_dt),
                    valid_labels,
                )
            )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    return model


def evaluate(
    model: nn.Module,
    *,
    seed: int,
    gap: float,
    n: int,
) -> tuple[float, float, float]:
    content, dt, labels = make_data(
        n,
        seed=seed,
        gap_low=gap,
        gap_high=gap,
    )

    model.eval()
    with torch.no_grad():
        prediction = (torch.sigmoid(model(content, dt)) > 0.5).float()

    per_head = (prediction == labels).float().mean(dim=0).cpu().numpy()
    joint = float((prediction == labels).all(dim=1).float().mean())
    return float(per_head[0]), float(per_head[1]), joint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--train", type=int, default=3500)
    parser.add_argument("--valid", type=int, default=900)
    parser.add_argument("--test", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument(
        "--rate-half-widths",
        type=float,
        nargs="+",
        default=[0.0, 0.15],
        help="training base-gap half-widths around 0.5 s/event",
    )
    args = parser.parse_args()

    kinds = ("event", "dt", "decay", "dual")

    print("dual-yoked learned memory attack")
    print("metrics average compressed and stretched test conditions")
    print("columns: physical-time head, structure head, joint")
    print()

    for width in args.rate_half_widths:
        print(f"training rate half-width = {width:.3f} s/event")
        for kind in kinds:
            rows: list[tuple[float, float, float]] = []
            rate_summaries: list[tuple[float, float, float]] = []

            for seed in range(args.seeds):
                model = train_model(
                    kind,
                    seed=10 + seed,
                    rate_half_width=width,
                    hidden=args.hidden,
                    train_n=args.train,
                    valid_n=args.valid,
                    epochs=args.epochs,
                )

                compressed = evaluate(
                    model,
                    seed=4000 + 100 * seed,
                    gap=TEST_GAPS["compressed"],
                    n=args.test,
                )
                stretched = evaluate(
                    model,
                    seed=5000 + 100 * seed,
                    gap=TEST_GAPS["stretched"],
                    n=args.test,
                )
                rows.append(
                    tuple((compressed[i] + stretched[i]) / 2.0 for i in range(3))
                )

                if isinstance(model, DecayGRU):
                    rates = model.rates()
                    rate_summaries.append(
                        (float(rates.min()), float(rates.max()), float(rates.mean()))
                    )

            values = np.asarray(rows)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            print(
                f"  {kind:>5s}: "
                f"time={mean[0]:.3f}+/-{std[0]:.3f}  "
                f"structure={mean[1]:.3f}+/-{std[1]:.3f}  "
                f"joint={mean[2]:.3f}+/-{std[2]:.3f}"
            )

            if rate_summaries:
                rs = np.asarray(rate_summaries).mean(axis=0)
                print(
                    f"         learned decay rates: "
                    f"min={rs[0]:.3f} max={rs[1]:.3f} mean={rs[2]:.3f} /s"
                )
        print()

    print("Interpretation discipline:")
    print("  a dual-state win here is evidence for preserving both invariants on this toy")
    print("  it is not evidence that the split is novel or uniquely necessary")
    print("  CT-GRU / continuous-time SSM baselines remain mandatory attackers")


if __name__ == "__main__":
    main()
