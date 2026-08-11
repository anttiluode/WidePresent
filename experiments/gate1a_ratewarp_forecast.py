"""Gate 1A: equal-information rate-warp forecasting.

Three small models observe the same irregular samples from a continuous signal:

1. EventGRU: values only (negative control; no elapsed-time information).
2. DtGRU: values + exact dt between events.
3. WideMLP: the same samples placed on a fixed relative-time grid with value/mask
   channels. It has an explicit age coordinate because grid position means time.

All models forecast the latent signal at a fixed horizon beyond `now`.
Train and test use different observation-rate ranges.

This is deliberately a small inductive-bias test, not a final architecture claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class Spec:
    window: float = 2.0
    horizon: float = 0.35
    grid_dt: float = 0.05
    min_events: int = 5


def signal_value(t, amps, freqs, phases, trend):
    y = trend * t
    for a, f, p in zip(amps, freqs, phases):
        y += a * np.sin(2 * np.pi * f * t + p)
    return y


def one_trial(rng: np.random.Generator, rate_range, spec: Spec):
    amps = rng.uniform(0.35, 1.0, size=2)
    freqs = rng.uniform([0.35, 0.8], [0.75, 1.6])
    phases = rng.uniform(0, 2 * np.pi, size=2)
    trend = rng.uniform(-0.12, 0.12)

    rate = rng.uniform(*rate_range)
    times = [0.0]
    while times[-1] < spec.window:
        times.append(times[-1] + rng.exponential(1.0 / rate))
    times = np.asarray(times[:-1], dtype=np.float32)
    times = times[(times >= 0) & (times <= spec.window)]
    if len(times) < spec.min_events:
        times = np.linspace(0.0, spec.window, spec.min_events, endpoint=False, dtype=np.float32)

    values = signal_value(times, amps, freqs, phases, trend).astype(np.float32)
    values += rng.normal(0.0, 0.03, size=len(values)).astype(np.float32)

    target_t = spec.window + spec.horizon
    target = np.float32(signal_value(target_t, amps, freqs, phases, trend))

    dts = np.diff(np.concatenate(([0.0], times))).astype(np.float32)

    ngrid = int(round(spec.window / spec.grid_dt)) + 1
    sums = np.zeros(ngrid, dtype=np.float32)
    counts = np.zeros(ngrid, dtype=np.float32)
    idx = np.clip(np.rint(times / spec.grid_dt).astype(int), 0, ngrid - 1)
    for i, v in zip(idx, values):
        sums[i] += v
        counts[i] += 1
    mask = (counts > 0).astype(np.float32)
    grid_vals = np.divide(sums, np.maximum(counts, 1.0), dtype=np.float32)
    grid = np.stack([grid_vals, mask], axis=-1)

    return values, dts, grid, target


class RateDataset(Dataset):
    def __init__(self, n, seed, rate_range, spec):
        rng = np.random.default_rng(seed)
        self.items = [one_trial(rng, rate_range, spec) for _ in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    vals, dts, grids, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in vals], dtype=torch.long)
    maxlen = int(lengths.max())
    vpad = torch.zeros(len(batch), maxlen, 1)
    dtpad = torch.zeros(len(batch), maxlen, 1)
    for i, (v, d) in enumerate(zip(vals, dts)):
        n = len(v)
        vpad[i, :n, 0] = torch.from_numpy(v)
        dtpad[i, :n, 0] = torch.from_numpy(d)
    grid = torch.from_numpy(np.stack(grids))
    y = torch.tensor(targets, dtype=torch.float32)[:, None]
    return vpad, dtpad, lengths, grid, y


class GRUForecast(nn.Module):
    def __init__(self, with_dt: bool, hidden=32):
        super().__init__()
        self.with_dt = with_dt
        self.gru = nn.GRU(2 if with_dt else 1, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, values, dts, lengths, grid):
        x = torch.cat([values, dts], dim=-1) if self.with_dt else values
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1])


class TimeTransformer(nn.Module):
    """Direct-access control: irregular events + explicit objective timestamps."""
    def __init__(self, window=2.0, d_model=16, nhead=4, ff=32):
        super().__init__()
        self.window = float(window)
        self.inp = nn.Linear(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, values, dts, lengths, grid):
        times = torch.cumsum(dts, dim=1)
        age = times - self.window
        x = self.inp(torch.cat([values, age], dim=-1))
        steps = torch.arange(x.shape[1], device=x.device)[None, :]
        pad = steps >= lengths[:, None]
        z = self.enc(x, src_key_padding_mask=pad)
        valid = (~pad).unsqueeze(-1).to(z.dtype)
        pooled = (z * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


class GridGRU(nn.Module):
    """GRU control that receives exactly the same fixed grid as WideMLP."""
    def __init__(self, hidden=28):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, values, dts, lengths, grid):
        _, h = self.gru(grid)
        return self.head(h[-1])


class WideMLP(nn.Module):
    def __init__(self, ngrid, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ngrid * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, values, dts, lengths, grid):
        return self.net(grid)


def params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_model(model, train_loader, epochs=12, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossfn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for values, dts, lengths, grid, y in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(values, dts, lengths, grid)
            loss = lossfn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    se = 0.0
    ae = 0.0
    n = 0
    for values, dts, lengths, grid, y in loader:
        pred = model(values, dts, lengths, grid)
        err = pred - y
        se += float((err ** 2).sum())
        ae += float(err.abs().sum())
        n += len(y)
    return math.sqrt(se / n), ae / n


def run(seed=0, epochs=12, n_train=5000, n_test=1200):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    spec = Spec()
    train = RateDataset(n_train, seed + 100, (7.0, 14.0), spec)
    iid = RateDataset(n_test, seed + 200, (7.0, 14.0), spec)
    slow = RateDataset(n_test, seed + 300, (2.0, 4.0), spec)
    fast = RateDataset(n_test, seed + 400, (22.0, 35.0), spec)

    train_loader = DataLoader(train, batch_size=128, shuffle=True, collate_fn=collate)
    loaders = {
        "iid": DataLoader(iid, batch_size=256, collate_fn=collate),
        "slow_ood": DataLoader(slow, batch_size=256, collate_fn=collate),
        "fast_ood": DataLoader(fast, batch_size=256, collate_fn=collate),
    }

    ngrid = int(round(spec.window / spec.grid_dt)) + 1
    models = {
        "event_gru": GRUForecast(False, hidden=32),
        "dt_gru": GRUForecast(True, hidden=32),
        "time_transformer": TimeTransformer(window=spec.window, d_model=16, nhead=4, ff=32),
        "grid_gru": GridGRU(hidden=28),
        "wide_mlp": WideMLP(ngrid, hidden=32),
    }

    results = {}
    for name, model in models.items():
        train_model(model, train_loader, epochs=epochs)
        results[name] = {"params": params(model)}
        for split, loader in loaders.items():
            rmse, mae = evaluate(model, loader)
            results[name][split] = {"rmse": rmse, "mae": mae}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--train", type=int, default=5000)
    p.add_argument("--test", type=int, default=1200)
    args = p.parse_args()
    r = run(args.seed, args.epochs, args.train, args.test)
    print("Gate 1A — rate-warp fixed-horizon forecasting")
    for name, d in r.items():
        print(f"\n{name} params={d['params']}")
        for split in ("iid", "slow_ood", "fast_ood"):
            print(f"  {split:9s} RMSE={d[split]['rmse']:.4f} MAE={d[split]['mae']:.4f}")


if __name__ == "__main__":
    main()
