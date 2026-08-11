"""Minimal clock-first substrate for WidePresent experiments.

This module intentionally contains no claim about consciousness. It implements
one mechanical distinction: state is indexed by elapsed physical time rather
than by the number of input events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Snapshot:
    """A centered temporal state around a privileged `now`.

    `past[k]` is the observation k+1 ticks ago.
    `now` is the observation at the current tick.
    `future[k]` is a prediction scheduled for k+1 ticks ahead.
    """

    past: np.ndarray
    now: np.ndarray
    future: np.ndarray
    tick: int
    dt: float

    @property
    def relative_times(self) -> np.ndarray:
        p = self.past.shape[0]
        f = self.future.shape[0]
        return np.arange(-p, f + 1, dtype=float) * self.dt

    def matrix(self) -> np.ndarray:
        return np.concatenate((self.past[::-1], self.now[None, :], self.future), axis=0)


class WidePresent:
    """Fixed-pitch past/now/future register.

    The clock is deliberately content-blind: every call to ``tick`` advances
    exactly one step. Inputs may alter state *at* a tick, but cannot decide
    whether a tick happened.

    Predictions can be placed into future slots. As time advances they move
    toward zero; when a slot reaches `now`, ``tick`` returns it as the matured
    prediction so it can be compared with the new observation.
    """

    def __init__(self, dim: int, dt: float = 0.02, past_ticks: int = 50, future_ticks: int = 25):
        if dim < 1 or dt <= 0 or past_ticks < 0 or future_ticks < 0:
            raise ValueError("invalid WidePresent dimensions")
        self.dim = int(dim)
        self.dt = float(dt)
        self.past_ticks = int(past_ticks)
        self.future_ticks = int(future_ticks)
        self.past = np.zeros((past_ticks, dim), dtype=np.float64)
        self.now = np.zeros(dim, dtype=np.float64)
        self.future = np.zeros((future_ticks, dim), dtype=np.float64)
        self.clock_tick = -1

    def reset(self) -> None:
        self.past.fill(0.0)
        self.now.fill(0.0)
        self.future.fill(0.0)
        self.clock_tick = -1

    def schedule(self, value: np.ndarray, ticks_ahead: int) -> None:
        """Place/replace a prediction at an absolute temporal offset."""
        if ticks_ahead < 1 or ticks_ahead > self.future_ticks:
            raise ValueError("ticks_ahead outside future register")
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.dim,):
            raise ValueError(f"expected value shape {(self.dim,)}, got {value.shape}")
        self.future[ticks_ahead - 1] = value

    def tick(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance one content-blind clock tick and return prediction due now."""
        matured = self.future[0].copy() if self.future_ticks else np.zeros(self.dim)

        if self.past_ticks:
            if self.past_ticks > 1:
                self.past[1:] = self.past[:-1]
            self.past[0] = self.now

        if self.future_ticks:
            if self.future_ticks > 1:
                self.future[:-1] = self.future[1:]
            self.future[-1] = 0.0

        if observation is None:
            self.now.fill(0.0)
        else:
            observation = np.asarray(observation, dtype=np.float64)
            if observation.shape != (self.dim,):
                raise ValueError(f"expected observation shape {(self.dim,)}, got {observation.shape}")
            self.now = observation.copy()

        self.clock_tick += 1
        return matured

    def snapshot(self) -> Snapshot:
        return Snapshot(self.past.copy(), self.now.copy(), self.future.copy(), self.clock_tick, self.dt)
