"""Bitemporal working-state substrate for WidePresent.

The v0.1 `widepresent.py` has one clock axis. This module adds the distinction
that real online systems need:

- world/event time: when the represented event belongs in the world;
- knowledge/arrival time: when the agent acquired that item;
- now: the moving reference used to project a working present.

This is established temporal-system machinery brought into an agent-facing
state representation. It is not claimed as a new notion of time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

Kind = Literal["observation", "prediction"]


@dataclass(frozen=True)
class TemporalItem:
    value: np.ndarray
    world_tick: int
    known_tick: int
    source: str
    kind: Kind
    uncertainty: float = 0.0


@dataclass
class BitemporalSnapshot:
    """Dense projection of a sparse temporal ledger around `now`.

    Rows run from `-past_ticks` through zero to `+future_ticks` in world time.
    Knowledge age is separate: a newly arrived old observation therefore
    appears in an old world-time row with knowledge_age == 0.
    """

    relative_ticks: np.ndarray
    observation_value: np.ndarray
    observation_mask: np.ndarray
    observation_knowledge_age: np.ndarray
    prediction_value: np.ndarray
    prediction_mask: np.ndarray
    prediction_knowledge_age: np.ndarray
    completeness: np.ndarray
    now_tick: int
    dt: float

    @property
    def relative_seconds(self) -> np.ndarray:
        return self.relative_ticks.astype(float) * self.dt


class BitemporalPresent:
    """Sparse bitemporal ledger + fixed-width working projection.

    `sources` should list evidence-producing sources whose event-time progress
    can be tracked by watermarks. Predictions may use arbitrary source names
    without becoming evidence sources.
    """

    _NEG_INF = -(10**18)

    def __init__(self, dim: int, dt: float = 0.02, sources: Optional[list[str]] = None):
        if dim < 1 or dt <= 0:
            raise ValueError("invalid BitemporalPresent dimensions")
        self.dim = int(dim)
        self.dt = float(dt)
        self.now_tick = 0
        self.items: list[TemporalItem] = []
        self.evidence_sources = list(dict.fromkeys(sources or []))
        self.watermarks: dict[str, int] = {s: self._NEG_INF for s in self.evidence_sources}

    def reset(self) -> None:
        self.now_tick = 0
        self.items.clear()
        for source in self.watermarks:
            self.watermarks[source] = self._NEG_INF

    def advance(self, ticks: int = 1) -> None:
        """Advance objective `now`; content cannot suppress this transition."""
        if ticks < 0:
            raise ValueError("ticks must be nonnegative")
        self.now_tick += int(ticks)

    def _add(
        self,
        value: np.ndarray,
        world_tick: int,
        source: str,
        kind: Kind,
        uncertainty: float = 0.0,
        known_tick: Optional[int] = None,
    ) -> TemporalItem:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.dim,):
            raise ValueError(f"expected value shape {(self.dim,)}, got {value.shape}")
        known = self.now_tick if known_tick is None else int(known_tick)
        if known > self.now_tick:
            raise ValueError("cannot already know an item at a future knowledge tick")
        if uncertainty < 0:
            raise ValueError("uncertainty must be nonnegative")

        item = TemporalItem(
            value=value.copy(),
            world_tick=int(world_tick),
            known_tick=known,
            source=str(source),
            kind=kind,
            uncertainty=float(uncertainty),
        )
        self.items.append(item)
        return item

    def observe(
        self,
        value: np.ndarray,
        world_tick: int,
        source: str = "sensor",
        uncertainty: float = 0.0,
        known_tick: Optional[int] = None,
    ) -> TemporalItem:
        """Add evidence at its world time, even when it arrives late."""
        if world_tick > self.now_tick:
            raise ValueError("an observation cannot belong to future world time")
        if source not in self.watermarks:
            self.evidence_sources.append(source)
            self.watermarks[source] = self._NEG_INF
        return self._add(value, world_tick, source, "observation", uncertainty, known_tick)

    def predict(
        self,
        value: np.ndarray,
        world_tick: int,
        source: str = "model",
        uncertainty: float = 0.0,
        known_tick: Optional[int] = None,
    ) -> TemporalItem:
        """Add a belief whose world-valid time may lie in the future."""
        return self._add(value, world_tick, source, "prediction", uncertainty, known_tick)

    def set_watermark(self, source: str, world_tick: int) -> None:
        """Mark event-time progress for one evidence source.

        Semantics: the source declares that no future arrival is expected whose
        world_tick is <= this watermark. Watermarks must move monotonically.
        """
        if source not in self.watermarks:
            self.evidence_sources.append(source)
            self.watermarks[source] = self._NEG_INF
        world_tick = int(world_tick)
        if world_tick < self.watermarks[source]:
            raise ValueError("watermarks must be monotone")
        if world_tick > self.now_tick:
            raise ValueError("watermark cannot exceed now")
        self.watermarks[source] = world_tick

    def completeness_at(self, world_tick: int, sources: Optional[list[str]] = None) -> float:
        """Fraction of evidence sources whose watermark has passed a world tick."""
        srcs = self.evidence_sources if sources is None else list(sources)
        if not srcs:
            return 0.0
        return float(np.mean([self.watermarks.get(s, self._NEG_INF) >= world_tick for s in srcs]))

    def _latest(self, world_tick: int, kind: Kind) -> Optional[TemporalItem]:
        candidates = [
            (i, item)
            for i, item in enumerate(self.items)
            if item.world_tick == world_tick
            and item.kind == kind
            and item.known_tick <= self.now_tick
        ]
        if not candidates:
            return None
        # Latest acquired item wins. Insertion order breaks same-tick ties.
        return max(candidates, key=lambda pair: (pair[1].known_tick, pair[0]))[1]

    def project(self, past_ticks: int, future_ticks: int) -> BitemporalSnapshot:
        """Project the ledger into a temporally typed matrix centered on now."""
        if past_ticks < 0 or future_ticks < 0:
            raise ValueError("window sizes must be nonnegative")

        rel = np.arange(-past_ticks, future_ticks + 1, dtype=int)
        n = len(rel)
        observation_value = np.zeros((n, self.dim), dtype=np.float64)
        observation_mask = np.zeros(n, dtype=np.float64)
        observation_knowledge_age = np.zeros(n, dtype=np.float64)
        prediction_value = np.zeros((n, self.dim), dtype=np.float64)
        prediction_mask = np.zeros(n, dtype=np.float64)
        prediction_knowledge_age = np.zeros(n, dtype=np.float64)
        completeness = np.zeros(n, dtype=np.float64)

        for i, relative_tick in enumerate(rel):
            world_tick = self.now_tick + int(relative_tick)

            observation = self._latest(world_tick, "observation")
            if observation is not None:
                observation_value[i] = observation.value
                observation_mask[i] = 1.0
                observation_knowledge_age[i] = self.now_tick - observation.known_tick

            prediction = self._latest(world_tick, "prediction")
            if prediction is not None:
                prediction_value[i] = prediction.value
                prediction_mask[i] = 1.0
                prediction_knowledge_age[i] = self.now_tick - prediction.known_tick

            # Future world time is by definition not evidence-complete yet.
            completeness[i] = 0.0 if world_tick > self.now_tick else self.completeness_at(world_tick)

        return BitemporalSnapshot(
            relative_ticks=rel,
            observation_value=observation_value,
            observation_mask=observation_mask,
            observation_knowledge_age=observation_knowledge_age,
            prediction_value=prediction_value,
            prediction_mask=prediction_mask,
            prediction_knowledge_age=prediction_knowledge_age,
            completeness=completeness,
            now_tick=self.now_tick,
            dt=self.dt,
        )

    def due_predictions(self) -> list[TemporalItem]:
        """Return predictions whose world-valid time has reached/passed now."""
        return [
            item
            for item in self.items
            if item.kind == "prediction"
            and item.world_tick <= self.now_tick
            and item.known_tick <= self.now_tick
        ]
