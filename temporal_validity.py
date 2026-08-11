"""Minimal temporal-validity runtime for tool-using agents.

This module is intentionally boring.

It separates three concerns that were repeatedly conflated in WidePresent:

1. representation: preserve candidate temporal coordinates;
2. validity semantics: map those coordinates to P(still usable now);
3. action policy: trade stale-reuse risk against refresh cost.

No neural architecture is required. The language model can consume the resolved
state or probability instead of repeatedly rediscovering timestamp arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class Evidence:
    """One cached piece of evidence.

    valid_time
        When the value was true / observed in the world.

    known_time
        When the agent learned or received it.

    known_structural_index
        Conversation/event index when the agent learned it. Structural age is
        measured from this point.

    source_version
        Optional source/change epoch captured with the evidence. Until-change
        validity can compare this against CurrentContext.source_versions.
    """

    source: str
    key: str
    value: Any
    valid_time: datetime
    known_time: datetime
    known_structural_index: int
    source_version: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurrentContext:
    time: datetime
    structural_index: int
    source_versions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemporalCoordinates:
    world_age_seconds: float
    knowledge_age_seconds: float
    event_age: int
    invalidated: bool


def coordinates(evidence: Evidence, now: CurrentContext) -> TemporalCoordinates:
    world_age = (now.time - evidence.valid_time).total_seconds()
    knowledge_age = (now.time - evidence.known_time).total_seconds()
    event_age = now.structural_index - evidence.known_structural_index

    if world_age < -1e-9:
        raise ValueError("current time precedes evidence valid_time")
    if knowledge_age < -1e-9:
        raise ValueError("current time precedes evidence known_time")
    if event_age < 0:
        raise ValueError("current structural index precedes evidence index")

    invalidated = False
    if evidence.source_version is not None and evidence.source in now.source_versions:
        invalidated = now.source_versions[evidence.source] != evidence.source_version

    return TemporalCoordinates(
        world_age_seconds=max(0.0, world_age),
        knowledge_age_seconds=max(0.0, knowledge_age),
        event_age=event_age,
        invalidated=invalidated,
    )


class ValidityModel(Protocol):
    def probability(
        self,
        evidence: Evidence,
        now: CurrentContext,
        age: TemporalCoordinates,
    ) -> float:
        ...

    def describe(self) -> str:
        ...


@dataclass(frozen=True)
class WorldTimeTTL:
    seconds: float
    kill_on_invalidation: bool = True

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError("seconds must be non-negative")

    def probability(self, evidence: Evidence, now: CurrentContext, age: TemporalCoordinates) -> float:
        if self.kill_on_invalidation and age.invalidated:
            return 0.0
        return 1.0 if age.world_age_seconds <= self.seconds + 1e-12 else 0.0

    def describe(self) -> str:
        return f"world_time_ttl({self.seconds:g}s)"


@dataclass(frozen=True)
class EventDistanceTTL:
    events: int
    kill_on_invalidation: bool = True

    def __post_init__(self) -> None:
        if self.events < 0:
            raise ValueError("events must be non-negative")

    def probability(self, evidence: Evidence, now: CurrentContext, age: TemporalCoordinates) -> float:
        if self.kill_on_invalidation and age.invalidated:
            return 0.0
        return 1.0 if age.event_age <= self.events else 0.0

    def describe(self) -> str:
        return f"event_distance_ttl({self.events})"


@dataclass(frozen=True)
class UntilChange:
    """Evidence remains valid indefinitely until its source version changes."""

    def probability(self, evidence: Evidence, now: CurrentContext, age: TemporalCoordinates) -> float:
        return 0.0 if age.invalidated else 1.0

    def describe(self) -> str:
        return "until_change"


@dataclass(frozen=True)
class ExponentialAgePlane:
    """Probabilistic fading over (world age, event distance).

    p(valid) = exp(-a * delta_t - b * delta_n)

    This is included because it is the smallest generic age-plane hazard used in
    the research attacks. It is not presented as a novel model.
    """

    per_second: float = 0.0
    per_event: float = 0.0
    kill_on_invalidation: bool = True

    def __post_init__(self) -> None:
        if self.per_second < 0 or self.per_event < 0:
            raise ValueError("hazards must be non-negative")

    def probability(self, evidence: Evidence, now: CurrentContext, age: TemporalCoordinates) -> float:
        if self.kill_on_invalidation and age.invalidated:
            return 0.0
        p = math.exp(
            -self.per_second * age.world_age_seconds
            -self.per_event * age.event_age
        )
        return min(1.0, max(0.0, p))

    def describe(self) -> str:
        return (
            "exp_age_plane("
            f"a={self.per_second:g}/s,b={self.per_event:g}/event)"
        )


@dataclass(frozen=True)
class DecisionUtilities:
    valid_reuse: float = 1.0
    stale_reuse: float = -1.5
    refresh: float = 0.55

    @property
    def reuse_probability_threshold(self) -> float:
        denom = self.valid_reuse - self.stale_reuse
        if denom <= 0:
            raise ValueError("valid_reuse must exceed stale_reuse")
        threshold = (self.refresh - self.stale_reuse) / denom
        return min(1.0, max(0.0, threshold))

    def expected_reuse_utility(self, p_valid: float) -> float:
        return (
            p_valid * self.valid_reuse
            + (1.0 - p_valid) * self.stale_reuse
        )


@dataclass(frozen=True)
class ValidityDecision:
    action: str
    p_valid: float
    threshold: float
    expected_reuse_utility: float
    refresh_utility: float
    coordinates: TemporalCoordinates
    model: str

    def as_runtime_state(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "p_valid": self.p_valid,
            "reuse_threshold": self.threshold,
            "model": self.model,
            "world_age_seconds": self.coordinates.world_age_seconds,
            "knowledge_age_seconds": self.coordinates.knowledge_age_seconds,
            "event_age": self.coordinates.event_age,
            "invalidated": self.coordinates.invalidated,
        }


class TemporalValidityRuntime:
    def __init__(
        self,
        models: Mapping[str, ValidityModel],
        *,
        utilities: DecisionUtilities | None = None,
    ) -> None:
        self.models = dict(models)
        self.utilities = utilities or DecisionUtilities()

    def evaluate(self, evidence: Evidence, now: CurrentContext) -> ValidityDecision:
        if evidence.source not in self.models:
            raise KeyError(f"no validity model registered for source {evidence.source!r}")

        age = coordinates(evidence, now)
        model = self.models[evidence.source]
        p_valid = float(model.probability(evidence, now, age))
        if not math.isfinite(p_valid) or not (0.0 <= p_valid <= 1.0):
            raise ValueError("validity model returned probability outside [0, 1]")

        threshold = self.utilities.reuse_probability_threshold
        reuse_u = self.utilities.expected_reuse_utility(p_valid)
        action = "reuse" if reuse_u >= self.utilities.refresh else "refresh"

        return ValidityDecision(
            action=action,
            p_valid=p_valid,
            threshold=threshold,
            expected_reuse_utility=reuse_u,
            refresh_utility=self.utilities.refresh,
            coordinates=age,
            model=model.describe(),
        )
