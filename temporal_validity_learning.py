"""Learn uncertainty over temporal validity semantics from audited outcomes.

This module complements :mod:`temporal_validity`.

The base runtime assumes that a source validity model is already known.  Here we
represent uncertainty over a deliberately small semantic hypothesis family:

* ``world_hazard`` -- validity fades with world/source age in seconds;
* ``event_hazard`` -- validity fades with structural/event distance;
* ``until_change`` -- validity does not age and is killed by explicit invalidation.

For the two hazard hypotheses we integrate over a grid of decay rates.  The grid
is log-spaced and receives equal discrete prior mass (approximately a log-uniform
rate prior over the covered interval).

This is ordinary Bayesian model averaging / robust decision machinery, not a
novel temporal-learning algorithm.  It exists here because the WidePresent
identifiability attacks showed that hard semantic selection is brittle when wall
time and event distance are still statistically confounded.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Mapping

import numpy as np

from temporal_validity import (
    CurrentContext,
    Evidence,
    TemporalCoordinates,
    ValidityModel,
)


SemanticName = Literal["world_hazard", "event_hazard", "until_change"]
Strategy = Literal["average", "map", "robust"]
SEMANTICS: tuple[SemanticName, ...] = (
    "world_hazard",
    "event_hazard",
    "until_change",
)


@dataclass(frozen=True)
class AuditObservation:
    """One supervised observation of whether cached evidence was still valid."""

    world_age_seconds: float
    event_age: int
    invalidated: bool
    still_valid: bool

    def __post_init__(self) -> None:
        if self.world_age_seconds < 0:
            raise ValueError("world_age_seconds must be non-negative")
        if self.event_age < 0:
            raise ValueError("event_age must be non-negative")


@dataclass
class SemanticPosterior:
    """Posterior over validity semantics plus nuisance hazard rates."""

    lambda_grid: np.ndarray
    model_weights: dict[SemanticName, float]
    lambda_weights: dict[SemanticName, np.ndarray | None]
    log_evidence: dict[SemanticName, float]

    def __post_init__(self) -> None:
        if self.lambda_grid.ndim != 1 or len(self.lambda_grid) == 0:
            raise ValueError("lambda_grid must be a non-empty vector")
        if np.any(self.lambda_grid <= 0):
            raise ValueError("lambda_grid values must be positive")
        total = sum(self.model_weights.values())
        if not math.isfinite(total) or abs(total - 1.0) > 1e-8:
            raise ValueError("model_weights must sum to one")

    def axis_predictive(self, semantic: SemanticName, age: TemporalCoordinates) -> float:
        if semantic == "until_change":
            return 0.0 if age.invalidated else 1.0

        lambda_weights = self.lambda_weights[semantic]
        if lambda_weights is None:
            raise RuntimeError(f"missing lambda posterior for {semantic}")

        scalar_age = (
            age.world_age_seconds
            if semantic == "world_hazard"
            else float(age.event_age)
        )
        p = float(
            np.sum(lambda_weights * np.exp(-self.lambda_grid * scalar_age))
        )
        return min(1.0, max(0.0, p))

    @property
    def map_semantic(self) -> SemanticName:
        return max(SEMANTICS, key=lambda name: self.model_weights[name])

    def probability(
        self,
        age: TemporalCoordinates,
        *,
        strategy: Strategy = "average",
        plausible_weight: float = 0.10,
    ) -> float:
        """Posterior predictive validity under a semantic decision strategy.

        ``average``
            Bayesian model average.  This is the ordinary expected-utility choice
            when model uncertainty is treated probabilistically.

        ``map``
            Commit to the highest-posterior semantic hypothesis.

        ``robust``
            Take the minimum validity probability across semantic hypotheses
            whose posterior mass is at least ``plausible_weight``.  This is a
            deliberately conservative safety attacker, not a Bayesian optimum.
        """

        predictions = {
            name: self.axis_predictive(name, age)
            for name in SEMANTICS
        }

        if strategy == "map":
            return predictions[self.map_semantic]

        if strategy == "average":
            return float(
                sum(
                    self.model_weights[name] * predictions[name]
                    for name in SEMANTICS
                )
            )

        if strategy == "robust":
            if not (0.0 <= plausible_weight <= 1.0):
                raise ValueError("plausible_weight must be in [0, 1]")
            plausible = [
                predictions[name]
                for name in SEMANTICS
                if self.model_weights[name] >= plausible_weight
            ]
            if not plausible:
                plausible = [predictions[self.map_semantic]]
            return float(min(plausible))

        raise ValueError(f"unknown strategy {strategy!r}")

    def summary(self) -> dict[str, object]:
        return {
            "map_semantic": self.map_semantic,
            "model_weights": dict(self.model_weights),
            "log_evidence": dict(self.log_evidence),
        }


@dataclass(frozen=True)
class PosteriorValidityModel:
    """Adapter making a learned semantic posterior usable by the base runtime."""

    posterior: SemanticPosterior
    strategy: Strategy = "average"
    plausible_weight: float = 0.10

    def probability(
        self,
        evidence: Evidence,
        now: CurrentContext,
        age: TemporalCoordinates,
    ) -> float:
        return self.posterior.probability(
            age,
            strategy=self.strategy,
            plausible_weight=self.plausible_weight,
        )

    def describe(self) -> str:
        weights = self.posterior.model_weights
        return (
            f"semantic_posterior(strategy={self.strategy},"
            f"world={weights['world_hazard']:.3f},"
            f"event={weights['event_hazard']:.3f},"
            f"change={weights['until_change']:.3f})"
        )


def _log_mean_exp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.mean(np.exp(values - maximum))))


def _softmax(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    return weights / np.sum(weights)


def _hazard_likelihood(
    age: np.ndarray,
    y: np.ndarray,
    lambda_grid: np.ndarray,
    epsilon: float,
) -> tuple[float, np.ndarray]:
    probabilities = np.exp(-np.outer(lambda_grid, age))
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    log_likelihood = (
        y[None, :] * np.log(probabilities)
        + (1.0 - y[None, :]) * np.log(1.0 - probabilities)
    ).sum(axis=1)

    log_evidence = _log_mean_exp(log_likelihood)
    lambda_weights = _softmax(log_likelihood)
    return log_evidence, lambda_weights


def fit_semantic_posterior(
    observations: Iterable[AuditObservation],
    *,
    lambda_grid: np.ndarray | None = None,
    semantic_priors: Mapping[SemanticName, float] | None = None,
    epsilon: float = 1e-9,
) -> SemanticPosterior:
    """Fit a posterior over the three candidate temporal semantics.

    The function intentionally does not infer semantics from source names.  Call
    it separately for each source/content class using audited validity outcomes.
    """

    rows = list(observations)
    if not rows:
        raise ValueError("at least one audit observation is required")
    if not (0.0 < epsilon < 0.5):
        raise ValueError("epsilon must lie in (0, 0.5)")

    grid = (
        np.geomspace(0.001, 0.20, 240).astype(np.float64)
        if lambda_grid is None
        else np.asarray(lambda_grid, dtype=np.float64)
    )
    if grid.ndim != 1 or len(grid) == 0 or np.any(grid <= 0):
        raise ValueError("lambda_grid must contain positive rates")

    world_age = np.asarray([r.world_age_seconds for r in rows], dtype=np.float64)
    event_age = np.asarray([r.event_age for r in rows], dtype=np.float64)
    invalidated = np.asarray([r.invalidated for r in rows], dtype=bool)
    y = np.asarray([r.still_valid for r in rows], dtype=np.float64)

    log_evidence: dict[SemanticName, float] = {}
    lambda_weights: dict[SemanticName, np.ndarray | None] = {}

    world_evidence, world_lambda = _hazard_likelihood(
        world_age, y, grid, epsilon
    )
    event_evidence, event_lambda = _hazard_likelihood(
        event_age, y, grid, epsilon
    )
    log_evidence["world_hazard"] = world_evidence
    log_evidence["event_hazard"] = event_evidence
    lambda_weights["world_hazard"] = world_lambda
    lambda_weights["event_hazard"] = event_lambda

    change_probability = np.where(invalidated, 0.0, 1.0)
    change_probability = np.clip(change_probability, epsilon, 1.0 - epsilon)
    change_log_likelihood = float(
        np.sum(
            y * np.log(change_probability)
            + (1.0 - y) * np.log(1.0 - change_probability)
        )
    )
    log_evidence["until_change"] = change_log_likelihood
    lambda_weights["until_change"] = None

    if semantic_priors is None:
        priors = {name: 1.0 / len(SEMANTICS) for name in SEMANTICS}
    else:
        priors = {name: float(semantic_priors[name]) for name in SEMANTICS}
        if any(value <= 0 for value in priors.values()):
            raise ValueError("semantic priors must be strictly positive")
        total = sum(priors.values())
        priors = {name: value / total for name, value in priors.items()}

    log_posterior_unnormalized = np.asarray(
        [log_evidence[name] + math.log(priors[name]) for name in SEMANTICS],
        dtype=np.float64,
    )
    model_probability = _softmax(log_posterior_unnormalized)
    model_weights = {
        name: float(weight)
        for name, weight in zip(SEMANTICS, model_probability)
    }

    return SemanticPosterior(
        lambda_grid=grid,
        model_weights=model_weights,
        lambda_weights=lambda_weights,
        log_evidence=log_evidence,
    )
