"""Active probe scoring for temporal-semantic identification.

The functions here distinguish two kinds of uncertainty:

* state/freshness uncertainty: is this particular cached value still valid?
* semantic/model uncertainty: which temporal coordinate governs source validity?

A probe that is useful for one need not be useful for the other.

This module provides deliberately small acquisition scores for candidate probe
opportunities represented by TemporalCoordinates.  It uses the semantic posterior
from temporal_validity_learning.py and does not introduce a new learning model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np

from temporal_validity import TemporalCoordinates
from temporal_validity_learning import SEMANTICS, SemanticPosterior


ProbeScore = Literal[
    "freshness_uncertainty",
    "semantic_disagreement",
    "semantic_information_gain",
]


@dataclass(frozen=True)
class ProbeOpportunity:
    coordinates: TemporalCoordinates
    payload: object | None = None


def _entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 1e-15]
    if len(positive) == 0:
        return 0.0
    return -float(np.sum(positive * np.log(positive)))


def semantic_predictions(
    posterior: SemanticPosterior,
    age: TemporalCoordinates,
) -> tuple[np.ndarray, np.ndarray]:
    """Return semantic model weights and each model's predictive P(valid)."""
    model_weights = np.asarray(
        [posterior.model_weights[name] for name in SEMANTICS],
        dtype=np.float64,
    )
    predictions = np.asarray(
        [posterior.axis_predictive(name, age) for name in SEMANTICS],
        dtype=np.float64,
    )
    return model_weights, predictions


def freshness_uncertainty_score(
    posterior: SemanticPosterior,
    age: TemporalCoordinates,
    *,
    decision_threshold: float = 0.82,
) -> float:
    """Prefer probes whose model-averaged validity is near the action boundary."""
    p_valid = posterior.probability(age, strategy="average")
    return -abs(p_valid - decision_threshold)


def semantic_disagreement_score(
    posterior: SemanticPosterior,
    age: TemporalCoordinates,
) -> float:
    """Weighted variance of P(valid) across temporal semantic hypotheses."""
    weights, predictions = semantic_predictions(posterior, age)
    mean = float(np.dot(weights, predictions))
    return float(np.dot(weights, (predictions - mean) ** 2))


def semantic_information_gain_score(
    posterior: SemanticPosterior,
    age: TemporalCoordinates,
) -> float:
    """Expected one-step information gain about semantic *class*.

    Hazard-rate uncertainty is integrated inside each semantic model. The score
    therefore asks how much a binary valid/stale audit is expected to reduce
    entropy over:

        world_hazard / event_hazard / until_change

    rather than how much it teaches the exact nuisance hazard rate.
    """
    weights, predictions = semantic_predictions(posterior, age)
    current_entropy = _entropy(weights)

    p_valid = float(np.dot(weights, predictions))
    p_valid = min(1.0 - 1e-12, max(1e-12, p_valid))
    p_stale = 1.0 - p_valid

    posterior_if_valid = weights * predictions / p_valid
    posterior_if_stale = weights * (1.0 - predictions) / p_stale

    expected_entropy = (
        p_valid * _entropy(posterior_if_valid)
        + p_stale * _entropy(posterior_if_stale)
    )
    return current_entropy - expected_entropy


def score_probe(
    posterior: SemanticPosterior,
    opportunity: ProbeOpportunity,
    *,
    method: ProbeScore,
    decision_threshold: float = 0.82,
) -> float:
    if method == "freshness_uncertainty":
        return freshness_uncertainty_score(
            posterior,
            opportunity.coordinates,
            decision_threshold=decision_threshold,
        )
    if method == "semantic_disagreement":
        return semantic_disagreement_score(posterior, opportunity.coordinates)
    if method == "semantic_information_gain":
        return semantic_information_gain_score(posterior, opportunity.coordinates)
    raise ValueError(f"unknown probe score {method!r}")


def choose_probe(
    posterior: SemanticPosterior,
    opportunities: Iterable[ProbeOpportunity],
    *,
    method: ProbeScore,
    decision_threshold: float = 0.82,
) -> ProbeOpportunity:
    candidates = list(opportunities)
    if not candidates:
        raise ValueError("at least one probe opportunity is required")

    scores = [
        score_probe(
            posterior,
            candidate,
            method=method,
            decision_threshold=decision_threshold,
        )
        for candidate in candidates
    ]
    return candidates[int(np.argmax(scores))]
