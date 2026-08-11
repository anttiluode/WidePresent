"""Online/rolling semantic posterior for non-stationary temporal validity.

This is a small sufficient-statistics companion to temporal_validity_learning.py.
It supports either cumulative evidence or a fixed rolling audit window.

The goal is not to introduce a new change-detection algorithm.  It provides the
boring baseline needed when a source's previously learned temporal semantics may
drift: forget old audit evidence at a controlled rate and allow the semantic
posterior to reopen.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from temporal_validity_learning import (
    AuditObservation,
    SEMANTICS,
    SemanticPosterior,
)


def _softmax(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    return weights / np.sum(weights)


def _log_mean_exp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.mean(np.exp(values - maximum))))


@dataclass
class _Contribution:
    world: np.ndarray
    event: np.ndarray
    until_change: float


class OnlineSemanticAccumulator:
    """Maintain a semantic posterior using cumulative or rolling audit evidence."""

    def __init__(
        self,
        *,
        lambda_grid: np.ndarray | None = None,
        window: int | None = None,
        epsilon: float = 1e-9,
    ) -> None:
        self.lambda_grid = (
            np.geomspace(0.001, 0.20, 240).astype(np.float64)
            if lambda_grid is None
            else np.asarray(lambda_grid, dtype=np.float64)
        )
        if self.lambda_grid.ndim != 1 or len(self.lambda_grid) == 0:
            raise ValueError("lambda_grid must be a non-empty vector")
        if np.any(self.lambda_grid <= 0):
            raise ValueError("lambda_grid must contain positive values")
        if window is not None and window <= 0:
            raise ValueError("window must be positive or None")
        if not (0.0 < epsilon < 0.5):
            raise ValueError("epsilon must lie in (0, 0.5)")

        self.window = window
        self.epsilon = epsilon
        self._world_log_likelihood = np.zeros(len(self.lambda_grid), dtype=np.float64)
        self._event_log_likelihood = np.zeros(len(self.lambda_grid), dtype=np.float64)
        self._change_log_likelihood = 0.0
        self._history: deque[_Contribution] = deque()

    @property
    def n_observations(self) -> int:
        return len(self._history)

    def _contribution(self, observation: AuditObservation) -> _Contribution:
        y = 1.0 if observation.still_valid else 0.0

        def hazard(age: float) -> np.ndarray:
            p = np.exp(-self.lambda_grid * age)
            p = np.clip(p, self.epsilon, 1.0 - self.epsilon)
            return y * np.log(p) + (1.0 - y) * np.log(1.0 - p)

        change_p = 0.0 if observation.invalidated else 1.0
        change_p = min(1.0 - self.epsilon, max(self.epsilon, change_p))
        change_ll = y * math.log(change_p) + (1.0 - y) * math.log(1.0 - change_p)

        return _Contribution(
            world=hazard(observation.world_age_seconds),
            event=hazard(float(observation.event_age)),
            until_change=float(change_ll),
        )

    def update(self, observation: AuditObservation) -> None:
        contribution = self._contribution(observation)
        self._world_log_likelihood += contribution.world
        self._event_log_likelihood += contribution.event
        self._change_log_likelihood += contribution.until_change
        self._history.append(contribution)

        if self.window is not None and len(self._history) > self.window:
            old = self._history.popleft()
            self._world_log_likelihood -= old.world
            self._event_log_likelihood -= old.event
            self._change_log_likelihood -= old.until_change

    def posterior(self) -> SemanticPosterior:
        if not self._history:
            # No evidence: equal semantic prior and equal rate prior.
            model_weights = {name: 1.0 / len(SEMANTICS) for name in SEMANTICS}
            uniform_lambda = np.full(
                len(self.lambda_grid),
                1.0 / len(self.lambda_grid),
                dtype=np.float64,
            )
            return SemanticPosterior(
                lambda_grid=self.lambda_grid.copy(),
                model_weights=model_weights,
                lambda_weights={
                    "world_hazard": uniform_lambda.copy(),
                    "event_hazard": uniform_lambda.copy(),
                    "until_change": None,
                },
                log_evidence={name: 0.0 for name in SEMANTICS},
            )

        world_evidence = _log_mean_exp(self._world_log_likelihood)
        event_evidence = _log_mean_exp(self._event_log_likelihood)
        change_evidence = self._change_log_likelihood

        evidence = np.asarray(
            [world_evidence, event_evidence, change_evidence],
            dtype=np.float64,
        )
        model_probability = _softmax(evidence)

        return SemanticPosterior(
            lambda_grid=self.lambda_grid.copy(),
            model_weights={
                name: float(weight)
                for name, weight in zip(SEMANTICS, model_probability)
            },
            lambda_weights={
                "world_hazard": _softmax(self._world_log_likelihood),
                "event_hazard": _softmax(self._event_log_likelihood),
                "until_change": None,
            },
            log_evidence={
                "world_hazard": world_evidence,
                "event_hazard": event_evidence,
                "until_change": change_evidence,
            },
        )

    def copy_as_cumulative(self) -> "OnlineSemanticAccumulator":
        """Copy current sufficient statistics but remove any future window limit."""
        other = OnlineSemanticAccumulator(
            lambda_grid=self.lambda_grid.copy(),
            window=None,
            epsilon=self.epsilon,
        )
        other._world_log_likelihood = self._world_log_likelihood.copy()
        other._event_log_likelihood = self._event_log_likelihood.copy()
        other._change_log_likelihood = self._change_log_likelihood
        other._history = deque(self._history)
        return other
