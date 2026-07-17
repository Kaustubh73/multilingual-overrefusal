"""Inference-time conditional steering controller.

Implements the SafeConstellations decision rule: detect the task; if it is in the
benign set and detection confidence clears the threshold, steer the top-K layer
hidden states toward that task's non-refusal manifold. Otherwise leave the
representations untouched (so genuinely harmful inputs and low-confidence cases
are not nudged toward compliance).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.safeconstellations.config import SteeringConfig
from src.safeconstellations.linalg import remove_component, steer_additive
from src.safeconstellations.task_store import TaskEmbeddingsStore

logger = logging.getLogger(__name__)


@dataclass
class SteeringResult:
    detected_task: str
    confidence: float
    steered: bool
    hidden_states: np.ndarray


class SteeringController:
    def __init__(self, store: TaskEmbeddingsStore, config: SteeringConfig | None = None):
        self.store = store
        self.cfg = config or SteeringConfig()

    def _should_steer(self, task: str, confidence: float) -> bool:
        return task in self.cfg.benign_tasks and confidence >= self.cfg.detection_threshold

    def apply(self, hidden_states: np.ndarray) -> SteeringResult:
        """Detect task and conditionally steer the top-K layers (returns a copy)."""
        task, confidence = self.store.detect_task(hidden_states)
        steered = self._should_steer(task, confidence)
        out = hidden_states.copy()
        if not steered:
            return SteeringResult(task, confidence, False, out)

        for layer in self.store.top_k_layers:
            centroids = self.store.store[task][layer]
            if self.cfg.mode == "ablate":
                direction = centroids.steering_direction()
                out[layer] = remove_component(out[layer], direction)
            else:  # additive: push toward the non-refusal manifold
                out[layer] = steer_additive(
                    out[layer], centroids.steering_direction(), self.cfg.alpha
                )
        return SteeringResult(task, confidence, True, out)
