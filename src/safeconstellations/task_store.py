"""Task Embeddings Store: per-task refusal/target centroids at the top-K layers.

This is the fitted artifact of SafeConstellations. From labeled train hidden
states it stores, for each task and each steered layer:

- ``task_centroid``  -- mean over all of that task's examples (used for task
  detection at inference via cosine similarity), and
- ``refusal_centroid`` / ``target_centroid`` -- means over refusing vs
  non-refusing examples (their difference defines the steering direction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.safeconstellations.config import ToyWorldConfig
from src.safeconstellations.data import Example
from src.safeconstellations.hidden_states import HiddenStateModel
from src.safeconstellations.linalg import centroid, cosine_similarity, steering_direction

logger = logging.getLogger(__name__)


@dataclass
class TaskCentroids:
    """Centroids for one task at one layer."""

    task_centroid: np.ndarray
    refusal_centroid: np.ndarray
    target_centroid: np.ndarray

    def steering_direction(self) -> np.ndarray:
        """Unit vector from refusal toward non-refusal (target) manifold."""
        return steering_direction(self.refusal_centroid, self.target_centroid)


class TaskEmbeddingsStore:
    """Fitted store of per-(task, layer) centroids."""

    def __init__(self, top_k_layers: list[int], detection_layer: int):
        self.top_k_layers = list(top_k_layers)
        self.detection_layer = detection_layer
        # store[task][layer] = TaskCentroids
        self.store: dict[str, dict[int, TaskCentroids]] = {}

    @property
    def tasks(self) -> list[str]:
        return list(self.store)

    @classmethod
    def fit(
        cls,
        train_examples: list[Example],
        backend: HiddenStateModel,
        *,
        top_k_layers: list[int] | None = None,
        detection_layer: int | None = None,
        cfg: ToyWorldConfig | None = None,
    ) -> "TaskEmbeddingsStore":
        """Fit the store from labeled train examples.

        Refusal labels for the toy come from the backend's own decision rule on the
        *unsteered* hidden states; for real models, pass examples pre-labeled and
        adapt this method to read those labels.
        """
        n_layers = backend.n_layers
        if top_k_layers is None:
            k = (cfg or ToyWorldConfig()).top_k_layers
            top_k_layers = list(range(n_layers - k, n_layers))
        if detection_layer is None:
            detection_layer = top_k_layers[0]

        store = cls(top_k_layers=top_k_layers, detection_layer=detection_layer)

        # Cache hidden states + refusal label per example once.
        by_task: dict[str, list[tuple[np.ndarray, bool]]] = {}
        for ex in train_examples:
            hs = backend.hidden_states(ex)
            refused = backend.decide_refusal(hs)
            by_task.setdefault(ex.task, []).append((hs, refused))

        layers_needed = sorted(set(top_k_layers) | {detection_layer})
        for task, items in by_task.items():
            store.store[task] = {}
            for layer in layers_needed:
                all_vecs = np.stack([hs[layer] for hs, _ in items])
                refuse_vecs = [hs[layer] for hs, r in items if r]
                target_vecs = [hs[layer] for hs, r in items if not r]
                if not refuse_vecs or not target_vecs:
                    logger.warning(
                        "Task %r layer %d: only one refusal class present "
                        "(refuse=%d target=%d); steering direction may be unreliable.",
                        task, layer, len(refuse_vecs), len(target_vecs),
                    )
                refuse_c = centroid(np.stack(refuse_vecs)) if refuse_vecs else centroid(all_vecs)
                target_c = centroid(np.stack(target_vecs)) if target_vecs else centroid(all_vecs)
                store.store[task][layer] = TaskCentroids(
                    task_centroid=centroid(all_vecs),
                    refusal_centroid=refuse_c,
                    target_centroid=target_c,
                )
        logger.info(
            "Fitted Task Embeddings Store: tasks=%s, top_k_layers=%s, detection_layer=%d",
            store.tasks, store.top_k_layers, store.detection_layer,
        )
        return store

    def detect_task(self, hidden_states: np.ndarray) -> tuple[str, float]:
        """Return (best_task, confidence) via cosine to per-task detection centroids.

        Confidence is a softmax over cosine similarities so it lives in (0, 1) and
        can be thresholded like the paper's >=0.85 gate.
        """
        vec = hidden_states[self.detection_layer]
        tasks = self.tasks
        sims = np.array(
            [cosine_similarity(vec, self.store[t][self.detection_layer].task_centroid) for t in tasks]
        )
        # Temperature-scaled softmax to sharpen the margin between tasks.
        scaled = sims / 0.1
        scaled -= scaled.max()
        weights = np.exp(scaled)
        probs = weights / weights.sum()
        best = int(np.argmax(probs))
        return tasks[best], float(probs[best])
