"""SafeConstellations toy scaffold: representation-space refusal steering (CPU-only).

This package is a *runnable plumbing* of the SafeConstellations intervention
(arXiv:2508.11290): build per-task refusal/target centroids from hidden states,
detect the benign task at inference, and selectively steer representations toward
the non-refusal manifold.

By default everything runs on a tiny synthetic "toy world" so the full pipeline
exercises end-to-end on CPU with no model downloads. Swap in a real HuggingFace
model by setting ``HiddenStateConfig.backend = "hf"`` (gated, opt-in).
"""

from __future__ import annotations

from src.safeconstellations.config import HiddenStateConfig, SteeringConfig, ToyWorldConfig
from src.safeconstellations.task_store import TaskEmbeddingsStore
from src.safeconstellations.steering import SteeringController

__all__ = [
    "HiddenStateConfig",
    "SteeringConfig",
    "ToyWorldConfig",
    "TaskEmbeddingsStore",
    "SteeringController",
]
