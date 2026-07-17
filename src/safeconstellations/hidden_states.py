"""Hidden-state backends: synthetic toy world (default) and gated HF reader.

A backend maps an :class:`Example` to a per-layer hidden-state stack of shape
``(n_layers, hidden_dim)`` and can decide whether the model refuses given a
(possibly steered) stack.

The toy backend is fully deterministic (seeded) and runs on CPU with no
downloads. The HF backend is opt-in (``backend="hf"``) and lazily imports
torch/transformers; it is never exercised by the demo or tests.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from src.safeconstellations.config import HiddenStateConfig, ToyWorldConfig
from src.safeconstellations.data import Example
from src.safeconstellations.linalg import projection_scalar, unit_vector

logger = logging.getLogger(__name__)


class HiddenStateModel(Protocol):
    """Common interface for hidden-state producers."""

    n_layers: int
    hidden_dim: int

    def hidden_states(self, example: Example) -> np.ndarray:
        """Return a ``(n_layers, hidden_dim)`` stack of per-layer hidden states."""
        ...

    def decide_refusal(self, hidden_states: np.ndarray) -> bool:
        """Return True if the model would refuse given this (possibly steered) stack."""
        ...


class ToyHiddenStateModel:
    """Deterministic synthetic LLM stand-in.

    Construction precomputes, per layer, a random unit "refusal axis" and a set of
    per-task centers that are orthogonalized against that axis. A hidden state is::

        h[layer] = task_center[task, layer]
                   + axis[layer] * (refusal_logit * layer_gain[layer])
                   + small_noise

    where ``refusal_logit = base + w_looks*looks_harmful + w_harm*truly_harmful``.
    The model refuses when the final-layer projection onto its (true) refusal axis
    exceeds ``refusal_threshold``.
    """

    def __init__(self, config: ToyWorldConfig | None = None, *, tasks: tuple[str, ...] = ("translation", "sentiment")):
        self.cfg = config or ToyWorldConfig()
        self.n_layers = self.cfg.n_layers
        self.hidden_dim = self.cfg.hidden_dim
        self.tasks = tuple(tasks)
        rng = np.random.default_rng(self.cfg.seed)

        # Refusal emerges in later layers: gain ramps 0 -> 1 across depth.
        self._layer_gain = np.linspace(0.0, 1.0, self.n_layers)

        # Per-layer true refusal axis (unit).
        self._axis = np.stack([unit_vector(rng.standard_normal(self.hidden_dim)) for _ in range(self.n_layers)])

        # Per-(task, layer) center, orthogonalized against that layer's axis so the
        # task identity does not leak into the refusal projection.
        self._task_center: dict[str, np.ndarray] = {}
        for task in self.tasks:
            centers = []
            for layer in range(self.n_layers):
                c = rng.standard_normal(self.hidden_dim) * self.cfg.center_scale
                axis = self._axis[layer]
                c = c - np.dot(c, axis) * axis  # orthogonalize
                centers.append(c)
            self._task_center[task] = np.stack(centers)

        self._rng = rng

    def _refusal_logit(self, example: Example) -> float:
        return (
            self.cfg.base_refusal
            + self.cfg.w_looks_harmful * float(example.looks_harmful)
            + self.cfg.w_truly_harmful * float(example.truly_harmful)
        )

    def hidden_states(self, example: Example) -> np.ndarray:
        if example.task not in self._task_center:
            raise KeyError(f"Unknown task {example.task!r}; toy world knows {self.tasks}")
        logit = self._refusal_logit(example)
        centers = self._task_center[example.task]
        noise = self._rng.standard_normal((self.n_layers, self.hidden_dim)) * self.cfg.noise_scale
        signal = (self._axis.T * (logit * self._layer_gain)).T  # (n_layers, d)
        return centers + signal + noise

    def decide_refusal(self, hidden_states: np.ndarray) -> bool:
        """Refuse iff the final-layer projection onto the true axis exceeds threshold."""
        last = self.n_layers - 1
        proj = projection_scalar(hidden_states[last], self._axis[last])
        return proj > self.cfg.refusal_threshold


class HFHiddenStateModel:
    """Real HuggingFace backend (opt-in; lazily imports torch/transformers).

    Reads ``hidden_states`` from a forward pass on the prompt. Refusal labeling
    for real models should use the project's existing judge/classifier path; here
    ``decide_refusal`` raises so callers wire it explicitly.
    """

    def __init__(self, config: HiddenStateConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: F401  (lazy, gated)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.warning(
            "Loading real HF model %s for hidden-state extraction (opt-in path).",
            self.config.hf_model_id,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.hf_model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id, output_hidden_states=True
        )
        self._model.eval()
        self.n_layers = self._model.config.num_hidden_layers + 1
        self.hidden_dim = self._model.config.hidden_size

    def hidden_states(self, example: Example) -> np.ndarray:
        import torch

        self._load()
        inputs = self._tokenizer(
            example.text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.hf_max_length,
        )
        with torch.no_grad():
            out = self._model(**inputs)
        # Mean-pool each layer over tokens -> (n_layers, hidden_dim).
        layers = [hs[0].mean(dim=0).numpy() for hs in out.hidden_states]
        return np.stack(layers).astype(np.float64)

    def decide_refusal(self, hidden_states: np.ndarray) -> bool:  # pragma: no cover - gated
        raise NotImplementedError(
            "Real-model refusal labeling should use src.refusal_judge / "
            "src.response_classifier on generated text, not the toy decision rule."
        )


def build_backend(config: HiddenStateConfig, *, tasks: tuple[str, ...]) -> HiddenStateModel:
    """Factory: return a hidden-state backend for the configured mode."""
    if config.backend == "toy":
        return ToyHiddenStateModel(config.toy, tasks=tasks)
    if config.backend == "hf":
        return HFHiddenStateModel(config)
    raise ValueError(f"Unknown backend {config.backend!r}; expected 'toy' or 'hf'.")
