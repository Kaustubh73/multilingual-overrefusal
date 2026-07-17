"""Config dataclasses for the SafeConstellations toy scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field

# Tasks the steering store is fit for. The paper uses 5 English NLP frames;
# the toy keeps the two highest-OR tasks from Figure 3 (translation, sentiment).
BENIGN_TASKS: tuple[str, ...] = ("translation", "sentiment")


@dataclass(frozen=True)
class ToyWorldConfig:
    """Parameters of the synthetic hidden-state generator (CPU stand-in for an LLM).

    The toy world emits per-layer hidden states whose projection onto a hidden
    "refusal axis" equals a refusal logit. The logit rises when content *looks*
    harmful (the over-refusal trigger) and rises more when content is *truly*
    harmful (legitimate refusal). This lets the scaffold demonstrate that bounded
    steering removes over-refusal while preserving genuine refusals.
    """

    n_layers: int = 8
    hidden_dim: int = 16
    top_k_layers: int = 4
    center_scale: float = 5.0
    base_refusal: float = -1.0
    w_looks_harmful: float = 2.0
    w_truly_harmful: float = 4.0
    noise_scale: float = 0.15
    refusal_threshold: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class HiddenStateConfig:
    """Selects how hidden states are produced.

    backend="toy" (default) uses the synthetic generator (no downloads, CPU only).
    backend="hf" loads a real causal LM and reads ``hidden_states`` from a forward
    pass. The HF path is opt-in and never triggered by the demo/tests.
    """

    backend: str = "toy"
    hf_model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    hf_max_length: int = 64
    toy: ToyWorldConfig = field(default_factory=ToyWorldConfig)


@dataclass(frozen=True)
class SteeringConfig:
    """Inference-time steering knobs (mirrors the paper's conditional steering)."""

    alpha: float = 2.5
    detection_threshold: float = 0.85
    benign_tasks: tuple[str, ...] = BENIGN_TASKS
    mode: str = "additive"  # "additive" (push toward target) or "ablate" (remove refusal comp)
