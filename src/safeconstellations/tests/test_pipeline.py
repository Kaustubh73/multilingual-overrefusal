"""Integration tests for the toy SafeConstellations pipeline."""

from __future__ import annotations

import numpy as np

from src.safeconstellations.config import HiddenStateConfig, SteeringConfig, ToyWorldConfig
from src.safeconstellations.data import split_examples, toy_dataset
from src.safeconstellations.hidden_states import ToyHiddenStateModel
from src.safeconstellations.linalg import cosine_similarity
from src.safeconstellations.pipeline import run_pipeline
from src.safeconstellations.task_store import TaskEmbeddingsStore


def test_extracted_refusal_direction_aligns_with_true_axis():
    cfg = ToyWorldConfig(seed=1)
    backend = ToyHiddenStateModel(cfg)
    train = split_examples(toy_dataset(), "train")
    store = TaskEmbeddingsStore.fit(train, backend, cfg=cfg)

    last = backend.n_layers - 1
    true_axis = backend._axis[last]
    for task in store.tasks:
        rd = store.store[task][last].refusal_centroid - store.store[task][last].target_centroid
        # Extracted refusal direction should be (anti)parallel to the true axis.
        assert abs(cosine_similarity(rd, true_axis)) > 0.9


def test_steering_reduces_over_refusal_and_preserves_harmful_refusal():
    result = run_pipeline(
        hidden_config=HiddenStateConfig(toy=ToyWorldConfig(seed=0)),
        steering_config=SteeringConfig(alpha=2.5, detection_threshold=0.85),
    )
    s = result.summary
    baseline_or = s._pct(s.baseline_or)
    steered_or = s._pct(s.steered_or)
    baseline_hr = s._pct(s.baseline_hr)
    steered_hr = s._pct(s.steered_hr)

    # Baseline shows non-trivial over-refusal that steering reduces.
    assert baseline_or > 0.0
    assert steered_or < baseline_or
    # Genuinely harmful refusals are preserved (not steered away).
    assert steered_hr == baseline_hr == 1.0


def test_task_detection_picks_correct_task_with_high_confidence():
    cfg = ToyWorldConfig(seed=2)
    backend = ToyHiddenStateModel(cfg)
    examples = toy_dataset()
    store = TaskEmbeddingsStore.fit(split_examples(examples, "train"), backend, cfg=cfg)

    correct = 0
    test = split_examples(examples, "test")
    for ex in test:
        detected, conf = store.detect_task(backend.hidden_states(ex))
        if detected == ex.task:
            correct += 1
    # Toy task centers dominate, so detection should be essentially perfect.
    assert correct == len(test)


def test_no_steer_when_below_confidence_threshold():
    # An impossible threshold (>1) means nothing is ever steered.
    result = run_pipeline(
        hidden_config=HiddenStateConfig(toy=ToyWorldConfig(seed=0)),
        steering_config=SteeringConfig(detection_threshold=1.01),
    )
    assert all(not r.steered for r in result.summary.rows)
    s = result.summary
    # With no steering, baseline and steered refusal decisions are identical.
    assert s.baseline_or == s.steered_or
