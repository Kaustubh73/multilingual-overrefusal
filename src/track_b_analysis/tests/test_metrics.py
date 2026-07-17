"""Unit tests for Track B analysis metric functions (numpy/pandas/pytest)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.track_b_analysis import loaders, metrics
from src.track_b_analysis.synthetic import generate_synthetic


# --------------------------------------------------------------------------- #
# Wilson confidence interval
# --------------------------------------------------------------------------- #
def test_wilson_ci_known_value():
    # Classic textbook value: 0/10 successes, 95% Wilson upper ~ 0.278.
    low, high = metrics.wilson_ci(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.2775, abs=1e-3)


def test_wilson_ci_symmetric_half():
    low, high = metrics.wilson_ci(5, 10)
    assert low == pytest.approx(1 - high, abs=1e-9)  # symmetric around 0.5
    assert 0.5 == pytest.approx((low + high) / 2, abs=1e-9)


def test_wilson_ci_contains_point_estimate():
    for count, n in [(1, 5), (3, 7), (9, 24), (50, 100)]:
        low, high = metrics.wilson_ci(count, n)
        assert low <= count / n <= high


def test_wilson_ci_bounds_clamped():
    low, high = metrics.wilson_ci(10, 10)
    assert 0.0 <= low <= 1.0
    assert high == pytest.approx(1.0, abs=1e-9)
    low0, high0 = metrics.wilson_ci(0, 0)
    assert (low0, high0) == (0.0, 1.0)


def test_wilson_ci_narrows_with_n():
    _, hi_small = metrics.wilson_ci(1, 5)
    _, hi_large = metrics.wilson_ci(20, 100)
    width_small = hi_small - metrics.wilson_ci(1, 5)[0]
    width_large = hi_large - metrics.wilson_ci(20, 100)[0]
    assert width_large < width_small


def test_wilson_ci_confidence_widens():
    low90, high90 = metrics.wilson_ci(5, 20, confidence=0.90)
    low99, high99 = metrics.wilson_ci(5, 20, confidence=0.99)
    assert (high99 - low99) > (high90 - low90)


def test_wilson_ci_rejects_bad_input():
    with pytest.raises(ValueError):
        metrics.wilson_ci(5, 3)
    with pytest.raises(ValueError):
        metrics.wilson_ci(-1, 5)


def test_z_for_confidence_lookup_and_approx():
    assert metrics.z_for_confidence(0.95) == pytest.approx(1.959963984540054, abs=1e-9)
    # Non-tabulated level falls back to the inverse-normal approximation.
    assert metrics.z_for_confidence(0.80) == pytest.approx(1.2815515594, abs=1e-3)


# --------------------------------------------------------------------------- #
# proportion_stats
# --------------------------------------------------------------------------- #
def test_proportion_stats_fields():
    s = metrics.proportion_stats(2, 8)
    assert s["count"] == 2 and s["n"] == 8
    assert s["rate"] == pytest.approx(0.25)
    assert s["ci_low"] <= 0.25 <= s["ci_high"]


def test_proportion_stats_zero_n():
    s = metrics.proportion_stats(0, 0)
    assert math.isnan(s["rate"])
    assert (s["ci_low"], s["ci_high"]) == (0.0, 1.0)


# --------------------------------------------------------------------------- #
# Grouping / rate_by_group
# --------------------------------------------------------------------------- #
@pytest.fixture()
def toy_merged() -> pd.DataFrame:
    rows = [
        # english translation: 1 refusal of 2
        {"language": "english", "task_label": "translation", "ri": "direct_answer",
         "is_benign_task": "True", "text_type": "benign_instruction"},
        {"language": "english", "task_label": "translation", "ri": "direct_refusal",
         "is_benign_task": "True", "text_type": "benign_instruction"},
        # hindi translation: 2 refusals of 2
        {"language": "hindi", "task_label": "translation", "ri": "indirect_refusal",
         "is_benign_task": "True", "text_type": "benign_instruction"},
        {"language": "hindi", "task_label": "translation", "ri": "direct_refusal",
         "is_benign_task": "True", "text_type": "harmful_instruction"},
        # hindi sentiment: 0 refusals of 1, plus 1 invalid judgment (dropped)
        {"language": "hindi", "task_label": "sentiment", "ri": "direct_answer",
         "is_benign_task": "True", "text_type": "benign_instruction"},
        {"language": "hindi", "task_label": "sentiment", "ri": "parse_error",
         "is_benign_task": "True", "text_type": "benign_instruction"},
    ]
    return loaders.normalize_frame(pd.DataFrame(rows))


def test_rate_by_group_counts_and_groups(toy_merged):
    table = metrics.rate_by_group(
        toy_merged, success_col="refusal", group_cols=["language", "task_label"]
    )
    idx = table.set_index(["language", "task_label"])
    assert idx.loc[("english", "translation"), "count"] == 1
    assert idx.loc[("english", "translation"), "n"] == 2
    assert idx.loc[("hindi", "translation"), "rate"] == pytest.approx(1.0)


def test_over_refusal_by_group_drops_invalid_ri(toy_merged):
    table = metrics.over_refusal_by_group(toy_merged)
    idx = table.set_index(["language", "task_label"])
    # hindi sentiment: parse_error row dropped -> n == 1, count == 0
    assert idx.loc[("hindi", "sentiment"), "n"] == 1
    assert idx.loc[("hindi", "sentiment"), "count"] == 0
    # hindi translation: both refusals, both benign task -> rate 1.0
    assert idx.loc[("hindi", "translation"), "rate"] == pytest.approx(1.0)


def test_over_refusal_counts_harmful_content_refusal_as_or(toy_merged):
    # The harmful-content hindi translation row is still a benign *task*, so a
    # refusal there counts toward OR (matches SafeConstellations Eq. 9).
    table = metrics.over_refusal_by_group(toy_merged)
    idx = table.set_index(["language", "task_label"])
    assert idx.loc[("hindi", "translation"), "count"] == 2


def test_harmful_refusal_by_group(toy_merged):
    table = metrics.harmful_refusal_by_group(toy_merged)
    # Only one harmful-content row (hindi translation, refused) -> rate 1.0
    assert len(table) == 1
    assert table.iloc[0]["rate"] == pytest.approx(1.0)
    assert table.iloc[0]["language"] == "hindi"


def test_rate_by_group_empty_frame():
    empty = loaders.normalize_frame(
        pd.DataFrame(columns=["language", "task_label", "ri", "is_benign_task", "text_type"])
    )
    table = metrics.rate_by_group(empty, success_col="refusal", group_cols=["language"])
    assert table.empty


# --------------------------------------------------------------------------- #
# Language comparison
# --------------------------------------------------------------------------- #
def test_compare_languages_delta_direction(toy_merged):
    deltas = metrics.compare_languages(toy_merged, baseline="english")
    trans = deltas[(deltas["language"] == "hindi") & (deltas["task"] == "translation")]
    assert len(trans) == 1
    row = trans.iloc[0]
    # hindi translation OR = 1.0, english = 0.5 -> delta +0.5
    assert row["delta_or"] == pytest.approx(0.5)
    assert row["delta_ci_low"] <= row["delta_or"] <= row["delta_ci_high"]


def test_compare_languages_excludes_baseline():
    raw, judged = generate_synthetic(n_per_task=10, seed=1)
    merged = raw.merge(judged[["prompt_id", "ri"]], on="prompt_id", how="left")
    merged = loaders.normalize_frame(merged)
    deltas = metrics.compare_languages(merged, baseline="english")
    assert "english" not in set(deltas["language"])
    assert set(deltas["language"]) <= {"hindi", "hinglish"}


# --------------------------------------------------------------------------- #
# Judge agreement / calibration
# --------------------------------------------------------------------------- #
def test_agreement_rate_perfect():
    out = metrics.agreement_rate(["a", "b", "c"], ["a", "b", "c"])
    assert out["agreement_rate"] == pytest.approx(1.0)
    assert out["cohen_kappa"] == pytest.approx(1.0)


def test_agreement_rate_partial():
    out = metrics.agreement_rate(
        ["x", "x", "y", "y"], ["x", "y", "y", "x"]
    )
    assert out["agreements"] == 2
    assert out["agreement_rate"] == pytest.approx(0.5)


def test_agreement_rate_length_mismatch():
    with pytest.raises(ValueError):
        metrics.agreement_rate(["a"], ["a", "b"])


def test_agreement_rate_empty():
    out = metrics.agreement_rate([], [])
    assert out["n"] == 0
    assert math.isnan(out["agreement_rate"])


# --------------------------------------------------------------------------- #
# Synthetic data integrity
# --------------------------------------------------------------------------- #
def test_synthetic_shape_and_markers():
    raw, judged = generate_synthetic(n_per_task=24)
    # 24 sample_ids/task * 2 tasks * 3 languages
    assert len(raw) == 24 * 2 * 3
    assert len(judged) == len(raw)
    assert (raw["model"] == "SYNTHETIC-fixture").all()
    assert (raw["synthetic"] == "True").all()
    assert set(judged["ri"]) <= set(metrics.VALID_RI)


def test_synthetic_is_deterministic():
    raw_a, _ = generate_synthetic(n_per_task=8, seed=42)
    raw_b, _ = generate_synthetic(n_per_task=8, seed=42)
    pd.testing.assert_frame_equal(raw_a, raw_b)


def test_judge_label_distribution_sums_to_n(toy_merged):
    dist = metrics.judge_label_distribution(toy_merged, group_cols=["language"])
    for _, row in dist.iterrows():
        total = sum(int(row[lab]) for lab in metrics.VALID_RI)
        assert total == int(row["n"])
