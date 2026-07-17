"""Golden-set regression tests locking evaluator behavior (not publishable rates)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.baseline_comparison import render_baseline_table
from src.judge_ensemble import _majority_vote
from src.languages import ALL_LANGUAGES, EXTENDED_LANGUAGES
from src.refusal_agreement import (
    _cohen_kappa,
    compute_agreement_report,
    load_annotations_long,
    pairwise_cohen_kappa,
)
from src.refusal_judge import parse_judge_output
from src.response_classifier import classify_response
from src.translation_qa import qa_reviewed_translations

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "data" / "golden"


@pytest.fixture()
def judge_cases():
    return json.loads((GOLDEN_DIR / "judge_parse_cases.json").read_text(encoding="utf-8"))


@pytest.fixture()
def classifier_cases():
    return json.loads((GOLDEN_DIR / "classifier_cases.json").read_text(encoding="utf-8"))


@pytest.fixture()
def translation_qa_df():
    return pd.read_csv(GOLDEN_DIR / "translation_qa_cases.csv", dtype=str, keep_default_na=False)


def test_language_registry_has_five_plus_extensions():
    assert len(EXTENDED_LANGUAGES) >= 5
    assert "spanish" in ALL_LANGUAGES
    assert "japanese" in ALL_LANGUAGES


@pytest.mark.parametrize("case_idx", range(5))
def test_judge_parse_golden(judge_cases, case_idx):
    case = judge_cases[case_idx]
    ri, _ = parse_judge_output(case["raw"])
    assert ri == case["expected_ri"]


@pytest.mark.parametrize("case_idx", range(3))
def test_classifier_golden(classifier_cases, case_idx):
    case = classifier_cases[case_idx]
    l1, l2 = classify_response(
        case["response"],
        task=case["task"],
        language=case["language"],
        source_text=case["source_text"],
    )
    assert l1 == case["expected_level1"]
    assert l2 == case["expected_level2"]


def test_translation_qa_golden(translation_qa_df):
    report = qa_reviewed_translations(translation_qa_df, id_col="prompt_id", require_reviewed=True)
    by_id = {
        (row["prompt_id"], row["language"]): row["passed"]
        for _, row in report.iterrows()
    }

    for _, row in translation_qa_df.iterrows():
        expected_langs = str(row["expected_pass_languages"]).strip()
        if expected_langs == "none":
            failing = report[(report["prompt_id"] == row["prompt_id"]) & (~report["passed"])]
            assert len(failing) > 0
            continue
        for lang in expected_langs.split(";"):
            assert by_id[(row["prompt_id"], lang)] is True


def test_agreement_golden():
    path = GOLDEN_DIR / "agreement_long.csv"
    ann_df = load_annotations_long(path)

    pair_ab = ann_df[
        ann_df["annotator"].isin(["alice", "bob"])
        & ann_df["item_id"].astype(str).str.startswith("g1")
    ]
    pair = pairwise_cohen_kappa(pair_ab)
    ab = pair[(pair["annotator_a"] == "alice") & (pair["annotator_b"] == "bob")]
    assert not ab.empty
    assert ab.iloc[0]["agreement_rate"] == pytest.approx(2 / 3)
    assert 0.0 < ab.iloc[0]["cohen_kappa"] < 1.0

    three_rater = ann_df[ann_df["item_id"].astype(str).str.startswith("g2")]
    fleiss = compute_agreement_report(three_rater)
    fk = fleiss["fleiss_kappa"]["fleiss_kappa"]
    assert fk > 0.0


def test_ensemble_majority_vote():
    votes = {"m1": "direct_refusal", "m2": "direct_answer", "m3": "direct_refusal"}
    label, unanimous = _majority_vote(votes)
    assert label == "direct_refusal"
    assert unanimous is False


def test_baseline_table_has_no_fake_local_numbers():
    md = render_baseline_table()
    assert "36.4%" in md  # published reference
    assert "| — | — |" in md or "| — |" in md
    assert "smoke-test" in md.lower() or "smoke test" in md.lower() or "smoke" in md.lower()


def test_cohen_kappa_perfect():
    labels = ["direct_answer", "direct_refusal", "indirect_refusal"]
    assert _cohen_kappa(labels, labels) == pytest.approx(1.0)
