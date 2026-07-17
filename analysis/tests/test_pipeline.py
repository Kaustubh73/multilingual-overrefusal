"""Smoke tests for the unified analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.ingest import ingest_all, load_track_b
from analysis.pipeline import run_pipeline
from analysis.stats import chi_square_independence, cohens_h


def test_chi_square_2x2_independence():
    # Constructed table: strong association
    table = [[10, 2], [2, 10]]
    result = chi_square_independence(__import__("numpy").array(table))
    assert result["chi2"] > 0
    assert result["df"] == 1
    assert 0 <= result["p_value"] <= 1
    assert result["cramers_v"] > 0


def test_cohens_h_symmetry():
    assert cohens_h(0.5, 0.5) == pytest.approx(0.0)
    assert cohens_h(0.8, 0.2) == pytest.approx(-cohens_h(0.2, 0.8))


def test_ingest_track_b_not_empty():
    df = load_track_b()
    if not Path("results/track_b_raw.csv").exists():
        pytest.skip("track_b artifacts missing")
    assert not df.empty
    assert "track" in df.columns
    assert set(df["track"]) == {"track_b"}


def test_ingest_limit_reduces_rows():
    full = ingest_all(limit=None)
    limited = ingest_all(limit=5)
    if full.empty:
        pytest.skip("no results to ingest")
    assert len(limited) <= len(full)
    for track in limited["track"].unique():
        assert limited[limited["track"] == track]["sample_id"].nunique() <= 5


def test_pipeline_smoke_writes_artifacts(tmp_path: Path):
    result = run_pipeline(
        out_dir=tmp_path,
        limit=5,
        label="pytest smoke",
        write_report=False,
        write_figures=False,
    )
    if result.merged.empty:
        pytest.skip("no results to analyse")
    assert (tmp_path / "rates_by_language.csv").exists()
    assert (tmp_path / "chi_square.csv").exists()
    assert (tmp_path / "refusal_rates.tex").exists()
    meta = json.loads((tmp_path / "analysis_meta.json").read_text())
    assert meta["is_smoke"] is True
    assert meta["limit"] == 5
