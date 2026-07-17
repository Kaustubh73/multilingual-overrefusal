"""Tests for run manifests and reproduce helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.reproducibility.config_snapshot import load_config_snapshot, write_config_snapshot
from src.reproducibility.manifest import build_manifest, load_manifest, write_manifest
from src.reproducibility.paths import RUNS_DIR, config_path, manifest_path
from src.reproducibility.reproduce import build_track_b_config
from src.reproducibility.run_context import begin_run, finish_run, generate_run_id


class _TrackBArgs:
    model = "meta-llama/Llama-3.1-8B-Instruct"
    judge_model = "gpt-oss:20b"
    limit = None
    limit_per_task = 5
    skip_inference = False
    skip_judge = True
    no_resume = False
    smoke_stub = True
    force_inference = False
    no_report = True
    force_full = False


def test_generate_run_id_unique():
    a = generate_run_id("track-b")
    b = generate_run_id("track-b")
    assert a != b
    assert "track-b" in a


def test_build_manifest_has_required_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("src.reproducibility.manifest.PROJECT_ROOT", tmp_path)
    manifest = build_manifest(
        run_id="test-run",
        pipeline="track-b",
        command="track-b run",
        argv=["main.py", "track-b", "run", "--smoke-stub"],
        config={"pipeline": "track-b", "smoke_stub": True},
    )
    for key in (
        "run_id",
        "pipeline",
        "git",
        "python_version",
        "dependencies",
        "seeds",
        "hardware",
        "config",
    ):
        assert key in manifest
    assert manifest["seeds"]["torch"] == 42


def test_config_snapshot_roundtrip(tmp_path):
    cfg = {"pipeline": "track-b", "limit_per_task": 5, "smoke_stub": True}
    path = tmp_path / "config.yaml"
    write_config_snapshot(path, cfg)
    loaded = load_config_snapshot(path)
    assert loaded == cfg


def test_begin_finish_run_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("src.reproducibility.run_context.RUNS_DIR", tmp_path)
    monkeypatch.setattr("src.reproducibility.paths.RUNS_DIR", tmp_path)

    ctx = begin_run(
        pipeline="track-b",
        command="track-b run",
        argv=["main.py", "track-b", "run"],
        config={"pipeline": "track-b", "smoke_stub": True},
        run_id="unit-test-run",
    )
    assert manifest_path("unit-test-run").exists()
    assert config_path("unit-test-run").exists()
    finish_run(ctx, status="completed")
    manifest = json.loads(manifest_path("unit-test-run").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["finished_at"] is not None


def test_build_track_b_config_from_args():
    cfg = build_track_b_config(_TrackBArgs(), seeds={"torch": 7, "numpy": 7, "random": 7})
    assert cfg["pipeline"] == "track-b"
    assert cfg["limit_per_task"] == 5
    assert cfg["smoke_stub"] is True
    assert cfg["seeds"]["torch"] == 7
    assert cfg["catalog_path"].startswith("data/")


def test_load_manifest_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_manifest("nonexistent-run-id-xyz")
