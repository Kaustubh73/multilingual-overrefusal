"""Tests for reproduce command wiring."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.reproducibility.config_snapshot import write_config_snapshot
from src.reproducibility.manifest import write_manifest
from src.reproducibility.paths import config_path, manifest_path
from src.reproducibility.reproduce import cmd_reproduce


def _write_prior_run(tmp_path, monkeypatch, *, pipeline: str = "track-b"):
    monkeypatch.setattr("src.reproducibility.paths.RUNS_DIR", tmp_path)
    monkeypatch.setattr("src.reproducibility.run_context.RUNS_DIR", tmp_path)
    run_id = "prior-smoke-run"
    cfg = {
        "pipeline": pipeline,
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "judge_model": "gpt-oss:20b",
        "limit": None,
        "limit_per_task": 5,
        "skip_inference": False,
        "skip_judge": True,
        "resume": False,
        "smoke_stub": True,
        "force_inference": False,
        "run_metrics": False,
        "catalog_path": "data/track_b/track_b_catalog.csv",
        "raw_path": "results/track_b_raw.csv",
        "judgments_path": "results/track_b_judgments.csv",
        "seeds": {"torch": 42, "numpy": 42, "random": 42},
    }
    write_config_snapshot(config_path(run_id), cfg)
    manifest = {
        "run_id": run_id,
        "pipeline": pipeline,
        "command": "track-b run",
        "argv": ["main.py", "track-b", "run", "--smoke-stub"],
        "seeds": cfg["seeds"],
        "config": cfg,
    }
    write_manifest(manifest_path(run_id), manifest)
    return run_id


def test_reproduce_dry_run(tmp_path, monkeypatch, capsys):
    run_id = _write_prior_run(tmp_path, monkeypatch)
    cmd_reproduce(run_id=run_id, dry_run=True)
    out = capsys.readouterr().out
    assert "track-b" in out
    assert "prior-smoke-run" in out or "smoke" in out.lower()


def test_reproduce_invokes_track_b_pipeline(tmp_path, monkeypatch):
    run_id = _write_prior_run(tmp_path, monkeypatch)
    with patch("src.run_track_b.cmd_track_b_pipeline") as mock_pipeline:
        with patch("src.reproducibility.reproduce.begin_run") as mock_begin:
            with patch("src.reproducibility.reproduce.finish_run"):
                mock_begin.return_value.run_id = "repro-run"
                cmd_reproduce(run_id=run_id, dry_run=False, new_run_id="repro-run")
    mock_pipeline.assert_called_once()
    kwargs = mock_pipeline.call_args.kwargs
    assert kwargs["smoke_stub"] is True
    assert kwargs["limit_per_task"] == 5


def test_cmd_reproduce_handler_exists():
    from main import cmd_reproduce

    assert callable(cmd_reproduce)
