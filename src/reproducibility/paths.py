"""Canonical paths for reproducibility artifacts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"

MANIFEST_FILENAME = "manifest.json"
CONFIG_FILENAME = "config.yaml"


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def manifest_path(run_id: str) -> Path:
    return run_dir(run_id) / MANIFEST_FILENAME


def config_path(run_id: str) -> Path:
    return run_dir(run_id) / CONFIG_FILENAME
