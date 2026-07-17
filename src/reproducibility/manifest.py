"""Build and persist per-run manifests (git, deps, seeds, hardware)."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.reproducibility.paths import PROJECT_ROOT, manifest_path

logger = logging.getLogger(__name__)

KEY_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "pandas",
    "numpy",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
    "pytest",
    "PyYAML",
)

DEFAULT_SEEDS = {"torch": 42, "numpy": 42, "random": 42}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_git_info() -> dict[str, Any]:
    """Return git commit hash and dirty flag, or nulls when not in a repo."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"commit": None, "dirty": None}


def get_key_dependencies() -> dict[str, str | None]:
    """Installed versions for packages that affect numerical / model behavior."""
    versions: dict[str, str | None] = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def get_pip_freeze() -> str:
    """Full pip freeze output when available (best-effort)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return ""


def get_hardware_info() -> dict[str, Any]:
    """CPU/GPU summary for the machine running the experiment."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_implementation": platform.python_implementation(),
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        else:
            info["cuda_device_count"] = 0
            info["cuda_devices"] = []
    except ImportError:
        info["cuda_available"] = False
        info["mps_available"] = False
        info["cuda_device_count"] = 0
        info["cuda_devices"] = []
    return info


def apply_seeds(seeds: dict[str, int] | None = None) -> dict[str, int]:
    """Set global RNG seeds and return the effective seed map."""
    seeds = dict(seeds or DEFAULT_SEEDS)
    random.seed(seeds.get("random", 42))
    try:
        import numpy as np

        np.random.seed(seeds.get("numpy", 42))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seeds.get("torch", 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds.get("torch", 42))
    except ImportError:
        pass
    return seeds


def build_manifest(
    *,
    run_id: str,
    pipeline: str,
    command: str,
    argv: list[str],
    config: dict[str, Any],
    seeds: dict[str, int] | None = None,
    output_paths: dict[str, str] | None = None,
    status: str = "running",
    started_at: str | None = None,
    finished_at: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Assemble a manifest dict for a run."""
    seeds = apply_seeds(seeds)
    rel_config = f"runs/{run_id}/config.yaml"
    return {
        "run_id": run_id,
        "pipeline": pipeline,
        "command": command,
        "argv": argv,
        "started_at": started_at or _utc_now_iso(),
        "finished_at": finished_at,
        "status": status,
        "error": error,
        "git": get_git_info(),
        "python_version": sys.version,
        "dependencies": get_key_dependencies(),
        "pip_freeze": get_pip_freeze(),
        "seeds": seeds,
        "hardware": get_hardware_info(),
        "config_path": rel_config,
        "config": config,
        "output_paths": output_paths or {},
        "environment": {
            k: os.environ.get(k)
            for k in ("OVERREFUSAL_MODEL", "CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE")
            if os.environ.get(k) is not None
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    logger.info("Wrote run manifest to %s", path)


def load_manifest(run_id: str) -> dict[str, Any]:
    path = manifest_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"No manifest for run_id={run_id!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
