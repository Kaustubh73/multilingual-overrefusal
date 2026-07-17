"""Context manager that snapshots config and manifest at run boundaries."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.reproducibility.config_snapshot import write_config_snapshot
from src.reproducibility.manifest import build_manifest, write_manifest
from src.reproducibility.paths import RUNS_DIR, config_path, manifest_path

logger = logging.getLogger(__name__)


def generate_run_id(pipeline: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = pipeline.replace("_", "-").replace(" ", "-")
    return f"{ts}-{slug}-{uuid.uuid4().hex[:8]}"


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    config: dict[str, Any]
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_file(self) -> Path:
        return manifest_path(self.run_id)

    @property
    def config_file(self) -> Path:
        return config_path(self.run_id)


def begin_run(
    *,
    pipeline: str,
    command: str,
    argv: list[str],
    config: dict[str, Any],
    run_id: str | None = None,
    output_paths: dict[str, str] | None = None,
    seeds: dict[str, int] | None = None,
) -> RunContext:
    """Create run directory, write config + initial manifest."""
    rid = run_id or generate_run_id(pipeline)
    rdir = RUNS_DIR / rid
    rdir.mkdir(parents=True, exist_ok=True)

    cfg_path = config_path(rid)
    write_config_snapshot(cfg_path, config)

    manifest = build_manifest(
        run_id=rid,
        pipeline=pipeline,
        command=command,
        argv=argv,
        config=config,
        seeds=seeds,
        output_paths=output_paths,
        status="running",
    )
    write_manifest(manifest_path(rid), manifest)

    ctx = RunContext(run_id=rid, run_dir=rdir, config=config, manifest=manifest)
    logger.info("Started reproducible run %s (%s)", rid, pipeline)
    return ctx


def finish_run(
    ctx: RunContext,
    *,
    status: str = "completed",
    error: str | None = None,
    output_paths: dict[str, str] | None = None,
) -> None:
    """Update manifest with completion metadata."""
    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ctx.manifest["status"] = status
    ctx.manifest["finished_at"] = finished_at
    ctx.manifest["error"] = error
    if output_paths:
        ctx.manifest["output_paths"].update(output_paths)
    write_manifest(ctx.manifest_file, ctx.manifest)
    logger.info("Finished run %s with status=%s", ctx.run_id, status)
