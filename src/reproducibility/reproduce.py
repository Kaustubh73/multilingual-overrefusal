"""Re-run a prior experiment from manifest + config snapshot."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from src.reproducibility.config_snapshot import load_config_snapshot
from src.reproducibility.manifest import apply_seeds, load_manifest
from src.reproducibility.paths import config_path
from src.reproducibility.run_context import begin_run, finish_run

logger = logging.getLogger(__name__)


def _rel(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    from src.reproducibility.paths import PROJECT_ROOT

    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _track_b_from_config(
    cfg: dict[str, Any],
    *,
    resume: bool,
    run_id: str | None,
    argv: list[str],
) -> None:
    from src.run_track_b import cmd_track_b_pipeline

    catalog_path = _rel(cfg.get("catalog_path"))
    raw_path = _rel(cfg.get("raw_path"))
    judgments_path = _rel(cfg.get("judgments_path"))
    report_path = _rel(cfg.get("report_path"))
    metrics_path = _rel(cfg.get("metrics_path"))

    output_paths = {
        "catalog": str(catalog_path) if catalog_path else "",
        "raw": str(raw_path) if raw_path else "",
            "judgments": str(judgments_path) if judgments_path else "",
            "report": str(report_path) if report_path else "",
            "metrics": str(metrics_path) if metrics_path else "",
    }
    ctx = begin_run(
        pipeline="track-b",
        command="track-b run (reproduce)",
        argv=argv,
        config=cfg,
        run_id=run_id,
        output_paths=output_paths,
        seeds=cfg.get("seeds"),
    )
    try:
        cmd_track_b_pipeline(
            catalog_path=catalog_path,
            raw_path=raw_path,
            judgments_path=judgments_path,
            report_path=report_path,
            metrics_path=metrics_path,
            model_id=cfg.get("model_id", "meta-llama/Llama-3.1-8B-Instruct"),
            skip_inference=bool(cfg.get("skip_inference", False)),
            skip_judge=bool(cfg.get("skip_judge", False)),
            resume=resume,
            limit=cfg.get("limit"),
            limit_per_task=cfg.get("limit_per_task"),
            judge_model=cfg.get("judge_model", "gpt-oss:20b"),
            run_metrics=bool(cfg.get("run_metrics", True)),
            smoke_stub=bool(cfg.get("smoke_stub", False)),
            force_inference=bool(cfg.get("force_inference", False)),
        )
        finish_run(ctx, status="completed", output_paths=output_paths)
    except Exception as exc:
        finish_run(ctx, status="failed", error=str(exc), output_paths=output_paths)
        raise


def _replicate_from_config(
    cfg: dict[str, Any],
    *,
    resume: bool,
    run_id: str | None,
    argv: list[str],
) -> None:
    from src.run_replication import cmd_replicate_pipeline

    raw_path = _rel(cfg.get("raw_path"))
    judgments_path = _rel(cfg.get("judgments_path"))
    output_paths = {
        "raw": str(raw_path) if raw_path else "",
        "judgments": str(judgments_path) if judgments_path else "",
    }
    ctx = begin_run(
        pipeline="replicate",
        command="replicate (reproduce)",
        argv=argv,
        config=cfg,
        run_id=run_id,
        output_paths=output_paths,
        seeds=cfg.get("seeds"),
    )
    try:
        cmd_replicate_pipeline(
            split=cfg.get("split", "test"),
            model_id=cfg.get("model_id", "meta-llama/Llama-3.1-8B-Instruct"),
            raw_path=raw_path,
            judgments_path=judgments_path,
            skip_inference=bool(cfg.get("skip_inference", False)),
            skip_judge=bool(cfg.get("skip_judge", False)),
            skip_calibration=bool(cfg.get("skip_calibration", False)),
            resume=resume,
            limit=cfg.get("limit"),
            judge_model=cfg.get("judge_model", "gpt-oss:20b"),
            run_metrics=bool(cfg.get("run_metrics", True)),
        )
        finish_run(ctx, status="completed", output_paths=output_paths)
    except Exception as exc:
        finish_run(ctx, status="failed", error=str(exc), output_paths=output_paths)
        raise


def _project_rel(path: Path) -> str:
    from src.reproducibility.paths import PROJECT_ROOT

    return str(path) if not path.is_absolute() else str(path.relative_to(PROJECT_ROOT))


def build_track_b_config(args: Any, *, seeds: dict[str, int] | None = None) -> dict[str, Any]:
    """Serialize track-b CLI args into a reproducible config dict."""
    from src.run_track_b import DEFAULT_CATALOG_PATH, DEFAULT_JUDGMENTS_PATH, DEFAULT_RAW_PATH

    return {
        "pipeline": "track-b",
        "model_id": args.model,
        "judge_model": args.judge_model,
        "limit": args.limit,
        "limit_per_task": args.limit_per_task,
        "skip_inference": args.skip_inference,
        "skip_judge": args.skip_judge,
        "resume": not args.no_resume,
        "smoke_stub": args.smoke_stub,
        "force_inference": args.force_inference,
        "run_metrics": not args.no_report,
        "force_full": getattr(args, "force_full", False),
        "catalog_path": _project_rel(Path(getattr(args, "catalog", None) or DEFAULT_CATALOG_PATH)),
        "raw_path": _project_rel(Path(getattr(args, "raw", None) or DEFAULT_RAW_PATH)),
        "judgments_path": _project_rel(Path(getattr(args, "judgments", None) or DEFAULT_JUDGMENTS_PATH)),
        "report_path": _project_rel(Path(getattr(args, "report", None) or "results/track_b_report.md")),
        "metrics_path": _project_rel(Path(getattr(args, "metrics", None) or "results/track_b_metrics.csv")),
        "seeds": seeds,
    }


def build_replicate_config(args: Any, *, seeds: dict[str, int] | None = None) -> dict[str, Any]:
    from src.run_replication import DEFAULT_RAW_PATH, PROJECT_ROOT

    judgments_default = PROJECT_ROOT / "results" / "paper_judgments.csv"
    return {
        "pipeline": "replicate",
        "split": args.split,
        "model_id": args.model,
        "judge_model": args.judge_model,
        "limit": args.limit,
        "skip_inference": args.skip_inference,
        "skip_judge": args.skip_judge,
        "skip_calibration": args.skip_calibration,
        "resume": not args.no_resume,
        "run_metrics": not args.no_report,
        "raw_path": _project_rel(DEFAULT_RAW_PATH),
        "judgments_path": _project_rel(judgments_default),
        "seeds": seeds,
    }


def cmd_reproduce(
    *,
    run_id: str,
    resume: bool = False,
    dry_run: bool = False,
    new_run_id: str | None = None,
) -> None:
    """Load a prior run's manifest/config and re-execute the pipeline."""
    manifest = load_manifest(run_id)
    cfg_file = config_path(run_id)
    cfg = load_config_snapshot(cfg_file)

    pipeline = manifest.get("pipeline") or cfg.get("pipeline")
    if not pipeline:
        logger.error("Manifest %s has no pipeline field", run_id)
        sys.exit(1)

    seeds = manifest.get("seeds") or cfg.get("seeds")
    apply_seeds(seeds)

    argv = ["main.py", "reproduce", "--run-id", run_id]
    if resume:
        argv.append("--resume")
    if new_run_id:
        argv.extend(["--new-run-id", new_run_id])

    logger.info(
        "Reproducing run %s (pipeline=%s, resume=%s)",
        run_id,
        pipeline,
        resume,
    )
    if dry_run:
        print(f"pipeline: {pipeline}")
        print(f"config: {cfg_file}")
        print(f"seeds: {seeds}")
        print(f"would resume: {resume}")
        print(f"new run_id: {new_run_id or '(auto)'}")
        return

    # Reproduce defaults to a fresh rerun unless --resume is passed.
    effective_resume = resume

    if pipeline == "track-b":
        _track_b_from_config(cfg, resume=effective_resume, run_id=new_run_id, argv=argv)
    elif pipeline == "replicate":
        _replicate_from_config(cfg, resume=effective_resume, run_id=new_run_id, argv=argv)
    else:
        logger.error("Unsupported pipeline for reproduce: %s", pipeline)
        sys.exit(1)
