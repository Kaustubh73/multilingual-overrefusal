#!/usr/bin/env python3
"""CLI for Phase 1 multilingual task stability experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.dataset import (
    DEFAULT_REVIEWED_PATH,
    filter_reviewed,
    init_review_template,
    load_reviewed_translations,
    save_base_prompts,
)
from src.paper_benchmark import write_dataset_audit
from src.run_replication import cmd_replicate_pipeline
from src.run_track_b import cmd_track_b_pipeline
from src.run_validation import run_pipeline
from src.track_b_prompts import write_track_b_catalog
from src.track_b_select import write_selected_rows
from src.track_b_translate import prepare_track_b_translations
from src.translate import export_translation_requests

PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_prepare(args: argparse.Namespace) -> None:
    """Build base prompts, translation exports, and review template."""
    save_base_prompts()
    export_translation_requests()
    init_review_template(overwrite=args.overwrite_review)
    logger.info(
        "Prepare complete. Fill data/translations/reviewed_translations.csv before running."
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Build catalog, run inference, classify, and write summary tables."""
    reviewed_path = Path(args.reviewed) if args.reviewed else DEFAULT_REVIEWED_PATH
    if not reviewed_path.exists():
        logger.error(
            "Reviewed translations not found at %s. Run: python main.py prepare",
            reviewed_path,
        )
        sys.exit(1)

    reviewed_df = load_reviewed_translations(reviewed_path)
    if filter_reviewed(reviewed_df).empty:
        logger.error(
            "No reviewed rows (reviewed=true). Fill %s first.",
            reviewed_path,
        )
        sys.exit(1)

    from src.translation_qa import summarize_qa, write_qa_report

    qa_report, _ = write_qa_report(reviewed_df, id_col="prompt_id")
    qa_summary = summarize_qa(qa_report)
    logger.info("Translation QA before run: %s", qa_summary)

    n_prompts = 1 if args.dry_run else args.n_prompts
    run_pipeline(
        n_prompts=n_prompts,
        tasks=args.tasks,
        models=args.models,
        reviewed_path=reviewed_path,
        catalog_path=Path(args.catalog) if args.catalog else None,
        raw_path=Path(args.raw) if args.raw else None,
        classified_path=Path(args.classified) if args.classified else None,
        skip_inference=args.skip_inference,
        no_plots=args.no_plots,
        resume=not args.no_resume,
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Re-classify and summarize from existing raw outputs (no inference)."""
    args.skip_inference = True
    cmd_run(args)


def cmd_audit_paper(args: argparse.Namespace) -> None:
    """Write official benchmark audit vs legacy Phase-1 CSVs."""
    write_dataset_audit(
        split=args.split,
        out_path=Path(args.out) if args.out else None,
    )


def cmd_translation_qa(args: argparse.Namespace) -> None:
    """Run translation QA checks on reviewed_translations CSV."""
    from src.translation_qa import summarize_qa, write_qa_report

    path = Path(args.input)
    if not path.exists():
        logger.error("Input not found: %s", path)
        sys.exit(1)
    df = load_reviewed_translations(path)
    report, out = write_qa_report(
        df,
        out_path=Path(args.out) if args.out else None,
        id_col=args.id_col,
        require_reviewed=not args.include_unreviewed,
    )
    summary = summarize_qa(report)
    logger.info("QA summary: %s", summary)
    if args.strict and summary.get("by_language"):
        failed = any(v["passed"] < v["n"] for v in summary["by_language"].values())
        if failed:
            logger.error("QA strict mode: some translations failed checks")
            sys.exit(1)


def cmd_agreement(args: argparse.Namespace) -> None:
    """Compute inter-annotator agreement on refusal (ri) labels."""
    from src.refusal_agreement import compute_agreement_report, load_annotations_long

    path = Path(args.input)
    if not path.exists():
        logger.error("Input not found: %s", path)
        sys.exit(1)
    annotations = load_annotations_long(path)
    report = compute_agreement_report(annotations)
    if args.out:
        import json

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote agreement report to %s", out)
    else:
        logger.info("Agreement report: %s", report)


def cmd_baseline_table(args: argparse.Namespace) -> None:
    """Write benchmark comparison template vs published baselines."""
    from src.baseline_comparison import write_baseline_comparison

    write_baseline_comparison(out_path=Path(args.out) if args.out else None)


def cmd_track_b_prepare(args: argparse.Namespace) -> None:
    """Select Track B rows, translation exports, and build catalog."""
    import pandas as pd

    from src.dataset import filter_reviewed, load_reviewed_translations

    from src.track_b_translate import resolve_reviewed_path

    track_b_dir = PROJECT_ROOT / "data" / "track_b"
    selected_path = track_b_dir / "selected_rows.csv"
    reviewed_path = resolve_reviewed_path(track_b_dir / "reviewed_translations.csv")

    write_selected_rows(split=args.split, out_path=selected_path)
    prepare_track_b_translations(
        selected_path=selected_path,
        reviewed_path=reviewed_path,
        assist_translate=args.assist_translate,
        seed_smoke=args.seed_smoke,
        ollama_model=args.ollama_model,
        overwrite_review=args.overwrite_review,
    )

    reviewed_df = load_reviewed_translations(reviewed_path)
    reviewed = filter_reviewed(reviewed_df)
    if reviewed.empty:
        logger.warning(
            "No reviewed rows yet. Set reviewed=true in %s before full run.",
            reviewed_path,
        )
    else:
        from src.translation_qa import summarize_qa, write_qa_report

        qa_report, qa_path = write_qa_report(reviewed_df, id_col="sample_id")
        logger.info("Translation QA report: %s — %s", qa_path, summarize_qa(qa_report))
        selected_df = pd.read_csv(selected_path, dtype=str, keep_default_na=False)
        write_track_b_catalog(selected_df, reviewed_df)
        logger.info("Track B prepare complete (%d reviewed rows).", len(reviewed))


def cmd_track_b_run(args: argparse.Namespace) -> None:
    """Run Track B inference, judge, and report."""
    if (
        args.limit is None
        and args.limit_per_task is None
        and not args.force_full
        and not args.smoke_stub
    ):
        logger.error(
            "Refusing track-b run without --limit or --limit-per-task. "
            "Pass --force-full to run the full catalog (GPU-intensive)."
        )
        sys.exit(1)

    from src.reproducibility.manifest import apply_seeds
    from src.reproducibility.reproduce import build_track_b_config
    from src.reproducibility.run_context import begin_run, finish_run

    seeds = apply_seeds()
    config = build_track_b_config(args, seeds=seeds)
    argv = [Path(sys.argv[0]).name, *sys.argv[1:]]
    output_paths = {
        "catalog": config["catalog_path"],
        "raw": config["raw_path"],
        "judgments": config["judgments_path"],
        "report": config.get("report_path", ""),
        "metrics": config.get("metrics_path", ""),
    }
    ctx = begin_run(
        pipeline="track-b",
        command="track-b run",
        argv=argv,
        config=config,
        run_id=args.run_id,
        output_paths=output_paths,
        seeds=seeds,
    )
    try:
        cmd_track_b_pipeline(
            catalog_path=Path(args.catalog) if args.catalog else None,
            raw_path=Path(args.raw) if args.raw else None,
            judgments_path=Path(args.judgments) if args.judgments else None,
            report_path=Path(args.report) if args.report else None,
            metrics_path=Path(args.metrics) if args.metrics else None,
            model_id=args.model,
            skip_inference=args.skip_inference,
            skip_judge=args.skip_judge,
            resume=not args.no_resume,
            limit=args.limit,
            limit_per_task=args.limit_per_task,
            judge_model=args.judge_model,
            judge_models=getattr(args, "judge_models", None),
            ensemble_strategy=getattr(args, "ensemble_strategy", "majority"),
            run_metrics=not args.no_report,
            smoke_stub=args.smoke_stub,
            force_inference=args.force_inference,
        )
        finish_run(ctx, status="completed", output_paths=output_paths)
        logger.info("Run manifest: runs/%s/manifest.json", ctx.run_id)
    except Exception as exc:
        finish_run(ctx, status="failed", error=str(exc), output_paths=output_paths)
        raise


def cmd_replicate(args: argparse.Namespace) -> None:
    """Replicate SafeConstellations Table 2 on official test split."""
    from src.reproducibility.manifest import apply_seeds
    from src.reproducibility.reproduce import build_replicate_config
    from src.reproducibility.run_context import begin_run, finish_run

    seeds = apply_seeds()
    config = build_replicate_config(args, seeds=seeds)
    if args.raw:
        config["raw_path"] = args.raw
    if args.judgments:
        config["judgments_path"] = args.judgments
    argv = [Path(sys.argv[0]).name, *sys.argv[1:]]
    output_paths = {
        "raw": config["raw_path"],
        "judgments": config["judgments_path"],
    }
    ctx = begin_run(
        pipeline="replicate",
        command="replicate",
        argv=argv,
        config=config,
        run_id=args.run_id,
        output_paths=output_paths,
        seeds=seeds,
    )
    try:
        cmd_replicate_pipeline(
            split=args.split,
            model_id=args.model,
            raw_path=Path(args.raw) if args.raw else None,
            judgments_path=Path(args.judgments) if args.judgments else None,
            skip_inference=args.skip_inference,
            skip_judge=args.skip_judge,
            skip_calibration=args.skip_calibration,
            resume=not args.no_resume,
            limit=args.limit,
            judge_model=args.judge_model,
            run_metrics=not args.no_report,
        )
        finish_run(ctx, status="completed", output_paths=output_paths)
        logger.info("Run manifest: runs/%s/manifest.json", ctx.run_id)
    except Exception as exc:
        finish_run(ctx, status="failed", error=str(exc), output_paths=output_paths)
        raise


def cmd_reproduce(args: argparse.Namespace) -> None:
    """Re-run a prior experiment from its manifest and config snapshot."""
    from src.reproducibility.reproduce import cmd_reproduce as run_reproduce

    run_reproduce(
        run_id=args.run_id,
        resume=args.resume,
        dry_run=args.dry_run,
        new_run_id=args.new_run_id,
    )


def main() -> None:
    from src.tasks import TASKS

    parser = argparse.ArgumentParser(
        description="Phase 1 multilingual task pipeline (sentiment, rephrase, translation)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Build base prompts and translation exports")
    prep.add_argument(
        "--overwrite-review",
        action="store_true",
        help="Overwrite reviewed_translations.csv template",
    )
    prep.set_defaults(func=cmd_prepare)

    for name, help_text, handler in (
        ("run", "Catalog, inference, classification, summaries", cmd_run),
        ("analyze", "Classify/summarize existing raw_outputs.csv only", cmd_analyze),
    ):
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument(
            "--n-prompts",
            type=int,
            default=10,
            help="Number of reviewed prompt_ids (default: 10)",
        )
        p.add_argument(
            "--tasks",
            nargs="+",
            default=None,
            choices=list(TASKS),
            help=f"Tasks to include (default: all {list(TASKS)})",
        )
        p.add_argument(
            "--models",
            nargs="+",
            default=None,
            help="Model ids (default: Llama-3.1-8B, qwen2.5:7b, gemma3:12b via Ollama)",
        )
        p.add_argument("--dry-run", action="store_true", help="Use 1 prompt only")
        p.add_argument("--reviewed", type=str, default=None)
        p.add_argument("--catalog", type=str, default=None)
        p.add_argument("--raw", type=str, default=None)
        p.add_argument("--classified", type=str, default=None)
        p.add_argument(
            "--skip-inference",
            action="store_true",
            help="Skip model calls; use existing raw CSV",
        )
        p.add_argument("--no-plots", action="store_true")
        p.add_argument(
            "--no-resume",
            action="store_true",
            help="Do not skip completed rows in raw_outputs.csv",
        )
        p.set_defaults(func=handler)

    audit = subparsers.add_parser(
        "audit-paper",
        help="Audit official Sakonii benchmark vs legacy base_prompts CSV",
    )
    audit.add_argument("--split", default="test", choices=("test", "train", "with_harmful_response"))
    audit.add_argument("--out", type=str, default=None)
    audit.set_defaults(func=cmd_audit_paper)

    tqa = subparsers.add_parser("translation-qa", help="Validate reviewed translations before eval")
    tqa.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "translations" / "reviewed_translations.csv"),
    )
    tqa.add_argument("--out", type=str, default=None)
    tqa.add_argument("--id-col", type=str, default="prompt_id")
    tqa.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Do not require reviewed=true for QA",
    )
    tqa.add_argument("--strict", action="store_true", help="Exit 1 if any QA check fails")
    tqa.set_defaults(func=cmd_translation_qa)

    agr = subparsers.add_parser("agreement", help="Inter-annotator agreement on ri labels")
    agr.add_argument("--input", type=str, required=True, help="Annotations CSV (long or wide)")
    agr.add_argument("--out", type=str, default=None)
    agr.set_defaults(func=cmd_agreement)

    bl = subparsers.add_parser(
        "baseline-table",
        help="Write published-baseline comparison template (no fake local numbers)",
    )
    bl.add_argument("--out", type=str, default=None)
    bl.set_defaults(func=cmd_baseline_table)

    rep = subparsers.add_parser(
        "replicate",
        help="Paper replication: HF Llama on test split + gpt-oss judge",
    )
    rep.add_argument("--split", default="test")
    rep.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id for inference",
    )
    rep.add_argument("--judge-model", default="gpt-oss:20b", help="Ollama tag for OR-Bench judge")
    rep.add_argument("--limit", type=int, default=None, help="Max rows (smoke test)")
    rep.add_argument("--raw", type=str, default=None)
    rep.add_argument("--judgments", type=str, default=None)
    rep.add_argument("--skip-inference", action="store_true")
    rep.add_argument("--skip-judge", action="store_true")
    rep.add_argument("--skip-calibration", action="store_true")
    rep.add_argument("--no-resume", action="store_true")
    rep.add_argument("--no-report", action="store_true")
    rep.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for manifest (auto-generated if omitted)",
    )
    rep.set_defaults(func=cmd_replicate)

    repro = subparsers.add_parser(
        "reproduce",
        help="Re-run a prior experiment from runs/<run-id>/manifest.json",
    )
    repro.add_argument("--run-id", type=str, required=True, help="Prior run id to reproduce")
    repro.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output CSVs instead of fresh rerun",
    )
    repro.add_argument(
        "--dry-run",
        action="store_true",
        help="Print reproduce plan without executing",
    )
    repro.add_argument(
        "--new-run-id",
        type=str,
        default=None,
        help="Write a new manifest for this reproduction (default: auto)",
    )
    repro.set_defaults(func=cmd_reproduce)

    tb = subparsers.add_parser("track-b", help="Track B multilingual OR pipeline")
    tb_sub = tb.add_subparsers(dest="track_b_command", required=True)

    tb_prep = tb_sub.add_parser("prepare", help="Select rows, translations, build catalog")
    tb_prep.add_argument("--split", default="test")
    tb_prep.add_argument(
        "--seed-smoke",
        action="store_true",
        help="Auto-review 5 translate + 5 sentiment rows for smoke test",
    )
    tb_prep.add_argument(
        "--assist-translate",
        action="store_true",
        help="Fill Hindi/Hinglish via Ollama (reviewed=false unless --seed-smoke)",
    )
    tb_prep.add_argument("--ollama-model", default="gemma3:12b")
    tb_prep.add_argument(
        "--overwrite-review",
        action="store_true",
        help="Overwrite reviewed_translations.csv template",
    )
    tb_prep.set_defaults(func=cmd_track_b_prepare)

    tb_run = tb_sub.add_parser("run", help="Inference + judge + Track B report")
    tb_run.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id for inference",
    )
    tb_run.add_argument("--judge-model", default="gpt-oss:20b")
    tb_run.add_argument(
        "--judge-models",
        nargs="+",
        default=None,
        help="Ensemble: multiple Ollama judge models (enables disagreement logging)",
    )
    tb_run.add_argument(
        "--ensemble-strategy",
        default="majority",
        choices=("majority", "unanimous_or_answer"),
    )
    tb_run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max unique sample_ids (first N in catalog order; use --limit-per-task for balanced smoke)",
    )
    tb_run.add_argument(
        "--limit-per-task",
        type=int,
        default=None,
        help="Smoke: N translate + N sentiment sample_ids (e.g. 5 → up to 30 inference rows)",
    )
    tb_run.add_argument("--skip-inference", action="store_true")
    tb_run.add_argument("--skip-judge", action="store_true")
    tb_run.add_argument("--catalog", help="Catalog CSV (default: data/track_b/track_b_catalog.csv)")
    tb_run.add_argument("--raw", help="Raw-response CSV (default: results/track_b_raw.csv)")
    tb_run.add_argument("--judgments", help="Judge-output CSV (default: results/track_b_judgments.csv)")
    tb_run.add_argument("--report", help="Markdown report path (default: results/track_b_report.md)")
    tb_run.add_argument("--metrics", help="Metrics CSV path (default: results/track_b_metrics.csv)")
    tb_run.add_argument(
        "--smoke-stub",
        action="store_true",
        help="Bootstrap raw CSV: English from Track A paper_raw, stubs for hi/hing (no HF)",
    )
    tb_run.add_argument(
        "--force-inference",
        action="store_true",
        help="Re-run HF inference for all catalog rows (judge still resumes unless --no-resume)",
    )
    tb_run.add_argument("--no-resume", action="store_true")
    tb_run.add_argument("--no-report", action="store_true")
    tb_run.add_argument(
        "--force-full",
        action="store_true",
        help="Allow full-catalog run without --limit (GPU-intensive; human-approved only)",
    )
    tb_run.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for manifest (auto-generated if omitted)",
    )
    tb_run.set_defaults(func=cmd_track_b_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
