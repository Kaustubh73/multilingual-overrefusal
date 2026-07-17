"""Multi-model Phase 1 pipeline: catalog → inference → classify → summaries."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.analysis import (
    build_failure_distribution,
    build_summary_rates,
    generate_plots,
    save_summary_tables,
)
from src.dataset import DEFAULT_REVIEWED_PATH, load_reviewed_translations
from src.model_runner import DEFAULT_RAW_OUTPUTS_PATH, run_validation_batch
from src.response_classifier import DEFAULT_CLASSIFIED_PATH, save_classified
from src.tasks import DEFAULT_CATALOG_PATH, TASKS, build_prompt_catalog, save_prompt_catalog

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5:7b-instruct",
    "gemma3:12b",
]


def build_jobs(
    catalog_df: pd.DataFrame,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Expand catalog rows with one row per model."""
    models = models or VALIDATION_MODELS
    rows = []
    for _, row in catalog_df.iterrows():
        for model in models:
            rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "task": row["task"],
                    "language": row["language"],
                    "model": model,
                    "prompt": row["prompt"],
                    "source_text": row.get("source_text", ""),
                }
            )
    return pd.DataFrame(rows)


def run_pipeline(
    *,
    n_prompts: int = 10,
    tasks: list[str] | None = None,
    models: list[str] | None = None,
    reviewed_path: Path | None = None,
    catalog_path: Path | None = None,
    raw_path: Path | None = None,
    classified_path: Path | None = None,
    skip_inference: bool = False,
    no_plots: bool = False,
    resume: bool = True,
) -> None:
    reviewed_path = reviewed_path or DEFAULT_REVIEWED_PATH
    catalog_path = catalog_path or DEFAULT_CATALOG_PATH
    raw_path = raw_path or DEFAULT_RAW_OUTPUTS_PATH
    classified_path = classified_path or DEFAULT_CLASSIFIED_PATH
    tasks = tasks or list(TASKS)

    if not reviewed_path.exists():
        logger.error("Reviewed translations not found: %s", reviewed_path)
        sys.exit(1)

    reviewed_df = load_reviewed_translations(reviewed_path)
    prompt_ids = reviewed_df["prompt_id"].astype(str).unique()[:n_prompts].tolist()

    catalog_df = save_prompt_catalog(
        reviewed_df,
        catalog_path,
        tasks=tasks,
        prompt_ids=prompt_ids,
    )
    if catalog_df.empty:
        logger.error("Empty prompt catalog. Check reviewed translations.")
        sys.exit(1)

    jobs = build_jobs(catalog_df, models=models)
    logger.info(
        "Jobs: %d prompts, %d tasks, %d rows (%d langs × %d tasks × %d models)",
        catalog_df["prompt_id"].nunique(),
        catalog_df["task"].nunique(),
        len(jobs),
        catalog_df["language"].nunique(),
        catalog_df["task"].nunique(),
        len(models or VALIDATION_MODELS),
    )

    if skip_inference:
        if not raw_path.exists():
            logger.error("--skip-inference set but %s missing", raw_path)
            sys.exit(1)
        raw_df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    else:
        raw_df = run_validation_batch(jobs, out_path=raw_path, resume=resume)

    if "task" not in raw_df.columns:
        logger.warning(
            "raw_outputs.csv has no task column; treating all rows as sentiment (legacy)"
        )
        raw_df["task"] = "sentiment"

    merge_cols = ["prompt_id", "task", "language", "model", "source_text"]
    if "source_text" not in raw_df.columns:
        raw_df = raw_df.merge(
            jobs[merge_cols],
            on=["prompt_id", "task", "language", "model"],
            how="left",
        )
    elif raw_df["source_text"].eq("").any() or raw_df["source_text"].isna().any():
        raw_df = raw_df.drop(columns=["source_text"], errors="ignore")
        raw_df = raw_df.merge(
            jobs[merge_cols],
            on=["prompt_id", "task", "language", "model"],
            how="left",
        )

    classified_df = save_classified(raw_df, out_path=classified_path)
    rates_df, failure_df = save_summary_tables(classified_df)

    print("\n=== Summary rates (language × model) ===")
    print(rates_df.to_string(index=False))
    print("\n=== Failure distribution (among failures) ===")
    print(failure_df.to_string(index=False))

    if not no_plots:
        generate_plots(classified_df, summary_df=rates_df)

    print(f"\nCatalog: {catalog_path}")
    print(f"Raw outputs: {raw_path}")
    print(f"Classified: {classified_path}")
    print(f"Summary rates: {PROJECT_ROOT / 'results' / 'summary_rates.csv'}")
    print(f"Failure distribution: {PROJECT_ROOT / 'results' / 'failure_distribution.csv'}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Phase 1 multilingual task pipeline (multi-model)",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=10,
        help="Number of reviewed prompt_ids to use (default: 10)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=list(TASKS),
        help=f"Tasks to run (default: all {list(TASKS)})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override model list (default: Llama, qwen2.5, gemma3)",
    )
    parser.add_argument("--reviewed", type=str, default=None)
    parser.add_argument("--catalog", type=str, default=None)
    parser.add_argument("--raw", type=str, default=None)
    parser.add_argument("--classified", type=str, default=None)
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only classify/summarize/plot existing raw_outputs.csv",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip completed jobs in raw_outputs.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use 1 prompt only (quick smoke test)",
    )
    args = parser.parse_args()

    n_prompts = 1 if args.dry_run else args.n_prompts
    run_pipeline(
        n_prompts=n_prompts,
        tasks=args.tasks,
        models=args.models,
        reviewed_path=Path(args.reviewed) if args.reviewed else None,
        catalog_path=Path(args.catalog) if args.catalog else None,
        raw_path=Path(args.raw) if args.raw else None,
        classified_path=Path(args.classified) if args.classified else None,
        skip_inference=args.skip_inference,
        no_plots=args.no_plots,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
