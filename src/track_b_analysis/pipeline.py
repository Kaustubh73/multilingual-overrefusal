"""One-command Track B analysis pipeline.

Loads results, computes all metric tables, and (optionally) renders the figure
set to a directory. Designed so that when real n>=30 results land they can be
analysed in a single call:

    python -m src.track_b_analysis \
        --raw results/track_b_raw.csv \
        --judgments results/track_b_judgments.csv \
        --out results/analysis

or, on synthetic data for a dry run:

    python -m src.track_b_analysis --synthetic --out results/analysis_synthetic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import loaders, metrics, plots
from .synthetic import generate_synthetic

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class AnalysisResult:
    """Container for every computed table + the merged frame."""

    merged: pd.DataFrame
    over_refusal: pd.DataFrame
    harmful_refusal: pd.DataFrame
    language_deltas: pd.DataFrame
    judge_distribution: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        n = len(self.merged)
        n_valid = int(self.merged["ri_valid"].sum()) if "ri_valid" in self.merged else 0
        langs = ", ".join(sorted(self.merged["language"].unique()))
        return (
            f"rows={n} (valid judgments={n_valid}) | languages: {langs} | "
            f"label='{self.meta.get('label', 'unlabelled')}'"
        )


def analyze(
    merged: pd.DataFrame,
    *,
    confidence: float = 0.95,
    label: str | None = None,
) -> AnalysisResult:
    """Run the full metric suite on an already-merged/normalized frame."""
    merged = loaders.normalize_frame(merged)
    return AnalysisResult(
        merged=merged,
        over_refusal=metrics.over_refusal_by_group(merged, confidence=confidence),
        harmful_refusal=metrics.harmful_refusal_by_group(merged, confidence=confidence),
        language_deltas=metrics.compare_languages(merged, confidence=confidence),
        judge_distribution=metrics.judge_label_distribution(merged),
        meta={"label": label or "unlabelled", "n_rows": len(merged), "confidence": confidence},
    )


def analyze_files(
    raw_path: str | Path,
    judgments_path: str | Path,
    *,
    confidence: float = 0.95,
    label: str | None = None,
) -> AnalysisResult:
    """Load raw + judgments CSVs and run the full analysis."""
    merged = loaders.load_merged(raw_path, judgments_path)
    return analyze(merged, confidence=confidence, label=label)


def analyze_synthetic(*, confidence: float = 0.95, **gen_kwargs) -> AnalysisResult:
    """Generate synthetic data and analyse it (clearly labelled SYNTHETIC)."""
    raw, judged = generate_synthetic(**gen_kwargs)
    merged = raw.merge(
        judged[["prompt_id", "ri", "judge_model"]], on="prompt_id", how="left"
    )
    return analyze(merged, confidence=confidence, label="SYNTHETIC — not real model output")


def save_tables(result: AnalysisResult, out_dir: str | Path) -> dict[str, Path]:
    """Write every metric table to CSV under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, table in (
        ("over_refusal", result.over_refusal),
        ("harmful_refusal", result.harmful_refusal),
        ("language_deltas", result.language_deltas),
        ("judge_distribution", result.judge_distribution),
    ):
        path = out_dir / f"track_b_{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    meta_path = out_dir / "analysis_meta.json"
    meta_path.write_text(json.dumps(result.meta, indent=2), encoding="utf-8")
    written["meta"] = meta_path
    return written


def save_figures(result: AnalysisResult, out_dir: str | Path) -> dict[str, Path]:
    """Render and save the standard figure set as PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    plots.plot_or_bars(result.over_refusal, ax=ax)
    written["or_bars"] = out_dir / "or_bars.png"
    fig.savefig(written["or_bars"], dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    plots.plot_or_heatmap(result.over_refusal, ax=ax)
    written["or_heatmap"] = out_dir / "or_heatmap.png"
    fig.savefig(written["or_heatmap"], dpi=120, bbox_inches="tight")
    plt.close(fig)

    if not result.language_deltas.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plots.plot_delta_or(result.language_deltas, ax=ax)
        written["delta_or"] = out_dir / "delta_or.png"
        fig.savefig(written["delta_or"], dpi=120, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plots.plot_refusal_breakdown(result.over_refusal, result.harmful_refusal, ax=ax)
    written["refusal_breakdown"] = out_dir / "refusal_breakdown.png"
    fig.savefig(written["refusal_breakdown"], dpi=120, bbox_inches="tight")
    plt.close(fig)

    return written


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track B over-refusal analysis pipeline")
    p.add_argument("--raw", type=Path, default=PROJECT_ROOT / "results" / "track_b_raw.csv")
    p.add_argument("--judgments", type=Path, default=PROJECT_ROOT / "results" / "track_b_judgments.csv")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "analysis")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--label", type=str, default=None, help="free-text label, e.g. 'real n=30 run'")
    p.add_argument("--synthetic", action="store_true", help="ignore --raw/--judgments and use synthetic data")
    p.add_argument("--no-figures", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.synthetic:
        result = analyze_synthetic(confidence=args.confidence)
    else:
        if not Path(args.raw).exists() or not Path(args.judgments).exists():
            print(f"Missing input(s): {args.raw} / {args.judgments}")
            print("Tip: run with --synthetic for a dry run, or point --raw/--judgments at real results.")
            return 1
        result = analyze_files(args.raw, args.judgments, confidence=args.confidence, label=args.label)

    print(result.summary())
    print("\nOver-refusal by language × task:")
    print(result.over_refusal.to_string(index=False))
    tables = save_tables(result, args.out)
    if not args.no_figures:
        figs = save_figures(result, args.out)
        tables.update(figs)
    print("\nWrote:")
    for name, path in tables.items():
        print(f"  {name}: {path}")
    return 0
