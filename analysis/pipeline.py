"""End-to-end analysis orchestration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from analysis import figures, ingest, latex, metrics, report, stats
from analysis.config import DEFAULT_OUT_DIR, REPORT_PATH, SMOKE_DISCLAIMER

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """All tables, paths, and metadata from one pipeline run."""

    merged: pd.DataFrame
    by_language: pd.DataFrame
    by_model: pd.DataFrame
    by_category: pd.DataFrame
    by_prompt_type: pd.DataFrame
    full_breakdown: pd.DataFrame
    language_task: pd.DataFrame
    harmful: pd.DataFrame
    chi_square: pd.DataFrame
    language_effects: pd.DataFrame
    paths: dict[str, Path] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        n = len(self.merged)
        label = self.meta.get("label", "unlabelled")
        smoke = " [SMOKE — not publishable]" if self.meta.get("is_smoke") else ""
        return f"{label}{smoke}: rows={n}, tracks={sorted(self.merged['track'].unique().tolist())}"


def run_pipeline(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    include_tracks: tuple[str, ...] = ("track_a_replicate", "track_a_phase1", "track_b"),
    limit: int | None = None,
    confidence: float = 0.95,
    label: str | None = None,
    write_report: bool = True,
    write_figures: bool = True,
    report_path: Path = REPORT_PATH,
) -> PipelineResult:
    """Ingest → normalize → metrics → stats → figures → LaTeX → report."""
    is_smoke = limit is not None
    if label is None:
        label = f"SMOKE n={limit}" if is_smoke else "full run"

    merged = ingest.ingest_all(include_tracks=include_tracks, limit=limit)
    if merged.empty:
        logger.warning("No data ingested — check results/ artifacts.")

    by_language = metrics.refusal_rates_by_language(merged, confidence=confidence)
    by_model = metrics.refusal_rates_by_model(merged, confidence=confidence)
    by_category = metrics.refusal_rates_by_category(merged, confidence=confidence)
    by_prompt_type = metrics.refusal_rates_by_prompt_type(merged, confidence=confidence)
    full_breakdown = metrics.refusal_rates_full_breakdown(merged, confidence=confidence)
    language_task = metrics.language_task_rates(merged, confidence=confidence)
    harmful = metrics.harmful_refusal_rates(merged, confidence=confidence)
    chi_square = stats.run_all_statistical_tests(merged)
    language_effects = stats.pairwise_language_effects(merged, confidence=confidence)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # CSV tables
    for name, table in (
        ("normalized", merged),
        ("rates_by_language", by_language),
        ("rates_by_model", by_model),
        ("rates_by_category", by_category),
        ("rates_by_prompt_type", by_prompt_type),
        ("rates_full_breakdown", full_breakdown),
        ("rates_language_task", language_task),
        ("rates_harmful", harmful),
        ("chi_square", chi_square),
        ("language_effects", language_effects),
    ):
        p = out_dir / f"{name}.csv"
        table.to_csv(p, index=False)
        paths[name] = p

    meta = {
        "label": label,
        "is_smoke": is_smoke,
        "limit": limit,
        "n_rows": len(merged),
        "confidence": confidence,
        "include_tracks": list(include_tracks),
        "smoke_disclaimer": SMOKE_DISCLAIMER if is_smoke else None,
    }
    meta_path = out_dir / "analysis_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["meta"] = meta_path

    smoke_caption = label if is_smoke else None
    smoke_note = "Smoke test; not for publication." if is_smoke else ""
    if write_figures:
        fig_paths = figures.save_all_figures(
            by_language=by_language,
            by_model=by_model,
            by_category=by_category,
            by_prompt_type=by_prompt_type,
            language_task=language_task,
            language_effects=language_effects,
            harmful=harmful,
            out_dir=out_dir,
            smoke_label=smoke_caption,
        )
        paths.update(fig_paths)

    tex_path = latex.write_latex_tables(
        by_language=by_language,
        by_model=by_model,
        by_category=by_category,
        by_prompt_type=by_prompt_type,
        full_breakdown=full_breakdown,
        out_dir=out_dir,
        smoke_note=smoke_note,
    )
    paths["latex"] = tex_path

    if write_report:
        report_path = report.write_analysis_report(
            merged=merged,
            by_language=by_language,
            by_model=by_model,
            by_category=by_category,
            by_prompt_type=by_prompt_type,
            chi_square=chi_square,
            language_effects=language_effects,
            out_paths=paths,
            label=label,
            is_smoke=is_smoke,
            confidence=confidence,
            report_path=report_path,
        )
        paths["report"] = report_path

    return PipelineResult(
        merged=merged,
        by_language=by_language,
        by_model=by_model,
        by_category=by_category,
        by_prompt_type=by_prompt_type,
        full_breakdown=full_breakdown,
        language_task=language_task,
        harmful=harmful,
        chi_square=chi_square,
        language_effects=language_effects,
        paths=paths,
        meta=meta,
    )
