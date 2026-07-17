"""Track B multilingual over-refusal analysis toolkit.

Dependency-light (numpy + pandas + matplotlib; no scipy) reusable pipeline for
analysing Track B results: load → metrics (rates + Wilson CIs) → language
comparison → figures. See ``README.md`` in this package for usage.
"""

from __future__ import annotations

from . import loaders, metrics, plots, synthetic
from .loaders import load_merged, load_raw, load_judgments, normalize_frame
from .metrics import (
    agreement_rate,
    compare_languages,
    harmful_refusal_by_group,
    judge_label_distribution,
    over_refusal_by_group,
    proportion_stats,
    rate_by_group,
    wilson_ci,
)
from .pipeline import (
    AnalysisResult,
    analyze,
    analyze_files,
    analyze_synthetic,
    save_figures,
    save_tables,
)

__all__ = [
    "loaders",
    "metrics",
    "plots",
    "synthetic",
    "load_merged",
    "load_raw",
    "load_judgments",
    "normalize_frame",
    "wilson_ci",
    "proportion_stats",
    "rate_by_group",
    "over_refusal_by_group",
    "harmful_refusal_by_group",
    "compare_languages",
    "agreement_rate",
    "judge_label_distribution",
    "AnalysisResult",
    "analyze",
    "analyze_files",
    "analyze_synthetic",
    "save_tables",
    "save_figures",
]
