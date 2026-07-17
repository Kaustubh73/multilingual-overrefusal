"""Refusal-rate tables grouped by language, model, category, and prompt type."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.track_b_analysis import metrics as tb_metrics


def _eligible_over_refusal(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "ri_valid" in work.columns:
        work = work[work["ri_valid"].astype(bool)]
    work["_or_num"] = work["over_refusal"].astype(bool) & work["is_benign_task"].astype(bool)
    return work


def refusal_rates(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Over-refusal rate (benign-task refusals) per group with Wilson CIs."""
    work = _eligible_over_refusal(df)
    return tb_metrics.rate_by_group(
        work,
        success_col="_or_num",
        group_cols=group_cols,
        confidence=confidence,
    )


def refusal_rates_by_language(df: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    return refusal_rates(df, ["language"], confidence=confidence)


def refusal_rates_by_model(df: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    return refusal_rates(df, ["model"], confidence=confidence)


def refusal_rates_by_category(df: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    return refusal_rates(df, ["category"], confidence=confidence)


def refusal_rates_by_prompt_type(df: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    return refusal_rates(df, ["prompt_type"], confidence=confidence)


def refusal_rates_full_breakdown(df: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    """All four dimensions in one table."""
    return refusal_rates(
        df,
        ["track", "language", "model", "category", "prompt_type"],
        confidence=confidence,
    )


def harmful_refusal_rates(df: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    return tb_metrics.harmful_refusal_by_group(df, group_cols=("language", "category"), confidence=confidence)


def language_task_rates(df: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    return tb_metrics.over_refusal_by_group(df, group_cols=("language", "category"), confidence=confidence)
