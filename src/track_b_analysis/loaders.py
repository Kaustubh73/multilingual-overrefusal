"""Robust loaders for Track B result files.

Handles the real-world messiness of ``results/track_b_raw.csv`` and
``results/track_b_judgments.csv``:

- Judgments files accumulate duplicate rows across reruns; some rows store the
  per-language key in ``prompt_id`` and others (older rows) leave ``prompt_id``
  empty and put it in ``sample_id``. We normalize to a single ``row_key`` and
  de-duplicate (keeping the last occurrence).
- ``is_benign_task`` is stored as a string; we coerce to bool.
- We derive boolean helper columns used by :mod:`track_b_analysis.metrics`.

The merged frame is the canonical input to every metric/plot function.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REFUSAL_RI = ("direct_refusal", "indirect_refusal")
VALID_RI = ("direct_answer", "direct_refusal", "indirect_refusal")
_TRUE_STRINGS = ("true", "1", "yes", "t", "y")


def _coerce_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(_TRUE_STRINGS)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a Track B raw responses CSV as strings (no NaN coercion)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_judgments(path: str | Path) -> pd.DataFrame:
    """Load + normalize a Track B judgments CSV.

    Adds a ``row_key`` column (the per-language identifier) and de-duplicates,
    keeping the last judgment for each key.
    """
    judged = pd.read_csv(path, dtype=str, keep_default_na=False)

    has_prompt_id = "prompt_id" in judged.columns
    sample_id = judged["sample_id"] if "sample_id" in judged.columns else pd.Series([""] * len(judged))

    def _key(idx: int) -> str:
        pid = str(judged["prompt_id"].iloc[idx]).strip() if has_prompt_id else ""
        if pid:
            return pid
        # Older rows stored the per-language key in sample_id (e.g. test_0008__english).
        return str(sample_id.iloc[idx]).strip()

    judged = judged.copy()
    judged["row_key"] = [_key(i) for i in range(len(judged))]
    judged = judged[judged["row_key"] != ""]
    judged = judged.drop_duplicates(subset="row_key", keep="last")
    return judged


def add_derived_columns(merged: pd.DataFrame) -> pd.DataFrame:
    """Attach boolean helper columns used across metrics/plots."""
    merged = merged.copy()
    merged["is_benign_task"] = _coerce_bool(merged.get("is_benign_task", pd.Series(["false"] * len(merged))))
    ri = merged.get("ri", pd.Series([""] * len(merged))).astype(str)
    merged["ri"] = ri
    merged["ri_valid"] = ri.isin(VALID_RI)
    merged["refusal"] = ri.isin(REFUSAL_RI)
    merged["over_refusal"] = ri.isin(REFUSAL_RI)
    if "text_type" in merged.columns:
        merged["is_harmful_content"] = merged["text_type"].astype(str).str.contains("harmful", case=False)
    else:
        merged["is_harmful_content"] = False
    return merged


def load_merged(raw_path: str | Path, judgments_path: str | Path) -> pd.DataFrame:
    """Merge raw responses with judgments on the per-language key.

    Returns one row per ``prompt_id`` (i.e. per sample_id × language) with the
    judge ``ri`` attached and all derived boolean columns.
    """
    raw = load_raw(raw_path)
    judged = load_judgments(judgments_path)

    keep = [c for c in ("row_key", "ri", "judge_model", "judge_raw") if c in judged.columns]
    merged = raw.merge(judged[keep], left_on="prompt_id", right_on="row_key", how="left")
    if "row_key" in merged.columns:
        merged = merged.drop(columns="row_key")
    # Rows with no matching judgment get an empty ri (treated as invalid).
    if "ri" not in merged.columns:
        merged["ri"] = ""
    merged["ri"] = merged["ri"].fillna("")
    return add_derived_columns(merged)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to an already-merged/synthetic frame.

    Useful when a caller hands in a frame that already has ``language``,
    ``task_label``, ``ri``, ``is_benign_task``, ``text_type`` columns.
    """
    return add_derived_columns(df)
