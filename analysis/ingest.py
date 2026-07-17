"""Ingest and normalize Track A + Track B result files into one schema."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from analysis.config import (
    NORMALIZED_COLUMNS,
    TRACK_A_JUDGMENTS,
    TRACK_A_PHASE1_CLASSIFIED,
    TRACK_A_RAW,
    TRACK_B_JUDGMENTS,
    TRACK_B_RAW,
)
from src.track_b_analysis import loaders as tb_loaders

logger = logging.getLogger(__name__)

_TRUE_STRINGS = ("true", "1", "yes", "t", "y")


def _coerce_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(_TRUE_STRINGS)


def _empty_normalized() -> pd.DataFrame:
    return pd.DataFrame(columns=list(NORMALIZED_COLUMNS))


def _finalize(df: pd.DataFrame, track: str) -> pd.DataFrame:
    """Attach standard columns and derived booleans."""
    if df.empty:
        return _empty_normalized()

    out = df.copy()
    out["track"] = track
    if "row_id" not in out.columns:
        out["row_id"] = out.get("prompt_id", out.get("sample_id", "")).astype(str)
    out["sample_id"] = out.get("sample_id", out["row_id"]).astype(str)
    out["language"] = out.get("language", "english").astype(str).str.strip().str.lower()
    out["model"] = out.get("model", "unknown").astype(str)
    out["task_label"] = out.get("task_label", out.get("task", "unknown")).astype(str).fillna("unknown")
    out["text_type"] = out.get("text_type", "unknown").astype(str).fillna("unknown")
    out["category"] = out["task_label"]
    out["prompt_type"] = out["text_type"]
    out["is_benign_task"] = _coerce_bool(out.get("is_benign_task", pd.Series(["true"] * len(out))))

    ri = out.get("ri", pd.Series([""] * len(out))).astype(str).str.strip()
    out["ri"] = ri
    out["ri_valid"] = ri.isin(tb_loaders.VALID_RI)
    out["refusal"] = ri.isin(tb_loaders.REFUSAL_RI)
    out["over_refusal"] = out["refusal"] & out["is_benign_task"]
    if "judge_source" not in out.columns:
        out["judge_source"] = "or_bench"

    return out[list(NORMALIZED_COLUMNS)]


def _apply_limit(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Keep rows for the first ``limit`` unique ``sample_id`` values (catalog order)."""
    if limit is None or df.empty:
        return df
    ordered_ids = list(dict.fromkeys(df["sample_id"].astype(str)))
    keep = set(ordered_ids[:limit])
    return df[df["sample_id"].astype(str).isin(keep)].copy()


def load_track_a_replicate(
    raw_path: Path = TRACK_A_RAW,
    judgments_path: Path = TRACK_A_JUDGMENTS,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load paper-replication outputs (English-only, OR-Bench judge)."""
    if not raw_path.exists() or not judgments_path.exists():
        logger.warning("Track A replicate inputs missing: %s / %s", raw_path, judgments_path)
        return _empty_normalized()

    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    judged = pd.read_csv(judgments_path, dtype=str, keep_default_na=False)
    merged = raw.merge(
        judged[["sample_id", "ri", "judge_model"]],
        on="sample_id",
        how="left",
        suffixes=("", "_judge"),
    )
    merged["language"] = "english"
    merged["row_id"] = merged["sample_id"]
    merged = _apply_limit(merged, limit)
    return _finalize(merged, "track_a_replicate")


def load_track_a_phase1(
    classified_path: Path = TRACK_A_PHASE1_CLASSIFIED,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load Phase-1 multilingual outputs (rule-based classifier labels)."""
    if not classified_path.exists():
        logger.warning("Track A Phase-1 classified outputs missing: %s", classified_path)
        return _empty_normalized()

    df = pd.read_csv(classified_path, dtype=str, keep_default_na=False)
    df["sample_id"] = df["prompt_id"]
    df["row_id"] = df["prompt_id"]
    df["task_label"] = df["task"].fillna("unknown")
    df["text_type"] = "phase1_prompt"
    df["is_benign_task"] = "true"
    df["ri"] = df["label_level1"].map(
        {"REFUSAL": "direct_refusal", "SUCCESS": "direct_answer", "FAILURE": "indirect_refusal"}
    ).fillna("")
    df["judge_source"] = "rule_classifier"
    df = _apply_limit(df, limit)
    return _finalize(df, "track_a_phase1")


def load_track_b(
    raw_path: Path = TRACK_B_RAW,
    judgments_path: Path = TRACK_B_JUDGMENTS,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load Track B multilingual OR outputs."""
    if not raw_path.exists() or not judgments_path.exists():
        logger.warning("Track B inputs missing: %s / %s", raw_path, judgments_path)
        return _empty_normalized()

    merged = tb_loaders.load_merged(raw_path, judgments_path)
    merged["row_id"] = merged["prompt_id"]
    merged = _apply_limit(merged, limit)
    return _finalize(merged, "track_b")


def ingest_all(
    *,
    include_tracks: tuple[str, ...] = ("track_a_replicate", "track_a_phase1", "track_b"),
    limit: int | None = None,
    raw_a: Path = TRACK_A_RAW,
    judgments_a: Path = TRACK_A_JUDGMENTS,
    classified_phase1: Path = TRACK_A_PHASE1_CLASSIFIED,
    raw_b: Path = TRACK_B_RAW,
    judgments_b: Path = TRACK_B_JUDGMENTS,
) -> pd.DataFrame:
    """Load every available track and concatenate into one normalized frame."""
    frames: list[pd.DataFrame] = []
    if "track_a_replicate" in include_tracks:
        frames.append(load_track_a_replicate(raw_a, judgments_a, limit=limit))
    if "track_a_phase1" in include_tracks:
        frames.append(load_track_a_phase1(classified_phase1, limit=limit))
    if "track_b" in include_tracks:
        frames.append(load_track_b(raw_b, judgments_b, limit=limit))

    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return _empty_normalized()
    return pd.concat(non_empty, ignore_index=True)
