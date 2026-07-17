"""Select translate + sentiment_analysis rows from Sakonii test split for Track B."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.dataset import extract_quoted_text
from src.paper_benchmark import load_paper_benchmark
from src.replication_failure_audit import TRANSLATE_LANG_RE

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "track_b" / "selected_rows.csv"

TRACK_B_TASKS = frozenset({"translate", "sentiment_analysis"})
PREFERRED_LANGS = frozenset({"hindi", "nepali", "urdu"})
SMOKE_ROWS_PER_TASK = 5


def _extract_translate_target(plain_text: str) -> str:
    match = TRANSLATE_LANG_RE.search(plain_text)
    return match.group(1).strip() if match else ""


def _priority_tier(intended_task: str, translate_target_lang: str) -> str:
    if intended_task != "translate":
        return "standard"
    if translate_target_lang.lower() in PREFERRED_LANGS:
        return "preferred"
    return "standard"


def select_track_b_rows(*, split: str = "test") -> pd.DataFrame:
    """Filter benchmark to translate + sentiment_analysis rows with metadata."""
    bench = load_paper_benchmark(split=split)
    subset = bench[bench["intended_task"].isin(TRACK_B_TASKS)].copy()
    subset["source_text"] = subset["plain_text"].astype(str).map(extract_quoted_text)
    subset["translate_target_lang"] = subset["plain_text"].astype(str).map(_extract_translate_target)
    subset["priority_tier"] = subset.apply(
        lambda r: _priority_tier(str(r["intended_task"]), str(r["translate_target_lang"])),
        axis=1,
    )
    tier_order = {"preferred": 0, "standard": 1}
    subset["_tier_sort"] = subset["priority_tier"].map(tier_order)
    subset = subset.sort_values(["_tier_sort", "sample_id"]).drop(columns=["_tier_sort"])
    subset = subset.reset_index(drop=True)
    return subset[
        [
            "sample_id",
            "plain_text",
            "text_type",
            "intended_task",
            "task_label",
            "is_benign_task",
            "source_text",
            "translate_target_lang",
            "priority_tier",
        ]
    ]


def select_smoke_sample_ids(
    selected_df: pd.DataFrame,
    *,
    per_task: int = SMOKE_ROWS_PER_TASK,
) -> list[str]:
    """Return sample_ids for smoke catalog: per_task translate + per_task sentiment."""
    translate_ids = (
        selected_df.loc[selected_df["intended_task"] == "translate", "sample_id"]
        .head(per_task)
        .astype(str)
        .tolist()
    )
    sentiment_ids = (
        selected_df.loc[selected_df["intended_task"] == "sentiment_analysis", "sample_id"]
        .head(per_task)
        .astype(str)
        .tolist()
    )
    return translate_ids + sentiment_ids


def write_selected_rows(
    *,
    split: str = "test",
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Select rows and write data/track_b/selected_rows.csv."""
    out_path = out_path or DEFAULT_OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = select_track_b_rows(split=split)
    df.to_csv(out_path, index=False)
    n_translate = int((df["intended_task"] == "translate").sum())
    n_sentiment = int((df["intended_task"] == "sentiment_analysis").sum())
    n_preferred = int((df["priority_tier"] == "preferred").sum())
    logger.info(
        "Selected %d Track B rows (%d translate, %d sentiment, %d preferred) -> %s",
        len(df),
        n_translate,
        n_sentiment,
        n_preferred,
        out_path,
    )
    return df
