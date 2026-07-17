"""Task templates and multilingual prompt catalog for Phase 1 experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.dataset import DEFAULT_REVIEWED_PATH, filter_reviewed

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "prompts" / "prompt_catalog.csv"

from src.languages import ALL_LANGUAGES, LANGUAGE_COLUMNS, TASK_WRAPPERS

TASKS = ("sentiment", "rephrase", "translation")


def build_prompt_catalog(
    reviewed_df: pd.DataFrame,
    *,
    tasks: tuple[str, ...] | list[str] | None = None,
    prompt_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build prompt catalog: one row per (prompt_id, task, language).

    Columns: prompt_id, task, language, prompt, source_text
    """
    tasks = tuple(tasks) if tasks else TASKS
    for task in tasks:
        if task not in TASK_WRAPPERS:
            raise ValueError(f"Unknown task: {task}. Choose from {TASKS}")

    reviewed = filter_reviewed(reviewed_df)
    if reviewed.empty:
        return pd.DataFrame(
            columns=["prompt_id", "task", "language", "prompt", "source_text"]
        )

    if prompt_ids is not None:
        reviewed = reviewed[reviewed["prompt_id"].astype(str).isin(prompt_ids)].copy()

    rows = []
    for _, row in reviewed.iterrows():
        pid = str(row["prompt_id"])
        for language, col in LANGUAGE_COLUMNS.items():
            text = str(row[col]).strip()
            if not text or text.lower() == "nan":
                logger.warning("Skipping %s/%s: empty translation", pid, language)
                continue
            for task in tasks:
                wrapper = TASK_WRAPPERS[task][language]
                rows.append(
                    {
                        "prompt_id": pid,
                        "task": task,
                        "language": language,
                        "prompt": wrapper.format(text=text),
                        "source_text": text,
                    }
                )
    return pd.DataFrame(rows)


def save_prompt_catalog(
    reviewed_df: pd.DataFrame,
    out_path: Path | None = None,
    *,
    tasks: tuple[str, ...] | list[str] | None = None,
    prompt_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Build and save prompt_catalog.csv."""
    out_path = out_path or DEFAULT_CATALOG_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_prompt_catalog(reviewed_df, tasks=tasks, prompt_ids=prompt_ids)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d catalog rows to %s", len(df), out_path)
    return df


def load_prompt_catalog(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_CATALOG_PATH
    return pd.read_csv(path, dtype=str, keep_default_na=False)
