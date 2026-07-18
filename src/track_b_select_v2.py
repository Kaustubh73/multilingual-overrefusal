"""Stratified sample selection for Track B (v2)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.paper_benchmark import load_paper_benchmark

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "track_b" / "selected_rows_v2.csv"


def _largest_remainder(counts: dict[str, int], n_total: int) -> dict[str, int]:
    """Largest-remainder method for fair integer rounding."""
    total = sum(counts.values())
    raw = {k: v / total * n_total for k, v in counts.items()}
    floored = {k: int(v) for k, v in raw.items()}
    remainders = {k: raw[k] - floored[k] for k in raw}
    deficit = n_total - sum(floored.values())
    for k in sorted(remainders, key=remainders.get, reverse=True):
        if deficit <= 0:
            break
        floored[k] += 1
        deficit -= 1
    return floored


def stratified_select(
    *,
    n_translate: int = 15,
    n_sentiment: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified sample from Sakonii test split, matching text_type proportions.

    Returns DataFrame with columns: sample_id, intended_task, text_type, ...
    """
    test_df = load_paper_benchmark("test")
    rows: list[dict] = []

    for task, n in [("translate", n_translate), ("sentiment_analysis", n_sentiment)]:
        task_df = test_df[test_df["intended_task"] == task]
        tt_counts = task_df["text_type"].value_counts().to_dict()
        target_counts = _largest_remainder(tt_counts, n)

        for tt, target_n in target_counts.items():
            candidates = task_df[task_df["text_type"] == tt]
            if len(candidates) < target_n:
                logger.warning(
                    "Task %s text_type %s: need %d, have %d — taking all",
                    task,
                    tt,
                    target_n,
                    len(candidates),
                )
                selected = candidates
            else:
                selected = candidates.sample(n=target_n, random_state=seed)
            rows.append(selected)

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["intended_task", "sample_id"]).reset_index(drop=True)

    # Log summary
    summary = result.groupby(["intended_task", "text_type"]).size().unstack(fill_value=0)
    logger.info("Stratified selection:\n%s", summary)
    return result


def write_selected_rows_v2(
    out_path: Path | None = None,
    **kwargs,
) -> Path:
    out_path = out_path or DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = stratified_select(**kwargs)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-translate", type=int, default=15)
    parser.add_argument("--n-sentiment", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else None
    write_selected_rows_v2(
        out_path=out,
        n_translate=args.n_translate,
        n_sentiment=args.n_sentiment,
        seed=args.seed,
    )
