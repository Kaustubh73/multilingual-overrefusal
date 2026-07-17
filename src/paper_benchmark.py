"""Official SafeConstellations benchmark loader (Sakonii/task-over-refusal-dataset)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DATASET_ID = "Sakonii/task-over-refusal-dataset"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "data" / "paper" / "dataset_audit.md"

# Paper Eq. 7: benign-intent tasks included in over-refusal metric.
BENIGN_INTENDED_TASKS = frozenset(
    {
        "sentiment_analysis",
        "translate",
        "cryptanalysis",
        "rag_qa",
    }
)

INTENDED_TASK_LABELS = {
    "sentiment_analysis": "sentiment",
    "translate": "translation",
    "cryptanalysis": "cryptanalysis",
    "rag_qa": "rag_qa",
    "rephrase": "rephrase",
}

PAPER_TARGETS = {
    "llama_or_rate": 0.1777,
    "qwen_or_rate": 0.0815,
    "llama_sentiment_or": 0.364,
    "llama_translation_or": 0.467,
}


def _normalize_intended_task(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def load_paper_benchmark(split: str = "test") -> pd.DataFrame:
    """
    Load official benchmark rows; use plain_text as the model prompt (no re-wrapping).

    Columns: sample_id, plain_text, text_type, intended_task, task_label, is_benign_task
    """
    ds = load_dataset(PAPER_DATASET_ID, split=split)
    df = ds.to_pandas()
    required = {"plain_text", "text_type", "intended_task"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected columns in {PAPER_DATASET_ID}/{split}: {missing}")

    df = df.reset_index(drop=True)
    df["sample_id"] = [f"{split}_{i:04d}" for i in range(len(df))]
    df["intended_task"] = df["intended_task"].astype(str).map(_normalize_intended_task)
    df["task_label"] = df["intended_task"].map(
        lambda t: INTENDED_TASK_LABELS.get(t, t)
    )
    df["is_benign_task"] = df["intended_task"].isin(BENIGN_INTENDED_TASKS)
    return df[
        [
            "sample_id",
            "plain_text",
            "text_type",
            "intended_task",
            "task_label",
            "is_benign_task",
        ]
    ]


def summarize_benchmark(df: pd.DataFrame) -> dict:
    """Return count summaries for audit reporting."""
    return {
        "n_rows": len(df),
        "text_type": df["text_type"].value_counts().to_dict(),
        "intended_task": df["intended_task"].value_counts().to_dict(),
        "n_benign_task": int(df["is_benign_task"].sum()),
        "n_rephrase_excluded_from_or": int((~df["is_benign_task"]).sum()),
    }


def diff_legacy_base_prompts(
    paper_df: pd.DataFrame,
    base_path: Path | None = None,
    reviewed_path: Path | None = None,
) -> dict:
    """Compare legacy Phase-1 CSVs to the official benchmark."""
    from src.dataset import DEFAULT_BASE_PATH, DEFAULT_REVIEWED_PATH

    base_path = base_path or DEFAULT_BASE_PATH
    reviewed_path = reviewed_path or DEFAULT_REVIEWED_PATH
    out: dict = {"base_exists": base_path.exists(), "reviewed_exists": reviewed_path.exists()}

    if base_path.exists():
        base = pd.read_csv(base_path, dtype=str)
        out["base_n_rows"] = len(base)
        short = base["base_text"].astype(str).str.len() < 20
        out["base_corrupted_short_text"] = int(short.sum())

    if base_path.exists() and reviewed_path.exists():
        base = pd.read_csv(base_path, dtype=str)
        rev = pd.read_csv(reviewed_path, dtype=str)
        if "english" in rev.columns:
            merged = base.merge(rev[["prompt_id", "english"]], on="prompt_id", how="left")
            out["base_reviewed_mismatches"] = int(
                (merged["base_text"] != merged["english"]).sum()
            )

    out["paper_test_n"] = len(paper_df)
    out["paper_harmful_wrapped_rows"] = int(
        paper_df["text_type"]
        .astype(str)
        .str.contains("harmful|jailbreak|unsafe|decrypt", case=False, regex=True)
        .sum()
    )
    return out


def write_dataset_audit(
    split: str = "test",
    out_path: Path | None = None,
    base_path: Path | None = None,
    reviewed_path: Path | None = None,
) -> Path:
    """Write markdown audit comparing official benchmark vs legacy CSVs."""
    out_path = out_path or DEFAULT_AUDIT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_paper_benchmark(split=split)
    summary = summarize_benchmark(df)
    diff = diff_legacy_base_prompts(df, base_path=base_path, reviewed_path=reviewed_path)

    lines = [
        "# Paper dataset audit",
        "",
        f"Source: `{PAPER_DATASET_ID}` split `{split}`",
        "",
        "## Official benchmark",
        f"- Rows: **{summary['n_rows']}**",
        f"- Benign-intent tasks (OR numerator eligible): **{summary['n_benign_task']}**",
        f"- Rephrase / other (OR indicator always 0): **{summary['n_rephrase_excluded_from_or']}**",
        "",
        "### text_type counts",
        "",
    ]
    for k, v in sorted(summary["text_type"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {v}")

    lines.extend(["", "### intended_task counts", ""])
    for k, v in sorted(summary["intended_task"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {v}")

    lines.extend(
        [
            "",
            "## Legacy Phase-1 CSV comparison",
            "",
            f"- `base_prompts.csv` exists: {diff.get('base_exists')}",
            f"- `reviewed_translations.csv` exists: {diff.get('reviewed_exists')}",
        ]
    )
    if "base_n_rows" in diff:
        lines.append(f"- Legacy base rows: **{diff['base_n_rows']}** (paper test: **{diff['paper_test_n']}**)")
    if "base_corrupted_short_text" in diff:
        lines.append(f"- Legacy corrupted short base_text (<20 chars): **{diff['base_corrupted_short_text']}**")
    if "base_reviewed_mismatches" in diff:
        lines.append(
            f"- Legacy prompt_id english vs base_text mismatches: **{diff['base_reviewed_mismatches']}**"
        )
    lines.append(
        f"- Paper test rows with harmful/jailbreak/unsafe/decrypt text_type: **{diff.get('paper_harmful_wrapped_rows', 'n/a')}**"
    )

    lines.extend(
        [
            "",
            "## Implications",
            "",
            "- Replication must use `plain_text` from the official split, not `build_base_prompts()`.",
            "- OR metric uses all test rows in the denominator (rephrase rows contribute 0 to numerator).",
            "- Legacy 10-prompt pilot is not comparable to Table 2 (17.77%).",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote dataset audit to %s", out_path)
    return out_path
