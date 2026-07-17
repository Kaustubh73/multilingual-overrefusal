"""Default paths and constants for the end-to-end analysis pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUT_DIR = RESULTS_DIR / "analysis"
REPORT_PATH = PROJECT_ROOT / "ANALYSIS_REPORT.md"

# Track A (paper replication)
TRACK_A_RAW = RESULTS_DIR / "paper_raw.csv"
TRACK_A_JUDGMENTS = RESULTS_DIR / "paper_judgments.csv"

# Track A (Phase-1 multilingual task stability)
TRACK_A_PHASE1_CLASSIFIED = RESULTS_DIR / "classified_outputs.csv"
TRACK_A_PHASE1_RAW = RESULTS_DIR / "raw_outputs.csv"

# Track B (multilingual OR extension)
TRACK_B_RAW = RESULTS_DIR / "track_b_raw.csv"
TRACK_B_JUDGMENTS = RESULTS_DIR / "track_b_judgments.csv"

NORMALIZED_COLUMNS = (
    "track",
    "row_id",
    "sample_id",
    "language",
    "model",
    "task_label",
    "text_type",
    "category",
    "prompt_type",
    "is_benign_task",
    "ri",
    "refusal",
    "over_refusal",
    "ri_valid",
    "judge_source",
)

SMOKE_DISCLAIMER = (
    "**SMOKE TEST (n≤5 per track slice)** — not publishable. "
    "Wilson CIs will be wide; treat all numbers as pipeline validation only."
)
