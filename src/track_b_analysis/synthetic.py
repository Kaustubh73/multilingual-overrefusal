"""Synthetic Track B fixture generator.

Produces a deterministic, clearly-labelled synthetic dataset so the analysis
pipeline + notebook run end-to-end even when real results are sparse. The data
is NOT real model output: the model name is ``SYNTHETIC-fixture`` and every row
carries a ``synthetic=True`` marker.

The generator bakes in a plausible (but invented) signal — higher over-refusal
for Hindi/Hinglish on translation, near-correct refusal of harmful content — so
plots and deltas are visibly non-trivial for demos.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

LANGUAGES = ("english", "hindi", "hinglish")
TASKS = ("translation", "sentiment")
SYNTHETIC_MODEL = "SYNTHETIC-fixture"

# Per (language, task) probability of refusing a *benign-content* row.
# Invented to mimic the smoke-run direction (localized > English on translation).
_BENIGN_REFUSAL_P = {
    ("english", "translation"): 0.12,
    ("hindi", "translation"): 0.34,
    ("hinglish", "translation"): 0.22,
    ("english", "sentiment"): 0.05,
    ("hindi", "sentiment"): 0.12,
    ("hinglish", "sentiment"): 0.08,
}
# Refusal probability on harmful-content rows (desired behaviour ~ high).
_HARMFUL_REFUSAL_P = {
    "english": 0.95,
    "hindi": 0.88,
    "hinglish": 0.82,
}


def generate_synthetic(
    n_per_task: int = 24,
    *,
    harmful_fraction: float = 0.15,
    seed: int = 20260623,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic (raw, judgments) frames.

    Args:
        n_per_task: number of distinct sample_ids per task (>=30 sample_ids
            overall when summed across the two tasks; default 24×2 = 48).
        harmful_fraction: fraction of sample_ids whose content is harmful.
        seed: RNG seed for reproducibility.

    Returns:
        (raw_df, judgments_df) matching the real Track B schema, with synthetic
        markers.
    """
    rng = random.Random(seed)
    raw_rows: list[dict] = []
    judge_rows: list[dict] = []

    sid_counter = 0
    for task in TASKS:
        intended = "translate" if task == "translation" else "sentiment_analysis"
        for _ in range(n_per_task):
            sid = f"synth_{sid_counter:04d}"
            sid_counter += 1
            harmful = rng.random() < harmful_fraction
            text_type = "harmful_instruction" if harmful else "benign_instruction"
            for lang in LANGUAGES:
                prompt_id = f"{sid}__{lang}"
                if harmful:
                    refuse = rng.random() < _HARMFUL_REFUSAL_P[lang]
                else:
                    refuse = rng.random() < _BENIGN_REFUSAL_P[(lang, task)]
                if refuse:
                    ri = "direct_refusal" if rng.random() < 0.6 else "indirect_refusal"
                    response = "[SYNTHETIC refusal]"
                else:
                    ri = "direct_answer"
                    response = "[SYNTHETIC answer]"
                raw_rows.append({
                    "prompt_id": prompt_id,
                    "sample_id": sid,
                    "language": lang,
                    "plain_text": f"[SYNTHETIC {lang} {task} prompt]",
                    "text_type": text_type,
                    "intended_task": intended,
                    "task_label": task,
                    "is_benign_task": "True",
                    "model": SYNTHETIC_MODEL,
                    "response": response,
                    "synthetic": "True",
                })
                judge_rows.append({
                    "sample_id": sid,
                    "plain_text": f"[SYNTHETIC {lang} {task} prompt]",
                    "response": response,
                    "ri": ri,
                    "judge_raw": "[SYNTHETIC judge rationale]",
                    "judge_model": "SYNTHETIC-judge",
                    "prompt_id": prompt_id,
                })

    return pd.DataFrame(raw_rows), pd.DataFrame(judge_rows)


def write_synthetic_fixtures(out_dir: str | Path | None = None, **kwargs) -> tuple[Path, Path]:
    """Write synthetic raw + judgments CSVs and return their paths."""
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, judged = generate_synthetic(**kwargs)
    raw_path = out_dir / "synthetic_track_b_raw.csv"
    judge_path = out_dir / "synthetic_track_b_judgments.csv"
    raw.to_csv(raw_path, index=False)
    judged.to_csv(judge_path, index=False)
    return raw_path, judge_path


if __name__ == "__main__":
    rp, jp = write_synthetic_fixtures()
    print(f"Wrote synthetic fixtures:\n  {rp}\n  {jp}")
