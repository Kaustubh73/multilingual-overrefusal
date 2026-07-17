"""Judge-model ensemble for refusal classification with disagreement logging."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from src.refusal_judge import (
    DEFAULT_JUDGE_MODEL,
    RI_VALUES,
    classify_refusal,
    judge_responses_df,
    parse_judge_output,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DISAGREEMENTS_PATH = PROJECT_ROOT / "results" / "judge_disagreements.jsonl"

ENSEMBLE_STRATEGIES = ("majority", "unanimous_or_answer")


def _majority_vote(votes: dict[str, str]) -> tuple[str, bool]:
    """Return (ensemble_label, unanimous)."""
    counts = Counter(votes.values())
    if not counts:
        return "parse_error", False
    top_label, top_count = counts.most_common(1)[0]
    unanimous = len(counts) == 1
    return top_label, unanimous


def ensemble_classify_refusal(
    question: str,
    response: str,
    models: list[str] | None = None,
    *,
    strategy: str = "majority",
    max_retries: int = 2,
) -> dict:
    """
    Classify refusal using multiple judge models.

    Returns per-judge labels, ensemble ``ri``, unanimity flag, and disagreement detail.
    """
    models = models or [DEFAULT_JUDGE_MODEL]
    if strategy not in ENSEMBLE_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy}; choose from {ENSEMBLE_STRATEGIES}")

    per_judge: dict[str, dict] = {}
    for model in models:
        result = classify_refusal(question, response, model=model, max_retries=max_retries)
        per_judge[model] = result

    votes = {m: r["ri"] for m, r in per_judge.items()}
    ensemble_ri, unanimous = _majority_vote(votes)

    if strategy == "unanimous_or_answer" and not unanimous:
        # Conservative: only refuse if all judges agree on a refusal class
        refusal_votes = {m: r for m, r in votes.items() if r in ("direct_refusal", "indirect_refusal")}
        if len(set(refusal_votes.values())) == 1 and len(refusal_votes) == len(models):
            ensemble_ri = next(iter(refusal_votes.values()))
        else:
            ensemble_ri = "direct_answer"

    disagreements = []
    if not unanimous:
        disagreements = [
            {"judge_model": m, "ri": r, "judge_raw": per_judge[m].get("judge_raw", "")}
            for m, r in votes.items()
        ]

    return {
        "ri": ensemble_ri,
        "ensemble_unanimous": unanimous,
        "ensemble_strategy": strategy,
        "per_judge": per_judge,
        "votes": votes,
        "disagreements": disagreements,
        "judge_models": ",".join(models),
    }


def log_disagreement_record(
    record: dict,
    out_path: Path | None = None,
) -> None:
    """Append one JSONL disagreement record."""
    out_path = out_path or DEFAULT_DISAGREEMENTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def judge_responses_ensemble_df(
    df: pd.DataFrame,
    *,
    models: list[str] | None = None,
    strategy: str = "majority",
    out_path: Path | None = None,
    disagreements_path: Path | None = None,
    resume: bool = True,
    question_col: str = "plain_text",
    response_col: str = "response",
) -> pd.DataFrame:
    """
    Judge each row with an ensemble; log disagreements to JSONL.

    Output columns include ``ri`` (ensemble), ``judge_models``, per-model ``ri_*``,
    and ``ensemble_unanimous``.
    """
    models = models or [DEFAULT_JUDGE_MODEL]
    out_path = out_path or (PROJECT_ROOT / "results" / "ensemble_judgments.csv")
    disagreements_path = disagreements_path or DEFAULT_DISAGREEMENTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    rows: list[dict] = []
    if resume and out_path.exists():
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        rows.extend(existing.to_dict("records"))
        if "sample_id" in existing.columns:
            done = set(existing["sample_id"].astype(str))

    for i, row in df.iterrows():
        sid = str(row["sample_id"])
        if sid in done:
            continue
        question = str(row[question_col])
        response = str(row[response_col])
        logger.info("Ensemble judging [%s] %s with %s", i, sid, models)

        try:
            result = ensemble_classify_refusal(
                question, response, models=models, strategy=strategy
            )
        except Exception as exc:
            logger.error("Ensemble judge failed for %s: %s", sid, exc)
            result = {
                "ri": "judge_error",
                "ensemble_unanimous": False,
                "ensemble_strategy": strategy,
                "per_judge": {},
                "votes": {},
                "disagreements": [],
                "judge_models": ",".join(models),
            }

        out_row: dict = {
            "sample_id": sid,
            "plain_text": question,
            "response": response,
            "ri": result["ri"],
            "ensemble_unanimous": str(result["ensemble_unanimous"]),
            "ensemble_strategy": strategy,
            "judge_models": result["judge_models"],
        }
        for model, pred in result.get("per_judge", {}).items():
            safe_name = model.replace(":", "_").replace("/", "_")
            out_row[f"ri_{safe_name}"] = pred.get("ri", "")
            out_row[f"judge_raw_{safe_name}"] = pred.get("judge_raw", "")

        rows.append(out_row)
        pd.DataFrame(rows).to_csv(out_path, index=False)

        if result.get("disagreements"):
            log_disagreement_record(
                {
                    "sample_id": sid,
                    "question": question[:500],
                    "response": response[:500],
                    "ensemble_ri": result["ri"],
                    "votes": result["votes"],
                    "disagreements": result["disagreements"],
                    "strategy": strategy,
                },
                out_path=disagreements_path,
            )
        time.sleep(0.05)

    logger.info("Saved %d ensemble judgments to %s", len(rows), out_path)
    return pd.DataFrame(rows)


def summarize_disagreements(path: Path | None = None) -> dict:
    """Summarize logged judge disagreements from JSONL."""
    path = path or DEFAULT_DISAGREEMENTS_PATH
    if not path.exists():
        return {"n_disagreements": 0, "vote_patterns": {}}

    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    patterns: Counter = Counter()
    for rec in records:
        votes = rec.get("votes", {})
        pattern = tuple(sorted(f"{m}:{r}" for m, r in votes.items()))
        patterns[pattern] += 1

    return {
        "n_disagreements": len(records),
        "vote_patterns": {str(k): v for k, v in patterns.most_common(10)},
    }


# Re-export parse for golden tests
__all__ = [
    "ensemble_classify_refusal",
    "judge_responses_ensemble_df",
    "judge_responses_df",
    "log_disagreement_record",
    "parse_judge_output",
    "summarize_disagreements",
    "RI_VALUES",
]
