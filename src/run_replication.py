"""Paper replication: official test split + HF Llama inference."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.inference import generate_response, set_model_id
from src.paper_benchmark import load_paper_benchmark

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "results" / "paper_raw.csv"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CAL_PATH = PROJECT_ROOT / "results" / "judge_calibration.json"


def _load_calibration_json(path: Path = CAL_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if data.get("n_compared") else None


def _should_persist_calibration(new: dict, existing: dict | None) -> bool:
    """Keep prior calibration when a re-run fails (e.g. Ollama judge unavailable)."""
    if not existing:
        return True
    new_agreements = int(new.get("agreements", 0))
    new_compared = int(new.get("n_compared", 0))
    if new_compared == 0:
        return False
    if new_agreements == 0:
        details = new.get("details") or []
        all_errors = details and all(
            str(d.get("pred_ri", "")) in ("judge_error", "parse_error") for d in details
        )
        if all_errors and int(existing.get("agreements", 0)) > 0:
            logger.warning(
                "Calibration re-run returned 0/%d agreements (judge errors); keeping existing %d/%d",
                new_compared,
                existing.get("agreements", 0),
                existing.get("n_compared", 0),
            )
            return False
    return True


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return set(df["sample_id"].astype(str))


def run_paper_inference(
    *,
    split: str = "test",
    model_id: str = DEFAULT_MODEL_ID,
    out_path: Path | None = None,
    resume: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run HF inference on official plain_text prompts."""
    out_path = out_path or DEFAULT_RAW_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bench = load_paper_benchmark(split=split)
    if limit is not None:
        bench = bench.head(limit).copy()

    done = _completed_ids(out_path) if resume else set()
    rows: list[dict] = []
    if resume and out_path.exists():
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        rows.extend(existing.to_dict("records"))

    set_model_id(model_id)
    total = len(bench)

    for i, (_, row) in enumerate(bench.iterrows(), start=1):
        sid = str(row["sample_id"])
        if sid in done:
            continue
        prompt = str(row["plain_text"])
        logger.info(
            "Paper inference [%d/%d] %s task=%s type=%s",
            i,
            total,
            sid,
            row["intended_task"],
            row["text_type"],
        )
        try:
            response = generate_response(prompt, model_id=model_id)
        except Exception as exc:
            logger.error("Inference failed for %s: %s", sid, exc)
            response = f"[INFERENCE_ERROR: {exc}]"

        rows.append(
            {
                "sample_id": sid,
                "plain_text": prompt,
                "text_type": row["text_type"],
                "intended_task": row["intended_task"],
                "task_label": row["task_label"],
                "is_benign_task": row["is_benign_task"],
                "model": model_id,
                "response": response,
            }
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    logger.info("Saved %d paper responses to %s", len(df), out_path)
    return df


def cmd_replicate_pipeline(
    *,
    split: str = "test",
    model_id: str = DEFAULT_MODEL_ID,
    raw_path: Path | None = None,
    judgments_path: Path | None = None,
    skip_inference: bool = False,
    skip_judge: bool = False,
    skip_calibration: bool = False,
    resume: bool = True,
    limit: int | None = None,
    judge_model: str = "gpt-oss:20b",
    run_metrics: bool = True,
) -> None:
    """Full replication: infer → calibrate judge → judge → metrics report."""
    from src.paper_metrics import write_replication_report
    from src.refusal_judge import calibrate_vs_manual, judge_responses_df

    raw_path = raw_path or DEFAULT_RAW_PATH
    judgments_path = judgments_path or (PROJECT_ROOT / "results" / "paper_judgments.csv")

    if not skip_inference:
        run_paper_inference(
            split=split,
            model_id=model_id,
            out_path=raw_path,
            resume=resume,
            limit=limit,
        )
    elif not raw_path.exists():
        logger.error("Missing %s; run inference first.", raw_path)
        sys.exit(1)

    calibration = None
    skip_cal = skip_calibration or (skip_inference and skip_judge)
    if not skip_cal:
        logger.info("Calibrating judge vs manual audit (~30 rows)...")
        existing_cal = _load_calibration_json()
        calibration = calibrate_vs_manual(n_sample=30, model=judge_model)
        if _should_persist_calibration(calibration, existing_cal):
            CAL_PATH.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
            logger.info(
                "Calibration: %d/%d agreement (%.1f%%) -> %s",
                calibration.get("agreements", 0),
                calibration.get("n_compared", 0),
                100 * calibration.get("agreement_rate", 0),
                CAL_PATH,
            )
        else:
            calibration = existing_cal

    if not skip_judge:
        raw_df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
        if limit is not None:
            raw_df = raw_df.head(limit)
        judge_responses_df(
            raw_df,
            model=judge_model,
            out_path=judgments_path,
            resume=resume,
        )
    elif not judgments_path.exists():
        logger.error("Missing %s; run judge step first.", judgments_path)
        sys.exit(1)

    if run_metrics:
        write_replication_report(
            raw_path=raw_path,
            judgments_path=judgments_path,
            calibration=calibration,
        )
