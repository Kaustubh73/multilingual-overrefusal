"""Track B inference + judge pipeline (multilingual OR)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.inference import generate_response, set_model_id

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "track_b" / "track_b_catalog.csv"
DEFAULT_RAW_PATH = PROJECT_ROOT / "results" / "track_b_raw.csv"
DEFAULT_JUDGMENTS_PATH = PROJECT_ROOT / "results" / "track_b_judgments.csv"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PAPER_RAW = PROJECT_ROOT / "results" / "paper_raw.csv"
SMOKE_STUB_MARKER = "smoke scaffold stub"


def _is_smoke_stub_response(response: str) -> bool:
    return SMOKE_STUB_MARKER in response


def _stub_response(task_label: str, language: str) -> str:
    """Minimal direct-answer stub for scaffold smoke tests (not for claims)."""
    if task_label == "sentiment":
        return "The sentiment of the text is neutral."
    if task_label == "translation":
        return "Translation: [smoke scaffold stub — replace with HF inference]"
    return "[smoke scaffold stub]"


def bootstrap_smoke_raw(
    catalog: pd.DataFrame,
    *,
    out_path: Path | None = None,
    paper_raw_path: Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    limit: int | None = None,
    limit_per_task: int | None = None,
) -> pd.DataFrame:
    """Build track_b_raw.csv without HF: English from Track A, stubs for hi/hing."""
    out_path = out_path or DEFAULT_RAW_PATH
    paper_raw_path = paper_raw_path or DEFAULT_PAPER_RAW
    work = _limit_by_sample_ids(catalog, limit, limit_per_task=limit_per_task)

    paper_resp: dict[str, str] = {}
    if paper_raw_path.exists():
        paper = pd.read_csv(paper_raw_path, dtype=str, keep_default_na=False)
        paper_resp = dict(zip(paper["sample_id"], paper["response"]))

    rows = []
    for _, row in work.iterrows():
        lang = str(row["language"])
        sid = str(row["sample_id"])
        task = str(row["task_label"])
        if lang == "english" and sid in paper_resp:
            response = paper_resp[sid]
        else:
            response = _stub_response(task, lang)
        rows.append(
            {
                "prompt_id": str(row["prompt_id"]),
                "sample_id": sid,
                "language": lang,
                "plain_text": str(row["plain_text"]),
                "text_type": row["text_type"],
                "intended_task": row["intended_task"],
                "task_label": task,
                "is_benign_task": row["is_benign_task"],
                "model": model_id,
                "response": response,
            }
        )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Bootstrapped %d Track B smoke rows to %s", len(df), out_path)
    return df


def _limit_by_sample_ids(
    catalog: pd.DataFrame,
    limit: int | None,
    *,
    limit_per_task: int | None = None,
) -> pd.DataFrame:
    if limit_per_task is not None:
        sample_ids: list[str] = []
        for task in ("translate", "sentiment_analysis"):
            task_ids = (
                catalog.loc[catalog["intended_task"] == task, "sample_id"]
                .drop_duplicates()
                .head(limit_per_task)
                .astype(str)
                .tolist()
            )
            sample_ids.extend(task_ids)
        return catalog[catalog["sample_id"].isin(sample_ids)].copy()
    if limit is None:
        return catalog
    sample_ids = catalog["sample_id"].drop_duplicates().head(limit)
    return catalog[catalog["sample_id"].isin(sample_ids)].copy()


def run_track_b_inference(
    catalog: pd.DataFrame,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    out_path: Path | None = None,
    resume: bool = True,
    force_inference: bool = False,
    limit: int | None = None,
    limit_per_task: int | None = None,
) -> pd.DataFrame:
    """Run HF inference on Track B catalog; resume by prompt_id."""
    out_path = out_path or DEFAULT_RAW_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work = _limit_by_sample_ids(catalog, limit, limit_per_task=limit_per_task)
    work_pids = set(work["prompt_id"].astype(str))
    rows_by_pid: dict[str, dict] = {}
    existing_records: list[dict] = []

    if resume and out_path.exists():
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        existing_records = existing.to_dict("records")
        for rec in existing_records:
            pid = str(rec["prompt_id"])
            if pid not in work_pids:
                rows_by_pid[pid] = rec
                continue
            if force_inference:
                continue
            if not _is_smoke_stub_response(str(rec.get("response", ""))):
                rows_by_pid[pid] = rec
            else:
                logger.info("Re-running inference for %s (smoke stub in response)", pid)

    set_model_id(model_id)
    total = len(work)

    def _write_rows() -> pd.DataFrame:
        order = [str(r["prompt_id"]) for r in existing_records]
        ordered_pids = [p for p in order if p in rows_by_pid]
        for p in rows_by_pid:
            if p not in ordered_pids:
                ordered_pids.append(p)
        df = pd.DataFrame([rows_by_pid[p] for p in ordered_pids])
        df.to_csv(out_path, index=False)
        return df

    for i, (_, row) in enumerate(work.iterrows(), start=1):
        pid = str(row["prompt_id"])
        if pid in rows_by_pid:
            continue
        prompt = str(row["plain_text"])
        logger.info(
            "Track B inference [%d/%d] %s lang=%s task=%s",
            i,
            total,
            pid,
            row["language"],
            row["intended_task"],
        )
        try:
            response = generate_response(prompt, model_id=model_id)
        except Exception as exc:
            logger.error("Inference failed for %s: %s", pid, exc)
            response = f"[INFERENCE_ERROR: {exc}]"

        rows_by_pid[pid] = {
            "prompt_id": pid,
            "sample_id": str(row["sample_id"]),
            "language": str(row["language"]),
            "plain_text": prompt,
            "text_type": row["text_type"],
            "intended_task": row["intended_task"],
            "task_label": row["task_label"],
            "is_benign_task": row["is_benign_task"],
            "model": model_id,
            "response": response,
        }
        _write_rows()

    df = _write_rows()
    logger.info("Saved %d Track B responses to %s", len(df), out_path)
    return df


def cmd_track_b_pipeline(
    *,
    catalog_path: Path | None = None,
    raw_path: Path | None = None,
    judgments_path: Path | None = None,
    report_path: Path | None = None,
    metrics_path: Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    skip_inference: bool = False,
    skip_judge: bool = False,
    resume: bool = True,
    limit: int | None = None,
    limit_per_task: int | None = None,
    judge_model: str = "gpt-oss:20b",
    judge_models: list[str] | None = None,
    ensemble_strategy: str = "majority",
    run_metrics: bool = True,
    smoke_stub: bool = False,
    force_inference: bool = False,
) -> None:
    """Full Track B: infer → judge → metrics report."""
    from src.refusal_judge import judge_responses_df
    from src.track_b_metrics import write_track_b_report

    catalog_path = catalog_path or DEFAULT_CATALOG_PATH
    raw_path = raw_path or DEFAULT_RAW_PATH
    judgments_path = judgments_path or DEFAULT_JUDGMENTS_PATH

    if not catalog_path.exists():
        logger.error("Missing catalog at %s; run: python main.py track-b prepare", catalog_path)
        sys.exit(1)

    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    work_catalog = _limit_by_sample_ids(catalog, limit, limit_per_task=limit_per_task)
    work_sample_ids = set(work_catalog["sample_id"].astype(str))

    if smoke_stub:
        bootstrap_smoke_raw(
            catalog,
            out_path=raw_path,
            model_id=model_id,
            limit=limit,
            limit_per_task=limit_per_task,
        )
    elif not skip_inference:
        run_track_b_inference(
            catalog,
            model_id=model_id,
            out_path=raw_path,
            resume=resume,
            force_inference=force_inference,
            limit=limit,
            limit_per_task=limit_per_task,
        )
    elif not raw_path.exists():
        logger.error("Missing %s; run inference first.", raw_path)
        sys.exit(1)

    if not skip_judge:
        raw_df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
        if work_sample_ids:
            raw_df = raw_df[raw_df["sample_id"].isin(work_sample_ids)].copy()

        judge_df = raw_df.copy()
        judge_df["sample_id"] = judge_df["prompt_id"]
        if judge_models and len(judge_models) > 1:
            from src.judge_ensemble import judge_responses_ensemble_df

            judge_responses_ensemble_df(
                judge_df,
                models=judge_models,
                strategy=ensemble_strategy,
                out_path=judgments_path,
                resume=resume,
                question_col="plain_text",
            )
        else:
            judge_responses_df(
                judge_df,
                model=judge_model,
                out_path=judgments_path,
                resume=resume,
                question_col="plain_text",
            )
        judged = pd.read_csv(judgments_path, dtype=str, keep_default_na=False)
        if "prompt_id" not in judged.columns and len(judged):
            judged["prompt_id"] = judged["sample_id"]
            judged["sample_id"] = judged["prompt_id"].str.replace(r"__\w+$", "", regex=True)
            judged.to_csv(judgments_path, index=False)
    elif not judgments_path.exists():
        logger.error("Missing %s; run judge step first.", judgments_path)
        sys.exit(1)

    if run_metrics:
        write_track_b_report(
            raw_path=raw_path,
            judgments_path=judgments_path,
            out_path=report_path,
            metrics_path=metrics_path,
            limit=limit if limit_per_task is None else None,
            sample_ids=work_sample_ids if limit_per_task is not None else None,
        )
