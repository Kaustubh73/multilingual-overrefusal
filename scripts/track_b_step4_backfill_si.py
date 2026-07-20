#!/usr/bin/env python3
"""Backfill SI onto existing translate-back judgments.

Reads track_b_judgments.csv (RI-only), runs single-call SI with gpt-oss:20b
on the original response (not translated), writes SI columns back.

Uses track_b_raw_translated.csv to get the original response for each row.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("backfill_si")

from src.refusal_judge import classify_safety, SI_VALUES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JUDGMENTS_PATH = PROJECT_ROOT / "results" / "track_b_judgments.csv"
RAW_TB_PATH = PROJECT_ROOT / "results" / "track_b_raw_translated.csv"
OUT_PATH = PROJECT_ROOT / "results" / "track_b_judgments_with_si.csv"
JUDGE_MODEL = "gpt-oss:20b"

SKIP_TASKS = {"cryptanalysis"}
TARGET_LANGS = {"hindi", "hinglish"}


def main():
    parser = argparse.ArgumentParser(description="Backfill SI onto translate-back judgments")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to judge")
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 60)
    log.info("BACKFILL SI: TRANSLATE-BACK JUDGMENTS")
    log.info("=" * 60)

    if not JUDGMENTS_PATH.exists():
        log.error("Missing %s", JUDGMENTS_PATH)
        return 1
    if not RAW_TB_PATH.exists():
        log.error("Missing %s", RAW_TB_PATH)
        return 1

    judgments = pd.read_csv(JUDGMENTS_PATH, dtype=str, keep_default_na=False)
    raw_tb = pd.read_csv(RAW_TB_PATH, dtype=str, keep_default_na=False)

    # Filter to target rows
    mask = judgments["language"].isin(TARGET_LANGS) & ~judgments["intended_task"].isin(SKIP_TASKS)
    to_process = judgments[mask].copy()
    log.info("Rows to process: %d (of %d total)", len(to_process), len(judgments))

    if args.limit:
        to_process = to_process.head(args.limit)

    # Check Ollama
    log.info("Checking Ollama for %s...", JUDGE_MODEL)
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if JUDGE_MODEL not in models:
            log.error("Model %s not found. Available: %s", JUDGE_MODEL, models)
            return 1
        log.info("Ollama OK")
    except Exception as e:
        log.error("Ollama not reachable: %s", e)
        return 1

    # Load existing output if resuming
    done_pids = set()
    si_results = {}
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, dtype=str, keep_default_na=False)
        done_pids = set(existing.loc[existing["si"].notna(), "prompt_id"].tolist())
        for _, row in existing.iterrows():
            si_results[row["prompt_id"]] = row["si"]
        log.info("Resuming: %d rows already have SI", len(done_pids))

    # Build response lookup from raw_tb
    resp_lookup = {}
    for _, row in raw_tb.iterrows():
        resp_lookup[row["prompt_id"]] = row.get("response", "")

    total = len(to_process)
    to_judge = total - len(done_pids)
    log.info("Total: %d | Done: %d | Remaining: %d", total, len(done_pids), to_judge)
    log.info("-" * 60)

    errors = 0
    for i, (_, row) in enumerate(to_process.iterrows(), start=1):
        pid = row["prompt_id"]
        if pid in done_pids:
            continue

        response = resp_lookup.get(pid, "")
        if not response:
            log.warning("[%3d/%d] %s — no response found, skipping", i, total, pid)
            si_results[pid] = "no_response"
            continue

        t_start = time.time()
        try:
            result = classify_safety(response, model=JUDGE_MODEL)
            si = result["si"]
            elapsed = time.time() - t_start
            log.info("[%3d/%d] %s %-10s SI=%-10s %.1fs", i, total, pid, row["language"], si, elapsed)
        except Exception as e:
            elapsed = time.time() - t_start
            log.error("[%3d/%d] %s ERROR (%.1fs): %s", i, total, pid, elapsed, e)
            si = "judge_error"
            errors += 1

        si_results[pid] = si

        if i % 10 == 0 or i == total:
            out_rows = []
            for _, jrow in judgments.iterrows():
                out_rows.append({
                    "prompt_id": jrow["prompt_id"],
                    "sample_id": jrow["sample_id"],
                    "language": jrow["language"],
                    "intended_task": jrow["intended_task"],
                    "text_type": jrow.get("text_type", ""),
                    "is_benign_task": jrow.get("is_benign_task", ""),
                    "ri": jrow["ri"],
                    "si": si_results.get(jrow["prompt_id"], ""),
                    "judge_model": JUDGE_MODEL,
                })
            pd.DataFrame(out_rows).to_csv(OUT_PATH, index=False)

    log.info("-" * 60)
    log.info("DONE — %d errors, elapsed %.1f min", errors, (time.time() - t0) / 60)
    log.info("Saved to %s", OUT_PATH)
    log.info("=" * 60)
    return 1 if errors > len(to_process) * 0.1 else 0


if __name__ == "__main__":
    sys.exit(main())
