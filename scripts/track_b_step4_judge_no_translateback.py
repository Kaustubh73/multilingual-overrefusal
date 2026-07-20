#!/usr/bin/env python3
"""Ablation: Judge Hindi/Hinglish responses WITHOUT translate-back.

Reads track_b_raw.csv (raw responses), filters to Hindi+Hinglish excl cryptanalysis,
runs single-call RI + SI judges with gpt-oss:20b, writes
track_b_judgments_no_translateback.csv.

Ablation config (for results file):
  Judge model:      gpt-oss:20b
  Ensemble calls:   1 (single call, no ensemble)
  Rows judged:      420 (Hindi + Hinglish, excl cryptanalysis)
  Metrics:          RI {direct_answer, direct_refusal, indirect_refusal}
                    SI {benign, cautious, harmful}
  Translate back:   No — judge sees raw responses
  Prompt language:  Original (Hindi/Hinglish), not translated
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
log = logging.getLogger("track_b_judge_no_tb")

from src.refusal_judge import classify_refusal, classify_safety, RI_VALUES, SI_VALUES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "results" / "track_b_raw.csv"
OUT_PATH = PROJECT_ROOT / "results" / "track_b_judgments_no_translateback.csv"
JUDGE_MODEL = "gpt-oss:20b"

SKIP_TASKS = {"cryptanalysis"}
TARGET_LANGS = {"hindi", "hinglish"}


def main():
    parser = argparse.ArgumentParser(
        description="Ablation judge: no translate-back, RI + SI, single call"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to judge")
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 60)
    log.info("ABLATION JUDGE: NO TRANSLATE-BACK (RI + SI, single call)")
    log.info("=" * 60)
    log.info("Config:")
    log.info("  Judge model:    %s", JUDGE_MODEL)
    log.info("  Ensemble calls: 1 (single call)")
    log.info("  Translate back: No")
    log.info("  Languages:      %s", TARGET_LANGS)
    log.info("  Skip tasks:     %s", SKIP_TASKS)
    log.info("")

    if not INPUT_PATH.exists():
        log.error("Missing %s", INPUT_PATH)
        return 1

    df = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
    log.info("Loaded %d total rows from %s", len(df), INPUT_PATH)

    df = df[df["language"].isin(TARGET_LANGS)].copy()
    df = df[~df["intended_task"].isin(SKIP_TASKS)].copy()
    log.info(
        "Filtered to %d rows (languages=%s, excl tasks=%s)",
        len(df), TARGET_LANGS, SKIP_TASKS,
    )

    if args.limit:
        df = df.head(args.limit)
        log.info("Limited to %d rows", args.limit)

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

    # Check resume
    done_pids = set()
    rows = []
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, dtype=str, keep_default_na=False)
        done_pids = set(existing["prompt_id"].tolist())
        rows = existing.to_dict("records")
        log.info("Resuming: %d rows already judged", len(done_pids))

    total = len(df)
    to_judge = total - len(done_pids)
    log.info("Total: %d | Done: %d | Remaining: %d", total, len(done_pids), to_judge)
    log.info("-" * 60)

    errors = 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        pid = row["prompt_id"]
        if pid in done_pids:
            continue

        question = str(row["plain_text"])
        response = str(row["response"])

        t_start = time.time()
        try:
            ri_result = classify_refusal(question, response, model=JUDGE_MODEL)
            si_result = classify_safety(response, model=JUDGE_MODEL)
            elapsed = time.time() - t_start

            ri = ri_result["ri"]
            si = si_result["si"]
            log.info(
                "[%3d/%d] %s %-10s %-20s RI=%-17s SI=%-10s %.1fs",
                i, total, pid, row["language"], row["intended_task"],
                ri, si, elapsed,
            )
        except Exception as e:
            elapsed = time.time() - t_start
            log.error(
                "[%3d/%d] %s %-10s %-20s ERROR (%.1fs): %s",
                i, total, pid, row["language"], row["intended_task"], elapsed, e,
            )
            ri = "judge_error"
            si = "judge_error"
            errors += 1

        rows.append({
            "prompt_id": pid,
            "sample_id": row["sample_id"],
            "language": row["language"],
            "intended_task": row["intended_task"],
            "text_type": row.get("text_type", ""),
            "is_benign_task": row.get("is_benign_task", ""),
            "ri": ri,
            "si": si,
            "judge_model": JUDGE_MODEL,
        })
        done_pids.add(pid)

        if i % 10 == 0 or i == total:
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    log.info("-" * 60)
    log.info("DONE")
    log.info("  Total rows: %d", len(rows))
    log.info("  Errors:     %d", errors)
    log.info("  Elapsed:    %.1f minutes", (time.time() - t0) / 60)
    log.info("  Saved to:   %s", OUT_PATH)

    log.info("")
    log.info("RI distribution:")
    rdf = pd.DataFrame(rows)
    for lang in ["hindi", "hinglish"]:
        subset = rdf[rdf["language"] == lang]
        log.info("  %s RI: %s", lang, dict(subset["ri"].value_counts()))
        log.info("  %s SI: %s", lang, dict(subset["si"].value_counts()))

    log.info("=" * 60)
    return 1 if errors > len(df) * 0.1 else 0


if __name__ == "__main__":
    sys.exit(main())
