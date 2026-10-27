#!/usr/bin/env python3
"""Step 4: Judge all Track B rows.

Reads track_b_raw_translated_{split}.csv if available, else track_b_raw_{split}.csv.
Runs single-call judge (faster than 3-call ensemble), writes track_b_judgments_{split}.csv.

Flags:
  --no-translate-back : Skip translate-back, judge raw responses directly
  --single            : Use single judge call (default, fast)
  --ensemble          : Use 3-call ensemble (slower, more robust)
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
log = logging.getLogger("track_b_judge")

from src.refusal_judge import classify_refusal_ensemble, RI_VALUES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
JUDGE_MODEL = "gpt-oss:20b"


def main():
    parser = argparse.ArgumentParser(description="Step 4: Judge Track B rows")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to judge (smoke test)")
    parser.add_argument(
        "--split", choices=["test", "train"], default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--no-translate-back", action="store_true",
        help="Judge raw responses directly (skip translate-back step)",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Use 3-call ensemble judge (slower, default is single call)",
    )
    args = parser.parse_args()

    n_calls = 3 if args.ensemble else 1

    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP 4: JUDGE (split=%s, mode=%s)", args.split, f"{n_calls}-call")
    log.info("=" * 60)

    # Determine input file
    translated_path = RESULTS_DIR / f"track_b_raw_translated_{args.split}.csv"
    raw_path = RESULTS_DIR / f"track_b_raw_{args.split}.csv"

    if args.no_translate_back or not translated_path.exists():
        input_path = raw_path
        response_col = "response"
        log.info("Using raw responses from %s", input_path)
    else:
        input_path = translated_path
        response_col = "response_english"
        log.info("Using translated responses from %s", input_path)

    if not input_path.exists():
        log.error("Missing %s — run Step 2 first", input_path)
        return 1

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)
        log.info("Limited to %d rows", args.limit)
    log.info("Loaded %d rows from %s", len(df), input_path)

    out_path = RESULTS_DIR / f"track_b_judgments_{args.split}.csv"

    if response_col not in df.columns:
        log.error("Column '%s' not found in %s", response_col, input_path)
        return 1

    # Check Ollama
    log.info("Checking Ollama connection for judge model %s...", JUDGE_MODEL)
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if JUDGE_MODEL not in models:
            log.error("Model %s not found. Available: %s", JUDGE_MODEL, models)
            return 1
        log.info("Ollama OK — %s available", JUDGE_MODEL)
    except Exception as e:
        log.error("Ollama not reachable: %s", e)
        return 1

    # Check resume
    done_pids = set()
    rows = []
    if out_path.exists():
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        done_pids = set(existing["prompt_id"].tolist())
        rows = existing.to_dict("records")
        log.info("Resuming: %d rows already judged", len(done_pids))

    total = len(df)
    to_judge = total - len(done_pids)
    log.info("Total rows: %d | Already done: %d | Remaining: %d", total, len(done_pids), to_judge)
    log.info("")
    log.info("-" * 60)

    errors = 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        pid = row["prompt_id"]
        if pid in done_pids:
            continue

        question = str(row["plain_text"])
        response = str(row[response_col])

        t_start = time.time()
        try:
            result = classify_refusal_ensemble(
                question, response, model=JUDGE_MODEL, n_calls=n_calls
            )
            ri = result["ri"]
            agree = result["ensemble_agreement"]
            labels = result["ensemble_labels"]
            elapsed = time.time() - t_start

            log.info(
                "[%4d/%d] %s %-10s %-20s → %-17s %.1fs",
                i, total, pid, row["language"], row["intended_task"],
                ri, elapsed,
            )
        except Exception as e:
            elapsed = time.time() - t_start
            log.error(
                "[%4d/%d] %s %-10s %-20s ERROR (%.1fs): %s",
                i, total, pid, row["language"], row["intended_task"], elapsed, e,
            )
            ri = "judge_error"
            agree = 0.0
            labels = []
            errors += 1

        rows.append({
            "prompt_id": pid,
            "sample_id": row["sample_id"],
            "language": row["language"],
            "intended_task": row["intended_task"],
            "text_type": row.get("text_type", ""),
            "is_benign_task": row.get("is_benign_task", ""),
            "is_degenerate_translate": row.get("is_degenerate_translate", ""),
            "ri": ri,
            "ensemble_agreement": str(agree),
            "ensemble_labels": str(labels),
            "judge_model": JUDGE_MODEL,
        })
        done_pids.add(pid)

        # Save every 10 rows
        if i % 10 == 0 or i == total:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    log.info("-" * 60)
    log.info("JUDGE COMPLETE")
    log.info("  Total rows:   %d", len(rows))
    log.info("  Errors:       %d", errors)
    log.info("  Elapsed:      %.1f minutes", (time.time() - t0) / 60)
    log.info("  Saved to:     %s", out_path)

    # Quick label distribution
    log.info("")
    log.info("Label distribution:")
    rdf = pd.DataFrame(rows)
    for lang in ["english", "hindi", "hinglish"]:
        subset = rdf[rdf["language"] == lang]
        vc = subset["ri"].value_counts()
        log.info("  %s: %s", lang, dict(vc))

    log.info("=" * 60)
    return 1 if errors > len(df) * 0.1 else 0


if __name__ == "__main__":
    sys.exit(main())