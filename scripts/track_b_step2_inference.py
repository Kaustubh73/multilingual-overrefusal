#!/usr/bin/env python3
"""Step 2: Run inference on all rows via Ollama LLaMA.

Reads results/track_b_catalog_{split}.csv, runs inference on non-English rows
(and English rows for train split), writes results/track_b_raw_{split}.csv.

For test split: English rows are copied from paper_raw.csv (pre-existing).
For train split: All rows (English + Hindi + Hinglish) require fresh inference.
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
log = logging.getLogger("track_b_inference")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_RAW_PATH = RESULTS_DIR / "paper_raw.csv"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b-instruct-q8_0"
TIMEOUT_SEC = 300.0


def ollama_generate(prompt: str, model: str = MODEL, timeout: float = TIMEOUT_SEC) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def main():
    parser = argparse.ArgumentParser(description="Step 2: Ollama inference on Track B rows")
    parser.add_argument("--limit", type=int, default=None, help="Max sample_ids to process")
    parser.add_argument(
        "--split", choices=["test", "train"], default="test",
        help="Dataset split (default: test)",
    )
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP 2: INFERENCE (split=%s)", args.split)
    log.info("=" * 60)

    catalog_path = RESULTS_DIR / f"track_b_catalog_{args.split}.csv"
    raw_path = RESULTS_DIR / f"track_b_raw_{args.split}.csv"

    if not catalog_path.exists():
        log.error("Catalog not found: %s — run step1 first", catalog_path)
        return 1

    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    if args.limit:
        sample_ids = catalog["sample_id"].unique()[: args.limit]
        catalog = catalog[catalog["sample_id"].isin(sample_ids)].copy()
        log.info("Limited to %d sample_ids (--limit %d)", len(sample_ids), args.limit)
    log.info("Loaded catalog: %d rows, %d sample_ids", len(catalog), catalog["sample_id"].nunique())

    en_rows = catalog[catalog["language"] == "english"]
    non_en = catalog[catalog["language"] != "english"]
    hi_rows = catalog[catalog["language"] == "hindi"]
    hl_rows = catalog[catalog["language"] == "hinglish"]

    # Determine which rows need inference vs copy
    need_inference = non_en.copy()  # Always infer Hindi/Hinglish
    en_to_copy = pd.DataFrame()

    if args.split == "test" and PAPER_RAW_PATH.exists():
        # Test split: copy English from paper_raw.csv
        en_to_copy = en_rows
        log.info("  English rows:  %d (copying from paper_raw.csv)", len(en_to_copy))
    else:
        # Train split (or test without paper_raw): infer everything
        need_inference = catalog.copy()
        log.info("  English rows:  %d (inference needed)", len(en_rows))

    log.info("  Hindi rows:    %d (inference needed)", len(hi_rows))
    log.info("  Hinglish rows: %d (inference needed)", len(hl_rows))
    log.info("  Total inference: %d rows", len(need_inference))
    log.info("")

    # Check Ollama connectivity
    log.info("Checking Ollama connection...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if MODEL not in models:
            log.error("Model %s not found. Available: %s", MODEL, models)
            return 1
        log.info("Ollama OK — model %s available", MODEL)
    except Exception as e:
        log.error("Ollama not reachable: %s", e)
        return 1

    # Load existing raw if resuming
    done_ids = set()
    rows = []
    if raw_path.exists():
        existing = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
        done_ids = set(existing["prompt_id"].tolist())
        rows = existing.to_dict("records")
        log.info("Resuming: %d rows already done", len(done_ids))

    # Copy English rows from paper_raw.csv (test split only)
    if args.split == "test" and PAPER_RAW_PATH.exists() and len(en_to_copy) > 0:
        paper_raw = pd.read_csv(PAPER_RAW_PATH, dtype=str, keep_default_na=False)
        paper_by_sid = paper_raw.set_index("sample_id")
        copied = 0
        for _, erow in en_to_copy.iterrows():
            pid = erow["prompt_id"]
            if pid in done_ids:
                continue
            sid = erow["sample_id"]
            if sid in paper_by_sid.index:
                pr = paper_by_sid.loc[sid]
                rows.append({
                    "prompt_id": pid,
                    "sample_id": sid,
                    "language": "english",
                    "plain_text": erow["plain_text"],
                    "response": pr["response"],
                    "text_type": erow["text_type"],
                    "intended_task": erow["intended_task"],
                    "task_label": erow.get("task_label", ""),
                    "is_benign_task": erow.get("is_benign_task", ""),
                })
                done_ids.add(pid)
                copied += 1
        log.info("Copied %d English rows from paper_raw.csv", copied)
        pd.DataFrame(rows).to_csv(raw_path, index=False)

    # Run inference
    total_new = len(need_inference)
    log.info("")
    log.info("Starting inference on %d rows...", total_new)
    log.info("Model: %s | Timeout: %ds per call", MODEL, int(TIMEOUT_SEC))
    log.info("-" * 60)

    errors = 0
    for i, (_, row) in enumerate(need_inference.iterrows(), start=1):
        pid = row["prompt_id"]
        if pid in done_ids:
            continue

        sid = row["sample_id"]
        lang = row["language"]
        task = row["intended_task"]
        prompt = row["plain_text"]

        t_start = time.time()
        try:
            response = ollama_generate(prompt, model=MODEL, timeout=TIMEOUT_SEC)
            elapsed = time.time() - t_start
            preview = response[:80].replace("\n", " ")
            log.info(
                "[%4d/%d] %s %10s %-20s %.1fs | %s",
                i, total_new, pid, lang, task, elapsed, preview,
            )
        except Exception as e:
            elapsed = time.time() - t_start
            log.error(
                "[%4d/%d] %s %10s %-20s %.1fs | ERROR: %s",
                i, total_new, pid, lang, task, elapsed, e,
            )
            response = f"[INFERENCE_ERROR: {e}]"
            errors += 1

        rows.append({
            "prompt_id": pid,
            "sample_id": sid,
            "language": lang,
            "plain_text": prompt,
            "response": response,
            "text_type": row["text_type"],
            "intended_task": task,
            "task_label": row.get("task_label", ""),
            "is_benign_task": row.get("is_benign_task", ""),
        })
        done_ids.add(pid)

        # Save every 5 rows
        if i % 5 == 0 or i == total_new:
            pd.DataFrame(rows).to_csv(raw_path, index=False)

    log.info("-" * 60)
    log.info("INFERENCE COMPLETE")
    log.info("  Total rows:   %d", len(rows))
    log.info("  Errors:       %d", errors)
    log.info("  Elapsed:      %.1f minutes", (time.time() - t0) / 60)
    log.info("  Saved to:     %s", raw_path)
    log.info("=" * 60)

    return 1 if errors > total_new * 0.1 else 0


if __name__ == "__main__":
    sys.exit(main())
