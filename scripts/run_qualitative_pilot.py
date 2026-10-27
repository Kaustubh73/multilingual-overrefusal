#!/usr/bin/env python3
"""Qualitative overnight pilot: 16 Slack-spec prompts × instruct models (GGUF).

minty notes:
- GTX 1650 (~4GB) + broken NVML (driver 595.84 vs userspace 595.91) → HF
  transformers+bnb fails in caching_allocator_warmup.
- Use llama-cpp-python + Q4_K_M GGUF on CPU (optional GPU layers if CUDA works).
- Resumes per-model CSVs. Writes side-by-side comparison at end.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("qualitative_pilot")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = ROOT / "results" / "qualitative_pilot" / "prompts.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "qualitative_pilot"

# Slack qualitative set → bartowski Q4_K_M GGUFs (+ Aya expanse for multilingual).
DEFAULT_MODELS = [
    {
        "key": "Qwen2.5-7B-Instruct",
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
    },
    {
        "key": "Llama-3.1-8B-Instruct",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "key": "gemma-2-9b-it",
        "repo": "bartowski/gemma-2-9b-it-GGUF",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "hf_id": "google/gemma-2-9b-it",
    },
    {
        "key": "aya-expanse-8b",
        "repo": "bartowski/aya-expanse-8b-GGUF",
        "filename": "aya-expanse-8b-Q4_K_M.gguf",
        "hf_id": "CohereForAI/aya-expanse-8b",
    },
]

MAX_NEW_TOKENS = 256
# Short pilot prompts; raise via --n-ctx if needed
N_CTX = 4096


def _slug(key: str) -> str:
    return key.replace("/", "__")


def ensure_hf_token() -> None:
    tok_path = Path.home() / ".cache" / "huggingface" / "token"
    if tok_path.exists() and not os.environ.get("HF_TOKEN"):
        token = tok_path.read_text(encoding="utf-8").strip()
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def download_gguf(spec: dict) -> Path:
    log.info("Ensuring GGUF %s/%s", spec["repo"], spec["filename"])
    path = hf_hub_download(
        repo_id=spec["repo"],
        filename=spec["filename"],
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(path)


def load_llm(gguf_path: Path, n_gpu_layers: int, n_threads: int, n_ctx: int) -> Llama:
    log.info(
        "Loading %s (n_gpu_layers=%d n_threads=%d n_ctx=%d)",
        gguf_path.name,
        n_gpu_layers,
        n_threads,
        n_ctx,
    )
    return Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        logits_all=False,
        verbose=False,
    )


def generate(llm: Llama, prompt: str, max_new_tokens: int) -> str:
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    return (out["choices"][0]["message"]["content"] or "").strip()


def run_model(
    spec: dict,
    prompts: pd.DataFrame,
    out_csv: Path,
    max_new_tokens: int,
    limit: int | None,
    n_gpu_layers: int,
    n_threads: int,
    n_ctx: int,
) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done: set[str] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = {str(r["pilot_id"]) for r in rows if str(r.get("response", "")).strip()}
        log.info("Resume %s: %d done", spec["key"], len(done))

    work = prompts.copy()
    if limit is not None:
        work = work.head(limit)
    pending = work[~work["pilot_id"].astype(str).isin(done)]
    if pending.empty:
        log.info("Nothing pending for %s", spec["key"])
        return pd.DataFrame(rows)

    gguf = download_gguf(spec)
    llm = load_llm(gguf, n_gpu_layers=n_gpu_layers, n_threads=n_threads, n_ctx=n_ctx)
    try:
        for _, row in pending.iterrows():
            pid = str(row["pilot_id"])
            t0 = time.time()
            log.info(
                "[%s] %s %s/%s/%s",
                spec["key"],
                pid,
                row["lang"],
                row["intended_task"],
                row["label"],
            )
            err = ""
            try:
                response = generate(llm, str(row["prompt"]), max_new_tokens)
            except Exception as exc:  # noqa: BLE001
                response = ""
                err = f"{type(exc).__name__}: {exc}"
                log.exception("Fail %s %s", spec["key"], pid)

            elapsed = round(time.time() - t0, 2)
            rec = {
                "pilot_id": pid,
                "content_id": row["content_id"],
                "lang": row["lang"],
                "intended_task": row["intended_task"],
                "label": row["label"],
                "en_seed": row["en_seed"],
                "prompt": row["prompt"],
                "model_key": spec["key"],
                "model_id": spec["hf_id"],
                "gguf_repo": spec["repo"],
                "gguf_file": spec["filename"],
                "response": response,
                "error": err,
                "elapsed_sec": elapsed,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
            }
            rows = [r for r in rows if str(r.get("pilot_id")) != pid]
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            log.info(
                "Saved %s %s (%.1fs) resp_len=%d",
                spec["key"],
                pid,
                elapsed,
                len(response),
            )
    finally:
        del llm

    return pd.DataFrame(rows)


def write_comparison(out_dir: Path, prompts: pd.DataFrame, specs: list[dict]) -> Path:
    base = prompts[
        ["pilot_id", "lang", "intended_task", "label", "en_seed", "prompt"]
    ].copy()
    wide = base
    for spec in specs:
        slug = _slug(spec["key"])
        path = out_dir / f"responses__{slug}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)[["pilot_id", "response", "error", "elapsed_sec"]].rename(
            columns={
                "response": f"resp__{slug}",
                "error": f"err__{slug}",
                "elapsed_sec": f"sec__{slug}",
            }
        )
        wide = wide.merge(df, on="pilot_id", how="left")

    csv_path = out_dir / "comparison_wide.csv"
    wide.to_csv(csv_path, index=False)

    lines = [
        "# Qualitative pilot — side-by-side",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Models: {', '.join(s['hf_id'] + ' (Q4_K_M GGUF)' for s in specs)}",
        "",
        "Qualitative only — do not publish rates from this n=16 grid.",
        "",
    ]
    for _, row in wide.iterrows():
        lines.append(
            f"## {row['pilot_id']} · {row['lang']} · {row['intended_task']} · {row['label']}"
        )
        lines.append("")
        lines.append(f"**EN seed:** {row['en_seed']}")
        lines.append("")
        lines.append("<details><summary>Full prompt</summary>")
        lines.append("")
        lines.append("```")
        lines.append(str(row["prompt"]))
        lines.append("```")
        lines.append("</details>")
        lines.append("")
        for spec in specs:
            slug = _slug(spec["key"])
            col = f"resp__{slug}"
            if col not in wide.columns:
                continue
            resp = row.get(col, "")
            err = row.get(f"err__{slug}", "")
            sec = row.get(f"sec__{slug}", "")
            lines.append(f"### {spec['hf_id']} ({sec}s)")
            lines.append("")
            if isinstance(err, str) and err.strip() and err == err:  # not NaN
                lines.append(f"*Error:* `{err}`")
                lines.append("")
            lines.append("```")
            lines.append(str(resp) if pd.notna(resp) else "")
            lines.append("```")
            lines.append("")

    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s and %s", csv_path, md_path)
    return md_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Subset of model keys, e.g. Qwen2.5-7B-Instruct",
    )
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="llama.cpp GPU layers (0=CPU; -1=all / Metal on Mac)",
    )
    ap.add_argument("--n-threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--n-ctx", type=int, default=N_CTX)
    args = ap.parse_args()

    ensure_hf_token()
    prompts = pd.read_csv(args.prompts)
    specs = DEFAULT_MODELS
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s["key"] in wanted]
        if not specs:
            raise SystemExit(f"No models matched --only {args.only}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "models": [s["hf_id"] for s in specs],
        "ggufs": [f"{s['repo']}/{s['filename']}" for s in specs],
        "n_prompts": int(
            len(prompts) if args.limit is None else min(args.limit, len(prompts))
        ),
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "n_ctx": args.n_ctx,
        "backend": "llama-cpp-python Q4_K_M",
        "host": os.uname().nodename if hasattr(os, "uname") else "",
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Meta: %s", meta)

    for spec in specs:
        out_csv = args.out_dir / f"responses__{_slug(spec['key'])}.csv"
        try:
            run_model(
                spec,
                prompts,
                out_csv,
                args.max_new_tokens,
                args.limit,
                args.n_gpu_layers,
                args.n_threads,
                args.n_ctx,
            )
        except Exception:
            log.exception("Model failed entirely: %s — continuing", spec["key"])
        write_comparison(args.out_dir, prompts, specs)

    write_comparison(args.out_dir, prompts, specs)
    (args.out_dir / "DONE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    log.info("All done.")


if __name__ == "__main__":
    main()
