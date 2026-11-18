"""Small file helpers: JSONL read/write, resume bookkeeping, run.json, seeds."""
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path


def read_jsonl(path) -> list[dict]:
    """Read a JSONL file; a missing file gives an empty list (so first runs and resumes look alike).

    A line that is not valid JSON (a write cut off by a crash or a full disk) is skipped with a
    warning, so a resumed run redoes that row instead of failing to start.
    """
    p = Path(path)
    if not p.exists():
        return []
    rows, bad = [], 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"warning: skipped {bad} unreadable line(s) in {p}", file=sys.stderr)
    return rows


def append_jsonl(path, rows) -> None:
    """Append rows to a JSONL file. Append-only keeps a crash from losing finished rows.

    If the file ends in a cut-off line, a newline is written first so new rows start on their own line.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if p.exists() and p.stat().st_size:
        with p.open("rb") as f:
            f.seek(-1, 2)
            prefix = "" if f.read(1) == b"\n" else "\n"
    with p.open("a", encoding="utf-8") as f:
        f.write(prefix + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def input_hash(*parts) -> str:
    """Short hash of the inputs that produced a record. Resume reuses a record only if this matches."""
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def already_done_ids(path, key: str = "row_id") -> set:
    """Ids already written to `path`; a resumed run skips these."""
    return {r[key] for r in read_jsonl(path) if key in r}


def latest_matching(records: list[dict], key, expected: dict) -> dict:
    """key(rec) -> the last record whose `input_hash` equals expected[key(rec)].

    Records for keys not in `expected` (rows no longer in the input) and records made from
    different inputs (stale) are left out.
    """
    out = {}
    for rec in records:
        k = key(rec)
        if k in expected and rec.get("input_hash") == expected[k]:
            out[k] = rec
    return out


def set_seed(seed: int) -> None:
    """Seed python (and numpy if installed). Torch/MLX seeds are set where those libraries load."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_run_json(out_dir, config: dict, seed: int, extra: dict | None = None) -> Path:
    """Record how an output folder was produced: command, git hash, seed, config, time (UTC)."""
    p = Path(out_dir) / "run.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "command": " ".join(sys.argv),
        "git_hash": git_hash(),
        "seed": seed,
        "config": config,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    p.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return p
