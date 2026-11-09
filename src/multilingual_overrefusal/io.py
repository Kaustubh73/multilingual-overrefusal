"""Small file helpers: JSONL read/write, resume bookkeeping, run.json, seeds."""
import json
import random
import subprocess
import sys
import time
from pathlib import Path


def read_jsonl(path) -> list[dict]:
    """Read a JSONL file; a missing file gives an empty list (so first runs and resumes look alike)."""
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, rows) -> None:
    """Append rows to a JSONL file. Append-only keeps a crash from losing finished rows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def already_done_ids(path, key: str = "row_id") -> set:
    """Ids already written to `path`; a resumed run skips these."""
    return {r[key] for r in read_jsonl(path) if key in r}


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
