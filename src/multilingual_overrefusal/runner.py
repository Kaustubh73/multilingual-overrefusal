"""Run a pipeline from one TOML config: `steps` run in order, finished steps are skipped.

A step is a function `step(config: dict) -> None` registered in STEPS. Each module that owns a
stage (judges, generation, ...) registers its steps here as it is merged in. Until then a step
name in a config fails at startup, before any compute is spent.
"""
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Callable

from .io import set_seed, write_run_json

STEPS: dict[str, Callable[[dict], None]] = {}   # step name -> function, filled by stage modules
REQUIRED_KEYS = ("name", "output_dir", "steps")


def step(name: str):
    """Decorator: register a function as a pipeline step."""
    def wrap(fn):
        STEPS[name] = fn
        return fn
    return wrap


def load_config(path) -> dict:
    """Read the TOML and fail early on missing keys or unknown step names."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing key(s) {missing}")
    unknown = [s for s in cfg["steps"] if s not in STEPS]
    if unknown:
        raise ValueError(f"{path}: unknown step(s) {unknown}. Known: {sorted(STEPS)}")
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """`--set generate.backend=modal` -> cfg["generate"]["backend"] = "modal" (value parsed as TOML)."""
    for item in overrides:
        key, _, raw = item.partition("=")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw                      # bare words are plain strings
        *parents, last = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = value
    return cfg


def _marker(out_dir: Path, name: str) -> Path:
    return out_dir / ".done" / name


def run(cfg: dict, only: str | None = None, force: list[str] = ()) -> None:
    """Run the configured steps. A step is skipped if its .done marker exists, unless forced."""
    out_dir = Path(cfg["output_dir"])
    seed = cfg.get("seed", 0)
    set_seed(seed)
    write_run_json(out_dir, cfg, seed)
    for name in cfg["steps"]:
        if only and name != only:
            continue
        if _marker(out_dir, name).exists() and name not in force and name != only:
            print(f"skip {name} (already done)")
            continue
        print(f"run  {name}")
        STEPS[name](cfg)
        _marker(out_dir, name).parent.mkdir(parents=True, exist_ok=True)
        _marker(out_dir, name).touch()


def launch_detached(cmd: list[str], log_path) -> int:
    """Start a long job that survives this process dying (macOS has no setsid). Returns its PID."""
    log = open(log_path, "ab")
    p = subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT)
    return p.pid
