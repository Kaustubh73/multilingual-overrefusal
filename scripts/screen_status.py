#!/usr/bin/env python3
"""Progress snapshot for the harmful-seed screen (runs/harmful-screen-2026-09)."""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "runs" / "harmful-screen-2026-09"
TARGET = {"stage1": 861, "stage2": 50 * 10}


def fmt_h(sec: float) -> str:
    return f"{sec / 3600:.1f}h" if sec >= 3600 else f"{sec / 60:.0f}m"


def main():
    for stage in ("stage1", "stage2"):
        path = RUN / f"{stage}.jsonl"
        if not path.exists():
            print(f"{stage}: not started")
            continue
        recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        by_model: dict[str, list[dict]] = {}
        for r in recs:
            by_model.setdefault(r["model_id"], []).append(r)
        print(f"{stage}: {len(recs)} records")
        for m, rs in by_model.items():
            lat = [r["latency_s"] for r in rs[-50:]]
            med = statistics.median(lat)
            left = TARGET[stage] - len(rs)
            eta = f"ETA {fmt_h(left * med)}" if left > 0 else "done"
            ref = sum(r["refusal"] for r in rs) / len(rs)
            print(f"  {m:36s} {len(rs):4d}/{TARGET[stage]}  refuse={ref:5.1%}  med={med:5.1f}s  {eta}")
        last = recs[-1]
        resp = last["response"][:110].replace("\n", " ")
        print(f"  last: [{last['model_id'].split('/')[-1]}] refusal={last['refusal']} {last['latency_s']}s :: {resp}")
    print()
    mem = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout.splitlines()
    print("\n".join(mem[:3]))
    print(f"disk free: {shutil.disk_usage(RUN).free / 1e9:.1f} GB")
    print(Path("/proc/loadavg").read_text().strip())
    log = RUN / "minty.log"
    if log.exists():
        print("\n--- minty.log tail ---")
        print("\n".join(log.read_text(encoding="utf-8").splitlines()[-6:]))


if __name__ == "__main__":
    main()
