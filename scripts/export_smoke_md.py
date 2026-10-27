#!/usr/bin/env python3
"""Build smoke_results.md from responses.jsonl for Slack review."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "smoke-cover-2026-09"
JSONL = RUN / "responses.jsonl"
OUT = RUN / "smoke_results.md"
SLACK = RUN / "slack_update.md"


def main():
    rows = []
    with JSONL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    by_idx = defaultdict(list)
    models = sorted({r["model_id"] for r in rows})
    for r in rows:
        by_idx[r["idx"]].append(r)

    ok = sum(1 for r in rows if not r.get("error") and (r.get("response") or "").strip())
    err = sum(1 for r in rows if r.get("error"))

    lines = [
        "# smoke-cover-2026-09 qualitative results",
        "",
        f"N responses: {len(rows)} (ok={ok}, err={err})",
        f"Prompts: {len(by_idx)}",
        f"Models: {', '.join(models)}",
        "",
        "Decode: greedy, max_tokens=512, no system prompt.",
        "",
    ]

    for idx in sorted(by_idx):
        group = by_idx[idx]
        base = group[0]
        lines.append(
            f"## {idx}. {base['lang']} · {base['intended_task']} · {base['label']}"
        )
        lines.append("")
        lines.append(f"**EN seed:** {base['en_seed']}")
        lines.append("")
        lines.append(f"**Prompt (localized):** {base['prompt']}")
        lines.append("")
        for r in sorted(group, key=lambda x: x["model_id"]):
            lines.append(f"### {r['model_id']} ({r.get('latency_s', '?')}s)")
            lines.append("")
            if r.get("error"):
                lines.append(f"*ERROR:* `{r['error']}`")
            else:
                body = (r.get("response") or "").strip() or "*empty*"
                lines.append(body)
            lines.append("")
        lines.append("---")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")

    slack = [
        "Hi <@U0BU6MG5Y87|Utsav Maskey> — qualitative covering smoke is done.",
        "",
        f"**Run ID:** `smoke-cover-2026-09`",
        f"**N:** {len(rows)} gens ({len(by_idx)} prompts × {len(models)} models)",
        f"**Models:** {', '.join(f'`{m}`' for m in models)}",
        "**Decode:** greedy, max_tokens=512, localized dataset prompts",
        "",
        "Results in `runs/smoke-cover-2026-09/smoke_results.md` — happy to walk through response quality next.",
        "",
    ]
    SLACK.write_text("\n".join(slack), encoding="utf-8")
    print(f"Wrote {SLACK}")


if __name__ == "__main__":
    main()
