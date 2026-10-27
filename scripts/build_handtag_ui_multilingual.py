#!/usr/bin/env python3
"""Build the E1b (multilingual) blind hand-tag UI: rubric v1.1 with the "reject premise, then pivot" note.

  .venv/bin/python scripts/build_handtag_ui_multilingual.py          # -> results/e1b/handtag/_shown/handtag_ui.html
  .venv/bin/python scripts/handtag_ui_server_multilingual.py         # serve + save labels (http://localhost:8770)

English-only view (Kaustubh, 26 Oct 2026): each item shows the Google-Translate
English of the prompt AND the response (from aid_en.json) instead of the original
ur/lo/mt/yo text, with the original language named at the top of the prompt block.
Consequence: labels are tags on the machine translation, not the original — the
rubric's "judge the original when they disagree" caveat cannot apply in this view.

The build reads handtag_key.json ONLY for item -> lang (never model/task/label);
language is not blind-breaking (it is what the request is written in) and was
explicitly requested to be shown. The whitelist in build_handtag_ui.py still
drops every field except n/item/prompt/response, so nothing else can reach the page.

Does not clobber _shown/handtag_labels.jsonl (the live label file the server
appends to); copies from handtag_labels.jsonl only if the live file is absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_handtag_ui as B  # noqa: E402
import handtag  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "e1b" / "handtag"
RUBRIC = "handtag-v1.1-2026-10-22"

LANG_NAMES = {
    "yo": "Yoruba",
    "lo": "Lao",
    "mt": "Maltese",
    "ur": "Urdu",
}


def main() -> None:
    handtag.RUBRIC_VERSION = RUBRIC
    B.TEMPLATE = Path(__file__).with_name("handtag_ui_template_multilingual.html")
    aid_path = RUN / "aid_en.json"
    if not aid_path.exists():
        raise SystemExit(f"{aid_path} missing; English aids not built")
    aid = json.loads(aid_path.read_text())
    key = json.loads((RUN / "handtag_key.json").read_text())
    lang_of = {e["item"]: e["lang"] for e in key}  # item -> lang only

    raw = B.read_jsonl(RUN / "handtag_sheet.jsonl")
    shown = RUN / "_shown"
    shown.mkdir(exist_ok=True)
    for r in raw:
        a = aid.get(r["item"])
        if not a:
            raise SystemExit(f"{r['item']} has no English aid")
        lang = lang_of.get(r["item"])
        name = LANG_NAMES.get(lang, lang) if lang else "unknown"
        r["prompt"] = f"[language: {name} ({lang}) — shown in Google Translate English]\n\n{a['prompt_en']}"
        r["response"] = a["response_en"]
    (shown / "handtag_sheet.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in raw) + "\n"
    )

    # Preserve the live label file; seed it from handtag_labels.jsonl only if absent.
    live = shown / "handtag_labels.jsonl"
    seed = RUN / "handtag_labels.jsonl"
    if not live.exists() or not live.stat().st_size:
        live.write_text(seed.read_text() if seed.exists() else "")

    B.build(shown, shown / "handtag_ui.html")


if __name__ == "__main__":
    main()
