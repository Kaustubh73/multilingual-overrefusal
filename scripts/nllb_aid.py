#!/usr/bin/env python3
"""English reading aid for the E1b hand-tag sheet: NLLB-200 (distilled 600M) ur/lo/mt/yo -> en.

  .venv/bin/python scripts/nllb_aid.py   # -> results/e1b/handtag/aid_en.json

Machine translation, greedy, line by line (long lines split at sentence ends). Only an aid to
read the text: tag from the original text when the two disagree. Google Sheets was the first
choice but GOOGLETRANSLATE resolved ~1 cell/min on this sheet, so this runs locally.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
HD = ROOT / "results" / "e1b" / "handtag"
MODEL = "facebook/nllb-200-distilled-600M"
CODE = {"ur": "urd_Arab", "lo": "lao_Laoo", "mt": "mlt_Latn", "yo": "yor_Latn"}
MAX_CHARS = 280


def segments(text: str) -> list[list[str]]:
    """lines -> list of chunks (each <= MAX_CHARS); empty line -> []."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            out.append([])
            continue
        parts = re.split(r"(?<=[.!?۔។।])\s+", line)
        chunks, cur = [], ""
        for p in parts:
            if cur and len(cur) + len(p) + 1 > MAX_CHARS:
                chunks.append(cur)
                cur = p
            else:
                cur = (cur + " " + p).strip()
        if cur:
            chunks.append(cur)
        out.append([c[i : i + MAX_CHARS] for c in chunks for i in range(0, len(c), MAX_CHARS)])
    return out


def main() -> None:
    sheet = [json.loads(l) for l in (HD / "handtag_sheet.jsonl").open(encoding="utf-8")]
    key = {k["item"]: k for k in json.load(open(HD / "handtag_key.json"))}
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(dev).eval()
    out_path = HD / "aid_en.json"
    aid = json.loads(out_path.read_text()) if out_path.exists() else {}
    eng = tok.convert_tokens_to_ids("eng_Latn")

    def translate(chunks: list[str], lang: str) -> list[str]:
        tok.src_lang = CODE[lang]
        res = []
        for i in range(0, len(chunks), 16):
            b = tok(chunks[i : i + 16], return_tensors="pt", padding=True, truncation=True, max_length=200).to(dev)
            with torch.no_grad():
                g = model.generate(**b, forced_bos_token_id=eng, max_new_tokens=200, num_beams=1)
            res += tok.batch_decode(g, skip_special_tokens=True)
        return res

    for n, r in enumerate(sheet, 1):
        if r["item"] in aid:
            continue
        lang = key[r["item"]]["lang"]
        rec = {}
        for field in ("prompt", "response"):
            segs = segments(r[field])
            flat = [c for s in segs for c in s]
            tr = iter(translate(flat, lang)) if flat else iter(())
            rec[field + "_en"] = "\n".join(" ".join(next(tr) for _ in s) for s in segs)
        aid[r["item"]] = rec
        out_path.write_text(json.dumps(aid, ensure_ascii=False))
        print(n, r["item"], lang, flush=True)


if __name__ == "__main__":
    main()
