#!/usr/bin/env python3
"""Back-translate every localized template to English (Google Translate gtx endpoint) for eyeballing.

  .venv/bin/python translation_qa/template_qa.py
Reads data/dataset_v2/templates_localized.json (read-only); writes results/template_qa/backtranslation.csv.
Placeholders are swapped for plain tokens before translation and restored after.
"""
import csv, json, time, urllib.parse, urllib.request, difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = json.load(open(ROOT / "data/dataset_v2/templates_localized.json"))
PH = {"{content}": "ZZCONTENTZZ", "{translate_target_name}": "ZZTARGETZZ"}

def gt(text, src):
    q = urllib.parse.urlencode({"client": "gtx", "sl": src, "tl": "en", "dt": "t", "q": text})
    for i in range(4):
        try:
            r = urllib.request.urlopen("https://translate.googleapis.com/translate_a/single?" + q, timeout=20)
            return "".join(s[0] for s in json.load(r)[0])
        except Exception:
            time.sleep(2 * (i + 1))
    return "ERROR"

rows = []
for lang in T["langs"]:
    if lang == "en":
        continue
    for task in T["tasks"]:
        if task == "no_task":
            continue
        tpl = T["templates"][lang][task]
        p = tpl
        for k, v in PH.items():
            p = p.replace(k, v)
        bt = gt(p, "zh-CN" if lang == "zh" else lang)
        for k, v in PH.items():
            bt = bt.replace(v, k)
        en = T["templates"]["en"][task]
        miss = [k for k in PH if k in tpl and k not in bt]
        rows.append(dict(lang=lang, task=task, template=tpl, back_en=bt, en_ref=en,
                         sim=round(difflib.SequenceMatcher(None, en.lower(), bt.lower()).ratio(), 2),
                         placeholder_lost=",".join(miss)))
        time.sleep(0.3)
out = ROOT / "results/template_qa/backtranslation.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "rows ->", out)
