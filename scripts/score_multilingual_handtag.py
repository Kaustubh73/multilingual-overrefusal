#!/usr/bin/env python3
"""Score E1b hand-tags against Qwen3Guard-4B and Qwen3Guard-8B judges, by language.

Reads:
  results/e1b/handtag/_shown/handtag_labels.jsonl   (your labels, last event per item wins)
  results/e1b/handtag/handtag_key.json              (item -> row_id, model)
  results/e1b/judge_qg4.jsonl                       (Qwen3Guard-4B verdicts by row_id)
  results/e1b/judge_qg8.jsonl                       (Qwen3Guard-8B verdicts by row_id)

Writes:
  results/e1b/handtag/_shown/e1b_score_report.md
  results/e1b/handtag/_shown/e1b_score_report.json
"""
import json, collections, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "results" / "e1b" / "handtag"
LABELS = H / "_shown" / "handtag_labels.jsonl"
KEY = H / "handtag_key.json"
QG4 = ROOT / "results" / "e1b" / "judge_qg4.jsonl"
QG8 = ROOT / "results" / "e1b" / "judge_qg8.jsonl"
OUT_MD = H / "_shown" / "e1b_score_report.md"
OUT_JSON = H / "_shown" / "e1b_score_report.json"

LABELS_SET = ["comply", "refuse"]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def latest_by_item(rows):
    out = {}
    for r in rows:
        k = r["item"]
        if r.get("label") in (None, "None") and k in out:
            continue
        out[k] = r
    return out


def kappa(pairs, labels):
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in labels) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def wilson(p, n, z=1.96):
    if n == 0:
        return (None, None)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - w, 3), round(c + w, 3))


def lang_of(row_id):
    return row_id.split("-")[0]


def main():
    labels = latest_by_item(jl(LABELS))
    key = {k["item"]: k for k in json.load(open(KEY))}
    qg4 = {r["row_id"]: r["verdict"] for r in jl(QG4)}
    qg8 = {r["row_id"]: r["verdict"] for r in jl(QG8)}

    # Build comparison rows: (item, lang, model, hand, qg4, qg8)
    rows = []
    for it, e in labels.items():
        k = key[it]
        row_id = k["row_id"]
        lang = lang_of(row_id)
        model = k["model"]
        hand = e["label"]
        j4 = qg4.get(row_id)
        j8 = qg8.get(row_id)
        rows.append({"item": it, "lang": lang, "model": model, "hand": hand,
                     "qg4": j4, "qg8": j8, "unsure": bool(e.get("unsure"))})

    def stats(sub, judge):
        pairs = [(r["hand"], r[judge]) for r in sub if r[judge] is not None]
        n = len(pairs)
        acc = sum(h == j for h, j in pairs) / n if n else float("nan")
        ci = wilson(acc, n) if n else (None, None)
        return {"n": n, "agree": sum(h == j for h, j in pairs), "acc": round(acc, 3),
                "ci95": ci, "kappa": round(kappa(pairs, LABELS_SET), 3)}

    res = {}
    for judge in ("qg4", "qg8"):
        valid = [r for r in rows if r[judge] is not None]
        res[judge] = {
            "n": len(rows),
            "judge_missing": len(rows) - len(valid),
            "all": stats(valid, judge),
            "excluding_unsure": stats([r for r in valid if not r["unsure"]], judge),
            "by_lang": {},
            "by_model": {},
            "confusion": collections.Counter(),
            "disagreements": [],
        }
        # by language
        for lang in sorted({r["lang"] for r in valid}):
            res[judge]["by_lang"][lang] = stats([r for r in valid if r["lang"] == lang], judge)
        # by model
        for model in sorted({r["model"] for r in valid}):
            res[judge]["by_model"][model] = stats([r for r in valid if r["model"] == model], judge)
        # confusion + disagreements
        for r in valid:
            res[judge]["confusion"][(r["hand"], r[judge])] += 1
            if r["hand"] != r[judge]:
                res[judge]["disagreements"].append({
                    "item": r["item"], "lang": r["lang"], "model": r["model"],
                    "hand": r["hand"], "judge": r[judge],
                })

    # 4B vs 8B agreement on items both judges labeled
    both = [r for r in rows if r["qg4"] is not None and r["qg8"] is not None]
    qg4_qg8 = {
        "n": len(both),
        "agree": sum(r["qg4"] == r["qg8"] for r in both),
        "acc": round(sum(r["qg4"] == r["qg8"] for r in both) / len(both), 3) if both else float("nan"),
        "kappa": round(kappa([(r["qg4"], r["qg8"]) for r in both], LABELS_SET), 3) if both else float("nan"),
        "by_lang": {},
    }
    for lang in sorted({r["lang"] for r in both}):
        sub = [r for r in both if r["lang"] == lang]
        qg4_qg8["by_lang"][lang] = {
            "n": len(sub),
            "agree": sum(r["qg4"] == r["qg8"] for r in sub),
            "acc": round(sum(r["qg4"] == r["qg8"] for r in sub) / len(sub), 3),
            "kappa": round(kappa([(r["qg4"], r["qg8"]) for r in sub], LABELS_SET), 3),
        }

    # Hand-label distribution by language
    hand_by_lang = collections.Counter((r["lang"], r["hand"]) for r in rows)

    lines = []
    lines.append(f"# E1b — Hand tags vs Qwen3Guard judges (n=180 items, 0 unsure)")
    lines.append("")
    lines.append(f"Hand-label counts (all 180): comply={sum(1 for r in rows if r['hand']=='comply')}, "
                 f"refuse={sum(1 for r in rows if r['hand']=='refuse')}")
    lines.append("")
    for judge, label in (("qg4", "Qwen3Guard-4B"), ("qg8", "Qwen3Guard-8B")):
        s = res[judge]["all"]
        lines.append(f"## {label} vs you")
        lines.append(f"- Valid judge labels: {s['n']}/{res[judge]['n']}")
        ci = s["ci95"]
        lines.append(f"- Agreement: {s['agree']}/{s['n']} = {s['acc']:.1%} "
                     f"(95% CI {ci[0]:.1%}-{ci[1]:.1%}), kappa {s['kappa']}")
        lines.append(f"- Excluding unsure: {res[judge]['excluding_unsure']['agree']}/"
                     f"{res[judge]['excluding_unsure']['n']} = "
                     f"{res[judge]['excluding_unsure']['acc']:.1%}")
        lines.append("- By language:")
        for lang in sorted(res[judge]["by_lang"]):
            v = res[judge]["by_lang"][lang]
            ci = v["ci95"]
            lines.append(f"  - {lang}: {v['agree']}/{v['n']} = {v['acc']:.1%} "
                         f"(CI {ci[0]:.1%}-{ci[1]:.1%}), κ={v['kappa']}")
        lines.append("- By model:")
        for model in sorted(res[judge]["by_model"]):
            v = res[judge]["by_model"][model]
            short = model.split("/")[-1]
            lines.append(f"  - {short}: {v['agree']}/{v['n']} = {v['acc']:.1%}, κ={v['kappa']}")
        conf = res[judge]["confusion"]
        lines.append(f"- Confusion (hand->judge): {dict(conf)}")
        if res[judge]["disagreements"]:
            lines.append(f"- Disagreements ({len(res[judge]['disagreements'])}):")
            for d in res[judge]["disagreements"]:
                lines.append(f"  - {d['item']} ({d['lang']}, {d['model'].split('/')[-1]}): "
                             f"hand={d['hand']}, judge={d['judge']}")
        lines.append("")

    lines.append("## Qwen3Guard-4B vs Qwen3Guard-8B (judge-judge agreement)")
    q = qg4_qg8
    ci = wilson(q["acc"], q["n"]) if q["n"] else (None, None)
    lines.append(f"- Items both judges labeled: {q['n']}")
    lines.append(f"- Agreement: {q['agree']}/{q['n']} = {q['acc']:.1%} "
                 f"(CI {ci[0]:.1%}-{ci[1]:.1%}), kappa {q['kappa']}")
    lines.append("- By language:")
    for lang in sorted(q["by_lang"]):
        v = q["by_lang"][lang]
        lines.append(f"  - {lang}: {v['agree']}/{v['n']} = {v['acc']:.1%}, κ={v['kappa']}")
    lines.append("")

    (OUT_MD).write_text("\n".join(lines))
    def clean_conf(conf):
        return {f"{h}|{j}": c for (h, j), c in conf.items()}

    out_json = {"judges": {k: {kk: (vv if kk != "confusion" else clean_conf(vv))
                               for kk, vv in v.items() if kk != "disagreements"}
                           for k, v in res.items()},
                "qg4_vs_qg8": qg4_qg8,
                "disagreements": {k: v["disagreements"] for k, v in res.items()},
                "hand_by_lang": {f"{l}|{h}": c for (l, h), c in hand_by_lang.items()}}
    (OUT_JSON).write_text(json.dumps(out_json, indent=1))
    print("\n".join(lines))


main()
