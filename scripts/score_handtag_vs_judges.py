#!/usr/bin/env python3
"""Score hand-tags against the stage-2 judges (gpt-oss policy judge, Qwen3Guard).

  python scripts/score_handtag_vs_judges.py [--run-dir runs/handtag-stage2-english-2026-10]

Reads handtag_labels.jsonl (last event per item wins), handtag_key.json (item -> model/content/sample),
stage2_judge.jsonl and stage2_qwen3guard.jsonl. Read-only on those; writes score_report.md/.json in run-dir.
"""
import argparse, json, collections, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "runs" / "harmful-screen-2026-09"

def jl(p): return [json.loads(l) for l in open(p) if l.strip()]

def latest_by_key(rows):
    out = {}
    for r in rows:  # later rows override, but a None tag never overrides a valid tag
        k = (r["model_id"], r["content_id"], str(r["sample"]))
        if r.get("tag") in (None, "None") and k in out: continue
        out[k] = r
    return out

def kappa(pairs, labels):
    n = len(pairs)
    if not n: return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in labels) / n**2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "handtag-stage2-english-2026-10")
    a = ap.parse_args(); D = a.run_dir
    ev = {}
    for e in jl(D / "handtag_labels.jsonl"): ev[e["item"]] = e
    key = {k["item"]: k for k in json.load(open(D / "handtag_key.json"))}
    judges = {"gpt-oss": latest_by_key(jl(STAGE / "stage2_judge.jsonl")),
              "qwen3guard": latest_by_key(jl(STAGE / "stage2_qwen3guard.jsonl"))}
    LAB = ["comply", "refuse", "garble"]
    res = {}; lines = []
    n_items = len(ev); n_unsure = sum(bool(e.get("unsure")) for e in ev.values())
    hand_dist = collections.Counter(e["label"] for e in ev.values())
    lines += [f"# Hand-tag vs judges ({n_items} items, {n_unsure} marked unsure)", "", f"Hand-label counts: {dict(hand_dist)}", ""]
    for jn, jd in judges.items():
        rows = []
        for it, e in ev.items():
            k = key[it]; j = jd.get((k["model_id"], k["content_id"], str(k["sample"])))
            tag = j["tag"] if j else None
            rows.append((it, k["model_id"], e["label"], tag, bool(e.get("unsure"))))
        valid = [r for r in rows if r[3] in LAB]
        res[jn] = {"n": len(rows), "judge_missing_or_none": len(rows) - len(valid)}
        def stats(sub):
            pairs = [(r[2], r[3]) for r in sub]
            acc = sum(h == j for h, j in pairs) / len(pairs) if pairs else float("nan")
            # Wilson 95% CI
            n = len(pairs); p = acc
            if n:
                z = 1.96; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; w = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
                ci = (round(c-w, 3), round(c+w, 3))
            else: ci = (None, None)
            return {"n": n, "agree": sum(h == j for h, j in pairs), "acc": round(acc, 3), "ci95": ci, "kappa": round(kappa(pairs, LAB), 3)}
        res[jn]["all"] = stats(valid)
        res[jn]["excluding_unsure"] = stats([r for r in valid if not r[4]])
        res[jn]["by_model"] = {m: stats([r for r in valid if r[1] == m]) for m in sorted({r[1] for r in valid})}
        conf = collections.Counter((r[2], r[3]) for r in valid)
        res[jn]["confusion(hand->judge)"] = {f"{h}->{j}": c for (h, j), c in sorted(conf.items())}
        res[jn]["disagreements"] = [{"item": r[0], "model": r[1], "hand": r[2], "judge": r[3], "unsure": r[4]} for r in valid if r[2] != r[3]]
        s = res[jn]["all"]
        lines += [f"## {jn}", f"- Valid judge labels: {len(valid)}/{len(rows)}",
                  f"- Agreement with you: {s['agree']}/{s['n']} = {s['acc']:.1%} (95% CI {s['ci95'][0]:.1%}-{s['ci95'][1]:.1%}), kappa {s['kappa']}",
                  f"- Excluding your unsure items: {res[jn]['excluding_unsure']['agree']}/{res[jn]['excluding_unsure']['n']} = {res[jn]['excluding_unsure']['acc']:.1%}",
                  "- By model: " + "; ".join(f"{m.split('/')[-1]} {v['agree']}/{v['n']}" for m, v in res[jn]["by_model"].items()),
                  f"- Confusion (you->judge): {res[jn]['confusion(hand->judge)']}", ""]
    (D / "score_report.json").write_text(json.dumps(res, indent=1)); (D / "score_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
main()
