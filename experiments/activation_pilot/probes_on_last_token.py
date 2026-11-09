"""probes on pilot1 activations (Qwen3.5-9B, last prompt token) + refusal table from e1b. Pilot, descriptive only.
  python experiments/activation_pilot/probes_on_last_token.py   (run from repo root)
Uses 4 prompts only (2 harmful, 2 harmless): held-out-prompt splits have very few groups.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os
PILOT = os.environ.get("PILOT", "results/activation_pilot/pilot1"); FIGDIR = os.environ.get("FIGDIR", "figures/activation_pilot/pilot"); MODEL = os.environ.get("MODEL_LABEL", "Qwen3.5-9B"); TAG = os.environ.get("TAG", "")
R = Path("results/activation_pilot"); P1 = Path(PILOT)
LAYERS = [8, 16, 24, 32]
SEED = 0
SCRIPT = {"ar": "Arabic", "ur": "Arabic", "hi": "Devanagari", "ne": "Devanagari", "ja": "Han", "zh": "Han", "th": "Thai", "lo": "Lao"}

rows = sorted((json.loads(l) for l in open(P1 / "pilot.jsonl")), key=lambda r: r["row_id"])
X = np.stack([np.load(P1 / "acts" / f"{r['row_id']}.npz")["last_prompt"].astype(np.float32) for r in rows])
task = np.array([r["task"] for r in rows]); lang = np.array([r["lang"] for r in rows])
cid = np.array([r["content_id"] for r in rows]); grp = np.array([r["group"] for r in rows])
script = np.array([SCRIPT.get(l, "Latin") for l in lang])


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300, random_state=SEED))


def cv_acc(Z, y, groups, splitter):
    hit = []
    for tr, te in splitter.split(Z, y, groups):
        if len(set(y[tr])) < 2:
            continue
        m = clf().fit(Z[tr], y[tr]); hit.extend(m.predict(Z[te]) == y[te])
    return round(float(np.mean(hit)), 3)


res = {"model": os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B"), "n": len(rows), "chance": {"task": round(1 / len(set(task)), 3), "language": round(1 / len(set(lang)), 3), "script": round(1 / len(set(script)), 3), "group": 0.5}, "layers": {}}
for L in LAYERS:
    Z = X[:, L]
    out = {}
    # held-out prompt: leave one prompt out (4 folds)
    lop = LeaveOneGroupOut()
    out["task_heldout_prompt"] = cv_acc(Z, task, cid, lop)
    out["language_heldout_prompt"] = cv_acc(Z, lang, cid, lop)
    # task probe generalising to unseen languages (leave one language out, 16 folds)
    out["task_heldout_language"] = cv_acc(Z, task, lang, LeaveOneGroupOut())
    # language probe generalising to unseen tasks
    out["language_heldout_task"] = cv_acc(Z, lang, task, LeaveOneGroupOut())
    # script probe across held-out language (script mostly shared inside a pair, so lang held out removes one member)
    out["script_heldout_language"] = cv_acc(Z, script, lang, LeaveOneGroupOut())
    # harmful vs harmless, hold out one harmful + one harmless prompt (4 splits)
    hv, hl = sorted(set(cid[grp == "harmful"])), sorted(set(cid[grp == "harmless"]))
    hit = []
    for a in hv:
        for b in hl:
            te = (cid == a) | (cid == b)
            m = clf().fit(Z[~te], grp[~te]); hit.extend(m.predict(Z[te]) == grp[te])
    out["harmful_vs_harmless_heldout_pair"] = round(float(np.mean(hit)), 3)
    # harmful-minus-harmless mean direction per task: cosine agreement across tasks and across languages
    def dirs(key):
        d = {}
        for v in sorted(set(key)):
            m = key == v
            d[v] = Z[m & (grp == "harmful")].mean(0) - Z[m & (grp == "harmless")].mean(0)
        return d
    for name, key in (("task", task), ("language", lang)):
        d = dirs(key); V = np.stack(list(d.values())); V = V / np.linalg.norm(V, axis=1, keepdims=True)
        C = V @ V.T; out[f"harm_dir_cos_across_{name}"] = round(float(C[np.triu_indices(len(V), 1)].mean()), 3)
    res["layers"][L] = out
    print(L, out, flush=True)
json.dump(res, open(R / f"probes_on_last_token{TAG}.json", "w"), indent=1)

# figure: probe accuracy by layer
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, lab in [("task_heldout_prompt", "task, held-out prompt"), ("task_heldout_language", "task, held-out language"),
               ("language_heldout_prompt", "language, held-out prompt"), ("language_heldout_task", "language, held-out task"), ("script_heldout_language", "script, held-out language")]:
    ax[0].plot(LAYERS, [res["layers"][L][k] for L in LAYERS], "o-", label=lab)
ax[0].axhline(res["chance"]["task"], color="grey", ls=":", lw=1); ax[0].text(8, res["chance"]["task"] + .01, "task chance 0.10", fontsize=7, color="grey")
ax[0].set_xlabel("layer"); ax[0].set_ylabel("probe accuracy"); ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=7, frameon=False)
ax[0].set_title("Linear probes, last prompt token", fontsize=10)
ax[1].plot(LAYERS, [res["layers"][L]["harmful_vs_harmless_heldout_pair"] for L in LAYERS], "o-", color="#d62728", label="harmful vs harmless probe")
ax[1].plot(LAYERS, [res["layers"][L]["harm_dir_cos_across_task"] for L in LAYERS], "s-", color="#1f77b4", label="cos of harm direction across tasks")
ax[1].plot(LAYERS, [res["layers"][L]["harm_dir_cos_across_language"] for L in LAYERS], "^-", color="#2ca02c", label="cos of harm direction across languages")
ax[1].axhline(.5, color="grey", ls=":", lw=1); ax[1].set_xlabel("layer"); ax[1].set_ylim(-.1, 1.05); ax[1].legend(fontsize=7, frameon=False)
ax[1].set_title("Harmful vs harmless (2+2 prompts only)", fontsize=10)
fig.suptitle(MODEL + ", " + os.path.basename(PILOT) + " (640 rows), pilot data, not a finding", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/probes_on_last_token.png", dpi=150); plt.close(fig)

if TAG:  # the e1b refusal table below is Qwen-only data
    raise SystemExit(0)
# 6: refusal rate task x language from e1b, qg4 labels
E = [r for r in csv.DictReader(open(R / "response_length.csv")) if r["src"] == "e1b"]
tasks = sorted({r["task"] for r in E}); langs = sorted({r["lang"] for r in E})
cnt = defaultdict(lambda: [0, 0])
for r in E:
    c = cnt[(r["task"], r["lang"])]; c[0] += r["judge"] == "refuse"; c[1] += 1
M = np.array([[cnt[(t, l)][0] / cnt[(t, l)][1] for l in langs] for t in tasks])
N = np.array([[cnt[(t, l)][1] for l in langs] for t in tasks])
with open(R / "refusal_task_lang.csv", "w") as f:
    w = csv.writer(f); w.writerow(["task"] + [f"{l}_refuse_rate" for l in langs] + [f"{l}_n" for l in langs])
    for i, t in enumerate(tasks):
        w.writerow([t] + [round(x, 3) for x in M[i]] + list(N[i]))
fig, ax = plt.subplots(figsize=(6, 4.6))
im = ax.imshow(M, cmap="Reds", vmin=0, vmax=.6)
ax.set_xticks(range(len(langs))); ax.set_xticklabels(langs); ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks)
for i in range(len(tasks)):
    for j in range(len(langs)):
        ax.text(j, i, f"{M[i, j]:.0%}\n(n={N[i, j]})", ha="center", va="center", fontsize=7)
ax.set_title("Refuse rate (qg4) task x language, e1b\ngarble not separated; pilot data, not a finding", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/refusal_task_lang.png", dpi=150); plt.close(fig)
print("refusal table task rows:", dict(zip(tasks, M.mean(1).round(2))))
