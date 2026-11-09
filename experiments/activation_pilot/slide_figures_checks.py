"""Slide figures for the batching, garble, length and PCA checks (pilot data, descriptive only).
  python experiments/activation_pilot/slide_figures_checks.py   -> figures/activation_pilot/pilot/xNN_*.png
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results/activation_pilot")
OUT = Path("figures/activation_pilot/pilot"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C4, C8, GREY, RED = "#1f77b4", "#2ca02c", "#999999", "#d62728"
NOTE = "pilot data, not a finding"


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name); plt.close(fig)


def labels(ax, bars, fmt="{:.0%}"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, fmt.format(b.get_height()), ha="center", fontsize=9)


# ---- 01: hand vs judges, 60-row pilot
hv = json.load(open(R / "handtag_vs_judges.json"))
key = json.load(open(R / "handtag_pilot_key.json"))["key"]
rows = [(i, hv["hand"][i], key[i]["qg4"], hv["qg8"][i], key[i]["group"]) for i in hv["hand"]]
groups = ["all", "harmful", "harmless"]
fig, ax = plt.subplots(figsize=(6, 3.8))
for k, (idx, col, lab) in enumerate([(2, C4, "Qwen3Guard-4B"), (3, C8, "Qwen3Guard-8B")]):
    vals = []
    for g in groups:
        s = [r for r in rows if g == "all" or r[4] == g]
        vals.append(np.mean([r[1] == r[idx] for r in s]))
    b = ax.bar(np.arange(3) + (k - .5) * .36, vals, .36, color=col, label=lab); labels(ax, b)
ns = [len(rows)] + [sum(r[4] == g for r in rows) for g in groups[1:]]
ax.set_xticks(range(3)); ax.set_xticklabels([f"{g} (n={n})" for g, n in zip(groups, ns)])
ax.set_ylim(0, 1.2); ax.set_ylabel("agreement with hand tag"); ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
ax.set_title(f"Judges vs hand tags, 60-row pilot ({NOTE})", fontsize=10)
save(fig, "handtag_vs_judges.png")

# ---- 02: judge agreement by language, correct join (180 rows, garble forced)
feat = list(csv.DictReader(open(R / "garble_features.csv")))
langs = ["mt", "ur", "yo", "lo"]
fig, ax = plt.subplots(figsize=(6, 3.8))
for k, (j, col, lab) in enumerate([("qg4", C4, "Qwen3Guard-4B"), ("qg8", C8, "Qwen3Guard-8B")]):
    vals = [np.mean([r[j] == r["hand"] for r in feat if r["lang"] == l]) for l in langs]
    b = ax.bar(np.arange(4) + (k - .5) * .36, vals, .36, color=col, label=lab); labels(ax, b)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{l} (n={sum(r['lang'] == l for r in feat)})" for l in langs])
ax.set_ylim(0, 1.2); ax.set_ylabel("agreement with hand tag"); ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
ax.set_title(f"Judge agreement by language, 180 rows, garble counted\n({NOTE})", fontsize=10)
save(fig, "judge_agree_by_lang.png")

# ---- 03: Lao disagreements
lao = list(csv.DictReader(open(R / "lao_review.csv")))
cat = Counter()
for r in lao:
    if r["hand"] == "garble":
        cat["hand garble\n(judge has no garble class)"] += 1
    else:
        j = "both judges" if r["qg4"] == r["qg8"] else ("only qg4" if r["qg4"] != r["hand"] else "only qg8")
        cat[f"hand {r['hand']},\njudge refuse: {j}"] += 1
fig, ax = plt.subplots(figsize=(7, 3.8))
ks = sorted(cat, key=lambda k: -cat[k])
b = ax.barh(ks, [cat[k] for k in ks], color=[GREY if "garble" in k else RED for k in ks])
for bb in b:
    ax.text(bb.get_width() + .1, bb.get_y() + bb.get_height() / 2, int(bb.get_width()), va="center")
ax.invert_yaxis(); ax.set_xlabel("items")
ax.set_title(f"Lao: {len(lao)} of 45 items where a judge differs from hand\n({NOTE})", fontsize=10)
save(fig, "lao_disagreements.png")

# ---- 04: garble filter sweep (hand: 17 garble, 152 comply, 11 refuse)
z = [0.2, 0.25, 0.3, 0.35, 0.4]
caught = [7, 12, 14, 16, 17]; fp = [2, 11, 24, 31, 45]
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(z, np.array(caught) / 17, "o-", color=C8, label="garble caught (of 17)")
ax.plot(z, np.array(fp) / 152, "s-", color=RED, label="comply wrongly flagged (of 152)")
ax.axvline(0.25, color=GREY, ls="--", lw=1); ax.text(.252, .02, "used", color=GREY, fontsize=8)
ax.set_xlabel("zlib ratio threshold (repetition rule, n_chars>=100)"); ax.set_ylabel("fraction")
ax.set_ylim(0, 1.05); ax.legend(frameon=False, loc="upper left")
ax.set_title(f"Garble filter trade-off, in-sample on 180 hand tags", fontsize=10)
save(fig, "garble_filter_sweep.png")

# ---- 05/06: response length (e1b, n=720)
L = [r for r in csv.DictReader(open(R / "response_length.csv")) if r["src"] == "e1b"]


def agg(keyname):
    d = defaultdict(list)
    for r in L:
        d[r[keyname]].append(r)
    return d


tasks = agg("task")
order = sorted(tasks, key=lambda t: np.median([int(r["tokens"]) for r in tasks[t]]))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
a1.barh(order, [np.median([int(r["tokens"]) for r in tasks[t]]) for t in order], color=C4)
a1.set_xlabel("median response tokens"); a1.set_title("Length by task", fontsize=10)
a2.barh(order, [np.mean([r["hit"] == "True" for r in tasks[t]]) for t in order], color=RED)
a2.set_xlabel("hit rate (>=500 tokens)"); a2.set_title("Truncation proxy by task", fontsize=10)
fig.suptitle(f"e1b, n=720, 4 models x 4 langs ({NOTE})", fontsize=9)
save(fig, "length_by_task.png")

models = sorted(agg("model")); langs = sorted(agg("lang"))
M = np.array([[np.mean([r["hit"] == "True" for r in L if r["model"] == m and r["lang"] == l]) for l in langs] for m in models])
fig, ax = plt.subplots(figsize=(5.8, 3.6))
im = ax.imshow(M, cmap="Reds", vmin=0, vmax=.6)
ax.set_xticks(range(len(langs))); ax.set_xticklabels(langs); ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
for i in range(len(models)):
    for j in range(len(langs)):
        ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=10)
ax.set_title(f"Hit >=500 tokens, model x language (n=45 each; {NOTE})", fontsize=9)
save(fig, "truncation_model_lang.png")

# ---- 07: PCA, layer 16 by task + silhouette by layer
pca = json.load(open(R / "pca_points" / "pca_language_and_task.json"))
from sklearn.decomposition import PCA  # noqa: E402
pil = sorted([json.loads(l) for l in open(R / "pilot1" / "pilot.jsonl")], key=lambda r: r["row_id"])
pil = [r for r in pil if r["group"] == "harmful"]
X = np.stack([np.load(R / "pilot1" / "acts" / f"{r['row_id']}.npz")["last_prompt"][16].astype(np.float32) for r in pil])
cid = np.array([r["content_id"] for r in pil])
for c in set(cid):
    X[cid == c] -= X[cid == c].mean(0)
P = PCA(2).fit_transform(X)
tk = np.array([r["task"] for r in pil])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
cmap = plt.get_cmap("tab10")
for k, t in enumerate(sorted(set(tk))):
    m = tk == t; a1.scatter(P[m, 0], P[m, 1], s=14, color=cmap(k), label=t)
a1.set_xlabel("PC1"); a1.set_ylabel("PC2"); a1.legend(fontsize=6, ncol=2, frameon=False)
a1.set_title("Layer 16, last prompt token, coloured by task", fontsize=10)
Ls = list(pca["layers"]); w = .2
for k, (s, lab) in enumerate([("task", "task"), ("language", "language"), ("script", "script"), ("refusal", "refusal")]):
    a2.bar(np.arange(len(Ls)) + (k - 1.5) * w, [pca["layers"][l][f"silhouette_{s}"] for l in Ls], w, label=lab)
a2.axhline(0, color="k", lw=.6); a2.set_xticks(range(len(Ls))); a2.set_xticklabels([f"L{l}" for l in Ls])
a2.set_ylabel("silhouette (PCA-10)"); a2.legend(fontsize=8, frameon=False)
a2.set_title(f"Silhouette by layer (only {pca['refused']} of {pca['n']} refused)", fontsize=10)
fig.suptitle(f"Activation PCA, harmful prompts, prompt mean removed ({NOTE})", fontsize=9)
save(fig, "pca_task_clusters.png")

# ---- 08: batched vs unbatched
B = [json.loads(l) for l in open(R / "batching_check.jsonl")]
mods = sorted({b["model"] for b in B})
fig, ax = plt.subplots(figsize=(6, 3.6))
for k, m in enumerate(mods):
    s = [b for b in B if b["model"] == m]
    same = sum(b["same_text"] in (True, "True") for b in s)
    lab = sum(b["label_batched"] == b["label_unbatched"] for b in s)
    ax.bar(np.arange(2) + (k - .5) * .36, [same / len(s), lab / len(s)], .36, label=f"{m} (n={len(s)})")
ax.set_xticks([0, 1]); ax.set_xticklabels(["identical text", "same judge label"])
ax.set_ylim(0, 1.1); ax.legend(frameon=False, loc="upper left", fontsize=9); ax.set_ylabel("fraction")
ax.set_title(f"Batched vs unbatched generation ({NOTE})", fontsize=10)
save(fig, "batched_vs_unbatched.png")

# ---- 09: translated task-template shells, auto flags
S = list(csv.DictReader(open(R / "template_shell_checks.csv")))
fl = Counter(s["auto_flags"] or "ok" for s in S)
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ks = ["ok", "identical_to_en", "length_ratio"]
b = ax.bar(ks, [fl.get(k, 0) for k in ks], color=[C8, RED, RED])
for bb in b:
    ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + .5, int(bb.get_height()), ha="center")
ax.set_ylabel("template shells"); ax.set_title(f"Auto-check of {len(S)} translated task shells (machine aid only)", fontsize=10)
save(fig, "shell_flags.png")
print(*sorted(p.name for p in OUT.glob("*.png")), sep="\n")
