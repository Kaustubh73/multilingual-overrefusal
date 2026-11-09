"""Slide figures for the 4-prompt pilot (2 harmful + 2 harmless x 16 langs x 10 tasks, Qwen3.5-9B). Pilot data, not findings.
  python experiments/activation_pilot/slide_figures_pilot.py -> figures/activation_pilot/pilot/*.png (prompt selection, refusal heatmaps, PCA per layer)"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA

import os
PILOT = os.environ.get("PILOT", "results/activation_pilot/pilot1"); FIGDIR = os.environ.get("FIGDIR", "figures/activation_pilot/pilot"); MODEL = os.environ.get("MODEL_LABEL", "Qwen3.5-9B"); TAG = os.environ.get("TAG", "")
JUDGE = os.environ.get("JUDGE", "qg8"); JLAB = {"qg4": "Qwen3Guard-4B", "qg8": "Qwen3Guard-8B"}[JUDGE]
R = Path(PILOT); OUT = Path(FIGDIR); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
NOTE = "pilot data (n=4 prompts), not a finding"
NAME = {"c_08daf70eed01": "Harmful A: Agent Orange hoax article", "c_8c392956ac8f": "Harmful B: convince daughter, poaching",
        "c_0e4a8b733b9b": "Harmless A: essay on public transport", "c_b51530340549": "Harmless B: two data-mining tools"}
SHORT = {"c_08daf70eed01": "Harmful A", "c_8c392956ac8f": "Harmful B", "c_0e4a8b733b9b": "Harmless A", "c_b51530340549": "Harmless B"}
HARM = ["c_08daf70eed01", "c_8c392956ac8f"]
ORDER = list(NAME)

rows = pd.DataFrame([json.loads(l) for l in open(R / "pilot.jsonl")])
j = pd.DataFrame([json.loads(l) for l in open(R / f"judge_{JUDGE}.jsonl")])[["row_id", "verdict"]]
d = rows.merge(j, on="row_id"); d["ref"] = (d.verdict == "refuse").astype(int); d["trunc"] = d.done_reason == "length"
langs = sorted(d.lang.unique(), key=lambda l: (l != "en", l)); tasks = list(dict.fromkeys(d.task))

# ---- 13: prompt selection funnel
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6)
steps = [("861 English harmful seeds", "e3 stage 1: greedy,\nregex refusal, 6 models\n(Gemma-2/3/4, Llama,\nQwen3, Aya-23)"),
         ("Top 10 by mixedness", "e3 stage 2: 10 samples\nx T=1.0, Qwen3Guard-4B;\np(1-p) over\nGemma3 + Gemma4"),
         ("Gate on Qwen3.5-9B", "3 of the 10 tried,\n10 samples each:\nrefused 8/10, 10/10, 4/10;\nthe 10/10 prompt dropped"),
         ("2 harmful + 2 harmless", "harmful A (8/10 refused),\nB (4/10 refused);\n2 harmless controls,\nno regex refusal")]
xs = [0.1, 2.85, 5.6, 8.35]
for i, ((t, s), x) in enumerate(zip(steps, xs)):
    ax.add_patch(FancyBboxPatch((x, 2.1), 2.5, 1.9, boxstyle="round,pad=0.05", fc=["#dbe9f6", "#c4dcf0", "#a9cde9", "#f6d9a8"][i], ec="#444"))
    ax.text(x + 1.25, 3.6, t, ha="center", va="center", weight="bold", fontsize=10.5)
    ax.text(x + 1.25, 2.7, s, ha="center", va="center", fontsize=9)
    if i < 3: ax.annotate("", xy=(xs[i + 1] - .05, 3.05), xytext=(x + 2.55, 3.05), arrowprops=dict(arrowstyle="->", lw=1.6))
ax.text(5.5, 1.5, "Then: each prompt x 16 languages x 10 tasks = 640 generations (greedy, 512 tok) + hidden states at all 33 layers saved",
        ha="center", fontsize=10, weight="bold")
for k, c in enumerate(ORDER):
    ax.text(0.2, 1.0 - k * .25, f"{SHORT[c]}:  {NAME[c].split(': ')[1]}", fontsize=8.5, family="monospace")
ax.text(10.9, 0.05, NOTE, ha="right", fontsize=8, color="#666")
fig.tight_layout(); fig.savefig(OUT / "prompt_selection.png"); plt.close(fig)

# ---- 14: refusal heatmaps, lang x task, per prompt
fig, axs = plt.subplots(2, 2, figsize=(12, 9.6))
for ax, c in zip(axs.flat, ORDER):
    s = d[d.content_id == c]
    M = s.pivot(index="lang", columns="task", values="ref").loc[langs, tasks]
    T = s.pivot(index="lang", columns="task", values="trunc").loc[langs, tasks]
    ax.imshow(M.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    for (y, x), v in np.ndenumerate(T.values):
        if v: ax.text(x, y, "•", ha="center", va="center", fontsize=9, color="#555" if not M.values[y, x] else "white")
    ax.set_xticks(range(len(tasks)), [t.replace("_", " ") for t in tasks], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(langs)), langs, fontsize=8)
    ax.set_title(f"{NAME[c]}\nrefused {int(s.ref.sum())}/160 ({s.ref.mean():.0%})", fontsize=10)
    for sp in ax.spines.values(): sp.set_visible(False)
fig.suptitle(f"Refusal by language x task, {MODEL}, {JLAB} verdict (red = refuse, • = output hit 512-token cap)  [{NOTE}]", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "refusal_heatmaps.png"); plt.close(fig)

# ---- 15: comparisons by language and by task
fig, (a, b, c3) = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.5, 1.1, .8]})
col = {"c_08daf70eed01": "#c0392b", "c_8c392956ac8f": "#e59866", "c_0e4a8b733b9b": "#2e86c1", "c_b51530340549": "#85c1e9"}
y = np.arange(len(langs))
for c in ORDER:
    v = d[d.content_id == c].groupby("lang").ref.mean().reindex(langs)
    a.scatter(v.values, y, s=55, color=col[c], label=SHORT[c], alpha=.9, zorder=3, marker="o" if c in HARM else "s")
a.set_yticks(y, langs); a.invert_yaxis(); a.set_xlabel("refusal rate over 10 tasks"); a.grid(axis="x", alpha=.3)
a.set_xlim(-.05, 1.05); a.legend(frameon=False, fontsize=8, loc="lower right"); a.set_title("By language (each dot = 10 generations)", fontsize=10)
for c in ORDER:
    v = d[d.content_id == c].groupby("task").ref.mean().reindex(tasks)
    b.plot(v.values, range(len(tasks)), "o-" if c in HARM else "s--", color=col[c], label=SHORT[c], lw=1.3)
b.set_yticks(range(len(tasks)), [t.replace("_", " ") for t in tasks]); b.invert_yaxis()
b.set_xlabel("refusal rate over 16 languages"); b.grid(axis="x", alpha=.3); b.set_title("By task (each point = 16 generations)", fontsize=10)
v = [d[d.content_id == c].ref.mean() for c in ORDER]
bars = c3.bar(["H-A", "H-B", "L-A", "L-B"], v, color=[col[c] for c in ORDER])
for bb in bars: c3.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + .005, f"{bb.get_height():.0%}", ha="center", fontsize=9)
c3.set_ylabel("overall refusal rate"); c3.set_title("Overall (n=160)\nH=harmful, L=harmless", fontsize=10)
fig.suptitle(f"Same prompts across languages and tasks, {MODEL}, {JLAB} verdict  [{NOTE}]", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "refusal_lang_task_compare.png"); plt.close(fig)

# ---- 16: PCA at all layers (last prompt token, 640 rows)
rows = rows.sort_values("row_id").reset_index(drop=True)
A = np.stack([np.load(R / "acts" / f"{r}.npz")["last_prompt"] for r in rows.row_id]).astype(np.float32)  # [640, 33, 4096]
lang_c = {l: plt.get_cmap("tab20")(i) for i, l in enumerate(langs)}
task_c = {t: plt.get_cmap("tab10")(i) for i, t in enumerate(tasks)}
grp_c = {"harmful": "#c0392b", "harmless": "#2e86c1"}
spec = [("task", task_c), ("lang", lang_c), ("group", grp_c)]


def pcs(L):
    X = A[:, L] - A[:, L].mean(0)
    return PCA(2).fit_transform(X)


cols = [0, 4, 8, 12, 16, 20, 24, 28, 32]
fig, axs = plt.subplots(3, len(cols), figsize=(2.2 * len(cols), 7.4))
for ci, L in enumerate(cols):
    P = pcs(L)
    for ri, (k, cm) in enumerate(spec):
        ax = axs[ri, ci]; v = rows[k].values
        ax.scatter(P[:, 0], P[:, 1], c=[cm[x] for x in v], s=5, alpha=.7, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0: ax.set_title(f"layer {L}", fontsize=10)
        if ci == 0: ax.set_ylabel(f"coloured by {k}", fontsize=10)
for ri, (k, cm) in enumerate(spec):
    if k == "lang": continue
    axs[ri, -1].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=str(n).replace("_", " ")) for n, c in cm.items()],
                       fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
fig.suptitle(f"PCA of last-prompt-token hidden state, 640 rows, {MODEL}, layers 0-32 in steps of 4  [{NOTE}]", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "pca_all_layers.png"); plt.close(fig)

# appendix: every layer, coloured by task and by prompt
fig, axs = plt.subplots(5, 7, figsize=(17, 12.5))
pc = {c: i for i, c in enumerate(ORDER)}
for L, ax in zip(range(33), axs.flat):
    P = pcs(L)
    ax.scatter(P[:, 0], P[:, 1], c=[task_c[t] for t in rows.task], s=3, alpha=.7, linewidths=0)
    ax.set_title(f"layer {L}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
for ax in list(axs.flat)[33:]: ax.axis("off")
axs.flat[34].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=t.replace("_", " ")) for t, c in task_c.items()], fontsize=7, frameon=False, loc="center")
fig.suptitle(f"PCA at every layer, coloured by task  [{NOTE}]", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "pca_every_layer_task.png"); plt.close(fig)
print("ok")
