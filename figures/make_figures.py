#!/usr/bin/env python3
"""Generate varied, publication-ready figures for multilingual over-refusal slides."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("white")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
OUT = Path("figures"); OUT.mkdir(exist_ok=True)

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

# ============================================================
# FIGURE 1 — Judge accuracy (E1): grouped bar + kappa annotation
# ============================================================
fig, ax = plt.subplots(figsize=(5.4, 3.4))
judges = ["Qwen3Guard-4B", "Qwen3Guard-8B", "gpt-oss-20b"]
acc = [0.94, 0.93, 0.667]
kappa = [0.873, 0.847, 0.073]
colors = ["#2ca02c", "#1f77b4", "#d62728"]
bars = ax.bar(judges, acc, color=colors, width=0.55, edgecolor="white", linewidth=1)
ax.set_ylim(0, 1.08); ax.set_ylabel("Accuracy vs hand tags (n=100)")
ax.set_title("E1 — Judge validation on English hand tags", fontsize=12, fontweight="bold", pad=12)
for b, a, k in zip(bars, acc, kappa):
    ax.text(b.get_x()+b.get_width()/2, a+0.03, f"{a:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(b.get_x()+b.get_width()/2, a-0.09, f"κ={k:.2f}", ha="center", va="top", fontsize=9, color="#555")
ax.axhline(0.90, color="#d62728", ls="--", lw=1, alpha=0.7)
ax.text(0.02, 0.91, "0.90 bar", color="#d62728", fontsize=8, transform=ax.get_yaxis_transform())
plt.tight_layout(); plt.savefig(OUT / "slide_01_judge_accuracy.png"); plt.close()

# ============================================================
# FIGURE 2 — E1b hand-tag vs judge by language: diverging paired bars
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
langs = ["Lao (lo)", "Maltese (mt)", "Urdu (ur)", "Yoruba (yo)"]
acc4 = [0.378, 0.889, 0.889, 0.844]
acc8 = [0.356, 0.889, 0.867, 0.844]
x = np.arange(len(langs)); w = 0.36
axes[0].bar(x - w/2, acc4, w, label="Qwen3Guard-4B", color="#1f77b4", edgecolor="white")
axes[0].bar(x + w/2, acc8, w, label="Qwen3Guard-8B", color="#ff7f0e", edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(langs, rotation=12, ha="right", fontsize=9)
axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Agreement with hand tags")
axes[0].set_title("Judge vs your labels", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=8, loc="lower right")
for i, (a4, a8) in enumerate(zip(acc4, acc8)):
    axes[0].text(i - w/2, a4+0.02, f"{a4:.0%}", ha="center", fontsize=9, fontweight="bold")
    axes[0].text(i + w/2, a8+0.02, f"{a8:.0%}", ha="center", fontsize=9, fontweight="bold")

jj = [0.711, 1.0, 0.978, 1.0]
jj_k = [0.414, float("nan"), 0.789, 1.0]
bars = axes[1].bar(langs, jj, color="#2ca02c", width=0.5, edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(langs, rotation=12, ha="right", fontsize=9)
axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("4B = 8B agreement")
axes[1].set_title("Judge–judge agreement", fontsize=11, fontweight="bold")
for i, (v, k) in enumerate(zip(jj, jj_k)):
    txt = f"{v:.0%}" + (f"\nκ={k:.2f}" if k == k else "")
    axes[1].text(i, v+0.02, txt, ha="center", fontsize=9, fontweight="bold")
fig.suptitle("E1b — Qwen3Guard vs your 180 hand tags, by language", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(OUT / "slide_02_e1b_by_lang.png"); plt.close()

# ============================================================
# FIGURE 3 — E3 refusal rate by model: horizontal bar, sorted
# ============================================================
fig, ax = plt.subplots(figsize=(5.6, 3.4))
models = ["Gemma-2 9B", "Gemma4 E4B", "Llama-3.1 8B", "Gemma3 12B", "Qwen3 8B", "Aya 23-8B"]
rates = [0.983, 0.965, 0.92, 0.862, 0.836, 0.62]
order = np.argsort(rates)
models = [models[i] for i in order]
rates = [rates[i] for i in order]
colors = ["#d62728" if r < 0.8 else "#ff7f0e" if r < 0.95 else "#2ca02c" for r in rates]
bars = ax.barh(models, rates, color=colors, height=0.6, edgecolor="white")
ax.set_xlim(0, 1.05); ax.set_xlabel("Regex refusal rate")
ax.set_title("E3 — Refusal rate by model (861 EN harmful seeds)", fontsize=12, fontweight="bold", pad=12)
for b, r in zip(bars, rates):
    ax.text(r+0.02, b.get_y()+b.get_height()/2, f"{r:.0%}", va="center", fontsize=10, fontweight="bold")
ax.axvline(0.8, color="#555", ls=":", lw=1)
ax.text(0.81, 5.7, "0.80", color="#555", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "slide_03_refusal_by_model.png"); plt.close()

# ============================================================
# FIGURE 4 — E4 translation drift: heatmap (cosine + intent loss)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
langs = ["lo", "mt", "pt", "ur", "es", "hi", "sw", "yo"]
cosine = [0.926, 0.970, 0.946, 0.957, 0.947, 0.936, 0.931, 0.866]
fwd_loss = [np.nan, np.nan, 0.01, 0.023, np.nan, np.nan, np.nan, 0.627]
new_lang = ["new", "new", "new", "new", "v1", "v1", "v1", "v1"]

# Heatmap of cosine
data = np.array(cosine).reshape(1, -1)
im = axes[0].imshow(data, cmap="RdYlGn", vmin=0.8, vmax=1.0, aspect="auto")
axes[0].set_xticks(range(len(langs)))
new_lang = ["new", "new", "new", "new", "v1", "v1", "v1", "v1"]
tl = axes[0].set_xticklabels(langs, rotation=0, fontsize=9)
for lbl, nl in zip(tl, new_lang):
    lbl.set_color("#d62728" if nl == "new" else "#555")
    lbl.set_fontweight("bold" if nl == "new" else "normal")
axes[0].set_yticks([])
for j, c in enumerate(cosine):
    axes[0].text(j, 0, f"{c:.3f}", ha="center", va="center", color="black" if 0.85 < c < 0.95 else "white", fontsize=9, fontweight="bold")
axes[0].set_title("Round-trip cosine (en→L→en)", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

# Forward intent loss
ax2 = axes[1]
x = np.arange(len(langs))
colors = ["#d62728" if (not np.isnan(v) and v > 0.05) else "#2ca02c" for v in fwd_loss]
valid = [(i, v) for i, v in enumerate(fwd_loss) if not np.isnan(v)]
if valid:
    xi, vi = zip(*valid)
    ax2.bar(xi, vi, color=[colors[i] for i in xi], width=0.5, edgecolor="white")
ax2.set_xticks(x); ax2.set_xticklabels(langs, rotation=12, ha="right", fontsize=9)
ax2.set_ylim(0, 0.7); ax2.set_ylabel("Intent lost (fwd)")
ax2.set_title("Harmful intent lost in translation", fontsize=11, fontweight="bold")
for i, v in enumerate(fwd_loss):
    if not np.isnan(v):
        ax2.text(i, v+0.02, f"{v:.0%}", ha="center", fontsize=9, fontweight="bold")
ax2.axhline(0.10, color="#d62728", ls="--", lw=1)
ax2.text(0.02, 0.105, "0.10 flag", color="#d62728", fontsize=8, transform=ax2.get_yaxis_transform())
fig.suptitle("E4 — Translation QA for new languages (pt/mt/ur/lo)", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(OUT / "slide_04_translation_heatmap.png"); plt.close()

# ============================================================
# FIGURE 5 — E5 decoding: paired bars (truncation + k sensitivity)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
settings = ["greedy 256", "greedy 512", "T=1.0 512"]
trunc = [10/27, 6/27, 62/270]
bars = axes[0].bar(settings, trunc, color=["#1f77b4", "#1f77b4", "#d62728"], width=0.5, edgecolor="white")
axes[0].set_ylim(0, 0.28); axes[0].set_ylabel("Fraction truncated")
axes[0].set_title("Output truncation (done_reason=length)", fontsize=11, fontweight="bold")
for b, t in zip(bars, trunc):
    axes[0].text(b.get_x()+b.get_width()/2, t+0.01, f"{t:.0%}", ha="center", fontsize=9, fontweight="bold")

ks = [1, 2, 3, 5, 7]
err = [0.0296, 0.0198, 0.0148, 0.0099, 0.0063]
bars = axes[1].bar(ks, err, color="#2ca02c", width=0.5, edgecolor="white")
axes[1].set_xticks(ks); axes[1].set_xlabel("Samples k per cell")
axes[1].set_ylim(0, 0.035); axes[1].set_ylabel("Mean |error| vs k=10")
axes[1].set_title("Refusal-rate error vs k (Gemma3-12B)", fontsize=11, fontweight="bold")
for b, e in zip(bars, err):
    axes[1].text(b.get_x()+b.get_width()/2, e+0.001, f"{e:.3f}", ha="center", fontsize=9, fontweight="bold")
fig.suptitle("E5 — Decoding settings: truncation vs sample count", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(OUT / "slide_05_decoding.png"); plt.close()

# ============================================================
# FIGURE 6 — E2 task pilot: scatter (refusals vs well-formedness fails)
# ============================================================
fig, ax = plt.subplots(figsize=(5.6, 3.4))
tasks = ["no_task", "sentiment", "rephrase", "summarize", "translate", "pos_tagging", "word_shuffle", "char_count", "keyword"]
refuse = [0, 0, 0, 1, 1, 0, 0, 0, 0]
check_fail = [3, 3, 2, 0, 1, 0, 3, 3, 0]
colors = ["#d62728" if r > 0 else "#1f77b4" for r in refuse]
sizes = [60 + 100*c for c in check_fail]
sc = ax.scatter(check_fail, refuse, s=sizes, c=colors, alpha=0.7, edgecolor="white", linewidth=1)
for i, t in enumerate(tasks):
    ax.annotate(t, (check_fail[i], refuse[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("Well-formedness fails (of 3)")
ax.set_ylabel("Refusals (of 3)")
ax.set_title("E2 — Task pilot: refusal vs well-formedness (Gemma3-12B)", fontsize=12, fontweight="bold", pad=12)
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 1.5)
ax.text(1.5, 1.35, "bubble size = well-formedness fails", fontsize=8, color="#555", style="italic")
plt.tight_layout(); plt.savefig(OUT / "slide_06_task_scatter.png"); plt.close()

print("Figures written to", OUT)
