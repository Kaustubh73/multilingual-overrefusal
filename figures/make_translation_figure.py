#!/usr/bin/env python3
"""E4 translation quality figure for pt/mt/ur/lo."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

langs = ["Portuguese (pt)", "Maltese (mt)", "Urdu (ur)", "Lao (lo)"]
cosine = [0.946, 0.970, 0.957, 0.926]
chrF   = [81.9, 87.7, 84.1, 75.0]
bleu   = [64.0, 75.3, 69.3, 55.3]
back_lost = [2.0, 1.5, 1.5, 3.0]      # % back-translation intent lost
fwd_lost  = [1.0, 9.0, 2.3, 2.0]      # % forward intent lost

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

# Cosine
axes[0].bar(langs, cosine, color=["#1f77b4", "#2ca02c", "#1f77b4", "#ff7f0e"], width=0.55, edgecolor="white")
axes[0].set_ylim(0.85, 1.0); axes[0].set_ylabel("Round-trip cosine")
axes[0].set_title("Translation fidelity (en→L→en)", fontweight="bold")
for i, c in enumerate(cosine):
    axes[0].text(i, c+0.003, f"{c:.3f}", ha="center", fontsize=10, fontweight="bold")
axes[0].axhline(0.90, color="#d62728", ls="--", lw=1)
axes[0].text(0.02, 0.902, "0.90 threshold", color="#d62728", fontsize=8, transform=axes[0].get_yaxis_transform())

# chrF++ / BLEU
x = np.arange(len(langs)); w = 0.35
axes[1].bar(x - w/2, chrF, w, label="chrF++", color="#1f77b4", edgecolor="white")
axes[1].bar(x + w/2, bleu, w, label="BLEU", color="#ff7f0e", edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(langs, rotation=12, ha="right", fontsize=9)
axes[1].set_ylabel("Score"); axes[1].set_title("Surface-form match", fontweight="bold")
axes[1].legend(fontsize=9)
for i, (c, b) in enumerate(zip(chrF, bleu)):
    axes[1].text(i - w/2, c+1, f"{c:.0f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].text(i + w/2, b+1, f"{b:.0f}", ha="center", fontsize=9, fontweight="bold")

# Intent lost
axes[2].bar(langs, back_lost, width=w, label="Back-translation lost", color="#2ca02c", edgecolor="white")
axes[2].bar(langs, fwd_lost, width=w, bottom=back_lost, label="Forward lost", color="#d62728", alpha=0.7, edgecolor="white")
axes[2].set_ylabel("Intent lost (%)"); axes[2].set_title("Harmful intent not preserved", fontweight="bold")
for i, (bl, fl) in enumerate(zip(back_lost, fwd_lost)):
    axes[2].text(i, bl+fl+0.2, f"{bl+fl:.1f}%", ha="center", fontsize=9, fontweight="bold")
axes[2].axhline(10, color="#d62728", ls="--", lw=1)
axes[2].text(0.02, 10.5, "10% flag", color="#d62728", fontsize=8, transform=axes[2].get_yaxis_transform())

fig.suptitle("E4 — Translation QA: pt / mt / ur / lo vs v1 reference (399 harmful seeds each)", fontweight="bold", fontsize=12, y=1.02)
plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/e4_pt_mt_ur_lo_translation.png", bbox_inches="tight")
plt.close()
print("Saved figures/e4_pt_mt_ur_lo_translation.png")
