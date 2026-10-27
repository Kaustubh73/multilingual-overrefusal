#!/usr/bin/env python3
"""E4 left panel only: cosine bar chart for pt/mt/ur/lo."""
import matplotlib.pyplot as plt
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
colors = ["#1f77b4", "#2ca02c", "#1f77b4", "#ff7f0e"]

fig, ax = plt.subplots(figsize=(4.2, 3.0))
bars = ax.bar(langs, cosine, color=colors, width=0.55, edgecolor="white")
ax.set_ylim(0.85, 1.0)
ax.set_ylabel("Round-trip cosine")
ax.set_xticks(range(len(langs)))
ax.set_xticklabels(langs, rotation=0, fontsize=9)
ax.axhline(0.90, color="#d62728", ls="--", lw=1)
for i, c in enumerate(cosine):
    ax.text(i, c+0.003, f"{c:.3f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/e4_pt_mt_ur_lo_cosine.png", bbox_inches="tight")
plt.close()
print("Saved figures/e4_pt_mt_ur_lo_cosine.png")
