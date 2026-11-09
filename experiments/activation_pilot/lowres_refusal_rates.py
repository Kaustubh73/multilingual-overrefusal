"""Refusal rate of the 4 e1b models (ur/lo/mt/yo, 180 rows each), Qwen3Guard-8B verdict.
  python experiments/activation_pilot/lowres_refusal_rates.py -> figures/activation_pilot/pilot/lowres_refusal_rates.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = pd.read_csv("results/e1b/judge_pairs.csv")
d["r"] = d.qg8 == "refuse"
models = ["aya-expanse-8b", "llama-3.1-8b", "gemma-4-e4b", "qwen3.5-9b"]
names = {"aya-expanse-8b": "Aya-Expanse 8B", "llama-3.1-8b": "Llama-3.1 8B", "gemma-4-e4b": "Gemma4 E4B", "qwen3.5-9b": "Qwen3.5 9B"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})

x = np.arange(len(models)); w = 0.38
for i, (lab, col) in enumerate([("harmful", "#c0392b"), ("harmless", "#4a90c2")]):
    v = [d[(d.model == m) & (d.label == lab)].r.mean() for m in models]
    n = [d[(d.model == m) & (d.label == lab)].r.sum() for m in models]
    bars = a.bar(x + (i - .5) * w, v, w, color=col, label=f"{lab} prompts")
    for bar, k in zip(bars, n):
        a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .01, f"{bar.get_height():.0%}", ha="center", fontsize=9)
a.set_xticks(x, [names[m] for m in models], rotation=15); a.set_ylabel("Refusal rate (QG-8B)"); a.set_ylim(0, .45)
a.legend(frameon=False); a.set_title("By prompt label (n=108 harmful, 72 harmless)", fontsize=10)

langs = ["ur", "lo", "mt", "yo"]; w = 0.2
for i, l in enumerate(langs):
    v = [d[(d.model == m) & (d.lang == l)].r.mean() for m in models]
    bars = b.bar(x + (i - 1.5) * w, v, w, label=l)
    for bar in bars:
        b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .01, f"{bar.get_height():.0%}", ha="center", fontsize=7)
b.set_xticks(x, [names[m] for m in models], rotation=15); b.set_ylim(0, .75)
b.legend(title="language (n=45 each)", frameon=False, ncol=4, fontsize=8); b.set_title("By language, all prompts", fontsize=10)
fig.suptitle("e1b: refusal rate, same 45 prompts x 4 languages per model (pilot data, not a finding)", fontsize=10)
fig.tight_layout(); fig.savefig("figures/activation_pilot/pilot/lowres_refusal_rates.png"); print("ok")
