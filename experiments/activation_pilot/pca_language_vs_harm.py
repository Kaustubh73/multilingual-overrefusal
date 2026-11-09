"""PCA of last_prompt hidden state (pilot1, 640 rows, Qwen3.5-9B), coloured by language and by group (harmful / harmless).
Layer 18 = best task-silhouette layer from pca_best_layer.py. Global mean subtracted, as there. Pilot data, descriptive only.
  python experiments/activation_pilot/pca_language_vs_harm.py -> figures/activation_pilot/pilot/pca_language_vs_harm.png, results/activation_pilot/pca_language_vs_harm.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import os
PILOT = os.environ.get("PILOT", "results/activation_pilot/pilot1"); FIGDIR = os.environ.get("FIGDIR", "figures/activation_pilot/pilot"); MODEL = os.environ.get("MODEL_LABEL", "Qwen3.5-9B"); TAG = os.environ.get("TAG", "")
R = Path(PILOT)
LAYER = 18
rows = pd.DataFrame([json.loads(l) for l in open(R / "pilot.jsonl")]).sort_values("row_id").reset_index(drop=True)
A = np.stack([np.load(R / "acts" / f"{r}.npz")["last_prompt"][LAYER] for r in rows.row_id]).astype(np.float32)
X = A - A.mean(0)
pca = PCA(10, random_state=0)
Z = pca.fit_transform(X)
var = pca.explained_variance_ratio_
P = Z[:, :2]

sil = {k: float(silhouette_score(Z, rows[k])) for k in ["lang", "group"]}
json.dump({"layer": LAYER, "silhouette_pca10": sil, "var_ratio_pc1_2": var[:2].tolist()},
          open(f"results/activation_pilot/pca_language_vs_harm{TAG}.json", "w"), indent=1)
print("layer", LAYER, "silhouette", sil, "var PC1/PC2", var[:2].round(3))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

# Panel 1: language
langs = sorted(rows.lang.unique())
cmap = plt.get_cmap("tab20")
lc = {l: cmap(i % 20) for i, l in enumerate(langs)}
a = axes[0]
a.scatter(P[:, 0], P[:, 1], c=[lc[l] for l in rows.lang], s=12, alpha=.8, linewidths=0)
a.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=lc[l], label=l) for l in langs],
         fontsize=7.5, frameon=False, ncol=2, loc="best")
a.set_title(f"Layer {LAYER}, coloured by language (silhouette {sil['lang']:.2f})", fontsize=10)
a.set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); a.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")

# Panel 2: group
gc = {"harmful": "#c0392b", "harmless": "#2e86c1"}
b = axes[1]
b.scatter(P[:, 0], P[:, 1], c=[gc[g] for g in rows.group], s=12, alpha=.8, linewidths=0)
b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in gc.items()],
         fontsize=9, frameon=False, loc="best")
b.set_title(f"Layer {LAYER}, coloured by group (silhouette {sil['group']:.2f})", fontsize=10)
b.set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); b.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")

fig.suptitle("PCA of last-prompt-token hidden state, 640 rows, " + MODEL + " (pilot data, not a finding)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/pca_language_vs_harm.png")
