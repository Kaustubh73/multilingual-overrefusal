"""Silhouette by layer (PCA-10 of last_prompt, 640 rows) and zoom on the best layer. Pilot data.
  python experiments/activation_pilot/pca_best_layer.py -> figures/activation_pilot/pilot/pca_best_layer.png, results/activation_pilot/silhouette_by_layer.json"""
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
rows = pd.DataFrame([json.loads(l) for l in open(R / "pilot.jsonl")]).sort_values("row_id").reset_index(drop=True)
A = np.stack([np.load(R / "acts" / f"{r}.npz")["last_prompt"] for r in rows.row_id]).astype(np.float32)
sil = {k: [] for k in ["task", "lang", "group", "content_id"]}
for L in range(1, 33):
    X = A[:, L] - A[:, L].mean(0)
    Z = PCA(10, random_state=0).fit_transform(X)
    for k in sil: sil[k].append(float(silhouette_score(Z, rows[k])))
json.dump({k: dict(zip(range(1, 33), v)) for k, v in sil.items()}, open(f"results/activation_pilot/silhouette_by_layer{TAG}.json", "w"), indent=1)
best = int(np.argmax(sil["task"])) + 1; print("best task layer", best, {k: round(v[best - 1], 3) for k, v in sil.items()})
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig = plt.figure(figsize=(14, 4.8)); g = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.15])
a = fig.add_subplot(g[0])
for k, c in zip(sil, ["#1f77b4", "#2ca02c", "#d62728", "#7f7f7f"]):
    a.plot(range(1, 33), sil[k], "o-", ms=3, color=c, label=k.replace("content_id", "prompt"))
a.axvline(best, color="k", ls=":", lw=1); a.axhline(0, color="grey", lw=.6)
a.set_xlabel("layer"); a.set_ylabel("silhouette (PCA-10)"); a.legend(frameon=False, fontsize=8); a.set_title(f"Cluster quality by layer (best for task: {best})", fontsize=10)
X = A[:, best] - A[:, best].mean(0); P = PCA(2, random_state=0).fit_transform(X)
tasks = list(dict.fromkeys(rows.task)); tc = {t: plt.get_cmap("tab10")(i) for i, t in enumerate(tasks)}
b = fig.add_subplot(g[1]); b.scatter(P[:, 0], P[:, 1], c=[tc[t] for t in rows.task], s=9, alpha=.75, linewidths=0)
b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=t.replace("_", " ")) for t, c in tc.items()], fontsize=6.5, frameon=False, ncol=2, loc="best")
b.set_title(f"Layer {best}, coloured by task", fontsize=10); b.set_xticks([]); b.set_yticks([])
c = fig.add_subplot(g[2]); sh = {"c_08daf70eed01": "#c0392b", "c_8c392956ac8f": "#e59866", "c_0e4a8b733b9b": "#2e86c1", "c_b51530340549": "#85c1e9"}
nm = {"c_08daf70eed01": "Harmful A", "c_8c392956ac8f": "Harmful B", "c_0e4a8b733b9b": "Harmless A", "c_b51530340549": "Harmless B"}
c.scatter(P[:, 0], P[:, 1], c=[sh[x] for x in rows.content_id], s=9, alpha=.75, linewidths=0)
c.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=v, label=nm[k]) for k, v in sh.items()], fontsize=8, frameon=False)
c.set_title(f"Layer {best}, coloured by prompt", fontsize=10); c.set_xticks([]); c.set_yticks([])
fig.suptitle("PCA of last-prompt-token hidden state, 640 rows, " + MODEL + " (pilot data, not a finding)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/pca_best_layer.png")
