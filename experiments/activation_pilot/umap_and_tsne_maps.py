#!/usr/bin/env python3
"""UMAP and t-SNE of last-prompt-token activations at layers 8, 16, 24, 32, Qwen3.5-9B vs Llama-3.1-8B, coloured by
task and by language. Pilot data (4 prompts), visual only: 2D maps distort distances, do not read cluster sizes/gaps.
Run from repo root: python experiments/activation_pilot/umap_and_tsne_maps.py -> figures/activation_pilot/clustering/g07_umap.png, g08_tsne.png
Seeds fixed (0). umap-learn==0.5.9.post2, sklearn TSNE (perplexity 30), input = standardized -> PCA 50."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RUNS = {"Qwen3.5-9B": Path("results/activation_pilot/pilot1"), "Llama-3.1-8B": Path("results/activation_pilot/pilot_llama")}
OUT = Path("figures/activation_pilot/clustering"); OUT.mkdir(parents=True, exist_ok=True)
LAYERS = [8, 16, 24, 32]
rows = pd.DataFrame([json.loads(l) for l in open(RUNS["Qwen3.5-9B"] / "pilot.jsonl")]).sort_values("row_id").reset_index(drop=True)
ids = rows.row_id.tolist()
tasks = list(dict.fromkeys(rows.task)); langs = sorted(rows.lang.unique())
tc = {t: plt.cm.tab10(i) for i, t in enumerate(tasks)}; lc = {l: plt.cm.tab20(i) for i, l in enumerate(langs)}


def feats(m, L):
    X = np.stack([np.load(RUNS[m] / "acts" / f"{r}.npz")["last_prompt"][L].astype(np.float32) for r in ids])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return PCA(50, random_state=0).fit_transform(X)


emb = {"umap": {}, "tsne": {}}
for m in RUNS:
    for L in LAYERS:
        P = feats(m, L)
        emb["umap"][m, L] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(P)
        emb["tsne"][m, L] = TSNE(perplexity=30, init="pca", random_state=0).fit_transform(P)
    print("done", m, flush=True)

for kind, fname in (("umap", "g07_umap.png"), ("tsne", "g08_tsne.png")):
    fig, ax = plt.subplots(4, 4, figsize=(16, 15))
    for r, (m, by) in enumerate([(m, by) for m in RUNS for by in ("task", "lang")]):
        for c, L in enumerate(LAYERS):
            E = emb[kind][m, L]; col = [(tc if by == "task" else lc)[v] for v in rows[by]]
            ax[r, c].scatter(E[:, 0], E[:, 1], c=col, s=7, linewidths=0); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            ax[r, c].set_title(f"{m} L{L} by {by}", fontsize=9)
    for name, cm_, n in (("task", tc, tasks), ("language", lc, langs)):
        pass
    h1 = [plt.Line2D([], [], marker="o", ls="", color=tc[t], label=t) for t in tasks]
    h2 = [plt.Line2D([], [], marker="o", ls="", color=lc[l], label=l) for l in langs]
    fig.legend(handles=h1, loc="lower left", ncol=5, fontsize=8, title="task", frameon=False)
    fig.legend(handles=h2, loc="lower right", ncol=8, fontsize=8, title="language", frameon=False)
    fig.suptitle(f"{kind.upper()} of last-prompt activations  [pilot data (4 prompts), visual only, not a finding]", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97)); fig.savefig(OUT / fname, dpi=110); plt.close(fig)
print("done")
