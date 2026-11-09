#!/usr/bin/env python3
"""label-free / non-probe checks of how last-prompt-token activations group, Qwen3.5-9B vs Llama-3.1-8B.
Pilot data (640 rows = 4 prompts x 16 langs x 10 tasks), descriptive only, not findings.  Run from repo root:
  python experiments/activation_pilot/clustering_without_labels.py -> figures/activation_pilot/clustering/g*.png, results/activation_pilot/clustering_without_labels.json
Per layer, per label (language, task, harmful/harmless, prompt content):
  g01 kNN label purity (k=5, cosine, self excluded) vs shuffled-label chance
  g02 k-means (k = n classes) adjusted Rand index vs label
  g03 adjusted eta^2 (share of variance explained by label, bias-corrected for group count)
  g04 participation ratio (effective dimensionality)
  g05 language eta^2 / kNN purity after removing the task x prompt cell mean (language effect isolated)
  g06 hierarchical clustering of language centroids (cell-mean-removed), layers 16 and 32
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

RUNS = {"Qwen3.5-9B": Path("results/activation_pilot/pilot1"), "Llama-3.1-8B": Path("results/activation_pilot/pilot_llama")}
COL = {"Qwen3.5-9B": "#3b6fb6", "Llama-3.1-8B": "#e08a2c"}
OUT = Path("figures/activation_pilot/clustering"); OUT.mkdir(parents=True, exist_ok=True)
NOTE = "pilot data (4 prompts), not a finding"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
LABELS = {"language": "lang", "task": "task", "harmful/harmless": "group", "prompt content": "content_id"}
LCOL = dict(zip(LABELS, ["#2a9d8f", "#9b5de5", "#d1495b", "#6c757d"]))
res = {}

rows = {m: pd.DataFrame([json.loads(l) for l in open(p / "pilot.jsonl")]).sort_values("row_id").reset_index(drop=True) for m, p in RUNS.items()}
ids = rows["Qwen3.5-9B"].row_id.tolist()
assert all(rows[m].row_id.tolist() == ids for m in rows)
meta = rows["Qwen3.5-9B"]
N = len(ids)
langs = sorted(meta.lang.unique(), key=lambda l: (l != "en", l))
rng = np.random.default_rng(0)


def acts(m):
    return np.stack([np.load(RUNS[m] / "acts" / f"{r}.npz")["last_prompt"].astype(np.float32) for r in ids])  # [N, L1, H]


def standardize(X):
    X = X - X.mean(0)
    return X / (X.std(0) + 1e-6)


def knn_purity(X, y, k=5):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T; np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    return float((y[nn] == y[:, None]).mean())


def adj_eta2(X, y):
    tot = ((X - X.mean(0)) ** 2).sum()
    cl = np.unique(y); between = 0.0
    for c in cl:
        Xc = X[y == c]; between += len(Xc) * ((Xc.mean(0) - X.mean(0)) ** 2).sum()
    e = between / tot; k = len(cl)
    return float(1 - (1 - e) * (len(y) - 1) / (len(y) - k))


def part_ratio(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


def kmeans_ari(X, y, k):
    Z = np.linalg.svd(X - X.mean(0), full_matrices=False)
    P = (Z[0] * Z[1])[:, :30]
    return float(adjusted_rand_score(y, KMeans(k, n_init=10, random_state=0).fit_predict(P)))


def remove_cell_mean(X):
    """subtract mean over languages within each (task, prompt) cell -> leaves language effect + noise"""
    key = meta.task + "|" + meta.content_id
    Xr = X.copy()
    for k in key.unique():
        m = (key == k).values; Xr[m] -= X[m].mean(0)
    return Xr


Y = {name: pd.factorize(meta[col])[0] for name, col in LABELS.items()}
chance = {name: knn_purity(rng.standard_normal((N, 8)), rng.permutation(Y[name])) for name in LABELS}  # shuffled-label baseline
res["knn_chance_shuffled"] = chance

R = {}
for m in RUNS:
    A = acts(m); L1 = A.shape[1]
    R[m] = {n: {"knn": [], "ari": [], "eta2": []} for n in LABELS}; R[m]["pr"] = []; R[m]["lang_resid"] = {"knn": [], "eta2": []}
    cent = {}
    for L in range(L1):
        X = standardize(A[:, L]) if L > 0 else A[:, L] - A[:, L].mean(0)
        if not np.isfinite(X).all() or X.std() < 1e-8:
            for n in LABELS:
                for k in ("knn", "ari", "eta2"): R[m][n][k].append(np.nan)
            R[m]["pr"].append(np.nan); R[m]["lang_resid"]["knn"].append(np.nan); R[m]["lang_resid"]["eta2"].append(np.nan); continue
        for n in LABELS:
            R[m][n]["knn"].append(knn_purity(X, Y[n])); R[m][n]["eta2"].append(adj_eta2(X, Y[n]))
            R[m][n]["ari"].append(kmeans_ari(X, Y[n], len(np.unique(Y[n]))))
        R[m]["pr"].append(part_ratio(X))
        Xr = remove_cell_mean(X)
        R[m]["lang_resid"]["knn"].append(knn_purity(Xr, Y["language"])); R[m]["lang_resid"]["eta2"].append(adj_eta2(Xr, Y["language"]))
        if L in (16, L1 - 1):
            cent[L] = np.stack([Xr[(meta.lang == l).values].mean(0) for l in langs])
    R[m]["cent"] = cent
    print("done", m, flush=True)
res["metrics"] = {m: {k: v for k, v in R[m].items() if k != "cent"} for m in R}


def per_model_lines(key, title, fname, ylab, chance_line=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for a, m in zip(ax, R):
        for n in LABELS: a.plot(R[m][n][key], marker=".", color=LCOL[n], label=n)
        if chance_line:
            for n in LABELS: a.axhline(chance_line[n], color=LCOL[n], ls=":", lw=.8)
        a.set_title(m); a.set_xlabel("layer index")
    ax[0].set_ylabel(ylab); ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title}  [{NOTE}]", fontsize=10); fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)


per_model_lines("knn", "kNN label purity (k=5, cosine; dotted = shuffled-label chance)", "g01_knn_purity.png", "share of neighbours with same label", chance)
per_model_lines("ari", "k-means adjusted Rand index vs label (k = number of classes)", "g02_kmeans_ari.png", "ARI")
per_model_lines("eta2", "Adjusted eta^2: share of activation variance explained by label", "g03_variance_explained.png", "adjusted eta^2")

fig, ax = plt.subplots(figsize=(6, 3.6))
for m in R: ax.plot(R[m]["pr"], marker=".", color=COL[m], label=m)
ax.set_xlabel("layer index"); ax.set_ylabel("participation ratio"); ax.legend(frameon=False)
ax.set_title(f"Effective dimensionality of last-prompt activations  [{NOTE}]", fontsize=9); fig.tight_layout(); fig.savefig(OUT / "g04_participation_ratio.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for m in R:
    ax[0].plot(R[m]["lang_resid"]["eta2"], marker=".", color=COL[m], label=m)
    ax[1].plot(R[m]["lang_resid"]["knn"], marker=".", color=COL[m], label=m)
ax[0].set_ylabel("adjusted eta^2 (language)"); ax[1].set_ylabel("kNN purity (language)"); ax[1].axhline(chance["language"], color="gray", ls=":")
for a in ax: a.set_xlabel("layer index")
ax[0].legend(frameon=False)
fig.suptitle(f"Language grouping after removing the task x prompt cell mean  [{NOTE}]", fontsize=10); fig.tight_layout(); fig.savefig(OUT / "g05_language_cell_mean_removed.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for i, m in enumerate(R):
    for j, L in enumerate(sorted(R[m]["cent"])):
        Z = linkage(R[m]["cent"][L], "average", metric="cosine")
        dendrogram(Z, labels=langs, ax=ax[i, j], color_threshold=0, above_threshold_color=COL[m])
        ax[i, j].set_title(f"{m}, layer {L}")
fig.suptitle(f"Hierarchical clustering of language centroids (cell mean removed, cosine, average linkage)  [{NOTE}]", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "g06_language_dendrograms.png", dpi=150); plt.close(fig)
Path("results/activation_pilot/clustering_without_labels.json").write_text(json.dumps(res, default=lambda o: None if o != o else o))
print("done")
