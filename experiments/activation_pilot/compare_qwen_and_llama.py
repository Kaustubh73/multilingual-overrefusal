#!/usr/bin/env python3
"""Qwen3.5-9B vs Llama-3.1-8B-Instruct on the same 640-row pilot (2 harmful + 2 harmless seeds x 16 langs x 10 tasks).
Pilot data, descriptive only, not findings.  Run from repo root:
  python experiments/activation_pilot/compare_qwen_and_llama.py  -> figures/activation_pilot/qwen_vs_llama/c*.png, results/activation_pilot/compare_qwen_and_llama.json
Needs results/activation_pilot/pilot1 and results/activation_pilot/pilot_llama (pilot.jsonl, judge_qg4.jsonl, acts/, tf_acts/).
Judge = Qwen3Guard-4B for both models (Llama has no 8B-judge run). Cross-model hidden-state comparison uses linear CKA / RSA
because hidden sizes differ (Qwen vs 4096).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUNS = {"Qwen3.5-9B": Path("results/activation_pilot/pilot1"), "Llama-3.1-8B": Path("results/activation_pilot/pilot_llama")}
COL = {"Qwen3.5-9B": "#3b6fb6", "Llama-3.1-8B": "#e08a2c"}
OUT = Path("figures/activation_pilot/qwen_vs_llama"); OUT.mkdir(parents=True, exist_ok=True)
NOTE = "pilot data (4 prompts), not a finding"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
res = {}


def load(name, d):
    rows = pd.DataFrame([json.loads(l) for l in open(d / "pilot.jsonl")])
    j = pd.DataFrame([json.loads(l) for l in open(d / "judge_qg4.jsonl")])[["row_id", "verdict"]]
    df = rows.merge(j, on="row_id").sort_values("row_id").reset_index(drop=True)
    df["ref"] = (df.verdict == "refuse").astype(int); df["trunc"] = (df.done_reason == "length").astype(int)
    return df


D = {m: load(m, p) for m, p in RUNS.items()}
ids = D["Qwen3.5-9B"].row_id.tolist()
assert all(D[m].row_id.tolist() == ids for m in D), "row sets differ"
base = D["Qwen3.5-9B"]
langs = sorted(base.lang.unique(), key=lambda l: (l != "en", l)); tasks = list(dict.fromkeys(base.task))


def acts(m, key="last_prompt", sub="acts"):
    return np.stack([np.load(RUNS[m] / sub / f"{r}.npz")[key].astype(np.float32) for r in ids])  # [N, L1, H]


def savefig(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)


# ---- c01 refusal rate by language, harmful vs harmless
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for k, grp in enumerate(["harmful", "harmless"]):
    for i, m in enumerate(D):
        s = D[m][D[m].group == grp].groupby("lang").ref.mean().reindex(langs)
        ax[k].bar(np.arange(len(langs)) + (i - .5) * .38, s.values, .38, color=COL[m], label=m)
    ax[k].set_xticks(range(len(langs))); ax[k].set_xticklabels(langs); ax[k].set_title(f"{grp} prompts (40 rows per language)")
ax[0].set_ylabel("share judged refuse (Qwen3Guard-4B)"); ax[0].legend(frameon=False)
fig.suptitle(f"Refusal by language  [{NOTE}]", fontsize=10); savefig(fig, "c01_refusal_by_language.png")
res["refusal_rate"] = {m: {g: round(float(D[m][D[m].group == g].ref.mean()), 3) for g in ("harmful", "harmless")} for m in D}

# ---- c02 heatmaps lang x task (harmful), both models + difference
H = {m: D[m][D[m].group == "harmful"].pivot_table(index="task", columns="lang", values="ref", aggfunc="mean").reindex(index=tasks, columns=langs) for m in D}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, m in zip(ax[:2], D):
    im = a.imshow(H[m].values, cmap="Reds", vmin=0, vmax=1); a.set_title(f"{m}: share refuse (harmful)")
diff = H["Llama-3.1-8B"].values - H["Qwen3.5-9B"].values
im2 = ax[2].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1); ax[2].set_title("Llama minus Qwen")
for a in ax:
    a.set_xticks(range(len(langs))); a.set_xticklabels(langs, rotation=90); a.set_yticks(range(len(tasks))); a.set_yticklabels(tasks if a is ax[0] else [])
fig.colorbar(im, ax=ax[:2], shrink=.8); fig.colorbar(im2, ax=ax[2], shrink=.8)
fig.suptitle(f"Harmful-prompt refusal, language x task (2 prompts per cell)  [{NOTE}]", fontsize=10); savefig(fig, "c02_refusal_heatmaps_diff.png")

# ---- c03 length / truncation
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for i, m in enumerate(D):
    s = D[m].groupby("task").n_gen.median().reindex(tasks); ax[0].bar(np.arange(len(tasks)) + (i - .5) * .38, s.values, .38, color=COL[m], label=m)
    t = D[m].groupby("lang").trunc.mean().reindex(langs); ax[1].bar(np.arange(len(langs)) + (i - .5) * .38, t.values, .38, color=COL[m])
ax[0].set_xticks(range(len(tasks))); ax[0].set_xticklabels(tasks, rotation=60, ha="right"); ax[0].set_ylabel("median generated tokens"); ax[0].legend(frameon=False)
ax[1].set_xticks(range(len(langs))); ax[1].set_xticklabels(langs); ax[1].set_ylabel("share hitting 512-token cap")
fig.suptitle(f"Output length and truncation  [{NOTE}]", fontsize=10); savefig(fig, "c03_length_truncation.png")
res["trunc_rate"] = {m: round(float(D[m].trunc.mean()), 3) for m in D}

# ---- c04 per-row verdict agreement between models
a, b = D["Qwen3.5-9B"].ref.values, D["Llama-3.1-8B"].ref.values
ct = pd.crosstab(a, b); res["verdict_crosstab_llama_then_qwen"] = ct.to_dict()  # {llama_verdict: {qwen_verdict: n}}
agree = pd.Series(a == b).groupby(base.lang.values).mean().reindex(langs)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 2.4]})
ax[0].imshow(ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0).values, cmap="Blues")
for (i, j), v in np.ndenumerate(ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0).values):
    ax[0].text(j, i, v, ha="center", va="center", fontsize=11)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["comply", "refuse"]); ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["comply", "refuse"])
ax[0].set_xlabel("Llama"); ax[0].set_ylabel("Qwen"); ax[0].set_title("640 rows, same prompt")
ax[1].bar(range(len(langs)), agree.values, color="#777"); ax[1].set_xticks(range(len(langs))); ax[1].set_xticklabels(langs)
ax[1].set_ylabel("rows with same verdict"); ax[1].set_ylim(0, 1); ax[1].set_title("agreement by language")
fig.suptitle(f"Do the two models refuse the same rows?  [{NOTE}]", fontsize=10); savefig(fig, "c04_verdict_agreement.png")

# ---- representations
A = {m: acts(m) for m in D}
L1 = A["Qwen3.5-9B"].shape[1]; LAY = list(range(0, L1, 4)) + ([L1 - 1] if (L1 - 1) % 4 else [])
y_task = base.task.values; y_lang = base.lang.values; y_grp = (base.group == "harmful").astype(int).values; y_cid = base.content_id.values


def cv_acc(X, y, groups, splitter):
    accs = []
    for tr, te in splitter.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1)); clf.fit(X[tr], y[tr]); accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs))


# c05 probes by layer: harmful-vs-harmless and task (held-out language), language (held-out prompt)
probe = {m: {"group_heldout_lang": [], "task_heldout_lang": [], "lang_heldout_prompt": []} for m in D}
for m in D:
    for L in LAY:
        X = A[m][:, L]
        probe[m]["group_heldout_lang"].append(cv_acc(X, y_grp, y_lang, LeaveOneGroupOut()))
        probe[m]["task_heldout_lang"].append(cv_acc(X, y_task, y_lang, GroupKFold(4)))
        probe[m]["lang_heldout_prompt"].append(cv_acc(X, y_lang, y_cid, GroupKFold(4)))
    print("probes done", m, flush=True)
res["probes"] = {m: {k: dict(zip(LAY, np.round(v, 3).tolist())) for k, v in probe[m].items()} for m in D}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a_, (k, ttl, ch) in zip(ax, [("group_heldout_lang", "harmful vs harmless (held-out language)", .5), ("task_heldout_lang", "task (held-out language group)", .1),
                                 ("lang_heldout_prompt", "language (held-out prompt)", 1 / 16)]):
    for m in D:
        a_.plot(LAY, probe[m][k], "-o", ms=3, color=COL[m], label=m)
    a_.axhline(ch, ls=":", color="#999"); a_.set_title(ttl); a_.set_xlabel("layer"); a_.set_ylim(0, 1.02)
ax[0].set_ylabel("probe accuracy (last prompt token)"); ax[0].legend(frameon=False)
fig.suptitle(f"What is linearly readable by depth  [{NOTE}]", fontsize=10); savefig(fig, "c05_probe_accuracy_by_layer.png")


# c06 linear CKA layer x layer (Qwen vs Llama), last prompt token
def gram(X):
    X = X - X.mean(0); return X @ X.T


def cka(K, Lm):
    n = K.shape[0]; Hc = np.eye(n) - 1 / n; K = Hc @ K @ Hc; Lm = Hc @ Lm @ Hc
    return float((K * Lm).sum() / np.sqrt((K * K).sum() * (Lm * Lm).sum()))


G = {m: [gram(A[m][:, L]) for L in range(L1)] for m in D}
C = np.array([[cka(G["Qwen3.5-9B"][i], G["Llama-3.1-8B"][j]) for j in range(L1)] for i in range(L1)])
res["cka_diag"] = np.round(np.diag(C), 3).tolist()
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
im = ax[0].imshow(C, origin="lower", cmap="viridis", vmin=0, vmax=1); ax[0].set_xlabel("Llama layer"); ax[0].set_ylabel("Qwen layer"); fig.colorbar(im, ax=ax[0], shrink=.85)
ax[0].set_title("linear CKA, last prompt token, 640 rows")
ax[1].plot(range(L1), np.diag(C), color="#444"); ax[1].plot(range(L1), C.max(1), ls="--", color="#999", label="best Llama layer for each Qwen layer")
ax[1].set_xlabel("layer (same index)"); ax[1].set_ylabel("CKA"); ax[1].set_ylim(0, 1); ax[1].legend(frameon=False); ax[1].set_title("diagonal = same depth")
fig.suptitle(f"Representation similarity across models  [{NOTE}]", fontsize=10); savefig(fig, "c06_cka_layer_matrix.png")

# c07 RSA of language geometry and of task geometry
iu = np.triu_indices(16, 1)


def centroid_rdm(X, y, keys):
    Cn = np.stack([X[y == k].mean(0) for k in keys]); Cn = Cn - Cn.mean(0)
    Cn = Cn / (np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-9); return (1 - Cn @ Cn.T)


rsa = {}
for tag, y, keys in [("language", y_lang, langs), ("task", y_task, tasks)]:
    iu_k = np.triu_indices(len(keys), 1)
    rsa[tag] = [float(spearmanr(centroid_rdm(A["Qwen3.5-9B"][:, L], y, keys)[iu_k], centroid_rdm(A["Llama-3.1-8B"][:, L], y, keys)[iu_k])[0]) for L in range(L1)]
res["rsa"] = {k: np.round(v, 3).tolist() for k, v in rsa.items()}
fig, ax = plt.subplots(figsize=(6, 3.6))
for k, c in zip(rsa, ["#2a9d8f", "#9b5de5"]):
    ax.plot(range(L1), rsa[k], "-o", ms=3, color=c, label=f"{k} centroids")
ax.set_xlabel("layer (same index in both models)"); ax.set_ylabel("Spearman of centroid distance matrices"); ax.set_ylim(-.2, 1); ax.legend(frameon=False)
ax.set_title(f"Do both models organize languages/tasks alike?  [{NOTE}]", fontsize=9); savefig(fig, "c07_rsa_language_task.png")

# c08 silhouette by layer (language, task, group)
sil = {m: {} for m in D}
for m in D:
    for tag, y in [("language", y_lang), ("task", y_task), ("harmful/harmless", y_grp)]:
        sil[m][tag] = [float(silhouette_score(A[m][:, L], y, metric="cosine")) if L else float("nan") for L in LAY]
res["silhouette"] = {m: {k: dict(zip(LAY, np.round(v, 3).tolist())) for k, v in sil[m].items()} for m in D}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for a_, tag in zip(ax, ["language", "task", "harmful/harmless"]):
    for m in D:
        a_.plot(LAY, sil[m][tag], "-o", ms=3, color=COL[m], label=m)
    a_.set_title(f"silhouette by {tag}"); a_.set_xlabel("layer"); a_.axhline(0, color="#aaa", lw=.5)
ax[0].set_ylabel("cosine silhouette"); ax[0].legend(frameon=False)
fig.suptitle(f"Cluster separation by depth, last prompt token  [{NOTE}]", fontsize=10); savefig(fig, "c08_silhouette_by_layer.png")

# c09 harm direction from English transferred to other languages (AUC by layer)
auc = {m: [] for m in D}; auc_lang = {}
en = (y_lang == "en")
for m in D:
    for L in range(1, L1):
        X = A[m][:, L]; dvec = X[en & (y_grp == 1)].mean(0) - X[en & (y_grp == 0)].mean(0); s = X @ dvec
        auc[m].append(float(roc_auc_score(y_grp[~en], s[~en])))
    Lb = 16; X = A[m][:, Lb]; dvec = X[en & (y_grp == 1)].mean(0) - X[en & (y_grp == 0)].mean(0); s = X @ dvec
    auc_lang[m] = {l: float(roc_auc_score(y_grp[y_lang == l], s[y_lang == l])) for l in langs if l != "en"}
res["en_harm_direction_auc_nonEn"] = {m: np.round(v, 3).tolist() for m, v in auc.items()}
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
for m in D:
    ax[0].plot(range(1, L1), auc[m], color=COL[m], label=m)
ax[0].axhline(.5, ls=":", color="#999"); ax[0].set_xlabel("layer"); ax[0].set_ylabel("AUC, harmful vs harmless, non-English rows"); ax[0].set_ylim(.4, 1.02); ax[0].legend(frameon=False)
ax[0].set_title("English harm direction applied to other languages")
lk = [l for l in langs if l != "en"]
for i, m in enumerate(D):
    ax[1].bar(np.arange(len(lk)) + (i - .5) * .38, [auc_lang[m][l] for l in lk], .38, color=COL[m])
ax[1].axhline(.5, ls=":", color="#999"); ax[1].set_xticks(range(len(lk))); ax[1].set_xticklabels(lk); ax[1].set_ylim(.4, 1.02); ax[1].set_title("by language, layer 16")
fig.suptitle(f"Does a harm direction found in English carry over?  [{NOTE}]", fontsize=10); savefig(fig, "c09_english_harm_direction_transfer.png")

# c10 teacher-forced: same (Qwen) text through Llama
tf_dir = RUNS["Llama-3.1-8B"] / "tf_acts"
if tf_dir.exists() and len(list(tf_dir.glob("*.npz"))) == len(ids):
    TF = acts("Llama-3.1-8B", "gen_mean", "tf_acts"); own = acts("Llama-3.1-8B", "gen_mean"); Q = acts("Qwen3.5-9B", "gen_mean")
    Gq = [gram(Q[:, L]) for L in range(L1)]
    c_tf = [cka(Gq[L], gram(TF[:, L])) for L in range(L1)]; c_own = [cka(Gq[L], gram(own[:, L])) for L in range(L1)]
    cos = [float(np.mean(np.sum((TF[:, L] - TF[:, L].mean(0)) * (own[:, L] - own[:, L].mean(0)), 1) / (np.linalg.norm(TF[:, L] - TF[:, L].mean(0), axis=1) * np.linalg.norm(own[:, L] - own[:, L].mean(0), axis=1) + 1e-9))) for L in range(L1)]
    res["tf"] = dict(cka_qwen_vs_llama_teacherforced_qwen_text=np.round(c_tf, 3).tolist(), cka_qwen_vs_llama_own_text=np.round(c_own, 3).tolist(), centered_cos_llama_tf_vs_own=np.round(cos, 3).tolist())
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(range(L1), c_own, color="#c0392b", label="Qwen text-states vs Llama on its own text"); ax[0].plot(range(L1), c_tf, color="#27ae60", label="Qwen text-states vs Llama reading Qwen's text")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("linear CKA, mean over response tokens"); ax[0].set_ylim(0, 1); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("how much of the gap is the text itself?")
    ax[1].plot(range(L1), cos, color="#444"); ax[1].set_xlabel("layer"); ax[1].set_ylabel("mean centered cosine"); ax[1].set_ylim(-.1, 1); ax[1].set_title("Llama: own text vs Qwen text (same row)")
    fig.suptitle(f"Teacher-forced cross-read, response-token states  [{NOTE}]", fontsize=10); savefig(fig, "c10_teacher_forced_cross_read.png")
else:
    print("tf_acts incomplete; skipped c10")

res["note"] = NOTE; res["judge"] = "Qwen3Guard-4B (both)"
json.dump(res, open("results/activation_pilot/compare_qwen_and_llama.json", "w"), indent=1)
print("done")
