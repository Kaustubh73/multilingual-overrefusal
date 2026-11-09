"""PCA of last-prompt-token activations over language x task (pilot1, harmful prompts, 320 points).

Per-prompt mean vector subtracted before PCA (prompt identity confound). Descriptive only.
  python experiments/activation_pilot/pca_language_and_task.py results/activation_pilot/pilot1 results/activation_pilot/pca_points
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
SCRIPT = {"ar": "Arabic", "ur": "Arabic", "hi": "Devanagari", "ne": "Devanagari", "ja": "Han", "zh": "Han", "th": "Thai", "lo": "Lao"}
LAYERS = [8, 16, 24, 32]
NPC = 10

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(src / "pilot.jsonl")]
rows = [r for r in rows if r["group"] == "harmful"]
rows.sort(key=lambda r: r["row_id"])

import re
src_lines = (Path(__file__).resolve().parent / "collect_activations.py").read_text().split("\n")
i0 = next(i for i, l in enumerate(src_lines) if l.startswith("REFUSAL_RE = "))
ns = {"re": re}
exec("\n".join(src_lines[i0:i0 + 5]) + "\n\ndef looks_refusal(t):\n    return bool(REFUSAL_RE.search(t[:400]))", ns)  # English-only regex, copied from collect_activations.py
looks_refusal = ns["looks_refusal"]
refused = np.array([looks_refusal(r["response"]) for r in rows])

X = np.stack([np.load(src / "acts" / f"{r['row_id']}.npz")["last_prompt"].astype(np.float32) for r in rows])  # [N, 33, 4096]
lab = {
    "language": np.array([r["lang"] for r in rows]),
    "task": np.array([r["task"] for r in rows]),
    "script": np.array([SCRIPT.get(r["lang"], "Latin") for r in rows]),
    "refusal": np.array([str(bool(x)) for x in refused]) if refused is not None else None,
    "prompt": np.array([r["content_id"] for r in rows]),
}
cid = lab["prompt"]
report = {"n": len(rows), "refused": int(refused.sum()) if refused is not None else None, "layers": {}}
fig, ax = plt.subplots(len(LAYERS), 3, figsize=(15, 4 * len(LAYERS)))
for i, L in enumerate(LAYERS):
    Z = X[:, L].copy()
    for c in set(cid):
        Z[cid == c] -= Z[cid == c].mean(0)
    p = PCA(n_components=NPC).fit(Z)
    P = p.transform(Z)
    rec = {"var_explained": [round(float(v), 4) for v in p.explained_variance_ratio_]}
    for k, y in lab.items():
        if y is None or k == "prompt" or len(set(y)) < 2:
            continue
        rec[f"silhouette_{k}"] = round(float(silhouette_score(P, y)), 3)
    report["layers"][L] = rec
    for j, k in enumerate(["language", "task", "script"]):
        a = ax[i, j]
        for v in sorted(set(lab[k])):
            m = lab[k] == v
            a.scatter(P[m, 0], P[m, 1], s=10, label=v)
        a.set_title(f"layer {L} by {k} (PC1 {p.explained_variance_ratio_[0]:.2f}, PC2 {p.explained_variance_ratio_[1]:.2f})", fontsize=9)
        if i == 0:
            a.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(out / "pca_language_and_task.png", dpi=90)
json.dump(report, open(out / "pca_language_and_task.json", "w"), indent=1)
for L, r in report["layers"].items():
    print(L, {k: v for k, v in r.items() if k != "var_explained"}, "PC1-3 var", r["var_explained"][:3])
print("refused", report["refused"], "of", report["n"])
