# `track_b_analysis` — Track B over-refusal analysis toolkit

Reusable, dependency-light pipeline for analysing **Track B** (multilingual
over-refusal) results. Turns raw model responses + judge labels into rates,
confidence intervals, language comparisons, and figures.

**Dependencies:** `numpy`, `pandas`, `matplotlib` only. **No scipy** — Wilson
score intervals and the normal quantiles are computed analytically.

> ⚠️ This toolkit only *computes and plots*. It enforces nothing about sample
> size. Do **not** present small-n (e.g. smoke n=5) output as a finding — the
> Wilson CIs will be near-[0,1] and are there precisely to make that obvious.

## What it computes

| Function | Output |
| --- | --- |
| `wilson_ci(count, n, confidence=0.95)` | Wilson score interval `(low, high)` |
| `proportion_stats(count, n)` | `rate` + Wilson CI as a dict |
| `rate_by_group(df, success_col, group_cols, eligible_col=None)` | per-group rate + CI |
| `over_refusal_by_group(df)` | OR rate per `(language, task_label)` (benign-task numerator) |
| `harmful_refusal_by_group(df)` | refusal rate on `harmful_instruction` content |
| `compare_languages(df, baseline="english")` | Δ OR vs baseline, with Wald CI on the difference |
| `agreement_rate(pred, gold)` | judge↔human agreement + Cohen's κ |
| `judge_label_distribution(df)` | `ri` label mix per group |

### Over-refusal definition

Matches `src/track_b_metrics.py` / SafeConstellations Eq. 9: a refusal of a
**benign task** (translate, sentiment) counts as over-refusal even when the
*content* is harmful, because the *task* is benign. Harmful-*content* refusals
are reported separately by `harmful_refusal_by_group` (these are mostly desired
safety behaviour).

## Schema expected

The loader works on the standard Track B files:

- `results/track_b_raw.csv`: `prompt_id, sample_id, language, plain_text,
  text_type, intended_task, task_label, is_benign_task, model, response`
- `results/track_b_judgments.csv`: `sample_id, plain_text, response, ri,
  judge_raw, judge_model, prompt_id` where `ri ∈ {direct_answer,
  direct_refusal, indirect_refusal}`.

`load_merged` is robust to the real files: it de-duplicates judgment rows
accumulated across reruns and handles older rows that stored the per-language
key in `sample_id` instead of `prompt_id`.

## Usage

### One command (CLI)

```bash
# Smoke / current data
python -m src.track_b_analysis \
    --raw results/track_b_raw.csv \
    --judgments results/track_b_judgments.csv \
    --out results/analysis --label "SMOKE n=5"

# Synthetic dry run (no real data needed)
python -m src.track_b_analysis --synthetic --out results/analysis_synthetic
```

Writes `track_b_over_refusal.csv`, `track_b_harmful_refusal.csv`,
`track_b_language_deltas.csv`, `track_b_judge_distribution.csv`,
`analysis_meta.json`, and PNG figures to `--out`.

### Library

```python
from src.track_b_analysis import analyze_files, metrics, plots
import matplotlib.pyplot as plt

result = analyze_files("results/track_b_raw.csv", "results/track_b_judgments.csv",
                       label="n=30 reviewed run")
print(result.over_refusal)            # rates + Wilson CIs
print(result.language_deltas)         # Δ OR vs English

ax = plots.plot_or_bars(result.over_refusal)
plt.show()
```

### Notebook

`notebooks/track_b_analysis.ipynb` runs the whole thing inline with
interpretation cells. Execute headless with:

```bash
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace track_b_analysis.ipynb
```

## Pointing at real n≥30 results later

1. Produce real artifacts (no `--smoke-stub`, human-reviewed translations):
   ```bash
   python main.py track-b prepare
   python main.py track-b run --limit 90 --force-inference   # 30 ids × 3 langs
   ```
2. Re-run the analysis at the **same paths** — nothing else changes:
   ```bash
   python -m src.track_b_analysis --label "n=30 reviewed, 2027-xx"
   ```
   or edit the load cell + `DATA_LABEL` in the notebook.
3. With n≥30 per cell the Wilson CIs tighten enough to discuss directionally.
   Still re-run judge calibration on Hindi/Hinglish before trusting magnitudes.

## Synthetic fixtures

`synthetic.py` generates a deterministic, clearly-labelled fixture
(`model = "SYNTHETIC-fixture"`, `synthetic = "True"`) so the pipeline and
notebook run even when real data is sparse. Regenerate with:

```bash
python -m src.track_b_analysis.synthetic   # writes fixtures/synthetic_track_b_*.csv
```

## Tests

```bash
python -m pytest src/track_b_analysis/tests/ -q
```

Covers Wilson CIs (known values, monotonicity, clamping), grouping/denominator
logic, benign-vs-harmful separation, language deltas, judge agreement/κ, and
synthetic-data integrity.
