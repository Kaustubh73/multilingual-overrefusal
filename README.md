# Multilingual Phase 1 Pipeline

Exploratory study: do LLMs show **task adherence**, **refusal**, or **failure** patterns differently across **English**, **Hindi**, and **Hinglish** on benign prompts?

Uses [SafeConstellations](https://github.com/Sakonii/SafeConstellations) data sources and task templates only — **no** steering, hidden states, UMAP, or embedding interventions.

## Phase 1 tasks

| Task | Description |
|------|-------------|
| `sentiment` | Analyze sentiment of language-aligned text |
| `rephrase` | Rephrase and fix grammar |
| `translation` | EN→Hindi; HI/Hinglish→English |

RAG-QA is deferred.

## Setup

```bash
pip install -r requirements.txt
# Ollama (recommended): pull models
ollama pull llama3.1:8b-instruct
ollama pull qwen2.5:7b-instruct
ollama pull gemma3:12b
```

Optional HuggingFace fallback for Llama when Ollama is unavailable:

```bash
huggingface-cli login
```

## Paper replication (SafeConstellations Table 2)

Reproduce over-refusal on the official benchmark using HF Llama-3.1-8B and a local OR-Bench judge.

```bash
# Audit official dataset vs legacy CSVs
python main.py audit-paper

# Pull judge model (once)
ollama pull gpt-oss:20b

# Full replication: 270 test rows → paper_raw.csv → paper_judgments.csv → replication_report.md
python main.py replicate

# Smoke test
python main.py replicate --limit 5 --skip-calibration

# Resume / judge-only / metrics-only
python main.py replicate --skip-inference   # judge existing raw
python main.py replicate --skip-inference --skip-judge  # re-report only
```

Artifacts: `data/paper/dataset_audit.md`, `results/paper_raw.csv`, `results/paper_judgments.csv`, `results/replication_report.md`.

## Track B (multilingual over-refusal)

After Track A gate **CONTINUE**, run Hindi/Hinglish wrappers on translate + sentiment rows from the official test split.

```bash
# One-time setup: select 115 rows, translation CSVs, build catalog
python main.py track-b prepare --seed-smoke --assist-translate

# Smoke test: balanced tasks (5 translate + 5 sentiment sample_ids × 3 langs → ≤30 rows)
python main.py track-b run --limit-per-task 5

# Legacy: first N sample_ids in catalog order (translate-heavy)
python main.py track-b run --limit 10

# Scaffold smoke (no HF): English from Track A paper_raw, stubs for hi/hing
python main.py track-b run --limit 10 --smoke-stub

# Partial reruns
python main.py track-b run --limit 10 --skip-inference
python main.py track-b run --limit 10 --skip-inference --skip-judge
```

**Human review tooling (no GPU):**

```bash
# Prioritized worksheet for weekly 20-row review
python tools/triage_translations.py

# G-translate-60 / G-n30 progress (also --json for scripts)
python tools/translation_progress.py
```

Open `tools/translation_review.html` in a browser and load `data/track_b/reviewed_translations.csv` for inline editing.

**Prepare** creates:

- `data/track_b/selected_rows.csv` — 115 rows (60 translate + 55 sentiment)
- `data/track_b/translation_requests.csv` — manual/Ollama translation prompts
- `data/track_b/reviewed_translations.csv` — Hindi/Hinglish with `reviewed=true` required for inference
- `data/track_b/track_b_catalog.csv` — one row per `(sample_id, language)`

**Run** produces:

- `results/track_b_raw.csv`, `results/track_b_judgments.csv`
- `results/track_b_report.md` — OR_rate by language × task, Δ OR vs English
- `results/track_b_metrics.csv`

Use `--seed-smoke` to auto-review 5 translate + 5 sentiment rows for scaffold validation only (full 115-row run requires human review). See [data/paper/track_b_multilingual.md](data/paper/track_b_multilingual.md).

## Workflow

### 1. Prepare data

```bash
python main.py prepare
```

Creates:

- `data/raw/base_prompts.csv` — benign prompts (XSTest + Sakonii)
- `data/translations/translation_requests.csv`
- `data/translations/reviewed_translations.csv` — fill Hindi/Hinglish, set `reviewed=true`

### 2. Run experiment

```bash
python main.py run                  # 10 prompts × 3 tasks × 3 langs × 3 models
python main.py run --dry-run        # 1 prompt smoke test (27 inferences)
python main.py run --tasks sentiment rephrase
python main.py run --skip-inference # classify/summarize existing raw_outputs.csv
```

### 3. Re-analyze only

```bash
python main.py analyze
```

## Outputs

| File | Contents |
|------|----------|
| `data/prompts/prompt_catalog.csv` | `prompt_id`, `task`, `language`, `prompt`, `source_text` |
| `results/raw_outputs.csv` | `prompt_id`, `task`, `language`, `model`, `prompt`, `response` |
| `results/classified_outputs.csv` | above + `label_level1`, `label_level2` |
| `results/summary_rates.csv` | `language`, `model`, `success_rate`, `refusal_rate`, `failure_rate` |
| `results/failure_distribution.csv` | `language`, `model`, `task_dev`, `semantic`, `repetition`, `other` |
| `results/plots/` | optional matplotlib charts |

### Label schema

**Level 1:** `SUCCESS` | `FAILURE` | `REFUSAL`

**Level 2** (only if `FAILURE`): `TASK_DEVIATION` | `SEMANTIC_DISTORTION` | `REPETITION` | `OTHER`

Classification is rule-based ([`src/response_classifier.py`](src/response_classifier.py)). Spot-check with manual audit before strong claims.

## Reporting template

For each slice (language, model, task), document:

- **A. Observation** — descriptive rates only
- **B. Confounders** — small N, classifier limits, translation quality, Ollama version
- **C. Alternatives** — capability limits vs instruction-following vs heuristic mislabels
- **D. Confidence** — low / medium / high (calibrate with manual audit sample)

Do not infer causality or make paper claims from Phase 1 alone.

## Project layout

```
multilingual-overrefusal/
├── data/
│   ├── raw/base_prompts.csv
│   ├── translations/
│   ├── track_b/              # Track B selected rows, translations, catalog
│   └── prompts/prompt_catalog.csv
├── src/
│   ├── dataset.py
│   ├── tasks.py
│   ├── translate.py
│   ├── track_b_select.py
│   ├── track_b_translate.py
│   ├── track_b_prompts.py
│   ├── run_track_b.py
│   ├── track_b_metrics.py
│   ├── model_runner.py
│   ├── response_classifier.py
│   ├── analysis.py
│   ├── run_validation.py
│   ├── inference.py          # HF fallback
│   └── evaluate.py           # legacy over-refusal pilot only
├── results/
└── main.py
```

## Out of scope

- SafeConstellations steering, hidden-state extraction, UMAP, constellation analysis
- RAG-QA (Phase 1)
- Paper over-refusal-only evaluation as primary metric
- LLM-as-judge (future phase)

## Legacy

The original single-model pilot (`src/evaluate.py`, `data/results/`) used refusal-rate keywords only. The Phase 1 path is `python main.py run` → `results/`.
