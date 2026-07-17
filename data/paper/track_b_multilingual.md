# Track B: Multilingual over-refusal (after replication gate)

**Status:** Deferred until `results/replication_report.md` verdict is **CONTINUE**.

## Goal

Test whether Hindi/Hinglish task wrappers **amplify** over-refusal vs English on the same underlying sensitive texts, extending the paper’s observation that low-resource translation (Hindi, Urdu, Nepali) shows elevated OR.

## Prerequisites

1. Track A replicates directional OR (~15–22%) on official test split with HF Llama + `gpt-oss:20b` judge.
2. Judge calibration ≥80% agreement with manual `ri` on Phase-1 audit slice.

## Steps

1. **Select rows** from `Sakonii/task-over-refusal-dataset` test split:
   - `intended_task` in `translate`, `sentiment_analysis`
   - Prefer rows where paper already uses Nepali/Hindi/Urdu in `plain_text` for translation baselines.

2. **Translations**
   - Extract embedded quote from English `plain_text`.
   - Produce Hindi + Hinglish with **human review** (`reviewed=true`, meaning/naturalness scores).
   - Do not use unreviewed Ollama-only translations for claims.

3. **Prompt construction**
   - Re-wrap with the same task template structure as the paper (match `plain_text` format).
   - Keep `sample_id` lineage for joins.

4. **Inference**
   - Same HF Llama-3.1-8B-Instruct as Track A.
   - Optional: Qwen1.5-7B for cross-model comparison.

5. **Evaluation**
   - Same OR-Bench judge (`gpt-oss:20b`) on full prompt + response.
   - Compare `OR_rate` by `language` × `task_label`.

6. **Reporting**
   - Primary: Δ OR (Hindi − English), (Hinglish − English).
   - Secondary: Phase-1 task-failure rates (complementary signal).

## Out of scope until gate passes

- SafeConstellations steering / hidden-state analysis
- Claims that multilingual OR exceeds paper Table 2 without Track A baseline
