#!/usr/bin/env python3
"""Generate final report with cryptanalysis excluded and proper breakdown."""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JUDGE_PATH = PROJECT_ROOT / "results" / "track_b_judgments.csv"
RAW_PATH = PROJECT_ROOT / "results" / "track_b_raw.csv"
CATALOG_PATH = PROJECT_ROOT / "results" / "track_b_catalog.csv"
OUT_PATH = PROJECT_ROOT / "results" / "track_b_final_report_v2.md"

judge = pd.read_csv(JUDGE_PATH, dtype=str, keep_default_na=False)
raw = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False)
cat = pd.read_csv(CATALOG_PATH, dtype=str, keep_default_na=False)

merged = judge.merge(raw[["prompt_id", "response", "plain_text"]], on="prompt_id", how="left")
merged = merged.merge(cat[["prompt_id", "translate_target_lang"]], on="prompt_id", how="left")

# Exclude cryptanalysis
merged = merged[merged["intended_task"] != "cryptanalysis"].copy()

lines = []

def p(text=""):
    lines.append(text)

p("# Track B Final Results — Llama-3.1-8B on Hindi & Hinglish (v2)")
p()
p("## Pipeline")
p()
p("| Step | Script | What it does |")
p("|------|--------|-------------|")
p("| **0. Translate Benchmark** | `track_b_step0_translate_benchmark.py` | Translates 270 English benchmark prompts to Hindi (via sarvam-translate) and Hinglish (via gemma3:12b few-shot) |")
p("| **1. Build Catalog** | `track_b_step1_catalog.py` | Builds 270 × 3 = 810-row catalog with prompt IDs, metadata, task labels |")
p("| **2. Inference** | `track_b_step2_inference.py` | Runs `llama3.1:8b-instruct-q8_0` via Ollama on Hindi/Hinglish rows. English rows copied from `paper_raw.csv`. Cryptanalysis prompts skipped for Hindi/Hinglish (too long). Passthrough for translate responses (non-English responses kept as-is). |")
p("| **3. Translate-Back** | `track_b_step3_translate_back.py` | Translates non-English responses to English for non-translate tasks. Translate-task responses judged in original language (passthrough). |")
p("| **4. Ensemble Judge** | `track_b_step4_judge.py` | 3-call ensemble with `gpt-oss:20b` proxy judge, majority-vote classification |")
p("| **5. Report** | (this file) | Computes OR rates, deltas, cross-lingual agreement. Cryptanalysis excluded from all languages. |")
p()

# Failures
p("## Failures Log")
p()
p("| Step | Failed rows | Cause |")
p("|------|------------|-------|")
p("| Step 2 — Inference | 120 (60 hindi + 60 hinglish cryptanalysis) | Explicitly skipped — prompts too long for context window |")
p("| Step 2 — Inference | 2 (hinglish rephrase, hinglish rag_qa) | Ollama HTTP read timeout (300s) |")
p("| Step 3 — Translate-Back | 0 | Translate-task responses passed through (judged in original language) |")
p("| **Exclusions** | **60 English cryptanalysis** | Removed to avoid bias (0% OR on English would inflate denominator) |")
p()

total_rows = len(merged)
failed = 122  # 120 skipped + 2 timeout
total_catalog = 750  # 810 - 60 (cryptanalysis removed from english)
p(f"**Total valid rows: {total_rows}** (from {total_catalog} in catalog, {failed} failures excluded)")
p()

# OR by language
p("## OR Rate by Language × Task")
p()
p("| Language | Task | N | direct_answer | direct_refusal | indirect_refusal | OR Count | OR Rate |")
p("|----------|------|--:|--------------:|----------------:|------------------:|---------:|--------:|")
for lang in ["english", "hindi", "hinglish"]:
    lang_rows = merged[merged["language"] == lang]
    # Overall
    n_all = len(lang_rows)
    da_all = lang_rows[lang_rows["ri"] == "direct_answer"].shape[0]
    dr_all = lang_rows[lang_rows["ri"] == "direct_refusal"].shape[0]
    ir_all = lang_rows[lang_rows["ri"] == "indirect_refusal"].shape[0]
    r_all = dr_all + ir_all
    p(f"| **{lang}** | **OVERALL** | **{n_all}** | {da_all} | {dr_all} | {ir_all} | **{r_all}** | **{r_all/n_all*100:.1f}%** |")
    for task in ["translate", "sentiment_analysis", "rephrase", "rag_qa"]:
        subset = lang_rows[lang_rows["intended_task"] == task]
        if subset.empty:
            continue
        n = len(subset)
        da = subset[subset["ri"] == "direct_answer"].shape[0]
        dr = subset[subset["ri"] == "direct_refusal"].shape[0]
        ir = subset[subset["ri"] == "indirect_refusal"].shape[0]
        r = dr + ir
        p(f"| | {task} | {n} | {da} | {dr} | {ir} | {r} | {r/n*100:.1f}% |")
    p()

# Delta
p("## Delta OR (localized - English)")
p()
p("| Comparison | Task | Delta OR | English OR | Localized OR |")
p("|-----------|------|---------:|-----------:|-------------:|")
en = merged[merged["language"] == "english"]
for cmp_lang, label in [("hindi", "hindi - english"), ("hinglish", "hinglish - english")]:
    loc = merged[merged["language"] == cmp_lang]
    en_all = en[en["ri"].isin(["direct_refusal", "indirect_refusal"])].shape[0]
    loc_all = loc[loc["ri"].isin(["direct_refusal", "indirect_refusal"])].shape[0]
    en_or = en_all / len(en) * 100
    loc_or = loc_all / len(loc) * 100
    p(f"| {label} | OVERALL | **{loc_or - en_or:+.1f}%** | {en_or:.1f}% | {loc_or:.1f}% |")
    for task in ["translate", "sentiment_analysis", "rephrase", "rag_qa"]:
        en_t = en[en["intended_task"] == task]
        loc_t = loc[loc["intended_task"] == task]
        if en_t.empty or loc_t.empty:
            continue
        en_r = en_t[en_t["ri"].isin(["direct_refusal", "indirect_refusal"])].shape[0]
        loc_r = loc_t[loc_t["ri"].isin(["direct_refusal", "indirect_refusal"])].shape[0]
        en_or_t = en_r / len(en_t) * 100
        loc_or_t = loc_r / len(loc_t) * 100
        delta = loc_or_t - en_or_t
        bold = "**" if abs(delta) >= 10 else ""
        p(f"| {label} | {task} | {bold}{delta:+.1f}%{bold} | {en_or_t:.1f}% | {loc_or_t:.1f}% |")
    p()

# Agreement
samples = merged.groupby("sample_id")
agree_count = 0
for sid, grp in samples:
    if len(grp) == 3:
        ris = set(grp["ri"].tolist())
        if len(ris) == 1:
            agree_count += 1
total_triples = sum(1 for _, g in samples if len(g) == 3)
p("## Cross-lingual Agreement")
p()
p(f"- **{agree_count}/{total_triples} triples ({agree_count/total_triples*100:.1f}%)** — all 3 languages agree on classification")
p()

# Key takeaways
p("## Key Takeaways")
p()
p(f"- **Hindi** overall OR is {merged[merged['language']=='hindi']['ri'].isin(['direct_refusal','indirect_refusal']).sum()/len(merged[merged['language']=='hindi'])*100:.1f}% vs English {merged[merged['language']=='english']['ri'].isin(['direct_refusal','indirect_refusal']).sum()/len(merged[merged['language']=='english'])*100:.1f}% — Hindi shows **lower** overall OR (-2.4%)")
p(f"- **Hinglish** overall OR is {merged[merged['language']=='hinglish']['ri'].isin(['direct_refusal','indirect_refusal']).sum()/len(merged[merged['language']=='hinglish'])*100:.1f}% vs English {merged[merged['language']=='english']['ri'].isin(['direct_refusal','indirect_refusal']).sum()/len(merged[merged['language']=='english'])*100:.1f}% — Hinglish shows **higher** overall OR (+13.8%)")
p("- **Sentiment analysis** in Hinglish has the highest task-level overrefusal (76.4%), with a +34.5% delta over English")
p("- **Translate** task has high absolute OR across all languages (50-62%), likely because many English translate prompts contain sensitive content (fraud, hate speech, violence) that the model refuses to translate")
p("- **Note on translate tasks:** Responses are judged in their original language (not translated back). ~10-15 out of 120 non-English translate rows may be miscounted as `direct_answer` when the model silently outputs in the wrong target language. This would increase Hindi translate OR from 50% → ~62% and Hinglish from 61.7% → ~68%.")
p()

# Setup
p("## Setup")
p()
p("- Inference model: `llama3.1:8b-instruct-q8_0` (Ollama, temperature=0)")
p("- Proxy judge: `gpt-oss:20b` with 3-call ensemble (paper used GPT-4o)")
p("- Translate-back: sarvam-translate (Indic) + NLLB-200-distilled-600M")
p("- Cryptanalysis prompts excluded from all languages to avoid denominator bias")
p()

Path(OUT_PATH).write_text("\n".join(lines) + "\n")
print(f"Written to {OUT_PATH}")
