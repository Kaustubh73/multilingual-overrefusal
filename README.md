# multilingual-overrefusal

Refusal and over-refusal of LLMs across 15 languages, 8 tasks and harmful/harmless prompts.
Work in progress: nothing here is a final result.

## Layout

| Path | What |
|---|---|
| `dataset/` | Dataset build (`build_dataset.py`, localized templates), manifests, HF loader |
| `translation_qa/` | Back-translation and intent checks of the translated prompts |
| `scripts/` | Harmful-prompt screen, judges (gpt-oss, Qwen3Guard), multilingual judge, hand-tagging UI and scoring |
| `figures/` | Figure scripts (PNGs are not tracked) |
| `data/`, `results/`, `runs/` | Gitignored. Data comes from the Hub, see below |

## Data

Released dataset: [`kaustubhg73/multilingual-multitask-refusal`](https://huggingface.co/datasets/kaustubhg73/multilingual-multitask-refusal).

```bash
pip install -r requirements.txt
python dataset/load_dataset.py          # or pass a tag/revision, e.g. v1
```

English seeds come from [`Sakonii/Multitask-Multilingual-Refusal`](https://huggingface.co/datasets/Sakonii/Multitask-Multilingual-Refusal).
Scripts expect the build outputs under `data/dataset_v1/` and `data/dataset_v2/`.

## Branches

| Branch | What |
|---|---|
| `main` | Current project (this tree) |
| `safeconstellations-replication` | Replication of SafeConstellations over-refusal (Phase 1, Track A) |
| `safeconstellation-extensions` | Replication plus the Hindi/Hinglish extension (Track B) and the no-translate-back ablation |
| `task-pilot`, `decoding-truncation`, `covering-smoke` | Exploratory runs kept off main |
