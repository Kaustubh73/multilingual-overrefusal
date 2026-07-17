# SafeConstellations toy scaffold (CPU-only)

A **runnable plumbing** of the SafeConstellations intervention
([arXiv:2508.11290](https://arxiv.org/abs/2508.11290)) for the multilingual
over-refusal project. It demonstrates the full pipeline end-to-end on a tiny
synthetic dataset with **no GPU and no model downloads**, so you can verify the
mechanics before committing real inference budget (see the Phase plan in
`research/notes/safeconstellations-replication-plan.md`).

> This is scaffolding for understanding the intervention pipeline. The toy
> numbers are **not** results and must never be cited.

## What it implements

The same five stages as the paper, behind small/mockable interfaces:

| Stage | Module | What it does |
| --- | --- | --- |
| Load hidden states | `hidden_states.py` | Per-layer `(n_layers, hidden_dim)` stack. Toy synthetic backend (default) or gated HF backend. |
| Extract refusal/target subspaces | `task_store.py` | Per-`(task, layer)` refusal & target centroids → steering direction. |
| Compute steering vectors | `linalg.py` | Centroids, refusal/steering directions, projection, ablation, additive steering. |
| Apply steering at inference | `steering.py` | Detect task via cosine; conditionally steer top-K layers toward the non-refusal manifold (confidence ≥ threshold, task ∈ benign set). |
| Evaluate | `evaluate.py` | Over-refusal (OR, benign tasks) vs harmful-refusal (HR, truly harmful) — baseline vs steered. |

`pipeline.py` wires it all together; `data.py` is the tiny toy dataset;
`demo.py` is the runnable entry point.

## How the toy world works (so the result is meaningful)

The synthetic backend emits hidden states whose projection onto a hidden
"refusal axis" equals a refusal logit:

```
refusal_logit = base + w_looks·looks_harmful + w_truly·truly_harmful
```

- **benign + looks-harmful** → logit `+1` → model over-refuses (the OR bug).
- **truly harmful** → logit `+5` → legitimate refusal.

Bounded steering subtracts ~`alpha` from the refusal-axis projection, which
removes the small "looks harmful" bump (OR → answer) but leaves the much larger
genuine-harm signal above threshold (HR preserved). This mirrors the paper's
claim: task-conditioned steering reduces OR **without** a global safety tradeoff.

## Run it

From the project root (`Me/multilingual-overrefusal`):

```bash
# Toy demo (CPU, no downloads)
python -m src.safeconstellations.demo --quiet

# Sweep the knobs
python -m src.safeconstellations.demo --alpha 2.5 --threshold 0.85 --mode additive
python -m src.safeconstellations.demo --mode ablate

# Tests (vector math + integration)
python -m pytest src/safeconstellations/tests/ -q
```

Expected demo output: baseline OR ~50% → steered OR ~0%, harmful-refusal 100% in
both. (Uses this project's `.venv`, e.g. `.venv/bin/python -m ...`.)

## Swapping in the real experiment

The toy is intentionally a drop-in shape match for real components:

1. **Real hidden states.** Set `HiddenStateConfig(backend="hf", hf_model_id=...)`.
   `HFHiddenStateModel` reads `output_hidden_states` from a forward pass. Start
   with the open `SmolLM2-1.7B-Instruct`; switch to `Llama-3.1-8B-Instruct` for
   parity (gated — `huggingface-cli login`). **This path loads a real model; keep
   it off by default and run only when you have the compute.**
2. **Real refusal labels.** Replace the toy `decide_refusal` rule with this
   repo's existing judge/classifier path (`src/refusal_judge.py`,
   `src/response_classifier.py`) applied to generated text.
3. **Real data.** Replace `data.toy_dataset()` with rows from
   `Sakonii/task-over-refusal-dataset` (the Track A/B catalog already carries
   `intended_task` / `is_benign_task`), keeping the `Example` schema.
4. **Layer selection.** `TaskEmbeddingsStore.fit(top_k_layers=..., detection_layer=...)`
   — set the layer indices identified in Phase 0.2 of the replication plan.

## Layout

```
src/safeconstellations/
├── config.py          # ToyWorldConfig, HiddenStateConfig, SteeringConfig
├── data.py            # tiny synthetic over-refusal dataset (Example schema)
├── linalg.py          # vector math (unit-tested)
├── hidden_states.py   # toy backend (default) + gated HF backend
├── task_store.py      # Task Embeddings Store: per-(task, layer) centroids
├── steering.py        # conditional steering controller
├── evaluate.py        # OR / HR metrics, baseline vs steered
├── pipeline.py        # end-to-end orchestration + report formatting
├── demo.py            # runnable CPU demo
└── tests/             # pytest: test_linalg.py, test_pipeline.py
```

## Scope / guardrails

- No GPU inference and no large downloads by default (toy backend is pure numpy).
- Toy outputs are plumbing checks, not publishable numbers.
- Real steering replication is gated behind the Track B production gate per the
  replication plan; this scaffold only de-risks the engineering.
