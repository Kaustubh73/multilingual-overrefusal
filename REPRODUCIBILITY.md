# Reproducibility Guide

This project records every `track-b run` and `replicate` invocation under `runs/<run-id>/` with a **config snapshot** and **manifest** so experiments can be re-run or audited later.

## What gets recorded

Each run writes:

| File | Purpose |
|------|---------|
| `runs/<run-id>/config.yaml` | Effective knobs (models, limits, paths, seeds) frozen at run start |
| `runs/<run-id>/manifest.json` | Git commit (+ dirty flag), Python version, key dependency versions, full `pip freeze`, RNG seeds, hardware summary, CLI argv, output paths, timestamps, status |

**Run id format:** `YYYYMMDDTHHMMSSZ-<pipeline>-<8-char-hex>` (auto-generated unless you pass `--run-id`).

RNG seeds default to `42` for `torch`, `numpy`, and `random` and are applied at run start. HF inference uses greedy decoding (`do_sample=False`), so outputs are deterministic given the same model weights and environment.

## Local workflow

### 1. Run an experiment (manifest auto-written)

```bash
# Track B smoke scaffold (no GPU) — safe default for CI / laptops
python main.py track-b run --limit-per-task 5 --smoke-stub --skip-judge --no-report

# Track B with optional explicit run id
python main.py track-b run --limit-per-task 5 --smoke-stub --run-id my-smoke-001

# Paper replication smoke (needs existing raw CSV or HF for inference)
python main.py replicate --limit 5 --skip-calibration
```

After completion, note the printed path, e.g. `runs/20260623T120000Z-track-b-a1b2c3d4/manifest.json`.

### 2. Reproduce a prior run

```bash
# Inspect plan without executing
python main.py reproduce --run-id 20260623T120000Z-track-b-a1b2c3d4 --dry-run

# Fresh re-run with same config + seeds (writes a new manifest)
python main.py reproduce --run-id 20260623T120000Z-track-b-a1b2c3d4

# Resume from existing output CSVs instead of overwriting
python main.py reproduce --run-id 20260623T120000Z-track-b-a1b2c3d4 --resume

# Pin the new manifest id
python main.py reproduce --run-id 20260623T120000Z-track-b-a1b2c3d4 --new-run-id reproduction-attempt-2
```

`reproduce` reloads `config.yaml`, reapplies seeds, and dispatches to the original pipeline (`track-b` or `replicate`).

### 3. Inspect a manifest

```bash
cat runs/<run-id>/manifest.json | python -m json.tool
cat runs/<run-id>/config.yaml
```

## Docker

Build once:

```bash
docker compose build
```

### CPU smoke (no GPU)

Uses `--smoke-stub` so no HuggingFace weights are loaded. Judge step skipped for speed.

```bash
docker compose --profile cpu-smoke run --rm cpu-smoke
```

Artifacts land in mounted `./results` and `./runs`.

### CPU replication smoke

Assumes `results/paper_raw.csv` exists or use after a local inference step. Skips judge/calibration in the compose command.

```bash
docker compose --profile cpu-replicate run --rm cpu-replicate
```

### GPU full (limited track-b)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Runs `--limit-per-task 5` inference on GPU (judge skipped in default compose command).

```bash
docker compose --profile gpu-full run --rm gpu-full
```

### Environment variables

| Variable | Default | Notes |
|----------|---------|-------|
| `OVERREFUSAL_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | HF model id for inference |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU selection for `gpu-full` |
| `HF_HOME` | `/app/.cache/huggingface` (in container) | Model cache volume |

Volumes: `./data`, `./results`, `./runs`, and a named `huggingface_cache` volume.

Custom command example:

```bash
docker compose --profile cpu-smoke run --rm cpu-smoke \
  python main.py track-b run --limit 5 --smoke-stub --skip-judge
```

## Limitations

- **Output paths** default to shared `results/*.csv`; manifests record paths but do not isolate per-run output directories. Reproduce with default settings overwrites those files unless you use `--resume`.
- **Ollama judge** (`gpt-oss:20b`) is not containerized here; `--skip-judge` is used in Docker smoke profiles. Full judge reproducibility needs Ollama on the host with the same model tag.
- **Phase 1 `run` / `analyze`** commands do not yet write manifests (Track B + replicate only).
- **Model weights** must match (same HF revision / Ollama tag). Manifest records package versions, not per-weight checksums.
- **Smoke n=5 rates** are scaffolding only — not publishable results.

## Tests

```bash
python -m pytest src/reproducibility/tests/ -q
```
