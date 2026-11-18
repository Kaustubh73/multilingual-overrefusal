"""Pipeline steps `sample` and `generate`: build the prompt sample, then have models answer it.

sample    Draws a stratified sample of seed prompts, renders each in the target language with the
          localized task template, writes `sample.jsonl`. The same seeds are used in every language;
          the task a seed gets rotates by language so each language x task cell is filled evenly.
generate  Runs each model in `[generate].models` over the sample (greedy, resumable, one
          `gen_<model>.jsonl` each), then joins sample + responses into `responses.jsonl`, which the
          judge and hand_tag steps read.

Backends: mlx (Mac), ollama (local server), llamacpp (local llama-server, GGUF from Hugging Face).
Modal is not ported here: the old x12_modal.py is a stand-alone Modal app and stays on its branch.
"""
import gc
import importlib
import json
import random
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from .config import TASKS
from .io import append_jsonl, input_hash, latest_matching, read_jsonl
from .runner import step

BATCH_SIZE = 16   # MLX batch size. Padding changes float rounding, so keep one size per output file.

# Model key -> (backend, model id). A key may also be given as [generate.model_ids] in the config.
MODELS = {
    "qwen3.5-9b": ("mlx", "lmstudio-community/Qwen3.5-9B-MLX-8bit"),
    "llama-3.1-8b": ("mlx", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
    "aya-expanse-8b": ("mlx", "mlx-community/aya-expanse-8b-4bit"),
    # mlx_lm 0.31.3 cannot load Gemma 4 ("gemma4_unified"), so it runs through Ollama.
    "gemma-4-e4b": ("ollama", "gemma4:e4b"),
    "aya-expanse-8b-gguf": ("llamacpp", "bartowski/aya-expanse-8b-GGUF:Q4_K_M"),
}

DATA_DIR = Path("data")
TEMPLATES = DATA_DIR / "dataset_v2" / "templates_localized.json"
SEEDS = DATA_DIR / "dataset_v1" / "seeds_en.parquet"
NEW_SHEET = DATA_DIR / "dataset_v2" / "sheets" / "new4_forward.csv"          # ur, mt, lo forward translations
OLD_SHEET = DATA_DIR / "dataset_v1" / "sheets" / "translations_gt_full.parquet"   # yo (long format)
SHEET_LANGS = ("ur", "mt", "lo")
# Language the `translate` task asks for, per source language (assign_translate_targets, seed 42).
TRANSLATE_TARGET = {"ur": "zh", "lo": "tr", "mt": "id", "yo": "ar"}
HARMFUL_SHARE = 0.6          # old design: 27 harmful + 18 harmless of 45 seeds
MAX_SEED_CHARS = 300
BAD_CONTENT = re.compile(r"^(Loading\.\.\.|#ERROR!|#N/A)")   # spreadsheet leftovers


# ---------------------------------------------------------------- sample

def resolve_tasks(tasks) -> list[str]:
    """`"all"` is the nine text tasks of the pilot (schema_constrained_output has no pilot template)."""
    if tasks == "all":
        return [t for t in TASKS if t != "schema_constrained_output"]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s) {unknown}")
    return list(tasks)


def load_content(languages: list[str]):
    """DataFrame content_id, lang, content: the Google Translate translation of every seed."""
    import pandas as pd
    parts = []
    sheet_langs = [lang for lang in languages if lang in SHEET_LANGS]
    other = [lang for lang in languages if lang not in SHEET_LANGS]
    if sheet_langs:
        new = pd.read_csv(NEW_SHEET, keep_default_na=False)
        parts.append(pd.DataFrame([{"content_id": r.content_id, "lang": lang, "content": getattr(r, lang)}
                                   for r in new.itertuples() for lang in sheet_langs]))
    if other:
        old = pd.read_parquet(OLD_SHEET)
        parts.append(old[old.lang.isin(other)][["content_id", "lang", "content"]])
    return pd.concat(parts, ignore_index=True)


def build_sample(languages: list[str], tasks: list[str], seeds_per_cell: int, seed: int,
                 translate_targets: dict | None = None):
    """Rows for `sample.jsonl`. Every language x task cell gets `seeds_per_cell` rows."""
    import pandas as pd
    translate_targets = translate_targets or TRANSLATE_TARGET
    n_seeds = len(tasks) * seeds_per_cell
    seeds = pd.read_parquet(SEEDS)
    content = load_content(languages)
    ok = content[content.content.map(lambda s: bool(str(s).strip()) and not BAD_CONTENT.match(str(s)))]
    have = ok.groupby("content_id").lang.nunique()
    usable = set(have[have == len(languages)].index)      # seeds translated in every language
    pool = seeds[seeds.content_id.isin(usable) & (seeds.content.str.len() <= MAX_SEED_CHARS)]
    rng = random.Random(seed)
    harmful = sorted(pool[pool.label == "harmful"].content_id)
    harmless = sorted(pool[pool.label == "harmless"].content_id)
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    n_harmful = round(n_seeds * HARMFUL_SHARE)
    chosen = harmful[:n_harmful] + harmless[:n_seeds - n_harmful]   # rank order drives the task rotation
    if len(chosen) < n_seeds:
        raise ValueError(f"only {len(chosen)} usable seeds, need {n_seeds}")

    templ = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    names = templ["lang_names"]
    meta = seeds.set_index("content_id")
    cell = ok.set_index(["content_id", "lang"]).content
    rows = []
    for li, lang in enumerate(languages):
        for rank, cid in enumerate(chosen):
            task = tasks[(rank + li * 2) % len(tasks)]
            text = cell[(cid, lang)]
            tgt = translate_targets.get(lang, "") if task == "translate" else ""
            tname = names.get(lang, {}).get(tgt, tgt) if tgt else ""
            prompt = templ["templates"][lang][task].format(content=text, translate_target_name=tname)
            rows.append({"row_id": f"{lang}-{task}-{cid}", "content_id": cid, "lang": lang, "task": task,
                         "label": meta.loc[cid, "label"], "category": meta.loc[cid, "category"] or "",
                         "en_content": meta.loc[cid, "content"], "content": text, "prompt": prompt,
                         "translate_target": tgt})
    df = pd.DataFrame(rows)
    sizes = df.groupby(["lang", "task"]).size()
    if not (sizes.min() == sizes.max() == seeds_per_cell):
        raise ValueError(f"uneven cells:\n{sizes.describe()}")
    return df


@step("sample")
def sample_step(cfg: dict) -> None:
    """Config [data]: languages, tasks ("all" or a list), seeds_per_cell, translate_targets (optional)."""
    d = cfg["data"]
    tasks = resolve_tasks(d.get("tasks", "all"))
    df = build_sample(d["languages"], tasks, d["seeds_per_cell"], cfg.get("seed", 0), d.get("translate_targets"))
    out = Path(cfg["output_dir"]) / "sample.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", lines=True, force_ascii=False)
    print(f"{len(df)} rows ->", out, df.groupby("lang").label.value_counts().unstack().to_dict())


# ---------------------------------------------------------------- generation backends

def generate_in_batches(model, tokenizer, texts: list[str], max_tokens: int, batch_size: int = BATCH_SIZE):
    """Yield (index, text) for greedy MLX generation over already-templated prompts."""
    gm = importlib.import_module("mlx_lm.generate")
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        out = gm.batch_generate(model, tokenizer, [tokenizer.encode(t) for t in chunk],
                                max_tokens=max_tokens).texts
        for j, o in enumerate(out):
            yield i + j, o


def render_chat(tokenizer, prompt: str) -> str:
    """Chat-template one user turn. Thinking is switched off where the template supports it."""
    msg = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def ollama_chat(model: str, prompt: str, max_tokens: int, seed: int = 0, temperature: float = 0.0) -> str:
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens, "num_ctx": 4096}}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["message"].get("content", "")


def _gen_mlx(key: str, model_id: str, todo: list[dict], out: Path, max_tokens: int, batch_size: int,
             hashes: dict) -> None:
    from mlx_lm import load
    print(f"[{key}] loading {model_id}; {len(todo)} prompts (batch {batch_size})", flush=True)
    model, tok = load(model_id)
    texts = [render_chat(tok, r["prompt"]) for r in todo]
    t0 = time.time()
    for i, resp in generate_in_batches(model, tok, texts, max_tokens, batch_size):
        rid = todo[i]["row_id"]
        append_jsonl(out, [{"row_id": rid, "model": key, "response": resp, "batch": batch_size,
                            "input_hash": hashes[rid]}])
        if i % batch_size == batch_size - 1:
            print(f"[{key}] {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    del model
    gc.collect()
    try:
        importlib.import_module("mlx.core").clear_cache()
    except ImportError:
        pass


def _gen_ollama(key: str, model_id: str, todo: list[dict], out: Path, max_tokens: int, seed: int,
                temperature: float, hashes: dict) -> None:
    for r in todo:
        t0 = time.time()
        resp = ollama_chat(model_id, r["prompt"], max_tokens, seed, temperature)
        append_jsonl(out, [{"row_id": r["row_id"], "model": key, "response": resp,
                            "latency_s": round(time.time() - t0, 1), "input_hash": hashes[r["row_id"]]}])


def _gen_llamacpp(key: str, model_id: str, todo: list[dict], out: Path, max_tokens: int, seed: int,
                  hashes: dict, port: int = 8099) -> None:
    """Start llama-server on a Hugging Face GGUF (`repo:quant`), greedy via top_k=1, stop it afterwards."""
    log = open(out.with_suffix(".server.log"), "ab")
    srv = subprocess.Popen(["llama-server", "-hf", model_id, "--port", str(port), "--jinja", "-c", "2048",
                            "-ngl", "99"], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(1800):                       # up to an hour: the first run downloads the weights
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise TimeoutError("llama-server did not become healthy")
        for r in todo:
            body = json.dumps({"messages": [{"role": "user", "content": r["prompt"]}], "max_tokens": max_tokens,
                               "temperature": 0, "top_k": 1, "seed": seed}).encode()
            t0 = time.time()
            req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
            resp = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
            append_jsonl(out, [{"row_id": r["row_id"], "model": key, "response": resp,
                                "latency_s": round(time.time() - t0, 1), "input_hash": hashes[r["row_id"]]}])
    finally:
        srv.terminate()


def resolve_model(key: str, g: dict) -> tuple[str, str]:
    """(backend, model id): from [generate.model_ids], else the MODELS registry. `backend` in the
    config forces a backend for every model that is not in the registry."""
    if key in g.get("model_ids", {}):
        return g.get("backend", "mlx"), g["model_ids"][key]
    if key in MODELS:
        return MODELS[key]
    raise ValueError(f"unknown model {key!r}: add it to [generate.model_ids] or MODELS. Known: {sorted(MODELS)}")


@step("generate")
def generate_step(cfg: dict) -> None:
    """Config [generate]: models = [keys], max_new_tokens, temperature (only 0 is supported on mlx and
    llamacpp), batch_size (mlx), model_ids = {key = id} (optional)."""
    g = cfg["generate"]
    out_dir = Path(cfg["output_dir"])
    sample = read_jsonl(out_dir / "sample.jsonl")
    if not sample:
        raise FileNotFoundError(f"{out_dir / 'sample.jsonl'} missing or empty: run the sample step first")
    seed, max_tokens = cfg.get("seed", 0), g.get("max_new_tokens", 128)
    for key in g["models"]:
        backend, model_id = resolve_model(key, g)
        if backend in ("mlx", "llamacpp") and g.get("temperature", 0.0) != 0:
            raise ValueError(f"{backend} generation is greedy only; got temperature={g['temperature']}")
        out = out_dir / f"gen_{key}.jsonl"
        hashes = _gen_hashes(sample, backend, model_id, g, seed)
        done = latest_matching(read_jsonl(out), lambda rec: rec["row_id"], hashes)
        todo = [r for r in sample if r["row_id"] not in done]
        if not todo:
            continue
        if backend == "mlx":
            _gen_mlx(key, model_id, todo, out, max_tokens, g.get("batch_size", BATCH_SIZE), hashes)
        elif backend == "ollama":
            _gen_ollama(key, model_id, todo, out, max_tokens, seed, g.get("temperature", 0.0), hashes)
        elif backend == "llamacpp":
            _gen_llamacpp(key, model_id, todo, out, max_tokens, seed, hashes)
        else:
            raise ValueError(f"backend {backend!r} is not supported here (mlx | ollama | llamacpp)")
    _write_responses(out_dir, sample, g["models"], {key: _gen_hashes(sample, *resolve_model(key, g), g, seed)
                                                    for key in g["models"]})


def _gen_hashes(sample: list[dict], backend: str, model_id: str, g: dict, seed: int) -> dict:
    """row_id -> hash of everything that decides the reply: prompt, model, backend and decoding settings.
    A generation record is reused on resume only if its hash matches, so an edited prompt, a new
    sample or a changed setting is regenerated instead of silently kept."""
    settings = {"backend": backend, "model_id": model_id, "max_new_tokens": g.get("max_new_tokens", 128),
                "temperature": g.get("temperature", 0.0), "seed": seed}
    if backend == "mlx":
        settings["batch_size"] = g.get("batch_size", BATCH_SIZE)   # padding changes the output
    return {r["row_id"]: input_hash(r["prompt"], settings) for r in sample}


def _write_responses(out_dir: Path, sample: list[dict], models: list[str], hashes: dict[str, dict]) -> None:
    """responses.jsonl = every sample field + model + response, in model then sample order.

    hashes: model key -> {row_id: input hash}. Only the latest generation made from the current
    inputs is used; records for rows no longer in the sample, or from older inputs, are left out.
    """
    rows = []
    for key in models:
        gens = latest_matching(read_jsonl(out_dir / f"gen_{key}.jsonl"), lambda rec: rec["row_id"], hashes[key])
        missing = [r["row_id"] for r in sample if r["row_id"] not in gens]
        if missing:
            raise RuntimeError(f"{key}: {len(missing)} sample rows have no current generation (e.g. {missing[0]})")
        rows += [{**r, **gens[r["row_id"]]} for r in sample]
    (out_dir / "responses.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} responses -> {out_dir / 'responses.jsonl'}")
