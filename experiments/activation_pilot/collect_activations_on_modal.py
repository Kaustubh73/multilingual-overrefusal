"""Modal app for the pilot: weights download (CPU), CPU dry run (2 rows), GPU pilot job.

  modal run experiments/activation_pilot/collect_activations_on_modal.py::download            # CPU container, weights -> volume (no GPU minutes)
  modal run experiments/activation_pilot/collect_activations_on_modal.py --stage dryrun       # CPU container, 2 rows, batch 2 vs batch 1
  modal run --detach experiments/activation_pilot/collect_activations_on_modal.py --stage pilot --run-id pilot1
  Llama cross-model run (gated weights, HF secret "huggingface"; seeds from Qwen's gate.json; Qwen text teacher-forced):
  modal run experiments/activation_pilot/collect_activations_on_modal.py --stage download --model meta-llama/Llama-3.1-8B-Instruct
  modal run experiments/activation_pilot/collect_activations_on_modal.py --stage dryrun --model meta-llama/Llama-3.1-8B-Instruct
  modal run --detach experiments/activation_pilot/collect_activations_on_modal.py --stage pilot --run-id pilot_llama --model meta-llama/Llama-3.1-8B-Instruct

Spend guards: one container (max_containers=1), no retries, hard timeout, weights read offline from the volume.
Stop the app with `modal app stop <app-id>` when it ends. Prices not verified (L40S ~ $2/h from memory).
"""
import json
import subprocess
import threading
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3.5-9B"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE
app = modal.App("overrefusal-pilot")
vol = modal.Volume.from_name("overrefusal-pilot", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "transformers==5.3.0", "accelerate==1.13.0", "numpy", "huggingface_hub[hf_xet]")
    .add_local_file(HERE / "collect_activations.py", "/app/pilot_gen.py")
    .add_local_file(ROOT / "results/activation_pilot/pilot_inputs.jsonl", "/app/pilot_inputs.jsonl")
    .add_local_file(ROOT / "results/activation_pilot/pilot1/gate.json", "/app/qwen_gate.json")
    .add_local_file(ROOT / "results/activation_pilot/pilot1/pilot.jsonl", "/app/qwen_pilot.jsonl")
)
ENV = {"HF_HUB_CACHE": "/vol/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}


def _commit_loop(stop):
    while not stop.wait(90):
        try:
            vol.commit()
        except Exception as e:  # noqa: BLE001
            print("commit failed:", e, flush=True)


HF_SECRET = modal.Secret.from_name("huggingface")


def _run(args: list[str], git_hash: str, model: str) -> int:
    stop = threading.Event()
    t = threading.Thread(target=_commit_loop, args=(stop,), daemon=True)
    t.start()
    try:
        r = subprocess.run(["python", "/app/pilot_gen.py", "--inputs", "/app/pilot_inputs.jsonl", "--git-hash", git_hash, "--model", model, *args], env={**__import__("os").environ, **ENV})
        return r.returncode
    finally:
        stop.set()
        vol.commit()


@app.function(image=image, volumes={"/vol": vol}, secrets=[HF_SECRET], timeout=3600, cpu=4, memory=16384, retries=0)
def download(model: str):
    import os
    from huggingface_hub import snapshot_download
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    os.environ.pop("HF_HUB_OFFLINE", None)
    tok = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    p = snapshot_download(model, cache_dir="/vol/hf", token=tok, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken"])
    vol.commit()
    return p


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=8, memory=65536, retries=0, max_containers=1)
def dryrun(git_hash: str, model: str):
    out = {}
    tag = model.split("/")[-1]  # per-model dirs: collect_activations.py resumes from an existing out dir
    for b in (2, 1):
        rc = _run(["--out", f"/vol/out/dryrun_{tag}_b{b}", "--gate-file", "/app/qwen_gate.json", "--teacher-jsonl", "/app/qwen_pilot.jsonl", "--tf-batch", "2", "--device", "cpu", "--dtype", "float16", "--batch", str(b), "--max-new", "12", "--gate-samples", "2",
                   "--gate-max-new", "12", "--limit-rows", "2", "--ignore-gate-fail"], git_hash, model)
        out[b] = rc
        if rc:
            return out
    r = {b: [json.loads(l) for l in open(f"/vol/out/dryrun_{tag}_b{b}/pilot.jsonl")] for b in (2, 1)}
    out["same_response_b2_vs_b1"] = [x["response"] == y["response"] for x, y in zip(sorted(r[2], key=lambda z: z["row_id"]), sorted(r[1], key=lambda z: z["row_id"]))]
    out["rows_b2"] = [{k: v[k] for k in ("row_id", "response", "n_gen", "done_reason", "nonfinite_pooled", "gen_tokens_pooled")} for v in r[2]]
    import glob
    import numpy as np
    tf = {}
    for f in sorted(glob.glob(f"/vol/out/dryrun_{tag}_b2/tf_acts/*.npz")) + sorted(glob.glob(f"/vol/out/dryrun_{tag}_b1/tf_acts/*.npz")):
        z = np.load(f)
        tf[f.split("/out/")[1]] = dict(shape=list(z["gen_mean"].shape), finite=bool(z["finite"]), n_resp=int(z["n_resp_tokens"]), n_prompt=int(z["n_prompt_tokens"]))
    out["tf"] = tf
    if len(tf) == 4:  # same row, batch 2 vs batch 1: padding must not change pooled states
        a_, b_ = (np.load(f"/vol/out/dryrun_{tag}_b{k}/tf_acts/{r}.npz")["gen_mean"].astype("float32") for k, r in ((2, "en-no_task-c_08daf70eed01"), (1, "en-no_task-c_08daf70eed01")))
        out["tf_pad_maxabs_diff_b2_vs_b1"] = float(abs(a_ - b_).max())
    out["run_json"] = json.load(open(f"/vol/out/dryrun_{tag}_b2/run.json"))
    return out


@app.function(image=image, volumes={"/vol": vol}, gpu="L40S", timeout=45 * 60, retries=0, max_containers=1)
def run_pilot(git_hash: str, run_id: str, model: str, cross: bool):
    extra = ["--gate-file", "/app/qwen_gate.json", "--teacher-jsonl", "/app/qwen_pilot.jsonl"] if cross else []
    return _run(["--out", f"/vol/out/{run_id}", "--device", "cuda", "--dtype", "float16", "--batch", "16", *extra], git_hash, model)


@app.local_entrypoint()
def main(stage: str = "dryrun", run_id: str = "pilot1", model: str = MODEL):
    gh = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if stage == "dryrun":
        print(json.dumps(dryrun.remote(gh, model), indent=1, ensure_ascii=False))
    elif stage == "pilot":
        fc = run_pilot.spawn(gh, run_id, model, model != MODEL)
        print("FUNCTION_CALL_ID", fc.object_id, flush=True)
    elif stage == "download":
        print(download.remote(model))
    else:
        raise SystemExit("stage must be download, dryrun or pilot")
