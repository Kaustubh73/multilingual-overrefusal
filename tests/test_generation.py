import json

import pytest

from multilingual_overrefusal import generation as g


def test_resolve_tasks_all_is_nine_text_tasks():
    t = g.resolve_tasks("all")
    assert len(t) == 9 and "schema_constrained_output" not in t


def test_resolve_tasks_rejects_unknown():
    with pytest.raises(ValueError):
        g.resolve_tasks(["no_task", "dance"])


def test_resolve_model_registry_and_override():
    assert g.resolve_model("gemma-4-e4b", {})[0] == "ollama"
    assert g.resolve_model("mine", {"backend": "ollama", "model_ids": {"mine": "x:1"}}) == ("ollama", "x:1")
    with pytest.raises(ValueError):
        g.resolve_model("nope", {})


def test_write_responses_joins_sample_and_generations(tmp_path):
    sample = [{"row_id": "a", "prompt": "p", "lang": "ur"}]
    (tmp_path / "gen_m.jsonl").write_text(json.dumps({"row_id": "a", "model": "m", "response": "r",
                                                      "input_hash": "h1"}) + "\n")
    g._write_responses(tmp_path, sample, ["m"], {"m": {"a": "h1"}})
    row = json.loads((tmp_path / "responses.jsonl").read_text())
    assert row == {"row_id": "a", "prompt": "p", "lang": "ur", "model": "m", "response": "r", "input_hash": "h1"}


def test_write_responses_ignores_stale_and_removed_rows(tmp_path):
    """Rows no longer in the sample are dropped (no KeyError); the record matching the current inputs wins."""
    sample = [{"row_id": "a", "prompt": "p2"}]
    (tmp_path / "gen_m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"row_id": "old", "model": "m", "response": "gone", "input_hash": "x"},
        {"row_id": "a", "model": "m", "response": "stale", "input_hash": "h_old"},
        {"row_id": "a", "model": "m", "response": "fresh", "input_hash": "h_new"},
    ]))
    g._write_responses(tmp_path, sample, ["m"], {"m": {"a": "h_new"}})
    rows = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
    assert [r["response"] for r in rows] == ["fresh"]
    with pytest.raises(RuntimeError, match="no current generation"):
        g._write_responses(tmp_path, sample, ["m"], {"m": {"a": "h_other"}})


def test_generate_skips_finished_rows_and_rejects_nonzero_temperature(tmp_path):
    (tmp_path / "sample.jsonl").write_text(json.dumps({"row_id": "a", "prompt": "p"}) + "\n")
    cfg = {"output_dir": str(tmp_path), "generate": {"models": ["qwen3.5-9b"], "temperature": 0.0}}
    h = g._gen_hashes([{"row_id": "a", "prompt": "p"}], *g.resolve_model("qwen3.5-9b", cfg["generate"]),
                      cfg["generate"], 0)["a"]
    (tmp_path / "gen_qwen3.5-9b.jsonl").write_text(
        json.dumps({"row_id": "a", "model": "qwen3.5-9b", "response": "r", "input_hash": h}) + "\n")
    g.generate_step(cfg)                      # nothing to do: no model is loaded
    assert (tmp_path / "responses.jsonl").exists()
    cfg["generate"]["temperature"] = 0.7
    with pytest.raises(ValueError):
        g.generate_step(cfg)
