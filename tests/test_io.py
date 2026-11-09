import json

from multilingual_overrefusal.io import already_done_ids, append_jsonl, read_jsonl, write_run_json


def test_resume(tmp_path):
    p = tmp_path / "out.jsonl"
    assert read_jsonl(p) == [] and already_done_ids(p) == set()
    append_jsonl(p, [{"row_id": "a"}, {"row_id": "b"}])
    append_jsonl(p, [{"row_id": "c"}])
    assert already_done_ids(p) == {"a", "b", "c"}


def test_run_json(tmp_path):
    p = write_run_json(tmp_path, {"name": "t"}, seed=7)
    info = json.loads(p.read_text())
    assert info["seed"] == 7 and info["config"] == {"name": "t"} and "git_hash" in info
