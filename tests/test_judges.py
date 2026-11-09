import json

from multilingual_overrefusal import judges
from multilingual_overrefusal.io import read_jsonl


def test_parse_guard_output():
    assert judges.parse_guard_output("Safety: Unsafe\nCategories: Violent\nRefusal: No") == ("comply", "Unsafe")
    assert judges.parse_guard_output("Safety: Safe\nCategories: None\nRefusal: Yes") == ("refuse", "Safe")
    assert judges.parse_guard_output("") == (None, None)
    assert judges.parse_guard_output(None) == (None, None)


def test_parse_policy_label_takes_last_label():
    assert judges.parse_policy_label("label: comply") == "comply"
    assert judges.parse_policy_label("I might refuse.\nlabel: comply") == "comply"
    assert judges.parse_policy_label("The label is refuse.\ncomply") is None
    assert judges.parse_policy_label("<|channel|>final<|message|>label: refuse") == "refuse"
    assert judges.parse_policy_label("So label: comply.\n<|message|>label: refuse") == "refuse"


def test_policy_message_truncates_and_reads_task():
    row = {"prompt": "p", "response": "x" * 2000, "task": "summarize"}
    msg = judges.build_policy_user_message(row)
    assert "intended_task: summarize" in msg and "[response truncated]" in msg
    assert "intended_task: no_task" in judges.build_policy_user_message({"prompt": "p", "response": "ok"})


def test_garble_rules():
    sig = judges.garble_signals("", "en")
    assert judges.garble_reasons(sig, "rephrase") == ["empty"]
    loop = judges.garble_signals("lock " * 60, "en")
    assert judges.garble_reasons(loop, "rephrase") == ["repetition"]
    latin_for_lao = judges.garble_signals("This is plain English text, not Lao at all.", "lo")
    assert judges.garble_reasons(latin_for_lao, "rephrase") == ["wrong_script"]
    assert judges.garble_reasons(latin_for_lao, "translate") == []      # translate output is in another language
    assert judges.garble_reasons(judges.garble_signals("A normal answer in English.", "yo"), "rephrase") == []


def test_run_judge_resumes(tmp_path, monkeypatch):
    seen = []

    def fake(rows, name, backend, n_threads):
        for r in rows:
            seen.append(r["row_id"])
            yield r, "Safety: Safe\nRefusal: Yes"

    monkeypatch.setattr(judges, "judge_with_guard", fake)
    rows = [{"row_id": f"r{i}", "model": "m", "prompt": "p", "response": "r"} for i in range(3)]
    out = tmp_path / "judge_qwen3guard-4b.jsonl"
    judges.run_judge("qwen3guard-4b", rows[:2], out)
    judges.run_judge("qwen3guard-4b", rows, out)
    assert seen == ["r0", "r1", "r2"]
    assert [r["verdict"] for r in read_jsonl(out)] == ["refuse"] * 3


def test_judge_step_writes_garble_flags(tmp_path):
    rows = [{"row_id": "a", "model": "m", "lang": "en", "task": "rephrase", "prompt": "p", "response": ""}]
    (tmp_path / "responses.jsonl").write_text(json.dumps(rows[0]) + "\n")
    judges.judge_step({"output_dir": str(tmp_path), "judge": {"judges": [], "garble_filter": True}})
    flags = read_jsonl(tmp_path / "garble_flags.jsonl")
    assert flags[0]["reasons"] == ["empty"]
