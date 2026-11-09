import math

from multilingual_overrefusal import hand_tagging as ht


def test_statistics_known_values():
    h = ["refuse"] * 25 + ["comply"] * 25
    j = ["refuse"] * 20 + ["comply"] * 5 + ["refuse"] * 10 + ["comply"] * 15
    assert abs(ht.cohens_kappa(h, j) - 0.4) < 1e-9
    assert abs(ht.scotts_pi(h, j) - (0.7 - 0.505) / 0.495) < 1e-9
    lo, hi = ht.wilson_interval(9, 10)
    assert abs(lo - 0.5958) < 1e-3 and abs(hi - 0.9821) < 1e-3
    s = ht.agreement_stats(h, j)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (20, 10, 5, 15)
    assert abs(s["refuse_precision"] - 20 / 30) < 1e-9 and abs(s["refuse_recall"] - 0.8) < 1e-9
    assert abs(s["p_pos"] - 40 / 55) < 1e-9 and abs(s["p_neg"] - 30 / 45) < 1e-9
    assert ht.cohens_kappa(["a", "b"], ["a", "b"]) == 1.0
    lo, hi = ht.bootstrap_interval(h, j, ht.cohens_kappa)
    assert lo < 0.4 < hi
    assert math.isnan(ht.cohens_kappa([], []))


def _rows():
    return {f"{m}|c{i}|0": {"prompt": f"p{i}", "response": f"r{i}", "model_id": m}
            for m in ("A", "B") for i in range(40)}


def test_blind_sheet_shape_and_repeats():
    rows = _rows()
    sheet, key = ht.draw_blind_sheet(rows, lambda rid, r: (r["model_id"], "refuse" if int(rid.split("|")[1][1:]) % 2 else "comply"),
                                     per_stratum=5, repeats=3, seed=1)
    assert len(sheet) == 4 * 5 + 3 and [s["n"] for s in sheet] == list(range(1, 24))
    assert set(sheet[0]) == {"n", "item", "prompt", "response"}          # blind: nothing else leaks
    reps = {i: m["repeat_of"] for i, m in key["items"].items() if m["repeat_of"]}
    assert len(reps) == 3
    pos = {s["item"]: s["n"] for s in sheet}
    assert all(pos[r] - pos[o] >= 1 for r, o in reps.items())
    assert (sheet, key["items"]) == (ht.draw_blind_sheet(rows, lambda rid, r: (r["model_id"], "refuse" if int(rid.split("|")[1][1:]) % 2 else "comply"),
                                                         per_stratum=5, repeats=3, seed=1)[0], key["items"])


def test_latest_label_wins(tmp_path):
    from multilingual_overrefusal.io import append_jsonl
    p = tmp_path / "labels.jsonl"
    append_jsonl(p, [{"item": "h001", "label": "comply"}, {"item": "h001", "label": "refuse"}])
    assert ht.latest_labels(p)["h001"]["label"] == "refuse"


# ---------------------------------------------------------------- browser UI, server, report, steps

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer


def _sheet(n=4):
    return [{"n": i, "item": f"h{i:03d}", "prompt": f"p{i} <b>", "response": f"r{i}", "extra": "LEAK"}
            for i in range(1, n + 1)]


def test_ui_html_rubric_variants_and_no_leak():
    v1 = ht.build_ui_html(_sheet(), [], "handtag-v1-2026-09-30")
    v11 = ht.build_ui_html(_sheet(), [], "handtag-v1.1-2026-10-22")
    assert "Cut off at 128 tokens" in v1 and "premise" not in v1.lower().split("<ol>")[1].split("</ol>")[0]
    assert "Cut off at 512 tokens" in v11 and "Rejects the premise" in v11
    assert "LEAK" not in v1
    assert "<b>" not in v1.split('id="handtag-data"')[1].split("</script>")[0]      # "<" escaped in payload
    for bad in ("__CUTOFF_TOKENS__", "PREMISE_START", "__HANDTAG_DATA__"):
        assert bad not in v1 and bad not in v11
    try:
        ht.build_ui_html(_sheet(), [], "nope")
    except ValueError:
        pass
    else:
        raise AssertionError


def _start_server(tmp_path, rubric="handtag-v1.1-2026-10-22"):
    sheet = _sheet()
    (tmp_path / "sheet.jsonl").write_text("\n".join(json.dumps(s) for s in sheet) + "\n")
    (tmp_path / "ui.html").write_text(ht.build_ui_html(sheet, [], rubric))
    store = ht.LabelStore(tmp_path, rubric)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ht.make_handler(store))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, store, f"http://127.0.0.1:{srv.server_address[1]}"


def _post(url, body, headers=None):
    req = urllib.request.Request(url + "/save", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_server_saves_validates_and_dedups(tmp_path):
    srv, store, url = _start_server(tmp_path)
    try:
        ev = {"item": "h001", "label": "refuse", "unsure": True, "note": "x",
              "rubric": "handtag-v1.1-2026-10-22", "ts": "2026-11-09T10:00:00+00:00"}
        assert _post(url, {"events": [ev]})[1]["appended"] == 1
        assert _post(url, {"events": [ev]})[1]["duplicates"] == 1                   # same item+ts is not written twice
        assert _post(url, {"events": [dict(ev, item="nope")]})[0] == 400
        assert _post(url, {"events": [dict(ev, label="maybe")]})[0] == 400
        assert _post(url, {"events": [dict(ev, rubric="handtag-v1-2026-09-30")]})[0] == 400
        assert _post(url, {"events": [ev]}, {"Origin": "http://evil.example"})[0] == 403
        assert ht.latest_labels(tmp_path / "labels.jsonl")["h001"]["unsure"] is True
        with urllib.request.urlopen(url + "/") as r:
            assert b"handtag-data" in r.read()
        with urllib.request.urlopen(url + "/api/labels") as r:
            assert len(json.load(r)["events"]) == 1
    finally:
        srv.shutdown()


def _cfg(tmp_path, langs=("ur", "lo"), models=("mA", "mB"), per=30):
    rows, qg = [], []
    for lang in langs:
        for m in models:
            for i in range(per):
                rid = f"{lang}-{i}"
                rows.append({"row_id": rid, "model": m, "lang": lang, "task": "t", "prompt": f"p {rid}",
                             "response": f"r {rid} {m}"})
                qg.append({"row_id": rid, "model": m, "verdict": "refuse" if i % 3 == 0 else "comply"})
    (tmp_path / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "judge_qwen3guard-4b.jsonl").write_text("\n".join(json.dumps(r) for r in qg) + "\n")
    return {"name": "t", "seed": 0, "output_dir": str(tmp_path), "steps": ["hand_tag", "report"],
            "judge": {"judges": ["qwen3guard-4b"]}, "hand_tag": {"rows": 24, "repeats": 3, "ui": "terminal"}}


def test_hand_tag_and_report_steps(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    ht._draw_sheet(cfg, tmp_path / "hand_tag")
    sheet = ht.read_jsonl(tmp_path / "hand_tag" / "sheet.jsonl")
    key = json.loads((tmp_path / "hand_tag" / "key.json").read_text())
    assert len(sheet) == 4 * 2 * 3 + 3 and key["strata_judge"] == "qwen3guard-4b"
    # hand label = the judge's verdict, except every 4th item flipped: report must show the disagreement
    verdict = ht.judge_verdicts(tmp_path / "judge_qwen3guard-4b.jsonl")
    events = []
    for k, s in enumerate(sheet):
        j = verdict[key["items"][s["item"]]["row_id"]]
        lab = j if k % 4 else ("comply" if j == "refuse" else "refuse")
        events.append({"item": s["item"], "label": lab, "unsure": False, "note": "", "rubric": ht.DEFAULT_RUBRIC,
                       "ts": f"2026-11-09T10:{k:02d}:00+00:00"})
    (tmp_path / "hand_tag" / "labels.jsonl").write_text("\n".join(json.dumps(e) for e in events[:-1]) + "\n")
    try:
        ht.report_step(cfg)
    except SystemExit as e:
        assert "untagged" in str(e)
    else:
        raise AssertionError("report must refuse while items are untagged")
    ht.append_jsonl(tmp_path / "hand_tag" / "labels.jsonl", events[-1:])
    ht.report_step(cfg)
    rep = (tmp_path / "hand_tag" / "report.md").read_text()
    res = json.loads((tmp_path / "hand_tag" / "report.json").read_text())
    s = res["judges"]["qwen3guard-4b"]["comply/refuse only (primary)"]
    assert s["n"] == 24 and 0.6 < s["acc"] < 0.9 and "Per lang" in rep and "Per model" in rep
    assert "Intra-rater" in rep and res["intra_rater"]["n"] == 3
    # the step is not done until every item is labelled
    (tmp_path / "hand_tag" / "labels.jsonl").write_text("")
    monkeypatch.setattr(ht, "label_in_terminal", lambda *a, **k: None)
    try:
        ht.hand_tag_step(cfg)
    except SystemExit as e:
        assert "untagged" in str(e)
    else:
        raise AssertionError
