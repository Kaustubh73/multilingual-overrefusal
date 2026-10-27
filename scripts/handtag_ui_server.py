#!/usr/bin/env python3
"""Tiny local server for the hand-tagging UI (stdlib only, localhost only).

  python scripts/build_handtag_ui.py        # once / whenever the sheet changes
  python scripts/handtag_ui_server.py       # http://localhost:8770

Serves handtag_ui.html and appends labels to runs/.../handtag_labels.jsonl in
exactly the format `handtag.py label` writes, so `handtag.py report` reads it.

  GET  /              the page
  GET  /api/labels    {"events": [...]} current contents of handtag_labels.jsonl
  POST /save          {"events": [...]} validated, de-duplicated, appended
  GET  /api/health

Before the first write of each server run, handtag_labels.jsonl is copied to
handtag_labels.jsonl.bak. It only ever reads the blind sheet (for item ids);
it never opens the key or any judge file. If the server is not running the
page keeps working from localStorage and offers a download instead.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handtag  # noqa: E402

LOCK = threading.Lock()
MAX_BODY = 1_000_000


class Store:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.labels = run_dir / "handtag_labels.jsonl"
        self.bak = run_dir / "handtag_labels.jsonl.bak"
        self.html = run_dir / "handtag_ui.html"
        self.items = {r["item"] for r in handtag.read_jsonl(run_dir / "handtag_sheet.jsonl")}
        self.backed_up = False

    def events(self) -> list[dict]:
        return handtag.read_jsonl(self.labels)

    def validate(self, e: object) -> dict:
        if not isinstance(e, dict):
            raise ValueError("event must be an object")
        if e.get("item") not in self.items:
            raise ValueError(f"unknown item {e.get('item')!r}")
        if e.get("label") not in handtag.LABELS:
            raise ValueError(f"label must be one of {handtag.LABELS}")
        if e.get("rubric") != handtag.RUBRIC_VERSION:
            raise ValueError(f"rubric must be {handtag.RUBRIC_VERSION}")
        note = e.get("note", "")
        if not isinstance(note, str) or len(note) > 5000:
            raise ValueError("bad note")
        ts = e.get("ts") or datetime.now(timezone.utc).isoformat()
        datetime.fromisoformat(str(ts))
        # rebuild with the CLI's exact key order
        return {"item": e["item"], "label": e["label"], "unsure": bool(e.get("unsure", False)),
                "note": note, "rubric": handtag.RUBRIC_VERSION, "ts": str(ts)}

    def append(self, raw_events: list) -> dict:
        evs = [self.validate(e) for e in raw_events]
        with LOCK:
            existing = {(x["item"], x["ts"]) for x in self.events()}
            fresh, seen = [], set()
            for e in evs:
                k = (e["item"], e["ts"])
                if k not in existing and k not in seen:
                    fresh.append(e)
                    seen.add(k)
            if fresh:
                if not self.backed_up:
                    if self.labels.exists():
                        shutil.copy2(self.labels, self.bak)
                    self.backed_up = True
                prefix = ""
                if self.labels.exists() and self.labels.stat().st_size:
                    with self.labels.open("rb") as f:
                        f.seek(-1, os.SEEK_END)
                        if f.read(1) != b"\n":
                            prefix = "\n"
                blob = prefix + "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in fresh)
                with self.labels.open("a", encoding="utf-8") as f:
                    f.write(blob)
                    f.flush()
                    os.fsync(f.fileno())
            total = len(self.events())
        return {"ok": True, "appended": len(fresh), "duplicates": len(evs) - len(fresh), "total_events": total}


def make_handler(store: Store, port: int):
    allowed_hosts = {f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"}

    class H(BaseHTTPRequestHandler):
        server_version = "handtag-ui"

        def log_message(self, fmt, *a):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % a))

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:  # blocks DNS-rebinding and cross-site writes
            if self.headers.get("Host", "") not in allowed_hosts:
                self._json(403, {"ok": False, "error": "bad Host header"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] not in allowed_hosts:
                self._json(403, {"ok": False, "error": "bad Origin"})
                return False
            return True

        def do_GET(self):
            if not self._host_ok():
                return
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html", "/handtag_ui.html"):
                if not store.html.exists():
                    return self._send(404, b"handtag_ui.html missing: run scripts/build_handtag_ui.py", "text/plain")
                return self._send(200, store.html.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/labels":
                return self._json(200, {"events": store.events(), "path": str(store.labels)})
            if path == "/api/health":
                return self._json(200, {"ok": True, "rubric": handtag.RUBRIC_VERSION, "labels": str(store.labels)})
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._host_ok():
                return
            if self.path.split("?", 1)[0] != "/save":
                return self._send(404, b"not found", "text/plain")
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(415, {"ok": False, "error": "Content-Type must be application/json"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= MAX_BODY:
                    raise ValueError("bad Content-Length")
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                events = body["events"]
                if not isinstance(events, list):
                    raise ValueError("events must be a list")
                res = store.append(events)
            except (ValueError, KeyError, TypeError) as exc:
                return self._json(400, {"ok": False, "error": str(exc)})
            except OSError as exc:
                return self._json(500, {"ok": False, "error": f"write failed: {exc}"})
            self._json(200, res)

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=Path(os.environ.get("HANDTAG_RUN", handtag.RUN)))
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1", help="keep this on localhost; there is no auth")
    args = ap.parse_args()
    store = Store(args.run_dir)
    if not store.html.exists():
        raise SystemExit(f"{store.html} missing; run `python scripts/build_handtag_ui.py` first")
    try:
        srv = ThreadingHTTPServer((args.host, args.port), make_handler(store, args.port))
    except OSError as exc:
        raise SystemExit(f"cannot bind {args.host}:{args.port}: {exc}")
    print(f"hand-tag UI on http://localhost:{args.port}  (labels -> {store.labels}, backup -> {store.bak.name})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
