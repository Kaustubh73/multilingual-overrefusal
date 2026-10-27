#!/usr/bin/env python3
"""handtag_ui_server.py for the E1b sheet: same server, rubric string handtag-v1.1-2026-10-22.

  .venv/bin/python scripts/handtag_ui_server_multilingual.py   # http://localhost:8770, labels -> results/e1b/handtag/_shown/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handtag  # noqa: E402
import handtag_ui_server as S  # noqa: E402

handtag.RUBRIC_VERSION = "handtag-v1.1-2026-10-22"
if "--run-dir" not in sys.argv:
    sys.argv += ["--run-dir", str(Path(__file__).resolve().parents[1] / "results" / "e1b" / "handtag" / "_shown")]
S.main()
