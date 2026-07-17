"""Enable ``python -m src.track_b_analysis``."""

from __future__ import annotations

import sys

from .pipeline import main

if __name__ == "__main__":
    sys.exit(main())
