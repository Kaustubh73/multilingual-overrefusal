#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
for i in 1 2 3; do python3 decoding_truncation.py gen && break; done
../.venv/bin/python decoding_truncation.py report
