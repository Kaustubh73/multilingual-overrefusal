#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
python3 task_pilot.py gen && ../.venv/bin/python task_pilot.py report
