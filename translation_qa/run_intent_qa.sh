#!/bin/bash
# E4 intent stage + report (resumable). Run under caffeinate:  nohup caffeinate -dimsu translation_qa/run_intent_qa.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=translation_qa/intent_qa.py
echo "=== e4 intent (re)start $(date -u +%FT%TZ) ===" | tee -a results/e4/run.log
$PY -u $S intent 2>&1 | tee -a results/e4/run.log
$PY -u $S report 2>&1 | tee -a results/e4/run.log
echo "=== e4 done $(date -u +%FT%TZ) ===" | tee -a results/e4/run.log
