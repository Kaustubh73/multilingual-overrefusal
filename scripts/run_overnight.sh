#!/bin/bash
# Overnight chain (run under caffeinate): E1b generate -> judge -> hand-tag sheet -> report, then E4 intent -> report.
# Sequential on purpose: one MLX model in memory at a time. Every stage resumes from its jsonl.
#   nohup caffeinate -dimsu scripts/run_overnight.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
E1=scripts/multilingual_judge.py; E4=translation_qa/intent_qa.py
mkdir -p results/e1b results/e4
echo "=== e1b start $(date -u +%FT%TZ) ===" | tee -a results/e1b/run.log
for step in generate judge handtag report; do $PY -u $E1 $step 2>&1 | tee -a results/e1b/run.log; done
echo "=== e1b done $(date -u +%FT%TZ) ===" | tee -a results/e1b/run.log
echo "=== e4 intent start $(date -u +%FT%TZ) ===" | tee -a results/e4/run.log
$PY -u $E4 intent 2>&1 | tee -a results/e4/run.log
$PY -u $E4 report 2>&1 | tee -a results/e4/run.log
echo "=== e4 done $(date -u +%FT%TZ) ===" | tee -a results/e4/run.log
