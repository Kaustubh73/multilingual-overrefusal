#!/usr/bin/env bash
# From minty: sync pilot code+prompts to Mac and start Metal run.
# Requires Mac: System Settings → General → Sharing → Remote Login ON
set -euo pipefail
MAC_HOST="${MAC_HOST:-100.120.28.9}"
MAC_USER="${MAC_USER:-kaustubhg73}"
REMOTE_ROOT="${REMOTE_ROOT:-/Users/${MAC_USER}/Desktop/Speshal/projects/multilingual-overrefusal}"
LOCAL_ROOT="/home/kaustubhg73/Desktop/Speshal/projects/multilingual-overrefusal"

echo "Checking SSH $MAC_USER@$MAC_HOST ..."
ssh -o BatchMode=yes -o ConnectTimeout=8 "$MAC_USER@$MAC_HOST" 'hostname; uname -m; sysctl -n hw.memsize'

echo "Syncing scripts + prompts ..."
ssh "$MAC_USER@$MAC_HOST" "mkdir -p '$REMOTE_ROOT/scripts' '$REMOTE_ROOT/results/qualitative_pilot' '$REMOTE_ROOT/results/qualitative_pilot_mac'"
rsync -avz \
  "$LOCAL_ROOT/scripts/run_qualitative_pilot.py" \
  "$LOCAL_ROOT/scripts/run_qualitative_mac.sh" \
  "$MAC_USER@$MAC_HOST:$REMOTE_ROOT/scripts/"
rsync -avz \
  "$LOCAL_ROOT/results/qualitative_pilot/prompts.csv" \
  "$MAC_USER@$MAC_HOST:$REMOTE_ROOT/results/qualitative_pilot/prompts.csv"

# Copy HF token if Mac missing one
if [[ -f "$HOME/.cache/huggingface/token" ]]; then
  ssh "$MAC_USER@$MAC_HOST" 'mkdir -p ~/.cache/huggingface'
  rsync -avz "$HOME/.cache/huggingface/token" "$MAC_USER@$MAC_HOST:~/.cache/huggingface/token"
fi

echo "Launching on Mac ..."
ssh "$MAC_USER@$MAC_HOST" "chmod +x '$REMOTE_ROOT/scripts/run_qualitative_mac.sh' && bash '$REMOTE_ROOT/scripts/run_qualitative_mac.sh'"
echo "Started. Tail log:"
echo "  ssh $MAC_USER@$MAC_HOST 'tail -f $REMOTE_ROOT/results/qualitative_pilot_mac/mac_run.log'"
