#!/usr/bin/env bash
# Run on the Mac. minty holds the canonical copy; the Mac is a working copy for MLX runs.
#   scripts/mac_sync.sh pull   # minty -> Mac, before a run
#   scripts/mac_sync.sh push   # Mac -> minty, after a run
# Never deletes on either side. --update skips files that are newer on the receiver.
set -euo pipefail

REMOTE="${REMOTE:-kaustubhg73@minty}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/kaustubhg73/Desktop/Speshal/projects/multilingual-overrefusal}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

EXCLUDES=(--exclude='.venv/' --exclude='.venv.bak-macos/' --exclude='__pycache__/' --exclude='.DS_Store')

case "${1:-}" in
  pull)
    rsync -az --update --itemize-changes "${EXCLUDES[@]}" \
      "$REMOTE:$REMOTE_ROOT/" "$LOCAL_ROOT/"
    ;;
  push)
    # Commits happen on minty; never push the Mac's .git over it.
    rsync -az --update --itemize-changes "${EXCLUDES[@]}" --exclude='.git/' \
      "$LOCAL_ROOT/" "$REMOTE:$REMOTE_ROOT/"
    ;;
  *)
    echo "usage: $0 pull|push" >&2
    exit 2
    ;;
esac
