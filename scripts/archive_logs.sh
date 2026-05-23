#!/bin/bash
# Archive the current state of ~/drl_logs into a timestamped tarball, then
# re-extract metrics.csv. Run before a launch (or periodically during a run)
# so we don't lose data when launch_on_ws.sh truncates the log files.
#
# Usage: ./scripts/archive_logs.sh [tag]
#   tag (optional): suffix for the archive name. Default: timestamp only.

set -euo pipefail

LOGS_DIR="$HOME/drl_logs"
ARCHIVE_DIR="$HOME/drl_logs_archive"
mkdir -p "$ARCHIVE_DIR"

TAG="${1:-}"
TS=$(date +%Y%m%d_%H%M%S)
NAME="${TS}${TAG:+_$TAG}"
ARCHIVE="$ARCHIVE_DIR/$NAME.tar.gz"

cd "$LOGS_DIR"
tar -czf "$ARCHIVE" *.log 2>/dev/null || true

# Also extract metrics.csv at this snapshot
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
"$PROJ_DIR/scripts/extract_metrics.py" --out "$ARCHIVE_DIR/${NAME}_metrics.csv" >/dev/null

sz=$(du -h "$ARCHIVE" | cut -f1)
nrow=$(wc -l < "$ARCHIVE_DIR/${NAME}_metrics.csv")
echo "[archive] $NAME: tarball=$sz, csv=$((nrow-1)) rows"
