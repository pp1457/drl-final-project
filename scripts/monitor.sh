#!/bin/bash
# Print a status table for all distributed runs. Reads NFS-home logs so it
# works from any ws.
#
# Modes:
#   ./scripts/monitor.sh             # one snapshot
#   ./scripts/monitor.sh -w          # auto-refresh every 30s (watch mode)
#   ./scripts/monitor.sh -t <file>   # tail a specific run's log
set -euo pipefail

LOG_NFS="$HOME/drl_logs"

if [ "${1:-}" = "-t" ]; then
  shift
  exec tail -f "$LOG_NFS/$1"
fi

WATCH=false
if [ "${1:-}" = "-w" ]; then
  WATCH=true
fi

show() {
  clear || true
  echo "DRL training status @ $(date)"
  echo "Logs: $LOG_NFS"
  echo
  printf "%-25s %-10s %-22s %s\n" "RUN" "LAST_UPD" "STATUS" "LAST_LINE"
  printf "%-25s %-10s %-22s %s\n" "---" "--------" "------" "---------"
  shopt -s nullglob
  for log in "$LOG_NFS"/*.log; do
    name=$(basename "$log" .log)
    # Extract latest "upd N/M" line (PPO trainer prints these)
    last_upd=$(grep -oE 'upd [0-9]+/[0-9]+' "$log" 2>/dev/null | tail -1 | sed 's/upd //')
    # Detect "FINISHED" sentinel
    if grep -q "FINISHED" "$log" 2>/dev/null; then
      status="finished"
    elif [ -z "$last_upd" ]; then
      # No PPO updates yet -- still booting / running into errors
      if grep -qiE "traceback|error|fail" "$log" 2>/dev/null; then
        status="ERROR"
      else
        status="setup..."
      fi
    else
      # Updates landing but not finished -- training in progress
      mtime=$(stat -c %Y "$log" 2>/dev/null || echo 0)
      now=$(date +%s)
      age=$((now - mtime))
      if [ $age -gt 300 ]; then
        status="STALLED (${age}s)"
      else
        status="training"
      fi
    fi
    last_line=$(tail -1 "$log" 2>/dev/null | cut -c1-80)
    printf "%-25s %-10s %-22s %s\n" "$name" "${last_upd:-?}" "$status" "$last_line"
  done
  echo
  echo "Final checkpoints in \$HOME/drl_ckpts/:"
  ls -1 "$HOME/drl_ckpts/" 2>/dev/null | head -10 || echo "  (none yet)"
}

if $WATCH; then
  while true; do
    show
    sleep 30
  done
else
  show
fi
