#!/bin/bash
# Summarize v6 Android matrix status from NFS-shared ~/drl_logs/ws*_*.log files.
LOG_DIR="$HOME/drl_logs"
OUT="$HOME/drl_status_latest.txt"
{
  echo "Android v6 matrix status @ $(date)"
  echo "Logs: $LOG_DIR"
  echo
  printf "%-22s %-12s %-10s %s\n" "RUN" "LAST_UPD" "STATUS" "LAST_LINE"
  printf "%-22s %-12s %-10s %s\n" "---" "--------" "------" "---------"
  for f in "$LOG_DIR"/ws*_*.log; do
    [ -f "$f" ] || continue
    RUN=$(basename "$f" .log)
    LAST=$(tail -n 100 "$f" | grep -E "^upd [0-9]+/[0-9]+" | tail -n 1)
    LASTANY=$(tail -n 1 "$f")
    if echo "$LASTANY" | grep -q "FINISHED"; then
      STATUS="DONE"
      UPD="--"
    elif tail -n 30 "$f" | grep -qE "Traceback|RuntimeError|ValueError|TypeError|Killed|^Error:"; then
      STATUS="ERROR"
      UPD="?"
    elif [ -n "$LAST" ]; then
      STATUS="training"
      UPD=$(echo "$LAST" | grep -oE "upd [0-9]+/[0-9]+" | sed 's/upd //')
    else
      STATUS="starting"
      UPD="--"
    fi
    LINE_SNIP=$(echo "${LAST:-$LASTANY}" | cut -c1-130)
    printf "%-22s %-12s %-10s %s\n" "$RUN" "$UPD" "$STATUS" "$LINE_SNIP"
  done
} | tee "$OUT"
