#!/bin/bash
# Summarize current clone-matrix training state. Writes to drl_status_latest.txt
# for the at-a-glance view; prints same content to stdout.
LOG_DIR="$HOME/drl_logs"
OUT="$HOME/drl_status_latest.txt"
{
  echo "Clone matrix status @ $(date)"
  echo "Logs: $LOG_DIR"
  echo
  printf "%-22s %-12s %-10s %s\n" "RUN" "LAST_UPD" "STATUS" "LAST_LINE"
  printf "%-22s %-12s %-10s %s\n" "---" "--------" "------" "---------"
  for f in "$LOG_DIR"/clone_*.log; do
    [ -f "$f" ] || continue
    RUN=$(basename "$f" .log)
    LAST=$(tail -n 50 "$f" | grep -E "^upd [0-9]+/[0-9]+" | tail -n 1)
    LASTANY=$(tail -n 1 "$f")
    if echo "$LASTANY" | grep -q "FINISHED"; then
      STATUS="DONE"
      UPD="--"
    elif echo "$LASTANY" | grep -qiE "Traceback|Error|Killed"; then
      STATUS="ERROR"
      UPD="?"
    elif [ -n "$LAST" ]; then
      STATUS="training"
      UPD=$(echo "$LAST" | grep -oE "upd [0-9]+/[0-9]+" | sed 's/upd //')
    else
      STATUS="starting"
      UPD="--"
    fi
    LINE_SNIP=$(echo "${LAST:-$LASTANY}" | cut -c1-120)
    printf "%-22s %-12s %-10s %s\n" "$RUN" "$UPD" "$STATUS" "$LINE_SNIP"
  done
} | tee "$OUT"
