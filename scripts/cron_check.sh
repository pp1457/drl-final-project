#!/bin/bash
# Cron task: snapshot training status to NFS home every N minutes.
# Three artifacts in $HOME, all readable from any ws:
#   ~/drl_status_latest.txt   - most recent monitor snapshot
#   ~/drl_status_history.log  - per-run-state counts over time (one line/check)
#   ~/drl_ALERT.txt           - present iff any run is ERROR/STALLED (delete on recovery)
#
# Install with:  ./scripts/cron_register.sh
# Remove with:   ./scripts/cron_unregister.sh
#
# Designed to be safe under cron's quirks: no shell-init dependencies, absolute
# paths, swallows all non-fatal errors, no terminal escape codes.
set +e

PROJ="/tmp2/$USER/DRL_final_project"
LATEST="$HOME/drl_status_latest.txt"
HISTORY="$HOME/drl_status_history.log"
ALERT="$HOME/drl_ALERT.txt"

# monitor.sh injects ANSI clear-screen codes; strip them for file readability.
"$PROJ/scripts/monitor.sh" 2>&1 | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g; s/\x1b\[[0-9;]*[HJ]//g' > "$LATEST"

ts=$(date '+%Y-%m-%d %H:%M:%S')

# Tally how many runs are in each state. grep -c can exit 1 (no match)
# under set -e but we have set +e at the top; still, trim the result with
# tr -d so any stray newlines from || fallback don't break arithmetic.
count() {
  # only count data rows (start with 'ws<N>_'); avoids matching the header
  # 'DRL training status @...'
  local n
  n=$(grep -cE "^ws[0-9]+_.*$1" "$LATEST" 2>/dev/null)
  echo "${n:-0}" | tr -d '\n'
}
N_TRAIN=$(count 'training ')
N_SETUP=$(count 'setup\.\.\. ')
N_ERR=$(count '(ERROR|STALLED) ')
N_DONE=$(count 'finished ')
N_CKPT=$(ls -1 "$HOME/drl_ckpts/" 2>/dev/null | wc -l | tr -d ' \n')

printf '%s  setup=%s  training=%s  finished=%s  err=%s  ckpts=%s\n' \
  "$ts" "$N_SETUP" "$N_TRAIN" "$N_DONE" "$N_ERR" "$N_CKPT" >> "$HISTORY"

# Alert handling
if [ "$N_ERR" -gt 0 ]; then
  {
    echo "ALERT @ $ts  ($N_ERR run(s) in trouble):"
    grep -E ' (ERROR|STALLED) ' "$LATEST"
    echo
    echo "Full status:"
    cat "$LATEST"
  } > "$ALERT"
elif [ -f "$ALERT" ]; then
  rm -f "$ALERT"
fi
