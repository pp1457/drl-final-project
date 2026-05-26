#!/bin/bash
# Periodically rsync the latest training checkpoint from each of the 9
# v11 workers to a single safe directory on ws10 (/tmp2 is local but
# survives cluster-wide kills as long as ws10 itself stays up).
#
# Run from anywhere; uses SSH. Idempotent — overwrites with the latest
# checkpoint each time.
set -u
DST_DIR=/tmp2/$USER/v11_ckpts
mkdir -p "$DST_DIR"

declare -a JOBS=(
  "ws1 baseline 0"
  "ws2 baseline 1"
  "ws3 baseline 2"
  "ws4 oca 0"
  "ws5 oca 1"
  "ws6 oca 2"
  "ws7 dpr 0"
  "ws8 dpr 1"
  "ws10 dpr 2"
)

for job in "${JOBS[@]}"; do
  WS=$(echo "$job" | awk '{print $1}')
  MODE=$(echo "$job" | awk '{print $2}')
  SEED=$(echo "$job" | awk '{print $3}')
  REMOTE_CMD="bash -c 'ls -t /tmp2/$USER/DRL_final_project/checkpoints/bouncy_${MODE}_lam0.5_s${SEED}_*/step_*.pt 2>/dev/null | head -1'"
  LATEST=$(ssh -o ConnectTimeout=5 -o BatchMode=yes ${WS}.csie.ntu.edu.tw "$REMOTE_CMD" 2>/dev/null)
  if [ -n "$LATEST" ]; then
    BN=$(basename "$LATEST")
    DST="$DST_DIR/${WS}_${MODE}_s${SEED}_${BN}"
    if [ ! -f "$DST" ]; then
      scp -q -o BatchMode=yes ${WS}.csie.ntu.edu.tw:"$LATEST" "$DST" 2>&1
      echo "[preserve_v11] [$WS $MODE s$SEED] NEW $BN"
    fi
  fi
done

# Keep only the LATEST checkpoint per (ws, mode, seed); delete older ones.
for job in "${JOBS[@]}"; do
  WS=$(echo "$job" | awk '{print $1}')
  MODE=$(echo "$job" | awk '{print $2}')
  SEED=$(echo "$job" | awk '{print $3}')
  prefix="${WS}_${MODE}_s${SEED}_step_"
  # Sort by step number (numeric), keep last, delete rest.
  matches=$(ls -1 "$DST_DIR"/${prefix}*.pt 2>/dev/null | awk -F'step_' '{print $2" "$0}' | sort -n)
  count=$(echo "$matches" | grep -c .)
  if [ "$count" -gt 1 ]; then
    echo "$matches" | head -n -1 | awk '{print $2}' | xargs -r rm -v 2>&1
  fi
done

echo "[preserve_v11] done @ $(date +%H:%M)  | total files: $(ls "$DST_DIR" | wc -l)"
