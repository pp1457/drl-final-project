#!/bin/bash
# Dispatch the 9-run ablation matrix across 9 workstations.
# Run THIS from ws10 (or anywhere you can SSH from). Each (mode, seed) ->
# its own ws, started in the background. Returns immediately; check progress
# with ./scripts/monitor.sh.
#
# Prereq: ./scripts/deploy_one.sh wsN must have been run for each target ws.
#
# Usage:
#   ./scripts/dispatch_all.sh              # default 50000 steps
#   ./scripts/dispatch_all.sh 100000       # 100k steps per run
set -euo pipefail

STEPS="${1:-50000}"
PROJ="/tmp2/$USER/DRL_final_project"

# (mode, seed) -> ws assignment. We leave ws10 free as the controller; if you
# want to use it, swap something onto ws10. Adjust if any ws is down.
ASSIGNMENTS=(
  "baseline 0 ws1"
  "baseline 1 ws2"
  "baseline 2 ws3"
  "oca 0 ws4"
  "oca 1 ws5"
  "oca 2 ws6"
  "dpr 0 ws7"
  "dpr 1 ws8"
  "dpr 2 ws9"
)

for assignment in "${ASSIGNMENTS[@]}"; do
  read -r MODE SEED WS <<< "$assignment"
  HOST="${WS}.csie.ntu.edu.tw"
  echo "[dispatch] $WS  $MODE seed=$SEED  steps=$STEPS"
  # nohup + disown so the remote process survives our SSH closing.
  # Local /tmp log captures the SSH-side output (boot messages, etc).
  ssh "$HOST" "cd $PROJ && nohup ./scripts/launch_on_ws.sh $MODE $SEED $STEPS \
      > /tmp/launch_${MODE}_${SEED}.log 2>&1 & disown" &
done

wait
echo
echo "[dispatch] all 9 launches sent. They run in parallel."
echo "[dispatch] watch progress with:  ./scripts/monitor.sh"
echo "[dispatch] logs live at:         \$HOME/drl_logs/<host>_<mode>_s<seed>.log"
