#!/bin/bash
# Resume v9 finals for another 20k steps each. Total per seed: 40k.
# Pulls each ws's final .pt from NFS (~/drl_ckpts/) and passes as --resume.
set -euo pipefail

STEPS="${1:-40000}"  # total timesteps (cumulative); 40000 = 20k more from 20k checkpoint
N_ENVS="${2:-2}"
BACKEND="${3:-adb-motionevent}"
FRAME_SKIP="${4:-1}"
PROJ="/tmp2/$USER/DRL_final_project"
CKPTS="$HOME/drl_ckpts"

ASSIGNMENTS=(
  "baseline 0 ws1"
  "baseline 1 ws2"
  "baseline 2 ws3"
  "oca 0 ws4"
  "oca 1 ws5"
  "oca 2 ws6"
  "dpr 0 ws7"
  "dpr 1 ws8"
  "dpr 2 ws10"
)

for assignment in "${ASSIGNMENTS[@]}"; do
  read -r MODE SEED WS <<< "$assignment"
  HOST="${WS}.csie.ntu.edu.tw"
  CKPT="$CKPTS/${WS}_${MODE}_s${SEED}_final.pt"
  if [ ! -f "$CKPT" ]; then
    echo "[dispatch_resume] SKIP $WS: no checkpoint at $CKPT"
    continue
  fi
  echo "[dispatch_resume] $WS $MODE seed=$SEED resume=$CKPT steps=$STEPS"
  ssh -n "$HOST" "cd $PROJ && nohup ./scripts/launch_on_ws.sh $MODE $SEED $STEPS $N_ENVS $BACKEND $FRAME_SKIP $CKPT \
      > /tmp/launch_${MODE}_${SEED}.log 2>&1 < /dev/null & disown" &
done
wait
echo
echo "[dispatch_resume] all resume launches sent"
