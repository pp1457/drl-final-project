#!/bin/bash
# Launch a 15-run clone-env PPO matrix on ws10 (5 seeds × 3 configs).
# All runs parallel, single-machine (clone_env is CPU-only).
#
# Logs: ~/drl_logs/clone_<config>_s<seed>.log
# Status check: tail -n5 ~/drl_logs/clone_*.log
set -u

PROJ="/tmp2/$USER/DRL_final_project"
LOG_DIR="$HOME/drl_logs"
mkdir -p "$LOG_DIR"

# Archive any prior v5/v6 Android logs so the status report only reflects clone
mkdir -p "$LOG_DIR/_archive_$(date +%Y%m%d_%H%M)"
mv "$LOG_DIR"/ws*_*.log "$LOG_DIR/_archive_$(date +%Y%m%d_%H%M)/" 2>/dev/null || true

source "$PROJ/android_env.sh"
cd "$PROJ"

STEPS=500000
N_ENVS=4
NUM_STEPS=128

PIDS=()
for MODE in baseline oca dpr; do
  for SEED in 0 1 2 3 4; do
    TAG="clone_${MODE}_s${SEED}"
    LOG="$LOG_DIR/${TAG}.log"
    {
      echo "=== $(date) - $(hostname -s) - clone $MODE seed=$SEED steps=$STEPS num_envs=$N_ENVS ==="
      .venv/bin/python -u train.py --env-id clone --aux-mode "$MODE" \
        --seed "$SEED" --total-timesteps "$STEPS" --num-envs "$N_ENVS" \
        --num-steps "$NUM_STEPS" --num-minibatches 4 --update-epochs 4 \
        --ckpt-every 50
      echo "=== $(date) - $TAG FINISHED rc=$? ==="
    } > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "[launch] $TAG → pid=$! log=$LOG"
    sleep 1
  done
done

echo "[launch] all 15 clone runs spawned: ${PIDS[*]}"
echo "${PIDS[*]}" > "$LOG_DIR/.clone_matrix_pids"
