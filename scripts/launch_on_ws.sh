#!/bin/bash
# Run on a target ws: kill leftovers, launch N emulators, run one training
# config, log to NFS-home so monitor.sh can see progress from any machine.
#
# Usage (locally on the ws, usually invoked via dispatch_all.sh):
#   ./scripts/launch_on_ws.sh <mode> <seed> [steps] [num_envs] [backend] [frame_skip] [resume]
#
# Example:
#   ./scripts/launch_on_ws.sh oca 0 50000 4
#
# Interrupt-safe: on SIGTERM/SIGINT we still tear down emulators and mirror
# the latest checkpoint to NFS, so kill-then-eval works without further
# manual steps. Use SIGTERM (not -9 SIGKILL) for clean interruption.
set -uo pipefail  # NOTE: dropped -e so a killed train.py doesn't skip the mirror step

MODE="${1:?usage: launch_on_ws.sh <mode> <seed> [steps] [num_envs] [backend] [frame_skip] [resume]}"
SEED="${2:?missing seed}"
STEPS="${3:-50000}"
N_ENVS="${4:-4}"
BACKEND="${5:-adb}"
FRAME_SKIP="${6:-0}"
RESUME="${7:-}"   # optional path to checkpoint to resume from

PROJ="/tmp2/$USER/DRL_final_project"
LOG_NFS="$HOME/drl_logs"
mkdir -p "$LOG_NFS"
HOST=$(hostname -s)
LOG="$LOG_NFS/${HOST}_${MODE}_s${SEED}.log"

# Clean up any leftover qemu/python from prior attempts (this run owns the box)
pkill -u $USER -9 -f qemu-system-x86 2>/dev/null || true
pkill -u $USER -9 -f train.py 2>/dev/null || true
pkill -u $USER -9 -f orchestrate.py 2>/dev/null || true
sleep 3

# Start fresh: this stomps the log
{
  echo "=== $(date) - $HOST - $MODE seed=$SEED steps=$STEPS num_envs=$N_ENVS ==="
} > "$LOG"

source "$PROJ/android_env.sh"
cd "$PROJ"

# Cleanup trap: ALWAYS runs (normal exit, SIGTERM/SIGINT, error). Tears down
# emulators and mirrors the latest checkpoint, so the script can be interrupted
# mid-training and still leave usable artifacts on NFS.
HOMECKPT="$HOME/drl_ckpts"
mkdir -p "$HOMECKPT"
cleanup() {
  echo "[launch_on_ws] cleanup trap firing at $(date)" >> "$LOG"
  .venv/bin/python orchestrate.py kill >> "$LOG" 2>&1 || true
  LATEST=$(ls -t "$PROJ"/checkpoints/bouncy_${MODE}_*_s${SEED}_*/step_*.pt 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    cp "$LATEST" "$HOMECKPT/${HOST}_${MODE}_s${SEED}_final.pt"
    echo "[launch_on_ws] mirrored checkpoint -> $HOMECKPT/${HOST}_${MODE}_s${SEED}_final.pt" >> "$LOG"
  fi
}
trap cleanup EXIT INT TERM

# Launch emulator farm
EMU_LAUNCH_STAGGER_S=8 .venv/bin/python orchestrate.py launch --n "$N_ENVS" \
  >> "$LOG" 2>&1
echo "[launch_on_ws] emulator farm up" >> "$LOG"
adb devices >> "$LOG" 2>&1

# v2 path: push minicap/minitouch binaries to every emulator before training.
# Both 'minicap' and 'adb-minitouch' need at least minitouch on-device.
if [ "$BACKEND" = "minicap" ] || [ "$BACKEND" = "adb-minitouch" ]; then
  ./scripts/push_minicap.sh >> "$LOG" 2>&1
fi

# Run training (blocking).
# `-u` = unbuffered stdout/stderr so PPO updates flush to the log on every print,
# not when a ~4KB buffer fills (which would delay the first visible update by
# ~15 minutes when redirected to a file).
RESUME_FLAG=""
if [ -n "$RESUME" ]; then
  RESUME_FLAG="--resume $RESUME"
  echo "[launch_on_ws] resuming from $RESUME" >> "$LOG"
fi
# Note: `|| true` so a SIGTERM that exits non-zero doesn't skip the trap-fired
# cleanup. With `set -e` dropped earlier in this script, the `|| true` is
# redundant but kept for safety/clarity.
.venv/bin/python -u train.py --env-id bouncy --aux-mode "$MODE" \
  --seed "$SEED" --total-timesteps "$STEPS" --num-envs "$N_ENVS" \
  --num-steps 128 --num-minibatches 4 --update-epochs 4 \
  --ckpt-every 25 \
  --backend "$BACKEND" --frame-skip "$FRAME_SKIP" \
  --use-rnd --use-charge-dim $RESUME_FLAG \
  >> "$LOG" 2>&1 || true

RC=$?
echo "=== $(date) - $HOST - $MODE seed=$SEED FINISHED (rc=$RC) ===" >> "$LOG"
# Trap-fired cleanup runs implicitly on EXIT after this point.
exit $RC
