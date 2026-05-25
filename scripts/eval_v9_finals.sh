#!/bin/bash
# Eval all v9 final checkpoints against a side emulator.
# Run AFTER v9 has fully finished (all 9 _final.pt mirrored to ~/drl_ckpts/).
set -uo pipefail

PROJ=/tmp2/$USER/DRL_final_project
CKPT_DIR=$HOME/drl_ckpts
EPISODES="${1:-5}"  # default 5 episodes per checkpoint
MAX_STEPS="${2:-256}"  # cap each episode (steps); 256 ≈ ~2 min @ 500ms/step
OUT_DIR="$HOME/drl_eval_v9_$(date +%H%M)"
mkdir -p "$OUT_DIR"

cd "$PROJ"
source ./android_env.sh

SERIAL="emulator-6562"
if adb -s $SERIAL shell getprop sys.boot_completed 2>/dev/null | grep -q 1; then
  echo "[eval] $SERIAL already booted, reusing"
else
  echo "[eval] launching side emu at port 6562 with clean_boot_v8..."
  nohup .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from orchestrate import _launch_emulator
proc = _launch_emulator(rank=4, read_only=True, snapshot='clean_boot_v8')
print(f'pid={proc.pid}', flush=True); proc.wait()
" > /tmp/eval_emu.log 2>&1 &
  EMU_BG=$!
  echo "[eval] emu launcher bg pid=$EMU_BG"
  sleep 15
  until adb -s $SERIAL shell getprop sys.boot_completed 2>/dev/null | grep -q 1; do
    sleep 4
  done
  echo "[eval] $SERIAL booted"
fi
echo "[eval] starting eval: $EPISODES episodes × max $MAX_STEPS steps each"

# Eval each checkpoint
for CKPT in "$CKPT_DIR"/ws*_*_final.pt; do
  TAG=$(basename "$CKPT" _final.pt)
  echo "[eval] $TAG..."
  .venv/bin/python eval.py \
    --env-id bouncy --checkpoint "$CKPT" \
    --episodes "$EPISODES" --serial $SERIAL \
    --deterministic \
    --max-episode-steps "$MAX_STEPS" \
    > "$OUT_DIR/${TAG}.txt" 2>&1
done

echo "[eval] DONE → $OUT_DIR"
ls -la "$OUT_DIR"
