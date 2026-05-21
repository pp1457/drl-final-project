#!/bin/bash
# Copy the project code + AVD + SDK from this machine (ws10) to a target ws.
#
# Usage:
#   ./scripts/deploy_one.sh ws1
#
# Idempotent: rsync only sends what's changed. Takes ~30s on a warm cache,
# ~5 min on first deploy (3-4 GB of SDK + AVD + snapshots).
set -euo pipefail

WS="${1:?usage: deploy_one.sh <ws_short_hostname>}"
HOST="${WS}.csie.ntu.edu.tw"
SRC_FINAL="/tmp2/$USER/DRL_final"
SRC_PROJ="/tmp2/$USER/DRL_final_project"

echo "[deploy] $WS -- sanity checking remote..."
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$HOST" "mkdir -p $SRC_FINAL $SRC_PROJ && echo OK"

echo "[deploy] $WS -- syncing SDK + AVD + snapshots ($SRC_FINAL/)..."
rsync -az --delete --exclude='emu_logs/' --exclude='frames/' --exclude='*.log' \
  "$SRC_FINAL/" "$HOST:$SRC_FINAL/"

echo "[deploy] $WS -- syncing project code ($SRC_PROJ/)..."
rsync -az --delete \
  --exclude='__pycache__/' --exclude='runs/' --exclude='checkpoints/' \
  --exclude='.venv/' --exclude='.git/' \
  "$SRC_PROJ/" "$HOST:$SRC_PROJ/"

echo "[deploy] $WS -- rebuilding venv on target (system torch + project deps)..."
ssh "$HOST" "cd $SRC_PROJ && \
  python3 -m venv --system-site-packages .venv 2>/dev/null && \
  .venv/bin/python -m pip install --quiet gymnasium wandb tensorboard pygame pymunk && \
  echo 'venv OK'"

echo "[deploy] $WS DONE."
