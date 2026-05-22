#!/bin/bash
# Push minicap + minicap.so + minitouch to every emulator on the local ws's
# adb server. Run AFTER orchestrate.py launch so emulators are up.
#
# Usage: ./scripts/push_minicap.sh
#
# Idempotent: safe to re-run; adb push overwrites.
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$PROJ_DIR/scripts/minicap_binaries"

for f in minicap minicap.so minitouch; do
  if [ ! -f "$BIN_DIR/$f" ]; then
    echo "[push_minicap] missing $BIN_DIR/$f — repo not synced?" >&2
    exit 1
  fi
done

# Discover emulator serials via adb (uses whatever ADB_SERVER_SOCKET is set).
mapfile -t SERIALS < <(adb devices | awk '/^emulator-[0-9]+\sdevice/{print $1}')
if [ "${#SERIALS[@]}" -eq 0 ]; then
  echo "[push_minicap] no emulator serials found via adb devices" >&2
  exit 1
fi

for serial in "${SERIALS[@]}"; do
  echo "[push_minicap] -> $serial"
  adb -s "$serial" push "$BIN_DIR/minicap"    /data/local/tmp/minicap    >/dev/null
  adb -s "$serial" push "$BIN_DIR/minicap.so" /data/local/tmp/minicap.so >/dev/null
  adb -s "$serial" push "$BIN_DIR/minitouch"  /data/local/tmp/minitouch  >/dev/null
  adb -s "$serial" shell chmod 755 /data/local/tmp/minicap /data/local/tmp/minitouch
  # Sanity: each binary should run --help-style flag without error
  if ! adb -s "$serial" shell '/data/local/tmp/minitouch -h' 2>&1 | grep -q "Usage:"; then
    echo "[push_minicap] minitouch -h on $serial did not print usage; binary may be incompatible" >&2
    exit 1
  fi
done

echo "[push_minicap] done; ${#SERIALS[@]} emulator(s) prepared."
