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

# Restart each emulator's adbd as root so on-device processes minitouch
# launches (via `adb shell nohup ... &`) inherit root context. Without root,
# minitouch's open() on /dev/input/event* fails with EACCES under Android
# 12's SELinux policy — the shell:s0 context isn't allowed access even
# though the shell user is in the input UNIX group. `adb root` is per-device
# on the google_apis AVDs; iterate per serial.
echo "[push_minicap] adb root per-serial for SELinux access to /dev/input/*"
for serial in "${SERIALS[@]}"; do
  out=$(adb -s "$serial" root 2>&1)
  echo "   $serial: $out"
done
sleep 2
# Re-verify by checking 'id' shows uid=0
for serial in "${SERIALS[@]}"; do
  id_out=$(adb -s "$serial" shell id 2>&1 | head -1)
  echo "   $serial id: $id_out"
done

echo "[push_minicap] done; ${#SERIALS[@]} emulator(s) prepared."
