"""Multi-emulator orchestration.

Launches N read-only emulator instances, waits for each to boot, and exposes a
mapping from worker rank to EmulatorEndpoint. Also provides:

    bootstrap_single()  -- one-time setup: cold boot, install APK, save snapshot
    launch_farm(n)      -- start N emulators sharing one AVD via -read-only
    kill_all()          -- adb emu kill on all running serials
    supervisor()        -- background loop that restarts dead emulators

Port assignments (Android convention):
    instance i -> console port 5554+2i, adb serial "emulator-(5554+2i)"

Usage:
    # one-time, after env_setup.md steps 1-7:
    python orchestrate.py bootstrap --apk /path/to/BouncyBasketball.apk

    # before each training run:
    python orchestrate.py launch --n 8

    # after each training run:
    python orchestrate.py kill
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from env import EmulatorEndpoint


AVD_NAME = "pixel5_api31"
SNAPSHOT = "clean_boot"
EMU_LOG_DIR = Path("/tmp2") / os.environ.get("USER", "unknown") / "DRL_final" / "emu_logs"
ENDPOINTS_FILE = Path("/tmp2") / os.environ.get("USER", "unknown") / "DRL_final" / "endpoints.json"

# Base console port. We use an unusual range to avoid collisions with other
# users on the same shared workstation. Console port = BASE_PORT + 2*rank,
# adb port = console + 1.
BASE_PORT = int(os.environ.get("EMU_BASE_PORT", 6554))

# Emulator launch flags shared by every instance.
COMMON_FLAGS = [
    "-no-window",
    "-no-audio",
    "-no-boot-anim",
    "-gpu", "swiftshader_indirect",
    "-no-metrics",
    "-no-snapshot-save",   # do not overwrite the saved snapshot at exit
]


def _serial_for(rank: int) -> str:
    return f"emulator-{BASE_PORT + 2 * rank}"


def _adb(*args: str, timeout: float = 10.0, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", *args], capture_output=True, timeout=timeout, check=check)


def _adb_for(serial: str, *args: str, **kw) -> subprocess.CompletedProcess:
    return _adb("-s", serial, *args, **kw)


def _wait_boot(serial: str, timeout: float = 300.0) -> None:
    """Block until `sys.boot_completed == 1` on the given device."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _adb_for(serial, "shell", "getprop", "sys.boot_completed", timeout=5.0, check=False)
            if r.stdout.strip() == b"1":
                # Also wait until package manager is up
                r2 = _adb_for(serial, "shell", "service", "check", "package", timeout=5.0, check=False)
                if b"found" in r2.stdout:
                    return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(3)
    raise TimeoutError(f"{serial}: boot did not complete within {timeout}s")


def _launch_emulator(rank: int, *, read_only: bool, snapshot: Optional[str]) -> subprocess.Popen:
    """Spawn an emulator process for the given rank. Returns the Popen handle."""
    EMU_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EMU_LOG_DIR / f"emu_{rank}.log"

    port = BASE_PORT + 2 * rank
    cmd = ["emulator", "-avd", AVD_NAME, "-port", str(port), *COMMON_FLAGS]
    if read_only:
        cmd.append("-read-only")
    if snapshot:
        cmd += ["-snapshot", snapshot]

    f = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
    )
    print(f"[rank {rank}] pid={proc.pid} port={port} log={log_path}")
    return proc


# -------------------------------------------------------------------
# Public commands
# -------------------------------------------------------------------
def bootstrap_single(apk_path: str) -> None:
    """One-time emulator setup: cold boot the AVD, install APK, save snapshot.

    Run *once* after the env_setup.md steps that create the AVD (steps 1-7).
    Replaces the manual env_setup.md steps 8-10.
    """
    print("Launching single emulator for bootstrap (cold boot)...")
    proc = _launch_emulator(rank=0, read_only=False, snapshot=None)
    try:
        serial = _serial_for(0)
        print(f"Waiting for {serial} to finish booting (this can take ~5 minutes)...")
        _wait_boot(serial, timeout=600.0)
        print("Boot complete. Installing APK...")
        _adb_for(serial, "install", "-r", apk_path, timeout=120.0)
        print("APK installed. Saving snapshot...")
        _adb_for(serial, "emu", "avd", "snapshot", "save", SNAPSHOT, timeout=60.0)
        print(f"Snapshot '{SNAPSHOT}' saved.")
    finally:
        print("Shutting down bootstrap emulator...")
        _adb_for(serial, "emu", "kill", check=False)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()


def launch_farm(n: int) -> list[EmulatorEndpoint]:
    """Start N read-only emulators sharing the saved snapshot."""
    EMU_LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    for rank in range(n):
        procs.append(_launch_emulator(rank=rank, read_only=True, snapshot=SNAPSHOT))
        time.sleep(2.0)  # stagger to avoid thundering-herd on emulator startup

    endpoints: list[EmulatorEndpoint] = []
    for rank in range(n):
        serial = _serial_for(rank)
        print(f"[rank {rank}] waiting for {serial} to boot...")
        _wait_boot(serial, timeout=300.0)
        endpoints.append(
            EmulatorEndpoint(
                adb_serial=serial,
                minicap_port=0,         # not yet -- AdbBackend doesn't use these
                minitouch_port=0,
                snapshot_name=SNAPSHOT,
            )
        )

    # Persist endpoints so the trainer (in a separate process) can read them.
    ENDPOINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ENDPOINTS_FILE, "w") as f:
        json.dump([ep.__dict__ for ep in endpoints], f, indent=2)
    print(f"Wrote {len(endpoints)} endpoints to {ENDPOINTS_FILE}")
    print(f"Emulator PIDs: {[p.pid for p in procs]}")
    return endpoints


def kill_all() -> None:
    """Kill every running emulator we can see via `adb devices`."""
    r = _adb("devices", check=False)
    serials = []
    for line in r.stdout.decode().splitlines()[1:]:
        line = line.strip()
        if not line or "\tdevice" not in line:
            continue
        serials.append(line.split("\t")[0])
    for serial in serials:
        print(f"killing {serial}")
        _adb_for(serial, "emu", "kill", check=False, timeout=5.0)
    if ENDPOINTS_FILE.exists():
        ENDPOINTS_FILE.unlink()


def load_endpoints() -> list[EmulatorEndpoint]:
    """Read the endpoints written by launch_farm. Called by the trainer."""
    with open(ENDPOINTS_FILE) as f:
        raw = json.load(f)
    return [EmulatorEndpoint(**d) for d in raw]


def supervise_once(endpoint: EmulatorEndpoint) -> bool:
    """Check one emulator; if dead, relaunch from snapshot. Returns True if
    relaunch occurred (caller should reset its env after a relaunch)."""
    r = _adb_for(endpoint.adb_serial, "get-state", check=False, timeout=3.0)
    if r.stdout.strip() == b"device":
        return False
    rank = (int(endpoint.adb_serial.split("-")[1]) - BASE_PORT) // 2
    print(f"[supervisor] {endpoint.adb_serial} appears dead; relaunching")
    _launch_emulator(rank=rank, read_only=True, snapshot=SNAPSHOT)
    _wait_boot(endpoint.adb_serial, timeout=180.0)
    return True


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_boot = sub.add_parser("bootstrap")
    p_boot.add_argument("--apk", required=True)

    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--n", type=int, default=8)

    sub.add_parser("kill")

    p_check = sub.add_parser("supervise-once")

    args = ap.parse_args()
    if args.cmd == "bootstrap":
        bootstrap_single(args.apk)
    elif args.cmd == "launch":
        launch_farm(args.n)
    elif args.cmd == "kill":
        kill_all()
    elif args.cmd == "supervise-once":
        endpoints = load_endpoints()
        any_restarted = False
        for ep in endpoints:
            any_restarted |= supervise_once(ep)
        sys.exit(0 if not any_restarted else 1)


if __name__ == "__main__":
    main()
