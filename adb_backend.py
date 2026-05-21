"""Plain-adb implementation of EmulatorBackend.

Slow (~10-20 steps/sec/instance) but works without minicap/minitouch. Use this
to wire the training loop to a real emulator while Person A is bringing up the
fast I/O path. Once minicap+minitouch are pushed and Person A writes
MinicapMinitouchBackend, swap the constructor in orchestrate.py — the env code
above is unchanged.

Action -> touch coordinate mapping:
    The constants below are placeholders. Capture one screenshot of Bouncy
    Basketball running in the emulator, identify the on-screen buttons (left
    arrow, right arrow, jump button, shoot button), and update the dict.
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

import cv2
import numpy as np

from env import EmulatorBackend, EmulatorEndpoint, NO_PRESS, PRESS

# All action constants live in config.ACTIONS.
from config import ACTIONS

PRESS_COORD: tuple[int, int] = ACTIONS.press_coord
PRESS_FRAME_MS: int = ACTIONS.press_hold_ms


def _adb(serial: str, *args: str, timeout: float = 10.0) -> bytes:
    """Run `adb -s <serial> <args...>` and return stdout. Raises on failure."""
    cmd = ["adb", "-s", serial, *args]
    out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    if out.returncode != 0:
        raise RuntimeError(
            f"adb failed: {' '.join(cmd)}\nstdout:\n{out.stdout!r}\nstderr:\n{out.stderr!r}"
        )
    return out.stdout


class AdbBackend(EmulatorBackend):
    """EmulatorBackend using plain adb shell commands. Use as a fallback until
    minicap/minitouch are available."""

    def __init__(self, endpoint: EmulatorEndpoint):
        super().__init__(endpoint)
        self._pressing = False

    # ---- frame capture -------------------------------------------------
    def grab_frame(self) -> np.ndarray:
        """Returns RGB ndarray of shape (H, W, 3), dtype uint8.

        `adb exec-out screencap -p` writes a PNG to stdout. Latency: 30-80ms.
        """
        png_bytes = _adb(self.endpoint.adb_serial, "exec-out", "screencap", "-p")
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to decode screencap PNG from {self.endpoint.adb_serial}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ---- action injection ---------------------------------------------
    def send_action(self, action: int, hold_ms: int = PRESS_FRAME_MS) -> None:
        """Send NO_PRESS or PRESS to the emulator, holding for `hold_ms`.

        For PRESS we issue an `input swipe x y x y hold_ms` — this is a
        zero-length swipe that holds the touch down for the requested duration.
        The adb call blocks until the swipe completes, so wall-clock time per
        call ≈ hold_ms + adb roundtrip overhead.

        For NO_PRESS we sleep `hold_ms` so game time still advances by the
        same amount, keeping agent steps a consistent length regardless of
        action. (Without this, NO_PRESS steps would race through real time.)

        Transition PRESS -> NO_PRESS is the shoot trigger in Bouncy Basketball;
        no explicit RELEASE event is needed because the previous PRESS's swipe
        already released the touch when its duration expired.
        """
        if action == PRESS:
            x, y = PRESS_COORD
            _adb(
                self.endpoint.adb_serial,
                "shell", "input", "swipe",
                str(x), str(y), str(x), str(y), str(int(hold_ms)),
                timeout=max(10.0, hold_ms / 1000 + 5),
            )
            self._pressing = True
        else:  # NO_PRESS
            self._pressing = False
            time.sleep(hold_ms / 1000.0)

    # ---- utility taps (UI navigation, not RL actions) -----------------
    def send_tap(self, x: int, y: int) -> None:
        """One-shot tap at native landscape coords. Used by env.reset() to
        advance past menu/quarter-end screens after a snapshot load."""
        _adb(self.endpoint.adb_serial, "shell", "input", "tap", str(x), str(y))

    # ---- lifecycle ----------------------------------------------------
    def load_snapshot(self, name: Optional[str] = None) -> None:
        snap = name or self.endpoint.snapshot_names[0]
        _adb(
            self.endpoint.adb_serial,
            "emu", "avd", "snapshot", "load", snap,
            timeout=60.0,
        )
        # Give the snapshot a moment to settle before reading the first frame.
        time.sleep(0.5)

    def is_alive(self) -> bool:
        try:
            state = _adb(self.endpoint.adb_serial, "get-state", timeout=3.0).strip()
            return state == b"device"
        except (RuntimeError, subprocess.TimeoutExpired):
            return False

    def restart(self) -> None:
        """Kill the emulator. The orchestrator's supervisor relaunches it."""
        try:
            _adb(self.endpoint.adb_serial, "emu", "kill", timeout=10.0)
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        # Wait until adb agrees the device is gone, then return.
        for _ in range(20):
            if not self.is_alive():
                return
            time.sleep(0.5)
