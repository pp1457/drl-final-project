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
    """Run `adb -s <serial> <args...>` and return stdout. Raises on failure.

    NOTE: subprocess.TimeoutExpired is caught and re-raised as a plain
    RuntimeError. gymnasium's AsyncVectorEnv tries to reconstruct exceptions
    across the worker/parent pipe as `exctype(value)`, and
    `subprocess.TimeoutExpired.__init__` requires two positional args (`cmd`,
    `timeout`) — so the reconstruct fails with a TypeError that crashes the
    trainer (we saw this kill ws1 + ws6 ~upd 40). Plain RuntimeError pickles
    and reconstructs cleanly.
    """
    cmd = ["adb", "-s", serial, *args]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"adb timed out after {timeout}s: {' '.join(cmd)}") from e
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

    # ---- foreground / app state --------------------------------------
    def foreground_package(self) -> str:
        """Return the package name of the currently focused activity, or ''
        if it can't be parsed. Used to detect when the game has dropped out
        of foreground (back to the Pixel launcher home screen) — observed
        across all 9 ws around upd 40-50 in the May 22 run."""
        try:
            out = _adb(
                self.endpoint.adb_serial, "shell",
                "dumpsys window | grep mCurrentFocus",
                timeout=5.0,
            ).decode("utf-8", errors="ignore")
        except RuntimeError:
            return ""
        # Format: "mCurrentFocus=Window{... <package>/<activity>}"
        import re
        m = re.search(r"mCurrentFocus=.*\s+([a-zA-Z0-9_.]+)/", out)
        return m.group(1) if m else ""

    def launch_app(self, package: str) -> None:
        """Bring the given package's launcher activity to the foreground.

        Uses `am start -n` instead of `monkey -p ... LAUNCHER 1` because monkey
        on Android 12 can briefly grab the input subsystem (it's a touch
        fuzzer in its other modes), which knocks an already-running minitouch
        off /dev/input/event10. `am start` is a pure activity-manager call
        with no input-system side effects.
        """
        # Bouncy Basketball is a Unity game; its main activity is universally
        # com.unity3d.player.UnityPlayerActivity for all Unity 2017+ builds.
        activity = f"{package}/com.unity3d.player.UnityPlayerActivity"
        _adb(
            self.endpoint.adb_serial, "shell",
            f"am start -n {activity}",
            timeout=15.0,
        )

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
        except RuntimeError:
            return False

    def restart(self) -> None:
        """Kill the emulator. The orchestrator's supervisor relaunches it."""
        try:
            _adb(self.endpoint.adb_serial, "emu", "kill", timeout=10.0)
        except RuntimeError:
            pass
        # Wait until adb agrees the device is gone, then return.
        for _ in range(20):
            if not self.is_alive():
                return
            time.sleep(0.5)


# Module-level for the minitouch port allocation. Mirror what minicap_backend.py
# uses so the two can coexist (different port ranges).
_MINITOUCH_REMOTE = "/data/local/tmp/minitouch"


class AdbMinitouchBackend(AdbBackend):
    """Frames via adb screencap (slow but unrestricted by Android 12's
    SurfaceFlinger lockdown). Actions via minitouch — a state-based touch
    interface that lets a touch persist across env steps, so consecutive
    PRESS actions accumulate into a *true* held press of arbitrary duration.

    This is the v2-light backend: we couldn't get minicap working on
    Android 12 (the API-21 binary dies silently against the post-API-29
    SurfaceFlinger), but minitouch's input layer is unchanged and works
    fine. So we keep adb screencap for frames and unlock the action-space
    win for free.

    Setup pushes happen via scripts/push_minicap.sh; this class assumes
    /data/local/tmp/minitouch exists on the emulator.
    """

    def __init__(self, endpoint: EmulatorEndpoint):
        super().__init__(endpoint)
        import socket as _socket  # local to avoid polluting the AdbBackend path
        self._socket_mod = _socket
        self._minitouch_proc: Optional[subprocess.Popen] = None
        self._minitouch_sock: Optional[_socket.socket] = None
        # Reuse AdbBackend's _pressing? We override send_action so it's fine.
        self._mt_pressing = False

    def setup(self) -> None:
        """Start minitouch on the device + connect to it via adb forward."""
        # Local port per-serial: 20000 + (port_suffix % 1000) — same scheme as
        # minicap_backend.py.
        port_suffix = int(self.endpoint.adb_serial.split("-")[1])
        local_port = 20000 + port_suffix % 1000
        # Ensure adbd is running as root on this serial. minitouch needs root
        # to open /dev/input/event* under Android 12's SELinux policy.
        # CRITICAL: only call `adb root` if NOT already root — otherwise the
        # second env's setup() restarts adbd a second time which kills the
        # first env's already-established minitouch socket. (Earlier launch
        # died with BrokenPipeError on every 2nd send_tap because of this.)
        id_out = subprocess.run(
            ["adb", "-s", self.endpoint.adb_serial, "shell", "id"],
            capture_output=True, timeout=5.0,
        ).stdout.decode("ascii", errors="ignore")
        if "uid=0" not in id_out:
            subprocess.run(
                ["adb", "-s", self.endpoint.adb_serial, "root"],
                capture_output=True, timeout=10.0,
            )
            time.sleep(2.0)
            subprocess.run(
                ["adb", "-s", self.endpoint.adb_serial, "wait-for-device"],
                capture_output=True, timeout=10.0,
            )
        # Kill any leftover minitouch from a prior run, then fire & forget
        # via nohup so the on-device process survives our Python subprocess
        # going away.
        _adb(self.endpoint.adb_serial, "shell", "pkill -9 minitouch || true", timeout=5.0)
        time.sleep(0.3)
        _adb(self.endpoint.adb_serial, "forward", f"tcp:{local_port}", "localabstract:minitouch")
        # CRITICAL: pin minitouch to /dev/input/event3 (= virtio_input_multi_touch_1).
        # The Pixel 5 x86_64 AVD ships with ~12 virtio touch devices; minitouch's
        # auto-detect picks "multi_touch_8" (event10), whose IDC config routes
        # events to a *virtual* display (`touch.displayId =
        # virtual:com.android.emulator.multidisplay:1234568`) — NOT the main
        # display where the game is rendered. So taps went into the void.
        # multi_touch_1 (event3) has no displayId override → routes to main.
        # Verified by manual screen-tap test: tap at landscape (1170, 793)
        # via event3 advanced END-OF-QUARTER stats → Q2 gameplay.
        subprocess.run(
            ["adb", "-s", self.endpoint.adb_serial, "shell",
             "nohup /data/local/tmp/minitouch -d /dev/input/event3 >/dev/null 2>&1 &"],
            capture_output=True, timeout=5.0,
        )
        # Connect — minitouch needs a moment to bind its abstract socket.
        sock = self._socket_mod.socket(self._socket_mod.AF_INET, self._socket_mod.SOCK_STREAM)
        last_err = None
        for _ in range(50):  # up to ~5s
            try:
                sock.connect(("127.0.0.1", local_port))
                break
            except ConnectionRefusedError as e:
                last_err = e
                time.sleep(0.1)
        else:
            raise RuntimeError(f"failed to connect to minitouch on tcp:{local_port}: {last_err}")
        # Read + parse the banner: "v 1\n^ <max_contacts> <max_x> <max_y> <max_p>\n$ <pid>\n"
        # Critically, minitouch uses virtual 0..max_x/max_y coords, NOT screen
        # pixels. Save the scale so send_action can convert PRESS_COORD from
        # pixel space to minitouch's virtual space.
        sock.settimeout(2.0)
        banner = b""
        try:
            while b"$" not in banner:
                chunk = sock.recv(256)
                if not chunk:
                    break
                banner += chunk
        except self._socket_mod.timeout:
            pass
        sock.settimeout(None)
        self._minitouch_sock = sock
        # Default scale = 1:1 (assume coords == pixels) if parsing fails.
        self._mt_max_x = 32767
        self._mt_max_y = 32767
        for line in banner.decode("ascii", errors="ignore").splitlines():
            if line.startswith("^"):
                parts = line.split()
                if len(parts) >= 5:
                    self._mt_max_x = int(parts[2])
                    self._mt_max_y = int(parts[3])
                    break

    def teardown(self) -> None:
        if self._minitouch_sock is not None:
            try:
                self._minitouch_sock.close()
            except OSError:
                pass
            self._minitouch_sock = None
        if self._minitouch_proc is not None:
            self._minitouch_proc.terminate()
            self._minitouch_proc = None

    # Landscape display dimensions. With minitouch pinned to event3
    # (multi_touch_1, no displayId override → main display, orientationAware →
    # auto-rotates events for current display orientation), minitouch's virtual
    # (mx, my) in 32767x32767 space maps DIRECTLY to landscape pixels. No
    # rotation needed. Verified by visual test: tap at landscape (1170, 793)
    # via event3 advanced the END-OF-QUARTER stats → Q2 gameplay.
    _LANDSCAPE_W = 2340
    _LANDSCAPE_H = 1080

    def _to_mt_coords(self, px: int, py: int) -> tuple[int, int]:
        """Convert landscape pixel coords (matching adb screencap dims +
        PRESS_COORD) to minitouch's virtual 32767x32767 space."""
        mx = int(px * self._mt_max_x / self._LANDSCAPE_W)
        my = int(py * self._mt_max_y / self._LANDSCAPE_H)
        return mx, my

    def _send_minitouch(self, payload: bytes) -> None:
        """Send `payload` over the minitouch socket, with one reconnect retry
        on BrokenPipeError. Minitouch's socket can drop unexpectedly (e.g.
        after the game's first orientation change loses its input handle and
        re-acquires it). One reconnect typically recovers."""
        sock = self._minitouch_sock
        if sock is None:
            raise RuntimeError("AdbMinitouchBackend not initialized; call setup() first")
        try:
            sock.sendall(payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Reconnect: kill any orphan + restart minitouch + reconnect socket.
            try:
                sock.close()
            except OSError:
                pass
            self._minitouch_sock = None
            self._mt_pressing = False  # device state reset on reconnect
            self.setup()
            assert self._minitouch_sock is not None
            self._minitouch_sock.sendall(payload)

    def send_action(self, action: int, hold_ms: int = PRESS_FRAME_MS) -> None:
        """State-based touch: send DOWN on press-state transition, UP on
        release-transition, nothing on continuation."""
        if self._minitouch_sock is None:
            raise RuntimeError("AdbMinitouchBackend not initialized; call setup() first")
        if action == PRESS and not self._mt_pressing:
            mx, my = self._to_mt_coords(*PRESS_COORD)
            self._send_minitouch(f"d 0 {mx} {my} 50\nc\n".encode("ascii"))
            self._mt_pressing = True
        elif action == NO_PRESS and self._mt_pressing:
            self._send_minitouch(b"u 0\nc\n")
            self._mt_pressing = False
        # Brief sleep so the game can actually process hold_ms of game time
        # in the current touch state before we next observe / decide.
        if hold_ms > 0:
            time.sleep(hold_ms / 1000.0)

    def send_tap(self, x: int, y: int) -> None:
        """One-shot UI tap. Use minitouch (faster) when the socket is alive,
        fall back to plain `adb shell input tap` if minitouch isn't set up
        yet (matters during env.reset() before setup() has run)."""
        if self._minitouch_sock is None:
            super().send_tap(x, y)
            return
        mx, my = self._to_mt_coords(x, y)
        self._send_minitouch(f"d 0 {mx} {my} 50\nc\nu 0\nc\n".encode("ascii"))

    def restart(self) -> None:
        # Tear down minitouch socket before adb emu kill so we don't get
        # a hung send on the next setup().
        self.teardown()
        super().restart()
