"""High-throughput EmulatorBackend using minicap + minitouch.

This file is a *scaffold*. The TCP plumbing is fully implemented; what's
missing is the prebuilt minicap and minitouch x86_64 binaries that have to be
pushed to /data/local/tmp/ on the emulator. Once the binaries land, this
backend is a one-line swap in train.py:

    from minicap_backend import MinicapMinitouchBackend
    backend = MinicapMinitouchBackend(endpoint)

Throughput vs. AdbBackend:
    AdbBackend:                ~50-200 ms per screencap, ~100-200 ms per tap
    Minicap+minitouch (this):  ~16-30 ms per frame stream, ~5 ms per tap event

A typical 5-10× speedup, which is the main remaining lever after frame-skip
and parallel emulators.

=== Setup procedure (run once per emulator AVD) ===

1. Obtain x86_64 builds of minicap+minicap.so and minitouch from the
   openstf project: https://github.com/openstf/minicap and
   https://github.com/openstf/minitouch. Build for Android API 31 (Pixel 5
   system image) targeting x86_64 ABI. Pre-built artifacts also live in the
   STF ecosystem repos.

2. Push to the emulator (per-rank since each emulator has its own filesystem
   under -read-only overlays; with shared snapshot you can include them in
   the snapshot itself by pushing once and re-saving):
        adb -s emulator-XXXX push minicap     /data/local/tmp/minicap
        adb -s emulator-XXXX push minicap.so  /data/local/tmp/minicap.so
        adb -s emulator-XXXX push minitouch   /data/local/tmp/minitouch
        adb -s emulator-XXXX shell chmod 755  /data/local/tmp/minicap
        adb -s emulator-XXXX shell chmod 755  /data/local/tmp/minitouch

3. Add minicap/minitouch start steps to orchestrate.py's launch_farm() after
   the boot wait. The MinicapMinitouchBackend.setup() method below does this
   on a per-instance basis; call it once after construction before training.

=== Action interface alignment ===

The agent's action space is unchanged (Discrete(2): NO_PRESS, PRESS).
Minitouch lets us actually hold a touch across multiple env steps without
re-issuing the swipe each time, so consecutive PRESS actions become a true
sustained hold rather than the chained 130 ms swipes that AdbBackend has to
issue. This makes the jump charge behavior much smoother and likely improves
final policy performance, not just throughput.
"""

from __future__ import annotations

import socket
import struct
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import ACTIONS
from env import EmulatorBackend, EmulatorEndpoint, NO_PRESS, PRESS


# Binary names on-device. Push paths under /data/local/tmp/.
_REMOTE_MINICAP = "/data/local/tmp/minicap"
_REMOTE_MINICAP_SO = "/data/local/tmp/minicap.so"
_REMOTE_MINITOUCH = "/data/local/tmp/minitouch"

# Native display resolution from the AVD. Used in minicap's -P argument.
_DISPLAY_W = 1080
_DISPLAY_H = 2340

# Capture resolution we ask minicap to deliver (small = fast).
_CAPTURE_W = 540   # half native; downsample on-device to save bandwidth
_CAPTURE_H = 1170

# Orientation: 1 = landscape (matches our AVD's mCurrentOrientation=1).
_CAPTURE_ORIENTATION = 1


def _adb_args(serial: str, *args: str) -> list[str]:
    return ["adb", "-s", serial, *args]


def _adb(serial: str, *args: str, timeout: float = 10.0, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(_adb_args(serial, *args), capture_output=True, timeout=timeout, check=check)


def _adb_shell_bg(serial: str, *cmd: str) -> subprocess.Popen:
    """Run a long-lived adb shell command in the background. Returns Popen so
    callers can terminate the on-device process."""
    return subprocess.Popen(
        _adb_args(serial, "shell", *cmd),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


class MinicapMinitouchBackend(EmulatorBackend):
    """High-throughput backend using minicap (frames) + minitouch (taps).

    Lifecycle:
        backend = MinicapMinitouchBackend(endpoint)
        backend.setup()      # pushes nothing; starts on-device daemons + adb forwards
        ... use grab_frame / send_action / load_snapshot / restart ...
        backend.teardown()   # closes sockets, kills daemons
    """

    def __init__(self, endpoint: EmulatorEndpoint) -> None:
        super().__init__(endpoint)
        self._minicap_proc: Optional[subprocess.Popen] = None
        self._minitouch_proc: Optional[subprocess.Popen] = None
        self._minicap_sock: Optional[socket.socket] = None
        self._minitouch_sock: Optional[socket.socket] = None
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._pump_thread: Optional[threading.Thread] = None
        self._stop_pump = threading.Event()
        # The pressing flag tracks whether minitouch currently has a finger
        # held down. Lets us issue DOWN/UP only on press-state transitions.
        self._pressing = False

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------
    def setup(self) -> None:
        self._check_binaries_pushed()
        self._start_minicap()
        self._start_minitouch()

    def teardown(self) -> None:
        self._stop_pump.set()
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=2.0)
        for sock in (self._minicap_sock, self._minitouch_sock):
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass
        for proc in (self._minicap_proc, self._minitouch_proc):
            if proc is not None:
                proc.terminate()
        self._minicap_sock = self._minitouch_sock = None
        self._minicap_proc = self._minitouch_proc = None

    def _check_binaries_pushed(self) -> None:
        r = _adb(
            self.endpoint.adb_serial, "shell", "ls", "-l",
            _REMOTE_MINICAP, _REMOTE_MINICAP_SO, _REMOTE_MINITOUCH,
            check=False,
        )
        if b"No such file" in r.stdout + r.stderr:
            raise RuntimeError(
                f"minicap/minitouch binaries not present on {self.endpoint.adb_serial}. "
                "See the setup procedure docstring at the top of minicap_backend.py."
            )

    # ------------------------------------------------------------------
    # Minicap (frame stream)
    # ------------------------------------------------------------------
    def _start_minicap(self) -> None:
        # Project arg format: <real_w>x<real_h>@<virt_w>x<virt_h>/<orientation>
        proj = f"{_DISPLAY_W}x{_DISPLAY_H}@{_CAPTURE_W}x{_CAPTURE_H}/{_CAPTURE_ORIENTATION}"
        local_port = 1313 + (int(self.endpoint.adb_serial.split("-")[1]) - ACTIONS.press_coord[0])  # unique-ish
        local_port = 19000 + int(self.endpoint.adb_serial.split("-")[1]) % 1000
        _adb(self.endpoint.adb_serial, "forward", f"tcp:{local_port}", "localabstract:minicap")
        self._minicap_proc = _adb_shell_bg(
            self.endpoint.adb_serial,
            f"LD_LIBRARY_PATH=/data/local/tmp/ {_REMOTE_MINICAP} -P {proj}",
        )
        # Connect to the forwarded socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for _ in range(50):  # wait up to ~5s for minicap to bind
            try:
                sock.connect(("127.0.0.1", local_port))
                break
            except ConnectionRefusedError:
                time.sleep(0.1)
        else:
            raise RuntimeError(f"failed to connect to minicap on tcp:{local_port}")
        # Read the global header (24 bytes)
        header = sock.recv(24)
        if len(header) < 24:
            raise RuntimeError("minicap global header truncated")
        self._minicap_sock = sock
        self._stop_pump.clear()
        self._pump_thread = threading.Thread(target=self._frame_pump, daemon=True)
        self._pump_thread.start()

    def _frame_pump(self) -> None:
        """Background thread reading the minicap stream and stashing the
        latest decoded frame. Avoids head-of-line blocking on the policy."""
        assert self._minicap_sock is not None
        sock = self._minicap_sock
        while not self._stop_pump.is_set():
            # Per-frame header: 4-byte little-endian length
            size_bytes = self._recv_exact(sock, 4)
            if size_bytes is None:
                return
            (size,) = struct.unpack("<I", size_bytes)
            payload = self._recv_exact(sock, size)
            if payload is None:
                return
            # minicap delivers JPEG-encoded frames by default
            arr = np.frombuffer(payload, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._frame_lock:
                self._latest_frame = rgb

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def grab_frame(self) -> np.ndarray:
        # Block briefly until the first frame arrives, then return the
        # most-recent buffered frame (don't queue up stale ones).
        for _ in range(200):  # ~2s max
            with self._frame_lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
            time.sleep(0.01)
        raise RuntimeError("minicap produced no frames within 2s")

    # ------------------------------------------------------------------
    # Minitouch (touch injection)
    # ------------------------------------------------------------------
    def _start_minitouch(self) -> None:
        local_port = 20000 + int(self.endpoint.adb_serial.split("-")[1]) % 1000
        _adb(self.endpoint.adb_serial, "forward", f"tcp:{local_port}", "localabstract:minitouch")
        self._minitouch_proc = _adb_shell_bg(self.endpoint.adb_serial, _REMOTE_MINITOUCH)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for _ in range(50):
            try:
                sock.connect(("127.0.0.1", local_port))
                break
            except ConnectionRefusedError:
                time.sleep(0.1)
        else:
            raise RuntimeError(f"failed to connect to minitouch on tcp:{local_port}")
        # minitouch sends a banner: "v <version>\n^ <max_contacts> <max_x> <max_y> <max_pressure>\n$ <pid>\n"
        # Skip it -- we don't currently need the values.
        sock.settimeout(2.0)
        try:
            _banner = sock.recv(1024)
        except socket.timeout:
            pass
        sock.settimeout(None)
        self._minitouch_sock = sock

    def send_action(self, action: int, hold_ms: int = ACTIONS.press_hold_ms) -> None:
        """Real tap-and-hold via minitouch.

        Unlike AdbBackend (which blocks Python for `hold_ms` while the swipe
        runs), minitouch lets us hold a finger down asynchronously. We issue
        d (down) on press-state transitions and u (up) on the opposite. For
        consistency with AdbBackend's wall-clock-per-step we still sleep
        hold_ms here, but the touch is genuinely held throughout.
        """
        sock = self._minitouch_sock
        if sock is None:
            raise RuntimeError("minitouch not initialized; call setup() first")
        x, y = ACTIONS.press_coord
        if action == PRESS and not self._pressing:
            # finger 0 down at (x, y, pressure 50)
            sock.sendall(f"d 0 {x} {y} 50\nc\n".encode("ascii"))
            self._pressing = True
        elif action == NO_PRESS and self._pressing:
            sock.sendall(b"u 0\nc\n")
            self._pressing = False
        time.sleep(hold_ms / 1000.0)

    # ------------------------------------------------------------------
    # Lifecycle delegated to plain adb (minicap doesn't help here)
    # ------------------------------------------------------------------
    def load_snapshot(self, name: Optional[str] = None) -> None:
        snap = name or self.endpoint.snapshot_names[0]
        # Pause the frame pump while the VM state changes underneath it
        self._stop_pump.set()
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=2.0)
        _adb(self.endpoint.adb_serial, "emu", "avd", "snapshot", "load", snap, timeout=60.0)
        time.sleep(1.0)
        # Restart the frame pump
        self._stop_pump.clear()
        self._pump_thread = threading.Thread(target=self._frame_pump, daemon=True)
        self._pump_thread.start()

    def is_alive(self) -> bool:
        try:
            r = _adb(self.endpoint.adb_serial, "get-state", timeout=3.0, check=False)
            return r.stdout.strip() == b"device"
        except (RuntimeError, subprocess.TimeoutExpired):
            return False

    def restart(self) -> None:
        self.teardown()
        try:
            _adb(self.endpoint.adb_serial, "emu", "kill", timeout=10.0, check=False)
        except subprocess.TimeoutExpired:
            pass
        # Caller's supervisor relaunches; we'll re-setup() on next reset.

    def send_tap(self, x: int, y: int) -> None:
        """One-shot tap for UI navigation. Uses minitouch since it's available."""
        sock = self._minitouch_sock
        if sock is None:
            raise RuntimeError("minitouch not initialized")
        sock.sendall(f"d 0 {x} {y} 50\nc\nu 0\nc\n".encode("ascii"))
