"""Gymnasium-compatible env wrapping a single Bouncy Basketball emulator.

This file defines the integration contract between three subsystems:

    Person A's work:  EmulatorBackend (process lifecycle, minicap/minitouch I/O)
    Person B's work:  pose extraction (vision.py) + reward extraction (reward.py)
    Person C's work:  PPO trainer + shared encoder + three aux heads (train.py)

Concrete methods that touch the emulator raise NotImplementedError until Person A
plugs in the backend. The Env class itself, observation shape, action space, info
payload, and OCA target packing are stable — Persons B and C can develop and test
against this contract today.

Observation:
    np.ndarray, dtype=uint8, shape=(84, 84), grayscale. Frame-stacking k=4 is a
    wrapper applied outside this env (see VecEnvFrameStack in train.py), not
    inside, so the base env stays cheap to copy across worker processes.

Action space:
    Discrete(2) = {NO_PRESS, PRESS}. Bouncy Basketball is a one-button game:
    tap-and-hold anywhere = player jumps (longer hold = higher), release in the
    air = shoot. The player auto-moves toward the ball. Consecutive PRESS
    actions keep the touch held (charging the jump). The frame-skip wrapper
    (k=4) is applied externally; one env step ≈ 130 ms of game time.

info payload (per step):
    info["full_rgb"]:    np.ndarray, dtype=uint8, shape=(H, W, 3)
        The pre-resize RGB frame from the emulator. Used by:
            - the DPR aux head to compute reconstruction targets at native res
              (downsampled by the trainer if needed),
            - the OCA CV pipeline which runs on the full frame for accuracy.
    info["oca_target"]:  np.ndarray, dtype=float32, shape=(10,)
        [xb, yb, xp, yp, sinp, cosp, xo, yo, sino, coso] (normalized).
    info["oca_mask"]:    np.ndarray, dtype=float32, shape=(10,)
        Per-element mask; 1.0 where detection succeeded, 0.0 otherwise.
        The OCA loss multiplies the per-element squared error by this mask so
        frames with partial detections still contribute their valid components.
    info["raw_score"]:   int
        Cumulative game score since the last reset, as reported by the reward
        extractor. Used for evaluation logging; the per-step reward is the
        delta and is already returned as the env reward.

Reward:
    +1.0 on goal scored (delta in raw_score > 0 between consecutive frames),
    0.0 otherwise. Episode terminates on match end (detected from the scoreboard
    region; see reward.py).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

import cv2
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # tolerated only while pip-install is in progress
    import gym
    from gym import spaces  # type: ignore


# Action indices. Bouncy Basketball is a single-button game:
#   - tap-and-hold anywhere = player jumps (longer hold = higher jump)
#   - release in air = shoot
#   - movement is automatic (player auto-walks toward the ball)
# So the action space is binary: at each step, agent chooses NO_PRESS or PRESS.
# Press state is sticky across consecutive PRESS actions (= sustained hold).
# Transition PRESS -> NO_PRESS issues the release/shoot.
NO_PRESS, PRESS = range(2)
N_ACTIONS = 2

# Kept as aliases so legacy callers don't break while migrating. All map to PRESS.
JUMP = CHARGE = PRESS
NOOP = LEFT = RIGHT = NO_PRESS
RELEASE = NO_PRESS

# Observation resolution after resize.
OBS_H, OBS_W = 84, 84

# OCA target dimensionality: [xb, yb, xp, yp, sinp, cosp, xo, yo, sino, coso]
OCA_DIM = 10


@dataclasses.dataclass
class EmulatorEndpoint:
    """Where this env's emulator lives. Set by the orchestration script before
    constructing the env in the worker subprocess.

    `snapshot_names` is a list because we train against a *pool* of opponents
    rather than a single one — each reset picks a random snapshot from the pool
    so the agent sees diverse opposing AIs and doesn't overfit to one
    opponent's quirks. For backward compat we accept a single snapshot_name and
    promote it to a one-element list.
    """
    adb_serial: str           # e.g. "emulator-5554"
    minicap_port: int         # TCP port for raw frames
    minitouch_port: int       # TCP port for touch injection
    snapshot_names: list[str] = dataclasses.field(default_factory=lambda: ["clean_boot"])

    # Backward-compat alias; populates snapshot_names from a single string.
    snapshot_name: dataclasses.InitVar[Optional[str]] = None

    def __post_init__(self, snapshot_name: Optional[str]) -> None:
        if snapshot_name is not None:
            self.snapshot_names = [snapshot_name]


class EmulatorBackend:
    """Thin wrapper around a single emulator. Person A's implementation.

    This class is the only place that talks to adb / minicap / minitouch. All
    other code in the project sees the abstract methods below.
    """

    def __init__(self, endpoint: EmulatorEndpoint) -> None:
        self.endpoint = endpoint

    # --- frame capture ---
    def grab_frame(self) -> np.ndarray:
        """Return latest RGB frame from minicap, dtype=uint8, shape=(H, W, 3)."""
        raise NotImplementedError("Person A: connect to minicap socket")

    # --- action injection ---
    def send_action(self, action: int) -> None:
        """Inject a touch event via minitouch corresponding to `action`."""
        raise NotImplementedError("Person A: open minitouch socket, write events")

    # --- lifecycle ---
    def load_snapshot(self, name: Optional[str] = None) -> None:
        """Load a named snapshot. Defaults to the first in the endpoint's pool."""
        raise NotImplementedError("Person A: shell out to adb")

    def is_alive(self) -> bool:
        """`adb -s <serial> get-state` == 'device'. False after a crash."""
        raise NotImplementedError("Person A: probe adb state")

    def restart(self) -> None:
        """Kill the emulator and relaunch from snapshot. Called by the supervisor."""
        raise NotImplementedError("Person A: kill + relaunch")


class BouncyBasketballEnv(gym.Env):
    """One emulator = one env instance. Vector-wrap externally for parallelism.

    Frame skip is applied INSIDE this env (not via an external wrapper) because
    the agent's action — PRESS — is meant to be *held* across consecutive game
    frames to charge a jump. We repeat the action for `frame_skip` game frames
    per agent step, but emit only one obs/reward to the policy. Aux targets
    (OCA, DPR) come from the LAST of the skipped frames, since that's the
    obs the policy actually sees.

    With frame_skip=4 and ~1 game frame ~ 130ms, each agent step is ~520ms of
    game time, which is roughly the duration of a normal jump and lets the
    policy decide at a sensible cadence without micro-managing each frame.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: EmulatorBackend,
        pose_extractor,           # callable: (rgb_frame) -> dict (see vision.py)
        reward_extractor,         # callable: (rgb_frame) -> (score, done)
        max_episode_steps: Optional[int] = None,   # default from config.PPO
        frame_skip: Optional[int] = None,          # default from config.ACTIONS
    ) -> None:
        from config import ACTIONS, PPO  # local import keeps env.py importable
                                          # even when config isn't on path
        super().__init__()
        self.backend = backend
        self.pose_extractor = pose_extractor
        self.reward_extractor = reward_extractor
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else PPO.max_episode_steps
        self.frame_skip = max(1, int(frame_skip if frame_skip is not None else ACTIONS.frame_skip))

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_H, OBS_W), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._step_count = 0
        self._raw_score = 0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        # Pick a random snapshot from the endpoint's pool. Different snapshots
        # correspond to different opposing teams (HOU, LAC, ...), giving the
        # agent diverse opponents across episodes.
        rng = self.np_random
        snapshots = self.backend.endpoint.snapshot_names
        chosen = snapshots[int(rng.integers(len(snapshots)))]
        self.backend.load_snapshot(chosen)
        self._step_count = 0
        self._raw_score = 0
        # Advance from whatever screen the snapshot landed on (team-select OR
        # end-of-quarter stats) into active play. The PLAY button (team-select)
        # and the NEXT QUARTER button (stats screen) are both green and live at
        # landscape coords found via OpenCV — fire both, exactly one will hit.
        # On an in-game snapshot both taps hit empty floor (no-op).
        if hasattr(self.backend, "send_tap"):
            self.backend.send_tap(1493, 918)   # PLAY (team-select)
            time.sleep(2.0)
            self.backend.send_tap(1170, 793)   # NEXT QUARTER (stats screen)
            time.sleep(1.5)
        # Best-effort reset for stateful reward extractors.
        rs = getattr(self, "_reward_state", None)
        if rs is not None and hasattr(rs, "reset"):
            rs.reset()
        rgb = self.backend.grab_frame()
        obs = self._to_obs(rgb)
        info = self._build_info(rgb, score_delta=0)
        return obs, info

    # One game frame at the emulator's native ~30 fps. Used to convert
    # frame_skip into the touch-hold duration we pass to the backend.
    _GAME_FRAME_MS = 33

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply `action`, held for `frame_skip` game frames, then read one obs.

        Efficient frame skip: ONE adb call (holding the touch for the full
        skip duration) + ONE screencap per agent step. The action's effect
        plays out across the skipped frames inside the emulator while Python
        blocks on the swipe call.
        """
        hold_ms = self.frame_skip * self._GAME_FRAME_MS
        # send_action accepts hold_ms on AdbBackend; fall back to default on
        # backends that don't (e.g. FakeBackend).
        try:
            self.backend.send_action(int(action), hold_ms=hold_ms)
        except TypeError:
            self.backend.send_action(int(action))
        rgb = self.backend.grab_frame()
        score, episode_done = self.reward_extractor(rgb)
        reward = float(max(0, score - self._raw_score))
        self._raw_score = score
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        info = self._build_info(rgb, score_delta=reward)
        obs = self._to_obs(rgb)
        return obs, reward, bool(episode_done), bool(truncated), info

    def close(self) -> None:
        pass

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    @staticmethod
    def _to_obs(rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)

    def _build_info(self, rgb: np.ndarray, score_delta: float) -> dict[str, Any]:
        pose = self.pose_extractor(rgb)
        target, mask = _pack_oca_from_pose(pose, frame_w=rgb.shape[1], frame_h=rgb.shape[0])
        # NOTE: do NOT include `full_rgb` here. AsyncVectorEnv pickles the info
        # dict and ships it through a pipe to the main process on every step;
        # the 7.5 MB RGB frame costs ~500 ms per step of IPC overhead and the
        # main process doesn't even use it (only the worker uses it for pose
        # + reward extraction, both of which happen here before this return).
        return {
            "oca_target": target,
            "oca_mask": mask,
            "raw_score": self._raw_score,
            "score_delta": score_delta,
        }


def _pack_oca_from_pose(
    pose: dict, frame_w: int, frame_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of vision.pack_oca_target, kept here to break the import cycle
    when this module is loaded in worker subprocesses before vision.py is
    importable. Layout must match vision.pack_oca_target."""
    target = np.zeros(OCA_DIM, dtype=np.float32)
    mask = np.zeros(OCA_DIM, dtype=np.float32)
    if pose.get("ball") is not None:
        bx, by = pose["ball"]
        target[0:2] = (bx / frame_w, by / frame_h)
        mask[0:2] = 1.0
    if pose.get("player") is not None:
        px, py, sp, cp = pose["player"]
        target[2:6] = (px / frame_w, py / frame_h, sp, cp)
        mask[2:6] = 1.0
    if pose.get("opp") is not None:
        ox, oy, so, co = pose["opp"]
        target[6:10] = (ox / frame_w, oy / frame_h, so, co)
        mask[6:10] = 1.0
    return target, mask


# -----------------------------------------------------------------
# Fakes for offline development (lets Person C train without an emulator)
# -----------------------------------------------------------------
class FakeBackend(EmulatorBackend):
    """Returns synthetic frames so the env can be unit-tested without adb.

    Produces a 256x256 frame with a moving orange disk (the "ball"), a blue
    block (player), and a red block (opponent) following sinusoidal trajectories.
    The pose extractor + reward extractor will see real pixels and behave
    realistically. Use this for CI and for smoke-testing the PPO loop.
    """

    def __init__(self, endpoint: Optional[EmulatorEndpoint] = None, seed: int = 0) -> None:
        super().__init__(endpoint or EmulatorEndpoint("fake-0", 0, 0))
        self._t = 0
        self._rng = np.random.default_rng(seed)
        self._alive = True

    def grab_frame(self) -> np.ndarray:
        H, W = 256, 256
        frame = np.full((H, W, 3), 240, dtype=np.uint8)  # light background
        t = self._t
        # Ball: orange (255, 140, 0) following a parabola
        bx = int(40 + (t * 3) % (W - 80))
        by = int(80 + 60 * abs(np.sin(t * 0.05)))
        cv2.circle(frame, (bx, by), 8, (255, 140, 0), -1)
        # Player: blue shoes, two small rects on the ground
        px = int(60 + 40 * np.sin(t * 0.03))
        cv2.rectangle(frame, (px - 6, H - 30), (px - 2, H - 20), (30, 60, 200), -1)
        cv2.rectangle(frame, (px + 2, H - 30), (px + 6, H - 20), (30, 60, 200), -1)
        # Opponent: red shoes
        ox = int(W - 60 - 40 * np.sin(t * 0.03))
        cv2.rectangle(frame, (ox - 6, H - 30), (ox - 2, H - 20), (200, 30, 30), -1)
        cv2.rectangle(frame, (ox + 2, H - 30), (ox + 6, H - 20), (200, 30, 30), -1)
        self._t += 1
        return frame

    def send_action(self, action: int) -> None:
        pass

    def load_snapshot(self, name: Optional[str] = None) -> None:
        self._t = 0

    def is_alive(self) -> bool:
        return self._alive

    def restart(self) -> None:
        self._alive = True
        self._t = 0


def fake_reward_extractor(rgb: np.ndarray) -> tuple[int, bool]:
    """Synthetic reward source for the fake backend. Always returns 0/False."""
    return 0, False
