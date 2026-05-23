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

# OCA target dimensionality (18): [ball_xy] + 4 players × [x, y, sin θ, cos θ]
OCA_DIM = 18


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
        # Vision watchdog: count consecutive step()s that returned an all-zero
        # detection mask. If this exceeds _BLANK_STREAK_LIMIT we raise so the
        # trainer's _recover_envs path hard-restarts the emulator. Without this
        # watchdog, a degenerate game state (menu, blank intermission) trains
        # PPO on garbage frames indefinitely while reset() keeps thinking it
        # succeeded — observed on ws5_oca_s1 at upd 25-26 on the 2026-05-22 run.
        self._blank_streak = 0
        # Game-over overlay watchdog: the reward extractor's `done` flag fires
        # whenever the top of the screen has saturated red text. That catches
        # the real GAME OVER screen — but ALSO catches the very-frequent "OUT
        # OF BOUNDS" mid-game overlay AND the per-quarter "END OF X QUARTER"
        # transition screens (every ~65 steps = once per ~2 s of game time).
        # All three look identical to red-pixel counting. Strategy: when the
        # streak hits the limit, try the reset()-style advance taps once; if
        # the overlay clears we keep playing (it was a quarter end); if it
        # persists past a second LIMIT we terminate for real (GAME OVER).
        self._overlay_streak = 0
        self._advance_attempted = False

    # ≈ 6 s of game time at frame_skip=1 (30fps). Long enough that a brief
    # transition (e.g. SHOT clock animation) doesn't trip it, short enough
    # that we lose ~1/6 of a 256-step rollout at most before recovery kicks in.
    _BLANK_STREAK_LIMIT = 180

    # ≈ 1 s of sustained red overlay required before we declare the episode
    # actually over. OUT OF BOUNDS animations clear in ~0.5 s; GAME OVER stays
    # forever. At frame_skip=1 this is 30 frames.
    _OVERLAY_STREAK_LIMIT = 30

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
        self._blank_streak = 0
        self._overlay_streak = 0
        self._advance_attempted = False
        # Advance from whatever screen the snapshot landed on into active play.
        # Strategy: tap the three known "go" buttons (PLAY, NEXT QUARTER,
        # REMATCH), then POLL until we see a gameplay frame (=at least one
        # object detected and not on a GAME OVER screen). Hardcoded sleeps
        # weren't enough — game has loading screens / animations that don't
        # match a known button or detection but eventually advance.
        if hasattr(self.backend, "send_tap"):
            from reward import is_game_over as _is_over
            # If the game has dropped out of the foreground (the Pixel launcher
            # is showing because the OS killed/backgrounded the activity), no
            # amount of tapping landscape-coord buttons will help — the home
            # screen is portrait and those coords land on Google search or
            # wallpaper. Relaunch the package first.
            GAME_PKG = "com.DreamonStudios.BouncyBasketball"
            if hasattr(self.backend, "foreground_package"):
                fg = self.backend.foreground_package()
                if fg and fg != GAME_PKG:
                    if hasattr(self.backend, "launch_app"):
                        self.backend.launch_app(GAME_PKG)
                        time.sleep(3.0)  # splash + intro animation
            recovered = False
            for retry in range(6):
                self.backend.send_tap(1493, 918)   # PLAY        (team-select)
                time.sleep(0.5)
                self.backend.send_tap(1170, 793)   # NEXT QUARTER (stats screen)
                time.sleep(0.5)
                self.backend.send_tap(1366, 793)   # REMATCH     (GAME OVER)
                time.sleep(2.0)                    # wait for transition / loading
                # Multi-frame validation: single-frame check was too lenient —
                # it would accept a transient detection during a menu animation
                # and PPO would then train on hours of blank frames. Require
                # ≥3/5 frames over ~1s to actually contain a detection and
                # mostly avoid the GAME OVER overlay. We tolerate up to 1
                # "is_over" frame per poll window because OOB animations can
                # be tripped by the same red-pixel check in the middle of
                # otherwise-fine gameplay.
                detected_count = 0
                overlay_count = 0
                for poll in range(5):
                    if poll > 0:
                        time.sleep(0.2)
                    check_rgb = self.backend.grab_frame()
                    if _is_over(check_rgb):
                        overlay_count += 1
                        continue
                    check_pose = self.pose_extractor(check_rgb)
                    if any(check_pose.get(k) is not None for k in ("ball", "player", "opp")):
                        detected_count += 1
                if detected_count >= 3 and overlay_count <= 1:
                    recovered = True
                    break
                # If retries keep failing, the app may have dropped to the
                # launcher mid-game (saw this across all 9 ws in the May 22
                # run). Try relaunching on every second retry.
                if retry == 2 and hasattr(self.backend, "launch_app"):
                    self.backend.launch_app(GAME_PKG)
                    time.sleep(3.0)
            if not recovered:
                # Raise instead of silently returning a stuck frame. Previously
                # we silently returned the bad obs, and PPO would happily train
                # for hours against fh=0 / v=0 garbage (saw this kill 7/9 ws
                # around upd 40-50). The trainer's rollout loop catches this
                # and rebuilds the env (which can include relaunching the
                # emulator via supervise_once).
                raise RuntimeError(
                    f"env.reset(): emulator {getattr(self.backend, 'endpoint', '?')} "
                    f"stuck after 6 PLAY/NEXT_QUARTER/REMATCH retries — needs hard restart"
                )
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
        # Sustained-overlay gate. The reward extractor's `done` fires on any
        # red text in the top band — that catches GAME OVER, OUT OF BOUNDS,
        # and the "END OF Nth QUARTER" transitions. Quarter ends happen every
        # ~65 steps (~2 s of game time) and would terminate every episode if
        # untreated. Strategy:
        #   - <LIMIT consecutive `done` frames: gate it (probably OOB).
        #   - exactly LIMIT, first time: try tapping the same buttons reset()
        #     would (PLAY/NEXT_QUARTER/REMATCH) to advance the screen, and
        #     reset the streak. If it was a quarter end, the overlay clears
        #     and the episode continues into the next quarter.
        #   - LIMIT again (advance already attempted): propagate True — this
        #     is a real GAME OVER, not a quarter transition.
        if episode_done:
            self._overlay_streak += 1
            if self._overlay_streak >= self._OVERLAY_STREAK_LIMIT:
                if not self._advance_attempted:
                    if hasattr(self.backend, "send_tap"):
                        self.backend.send_tap(1493, 918)   # PLAY (team select)
                        time.sleep(0.3)
                        self.backend.send_tap(1170, 793)   # NEXT QUARTER
                        time.sleep(0.3)
                        self.backend.send_tap(1366, 793)   # REMATCH
                        time.sleep(1.0)
                    self._advance_attempted = True
                    self._overlay_streak = 0
                    episode_done = False
                # else: already tried; let episode_done propagate as real GAME OVER
            else:
                episode_done = False
        else:
            self._overlay_streak = 0
            self._advance_attempted = False
        # Per-step reward = delta of the cumulative net score. With the new
        # ScoreboardDiffReward this can be negative (opponent scored). The
        # old `max(0, ...)` truncation is gone — that was hiding the negative
        # signal we now want PPO to learn from.
        reward = float(score - self._raw_score)
        self._raw_score = score
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        info = self._build_info(rgb, score_delta=reward)
        # Vision watchdog. If `_pack_oca_from_pose` produced an all-zero mask,
        # vision found nothing this frame. A sustained streak means the game
        # is stuck on a non-gameplay screen (menu, loading, blank) and our
        # rollout is garbage — raise so _recover_envs hard-restarts the emu.
        if int(info["oca_mask"].sum()) == 0:
            self._blank_streak += 1
            if self._blank_streak >= self._BLANK_STREAK_LIMIT:
                raise RuntimeError(
                    f"env.step(): {self._BLANK_STREAK_LIMIT} consecutive blank "
                    f"frames from {getattr(self.backend, 'endpoint', '?')} — "
                    f"vision pipeline returned no detections; needs hard restart"
                )
        else:
            self._blank_streak = 0
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
    """Mirror of vision.pack_oca_target — kept here to break the import cycle
    when env is loaded by worker subprocesses before vision is importable.
    Layout must match vision.pack_oca_target (18-dim per-player schema)."""
    target = np.zeros(OCA_DIM, dtype=np.float32)
    mask   = np.zeros(OCA_DIM, dtype=np.float32)
    if pose.get("ball") is not None:
        bx, by = pose["ball"]
        target[0:2] = (bx / frame_w, by / frame_h)
        mask[0:2] = 1.0
    for i, p in enumerate((pose.get("chi") or [])[:2]):
        base = 2 + i * 4
        x, y, sin_t, cos_t = p
        target[base + 0] = x / frame_w
        target[base + 1] = y / frame_h
        mask[base + 0] = 1.0
        mask[base + 1] = 1.0
        if sin_t is not None and cos_t is not None:
            target[base + 2] = sin_t
            target[base + 3] = cos_t
            mask[base + 2] = 1.0
            mask[base + 3] = 1.0
    for i, p in enumerate((pose.get("hou") or [])[:2]):
        base = 10 + i * 4
        x, y, sin_t, cos_t = p
        target[base + 0] = x / frame_w
        target[base + 1] = y / frame_h
        mask[base + 0] = 1.0
        mask[base + 1] = 1.0
        if sin_t is not None and cos_t is not None:
            target[base + 2] = sin_t
            target[base + 3] = cos_t
            mask[base + 2] = 1.0
            mask[base + 3] = 1.0
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
