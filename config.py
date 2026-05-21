"""Central configuration for the DRL final project.

ALL tunable constants live here. Other modules import from this file rather
than defining their own constants, so:
  - changing a threshold doesn't require grepping across files
  - the same constants drive both runtime behavior and the values printed in
    paper appendices (just print VISION, REWARD, ACTIONS, etc.)
  - swapping in a different opponent pool, different HSV ranges, different
    NatureCNN dims, etc. is a one-line edit

Three layers of overridability:
  1. Edit the dataclass defaults below for a permanent project-wide change.
  2. Set an environment variable (we expose the most-likely-tuned ones).
  3. Pass a CLI arg to train.py / eval.py for the per-run knobs.

Style:
  - Each config is a frozen dataclass (immutable -> safe to share across
    workers, no accidental mutation mid-run).
  - Default factory functions used for any mutable defaults (lists/dicts).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional


# =============================================================================
# Vision
# =============================================================================
@dataclasses.dataclass(frozen=True)
class VisionConfig:
    """HSV ranges and spatial filters for the CV pose pipeline.

    Tuned 2026-05-21 on real Bouncy Basketball gameplay frames captured at
    2340x1080 from the pixel5_api31 AVD. Re-tune if the AVD or game version
    changes by sampling pixels from new frames in a notebook.
    """
    # HSV ranges: (h_min, s_min, v_min) .. (h_max, s_max, v_max)
    ball_hsv_lo:      tuple[int, int, int] = (8,   200, 180)
    ball_hsv_hi:      tuple[int, int, int] = (22,  255, 255)
    chi_red_hsv_lo:   tuple[int, int, int] = (0,   180, 100)
    chi_red_hsv_hi:   tuple[int, int, int] = (8,   255, 220)
    hou_white_hsv_lo: tuple[int, int, int] = (0,   0,   230)
    hou_white_hsv_hi: tuple[int, int, int] = (180, 30,  255)

    # Spatial filter (native landscape coords on 2340x1080 frame).
    # Anything outside [court_y_lo, court_y_hi] is filtered out as scoreboard /
    # below-floor area.
    court_y_lo: int = 200
    court_y_hi: int = 830
    # The basketball rims are saturated orange/red and produce false-positive
    # blobs. Exclude these x-bands from detection.
    left_rim_x:  tuple[int, int] = (80,   280)
    right_rim_x: tuple[int, int] = (2050, 2240)

    # Blob area bounds, in pixels at native 2340x1080.
    player_blob_min: int = 400
    player_blob_max: int = 3500
    ball_blob_min:   int = 50
    ball_blob_max:   int = 800


VISION = VisionConfig()


# =============================================================================
# Reward
# =============================================================================
@dataclasses.dataclass(frozen=True)
class RewardConfig:
    """Pixel-diff reward parameters on the CHI scoreboard region."""
    # ROI: (y0, y1, x0, x1) in the 2340x1080 frame. CHI score lives in a
    # red-bordered box containing both the "CHI" label and the score digits.
    chi_score_roi: tuple[int, int, int, int] = (260, 360, 970, 1120)

    # Tuned by inspection: clock-tick changes ~10-15 mean-abs-diff; a score
    # change is ~30-50. Threshold above the clock-tick floor.
    diff_threshold: float = 25.0

    # After a score change is detected, suppress further detections for N
    # steps to avoid double-counting the multi-frame scoreboard animation.
    cooldown_steps: int = 8


REWARD = RewardConfig()


# =============================================================================
# Actions / touch injection
# =============================================================================
@dataclasses.dataclass(frozen=True)
class ActionsConfig:
    """Touch coordinates and timings for AdbBackend.

    Bouncy Basketball is one-button: tap-anywhere works, but we send taps to
    the same on-screen spot for consistency (where the in-game tutorial
    hand-cursor sits during the first match).
    """
    press_coord: tuple[int, int] = (1900, 860)

    # One game frame ≈ 33ms at 30fps. frame_skip × frame_ms = touch hold
    # duration per agent step.
    game_frame_ms: int = 33
    frame_skip:    int = 4

    @property
    def press_hold_ms(self) -> int:
        return self.game_frame_ms * self.frame_skip


ACTIONS = ActionsConfig()


# =============================================================================
# Emulator orchestration
# =============================================================================
@dataclasses.dataclass(frozen=True)
class EmulatorConfig:
    """AVD, snapshot pool, ports, launch flags."""
    avd_name: str = "pixel5_api31"

    # Snapshot pool — each reset picks one uniformly. Add more by capturing
    # additional CHI vs <team> matchups (see Task #14).
    snapshot_pool: tuple[str, ...] = ("clean_boot", "clean_boot_lac")

    # Base console port for our emulators. Console port = base + 2*rank, adb
    # port = console + 1. Set EMU_BASE_PORT in the env to override (defaults
    # avoid colliding with other students on shared ws10).
    base_port: int = int(os.environ.get("EMU_BASE_PORT", 6554))

    # Stagger between emulator launches (seconds). ws10's per-user watchdog
    # kills the session if too many threads spawn in a tight window, so we
    # err on the slow side.
    launch_stagger_s: float = float(os.environ.get("EMU_LAUNCH_STAGGER_S", 5.0))

    # Per-emulator launch flags.
    common_flags: tuple[str, ...] = (
        "-no-window",
        "-no-audio",
        "-no-boot-anim",
        "-gpu", "swiftshader_indirect",
        "-no-metrics",
        "-no-snapshot-save",
    )

    @property
    def default_boot_snapshot(self) -> str:
        return self.snapshot_pool[0]


EMU = EmulatorConfig()


# =============================================================================
# Model architecture (shared NatureCNN + 3 heads)
# =============================================================================
@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Architecture knobs for the shared encoder and the three heads.

    Swap any of these to test different model sizes without touching the
    training loop. The Agent class in train.py reads from MODEL.
    """
    # Shared encoder
    frame_stack:      int   = 4
    latent_dim:       int   = 512
    cnn_channels:     tuple[int, int, int] = (32, 64, 64)
    cnn_kernels:      tuple[int, int, int] = (8, 4, 3)
    cnn_strides:      tuple[int, int, int] = (4, 2, 1)
    # Resulting feature-map H×W after the 3 conv layers (depends on input
    # 84x84). NatureCNN: 84 -> 20 -> 9 -> 7. The 64*7*7 below comes from this.
    cnn_flat_dim:     int   = 64 * 7 * 7

    # OCA head: MLP regressing 10-dim pose target
    oca_hidden_dim:   int   = 128
    oca_output_dim:   int   = 10
    # How many agent steps into the future OCA predicts the pose. With
    # frame_skip=4 and ~33ms/game frame, K=4 -> predict ~528ms ahead.
    # Earlier smoke tests showed K=1 lets the encoder solve the task trivially
    # (aux loss -> 0 by upd 6, killing the representation-shaping signal for
    # the remaining 91 updates). K>1 forces actual forward dynamics.
    oca_horizon_steps: int  = 4

    # DPR decoder: transposed convs to reconstruct an 84x84 grayscale frame
    dpr_decoder_in_channels: int = 64
    dpr_decoder_feature_hw:  int = 7   # latent reshape (B, 64, 7, 7) before deconv


MODEL = ModelConfig()


# =============================================================================
# PPO hyperparameters (per-run; CLI flags in train.py can still override)
# =============================================================================
@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """Defaults for PPO. train.py's argparse builds on top of these."""
    learning_rate:    float = 2.5e-4
    anneal_lr:        bool  = True
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_coef:        float = 0.1
    vf_coef:          float = 0.5
    ent_coef:         float = 0.01
    max_grad_norm:    float = 0.5
    num_steps:        int   = 128
    num_minibatches:  int   = 4
    update_epochs:    int   = 4

    # Auxiliary loss weight; swept across {0.1, 0.5, 1.0} in §3 sweep.
    aux_coef:         float = 0.5

    # Total environment steps per training run. The deadline forces us to a
    # modest count; we argue in the paper that the comparison is what matters.
    total_timesteps:  int   = 1_000_000

    # Episode length cap (per agent step). 1024 instead of 4096 because at
    # ~1 SPS/env each episode lasts ~17 minutes wall clock; with 1024 we get
    # an episode (and a usable 'ret' value for logging/eval) every ~4 minutes.
    max_episode_steps: int  = 1024


PPO = PPOConfig()


# =============================================================================
# Paths (derived from env vars, so they follow $USER on any workstation)
# =============================================================================
@dataclasses.dataclass(frozen=True)
class PathsConfig:
    project_root:    str = os.path.dirname(os.path.abspath(__file__))
    drl_final_root:  str = f"/tmp2/{os.environ.get('USER', 'unknown')}/DRL_final"

    @property
    def android_sdk(self) -> str:
        return f"{self.drl_final_root}/android-sdk"

    @property
    def emu_logs(self) -> str:
        return f"{self.drl_final_root}/emu_logs"

    @property
    def endpoints_file(self) -> str:
        return f"{self.drl_final_root}/endpoints.json"

    @property
    def frames_dir(self) -> str:
        return f"{self.drl_final_root}/frames"

    @property
    def avd_dir(self) -> str:
        return f"{self.drl_final_root}/.android/avd"


PATHS = PathsConfig()
