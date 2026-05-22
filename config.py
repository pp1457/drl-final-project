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
    # Calibrated 2026-05-22 against RGB-pipeline frames from the live emulator.
    # CHI red jersey: pixel samples gave (H≈4, S≈194, V≈233). V_max was 220 (too
    # restrictive) → extended to 255.
    # HOU white jersey: pixel samples gave (H≈0, S≈1, V≈220). V_min was 230 →
    # lowered to 200 so off-white shading on the jerseys is caught.
    ball_hsv_lo:      tuple[int, int, int] = (8,   200, 180)
    ball_hsv_hi:      tuple[int, int, int] = (22,  255, 255)
    chi_red_hsv_lo:   tuple[int, int, int] = (0,   180, 100)
    chi_red_hsv_hi:   tuple[int, int, int] = (8,   255, 255)
    hou_white_hsv_lo: tuple[int, int, int] = (0,   0,   200)
    hou_white_hsv_hi: tuple[int, int, int] = (180, 40,  255)
    # Shoes: dark + low-saturation pixels in the feet band. Used to derive
    # per-player body axis (vector from feet to torso). Sampled values in the
    # feet band of a Q4 frame: RGB=(42,42,42) HSV=(0,0,42) is typical shoe.
    shoe_hsv_lo:      tuple[int, int, int] = (0,   0,   0)
    shoe_hsv_hi:      tuple[int, int, int] = (180, 100, 80)

    # Spatial filter (native landscape coords on 2340x1080 frame).
    # Two-stage filter, applied to each blob centroid:
    #   1. Must be inside the court rectangle [court_y_lo, court_y_hi]
    #   2. Must NOT be inside any of the exclusion zones (scoreboard, rims)
    # The scoreboard exclusion is a rectangle, NOT just a y-strip, so we keep
    # detecting mid-air players at low y (jumping for dunks) as long as they're
    # outside the scoreboard's x range.
    # court_y_lo=410: the upper screen is dense with UI elements (CHI/HOU team
    # rosters in the two upper corners at y~378, basketball icon at y~310,
    # 'HOME'/'ROAD' labels around y~375) that mimic player colors at small
    # blob sizes. Even mid-air players' CENTROIDS stay around y~500 (feet at
    # 620, head at 380 during peak jump, midpoint ~500), so y_lo=410 is safe.
    court_y_lo: int = 410
    court_y_hi: int = 830
    # Rectangular exclusions: any blob centroid inside any of these is dropped.
    # (x_min, x_max, y_min, y_max)
    exclusion_zones: tuple[tuple[int, int, int, int], ...] = (
        (900,  1450, 0,   400),    # central scoreboard (CHI/HOU labels + digits)
        (80,   280,  0,   400),    # left basketball rim+net
        (2050, 2240, 0,   400),    # right basketball rim+net
    )
    # Kept for backward compatibility with existing code references.
    left_rim_x:       tuple[int, int] = (80,   280)
    right_rim_x:      tuple[int, int] = (2050, 2240)
    scoreboard_x:     tuple[int, int] = (900, 1450)
    scoreboard_y_max: int = 400

    # Blob area bounds, in pixels at native 2340x1080.
    # Calibrated 2026-05-22: a single jersey blob measured ~5000 px in a Q4
    # gameplay frame. Court floor's red/orange wood is ~60000 px when caught.
    # Tightened so the floor blob (way too large) is excluded but real jerseys
    # (up to ~8000) are caught.
    player_blob_min: int = 400
    player_blob_max: int = 8000
    ball_blob_min:   int = 50
    ball_blob_max:   int = 1200
    # Shoe-pair blob (two shoes merged): smaller than jersey blob.
    shoe_blob_min:   int = 80
    shoe_blob_max:   int = 2000
    # Feet band y range — shoes only valid in this strip.
    shoe_y_lo:       int = 660
    shoe_y_hi:       int = 760
    # Max vertical distance from jersey centroid to its paired shoe centroid.
    # Players' torso is ~120-180 px above their feet; allow some slack for
    # mid-air players.
    jersey_to_shoe_max_dy: int = 250

    # Downsample factor applied inside detect_pose / ScoreboardDiffReward
    # / is_game_over right before the expensive HSV + contour work. Linear
    # spatial constants scale by `vision_scale`; area constants by its
    # square. 1.0 = no downsample (matches v1). 0.5 ≈ 4× cheaper vision at
    # the cost of a smaller pixel margin for blob detection.
    vision_scale: float = 1.0


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

    # HOU score ROI — mirror-symmetric to CHI across the scoreboard center
    # (~x=960 on a 1920-wide frame). Verified visually on a Q3 gameplay frame
    # where HOU "24" sits at roughly x∈[770,870], y∈[260,360]. The 800-950
    # window has comfortable margin for 1-2 digit scores.
    hou_score_roi: tuple[int, int, int, int] = (260, 360, 800, 950)

    # Tuned by inspection: clock-tick changes ~10-15 mean-abs-diff; a score
    # change is ~30-50. Threshold above the clock-tick floor.
    diff_threshold: float = 25.0

    # After a score change is detected, suppress further detections for N
    # steps to avoid double-counting the multi-frame scoreboard animation.
    cooldown_steps: int = 8

    # Multiplier for the HOU (opponent) score event. Default -1.0 makes a
    # opponent basket cancel out one CHI basket; set to 0.0 for the
    # "score-only" baseline reward used in the May 22 matrix.
    opponent_score_weight: float = -1.0


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

    # OCA head: MLP regressing 18-dim pose target
    # Layout: [ball_xy] + [chi1_xy_sin_cos] + [chi2_xy_sin_cos] +
    #         [hou1_xy_sin_cos] + [hou2_xy_sin_cos]
    # = 2 + 4 + 4 + 4 + 4 = 18 dims.
    # rotation is body axis (feet → torso), encoded sin/cos to avoid θ
    # wrap-around discontinuity.
    oca_hidden_dim:   int   = 256       # bumped to handle the larger output
    oca_output_dim:   int   = 18
    # How many agent steps into the future OCA predicts the pose. With
    # frame_skip=4 and ~33ms/game frame, K=N -> predict ~132N ms ahead.
    # Empirical findings:
    #   K=1: encoder solves trivially in 6 updates (aux→0), no signal for the rest
    #   K=4: still solved fast (aux→0.0002 by upd 6), only marginally harder.
    #        The 4-frame stack input means the encoder can extrapolate linear
    #        motion 4 steps ahead with no genuine dynamics learning.
    #   K=8: ~1.06s ahead — bouncing trajectories curve significantly, opponent
    #        AI has time to react and change direction. This is the regime where
    #        next-pose prediction stops being trivial.
    oca_horizon_steps: int  = 8

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
    # ent_coef 0.01 → 0.05: previous training showed PPO committing to a
    # worse-than-random policy by upd ~30, then degrading further. The lowest-
    # entropy runs had the lowest returns. Increased entropy bonus prevents
    # premature commitment in this sparse-reward setting. Standard Atari uses
    # 0.01; this game's reward density is lower, so we need more exploration.
    ent_coef:         float = 0.05
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
