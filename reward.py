"""Reward extraction from emulator frames.

The reward signal for Bouncy Basketball is the player's score, detected by
template-matching digit sprites in the scoreboard region of the captured frame.

The default `score_extractor()` below returns (0, False) and is meant to be
replaced once Person B has captured real frames and:
    1. cropped the scoreboard ROI coordinates from a sample frame
    2. saved per-digit templates (0..9) to digits/<n>.png at the same scale
    3. tuned `THRESH` for cv2.matchTemplate against those templates

Validation gate (§5.5 of research_idea.md): run `score_extractor` on 10
manually-scored episodes; require >= 98% agreement on per-frame score.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclasses.dataclass
class ScoreRegion:
    x: int          # top-left in device pixel space
    y: int
    w: int
    h: int
    # Up to 4 digits, evenly spaced inside the region. Stored separately so
    # multi-digit matching can iterate over slots.
    n_digits: int = 3


# Placeholder. Tune to the actual Bouncy Basketball scoreboard once we have a
# screenshot.
PLAYER_SCORE_ROI = ScoreRegion(x=1050, y=40, w=180, h=80, n_digits=3)
MATCH_END_ROI: Optional[ScoreRegion] = None    # placeholder for a "game over" banner detector

THRESH = 0.85   # cv2.matchTemplate score above which a digit is accepted


class TemplateScoreExtractor:
    """Template-match digits to read the score off the scoreboard.

    Usage:
        ext = TemplateScoreExtractor("digits/")
        score, done = ext(rgb_frame)
    """

    def __init__(self, digits_dir: str | Path):
        self.digits_dir = Path(digits_dir)
        self._digits: dict[int, np.ndarray] = {}
        self._load_digits()
        self._last_score = 0

    def _load_digits(self) -> None:
        for d in range(10):
            path = self.digits_dir / f"{d}.png"
            if not path.exists():
                continue
            tpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if tpl is None:
                raise RuntimeError(f"failed to read digit template {path}")
            self._digits[d] = tpl

    def _read_digit(self, slot: np.ndarray) -> Optional[int]:
        best_d, best_score = None, -1.0
        for d, tpl in self._digits.items():
            if tpl.shape[0] > slot.shape[0] or tpl.shape[1] > slot.shape[1]:
                continue
            r = cv2.matchTemplate(slot, tpl, cv2.TM_CCOEFF_NORMED)
            s = float(r.max())
            if s > best_score:
                best_score = s
                best_d = d
        return best_d if best_score >= THRESH else None

    def __call__(self, rgb_frame: np.ndarray) -> tuple[int, bool]:
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        roi = PLAYER_SCORE_ROI
        H, W = gray.shape
        if roi.x + roi.w > W or roi.y + roi.h > H:
            return self._last_score, False
        region = gray[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        slot_w = roi.w // roi.n_digits
        digits: list[int] = []
        for i in range(roi.n_digits):
            slot = region[:, i * slot_w : (i + 1) * slot_w]
            d = self._read_digit(slot)
            if d is None:
                continue
            digits.append(d)
        if not digits:
            return self._last_score, False
        score = 0
        for d in digits:
            score = score * 10 + d
        self._last_score = score
        # Match-end detection deferred until Person B identifies the banner ROI.
        return score, False


def zero_extractor(_rgb: np.ndarray) -> tuple[int, bool]:
    """Use this until digit templates are captured. Always returns (0, False).
    The PPO loop still trains end-to-end (rewards are just always 0)."""
    return 0, False


# All reward constants live in config.REWARD.
from config import REWARD

from config import VISION  # for vision_scale, shared with vision.py

CHI_SCORE_ROI        = REWARD.chi_score_roi
HOU_SCORE_ROI        = REWARD.hou_score_roi
SCORE_DIFF_THRESHOLD = REWARD.diff_threshold
SCORE_COOLDOWN_STEPS = REWARD.cooldown_steps
OPPONENT_WEIGHT      = REWARD.opponent_score_weight


def _scaled_roi(roi: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    """Scale a (y0, y1, x0, x1) ROI by a linear factor."""
    y0, y1, x0, x1 = roi
    return (int(y0 * scale), int(y1 * scale), int(x0 * scale), int(x1 * scale))


# GAME OVER detection: the post-match screen has "GAME OVER" / "RED WINS!" /
# "BLUE WINS!" in saturated red text near y=20-120, x roughly across the
# middle. Detect by counting saturated-red pixels in that top strip — any
# normal gameplay frame has very little red text up there (just the small
# "CHI" scoreboard label which is ~5k red pixels at most). GAME OVER triples
# that with the big "GAME OVER" + "RED WINS!" headers.
_GAME_OVER_Y_BAND   = (20, 130)
# Measured Day 1: clean gameplay frames have ~0 red pixels in this top strip
# (the 'CHI' scoreboard label is at y=270, below this band). The GAME OVER
# screen has ~3600 saturated-red pixels from the 'GAME OVER' / 'WINS!' text.
# Threshold of 1500 cleanly separates the two with margin.
_GAME_OVER_PIXEL_THRESHOLD = 1500


def is_game_over(rgb_frame: np.ndarray, scale: Optional[float] = None) -> bool:
    """Return True if the frame looks like the post-match GAME OVER screen.

    Input MUST be RGB. If `scale` is provided (or VISION.vision_scale != 1.0),
    the y-band and pixel threshold are rescaled to match.
    """
    if scale is None:
        scale = VISION.vision_scale
    y0, y1 = _GAME_OVER_Y_BAND
    threshold = _GAME_OVER_PIXEL_THRESHOLD
    if scale != 1.0:
        y0 = int(y0 * scale)
        y1 = int(y1 * scale)
        threshold = int(threshold * scale * scale)
    H, W = rgb_frame.shape[:2]
    if y1 > H:
        return False
    hsv = cv2.cvtColor(rgb_frame[y0:y1], cv2.COLOR_RGB2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 180, 100]), np.array([8, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 180, 100]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(red1, red2)
    return int(red.sum() // 255) > threshold


class _ScoreboardRoiTracker:
    """Detect score increments inside a single fixed scoreboard ROI by
    pixel-diffing between consecutive frames. One instance per team."""

    def __init__(self, roi: tuple[int, int, int, int]):
        self._roi = roi
        self._prev_patch: Optional[np.ndarray] = None
        self._cooldown = 0

    def reset(self) -> None:
        self._prev_patch = None
        self._cooldown = 0

    def step(self, rgb_frame: np.ndarray) -> bool:
        """Returns True iff this frame shows a score-event for this ROI."""
        y0, y1, x0, x1 = self._roi
        H, W = rgb_frame.shape[:2]
        if y1 > H or x1 > W:
            return False
        patch = rgb_frame[y0:y1, x0:x1].astype(np.int16)
        if self._cooldown > 0:
            self._cooldown -= 1
        event = False
        if self._prev_patch is not None and self._cooldown == 0:
            diff = float(np.abs(patch - self._prev_patch).mean())
            if diff >= SCORE_DIFF_THRESHOLD:
                event = True
                self._cooldown = SCORE_COOLDOWN_STEPS
        self._prev_patch = patch
        return event


# 2026-05-26 fix: SWITCH-SIDES is *on* in clean_boot_v8 (despite our prior
# assumption it was off). Between quarters the HOU/CHI score panels swap
# physical sides on the scoreboard. The original code assumed HOU was
# always at the right slot, which inverted the reward signal in Q1/Q3.
#
# Fix: detect which side has the BLUE panel (HOU's color) per-frame.
# HSV blue counts in left/right slots cleanly separate the two layouts
# (Q2 normal: LEFT red=2347 BLUE=0, RIGHT blue=2693 red=0;
#  Q3 swap:   LEFT blue=2693 red=0, RIGHT blue=0 red=1334).
def _hou_is_on_left(rgb_frame: np.ndarray,
                    left_roi: tuple[int, int, int, int],
                    right_roi: tuple[int, int, int, int]) -> Optional[bool]:
    """Return True if HOU (blue) panel is currently on the LEFT slot,
    False if normal (HOU on RIGHT), None if neither slot has a clear
    team color (e.g. during the END-OF-QUARTER overlay)."""
    ly0, ly1, lx0, lx1 = left_roi
    ry0, ry1, rx0, rx1 = right_roi
    H, W = rgb_frame.shape[:2]
    if ly1 > H or lx1 > W or ry1 > H or rx1 > W:
        return None
    hsv_l = cv2.cvtColor(rgb_frame[ly0:ly1, lx0:lx1], cv2.COLOR_RGB2HSV)
    hsv_r = cv2.cvtColor(rgb_frame[ry0:ry1, rx0:rx1], cv2.COLOR_RGB2HSV)
    blue_l = cv2.inRange(hsv_l, np.array([100, 80, 50]),
                         np.array([130, 255, 255])).sum() / 255
    blue_r = cv2.inRange(hsv_r, np.array([100, 80, 50]),
                         np.array([130, 255, 255])).sum() / 255
    # Need a clear majority of blue pixels (>=300) on exactly one side.
    if blue_l >= 300 and blue_r < 100:
        return True
    if blue_r >= 300 and blue_l < 100:
        return False
    return None


class ScoreboardDiffReward:
    """Reward extractor that detects scoring events by pixel-diffing the
    two scoreboard score-digit ROIs, and assigns +1 / -1 polarity by which
    side currently holds the HOU (blue) panel.

    Reward semantics (HOU is the agent's team, CHI is the CPU opponent):
      - HOU scores  → +1.0 that step
      - CHI scores  → OPPONENT_WEIGHT (default -1.0)
      - otherwise   →  0.0

    Two ROIs are tracked, one at the LEFT pixel slot and one at the RIGHT
    pixel slot. Which slot is "HOU" is detected per-frame by checking
    panel colour (HOU = blue, CHI = red). When the side ordering flips
    (e.g. at a quarter boundary) we reset both trackers so the legitimate
    pixel change due to the panel swap is NOT counted as a score event.
    """

    def __init__(
        self,
        chi_roi: tuple[int, int, int, int] = CHI_SCORE_ROI,
        hou_roi: tuple[int, int, int, int] = HOU_SCORE_ROI,
        opponent_weight: float = OPPONENT_WEIGHT,
        scale: Optional[float] = None,
    ):
        s = VISION.vision_scale if scale is None else scale
        self._scale = s
        # The two physical slot ROIs. The historical names chi_roi and
        # hou_roi are pixel positions; we now treat them as left/right
        # positions and re-resolve which team owns each slot per-frame.
        # By convention: hou_roi is the LEFT slot, chi_roi is the RIGHT
        # slot (matches config.py x0 values: 1000 < 1250).
        if hou_roi[2] < chi_roi[2]:
            left_roi, right_roi = hou_roi, chi_roi
        else:
            left_roi, right_roi = chi_roi, hou_roi
        self._left_roi_unscaled = left_roi
        self._right_roi_unscaled = right_roi
        self._left = _ScoreboardRoiTracker(_scaled_roi(left_roi, s))
        self._right = _ScoreboardRoiTracker(_scaled_roi(right_roi, s))
        self._opponent_weight = float(opponent_weight)
        self._net_score = 0.0
        self._last_hou_left: Optional[bool] = None

    def reset(self) -> None:
        self._left.reset()
        self._right.reset()
        self._net_score = 0.0
        self._last_hou_left = None

    def __call__(self, rgb_frame: np.ndarray) -> tuple[float, bool]:
        if self._scale != 1.0:
            h, w = rgb_frame.shape[:2]
            rgb_frame = cv2.resize(
                rgb_frame,
                (int(w * self._scale), int(h * self._scale)),
                interpolation=cv2.INTER_AREA,
            )
        done = is_game_over(rgb_frame, scale=1.0)
        if done:
            return self._net_score, True

        # Per-frame side detection. Compare with the previous frame's
        # ordering; if it changed (quarter swap), reset both trackers
        # so the panel-swap pixel change doesn't fire a false score.
        # Use the scaled ROIs for color detection so the patches actually
        # cover the right pixels at the runtime resolution.
        hou_left_now = _hou_is_on_left(
            rgb_frame,
            _scaled_roi(self._left_roi_unscaled, self._scale),
            _scaled_roi(self._right_roi_unscaled, self._scale),
        )
        if hou_left_now is not None:
            if (self._last_hou_left is not None
                    and hou_left_now != self._last_hou_left):
                # Side ordering flipped — silence one frame of diff.
                self._left.reset()
                self._right.reset()
            self._last_hou_left = hou_left_now

        left_event = self._left.step(rgb_frame)
        right_event = self._right.step(rgb_frame)

        # Only commit reward when we know which side is which.
        if self._last_hou_left is True:
            # Swapped layout: LEFT = HOU (+1), RIGHT = CHI (opp weight).
            if left_event:
                self._net_score += 1.0
            if right_event:
                self._net_score += self._opponent_weight
        elif self._last_hou_left is False:
            # Normal layout: LEFT = CHI (opp), RIGHT = HOU (+1).
            if left_event:
                self._net_score += self._opponent_weight
            if right_event:
                self._net_score += 1.0
        # else: side ordering unknown (transition / overlay) → skip event
        # this frame; cooldown in tracker handles bouncing.

        return self._net_score, False
