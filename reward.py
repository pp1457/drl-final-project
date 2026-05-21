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

CHI_SCORE_ROI        = REWARD.chi_score_roi
SCORE_DIFF_THRESHOLD = REWARD.diff_threshold
SCORE_COOLDOWN_STEPS = REWARD.cooldown_steps


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


def is_game_over(rgb_frame: np.ndarray) -> bool:
    """Return True if the frame looks like the post-match GAME OVER screen.

    Input MUST be RGB (what EmulatorBackend.grab_frame() returns). Tested
    against Day-1 stuck frames (GAME OVER → ~3600 red pixels) and clean
    gameplay frames (~0 red pixels); separates cleanly via threshold 1500.
    """
    y0, y1 = _GAME_OVER_Y_BAND
    H, W = rgb_frame.shape[:2]
    if y1 > H:
        return False
    hsv = cv2.cvtColor(rgb_frame[y0:y1], cv2.COLOR_RGB2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 180, 100]), np.array([8, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 180, 100]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(red1, red2)
    return int(red.sum() // 255) > _GAME_OVER_PIXEL_THRESHOLD


class ScoreboardDiffReward:
    """Reward extractor that detects CHI score increments by pixel-diffing the
    CHI score ROI between consecutive frames.

    Pros over digit OCR:
      - no digit templates needed
      - works for arbitrary score values, including 2-digit numbers
      - tolerant of small visual noise (jpeg-like artifacts)

    Cons:
      - false-positive risk: any visual change in the CHI box (label
        re-renders, color flickers) triggers a reward; tuned threshold and
        cooldown to minimize this
      - cannot tell +1 vs +2 vs +3 point shots apart — every score event is
        worth +1 here. Acceptable for our research because PPO learns from
        the relative density of scoring events, not their exact point values.
    """

    def __init__(self, roi: tuple[int, int, int, int] = CHI_SCORE_ROI):
        self._roi = roi
        self._prev_patch: Optional[np.ndarray] = None
        self._score = 0
        self._cooldown = 0

    def reset(self) -> None:
        self._prev_patch = None
        self._score = 0
        self._cooldown = 0

    def __call__(self, rgb_frame: np.ndarray) -> tuple[int, bool]:
        y0, y1, x0, x1 = self._roi
        H, W = rgb_frame.shape[:2]
        if y1 > H or x1 > W:
            return self._score, False
        patch = rgb_frame[y0:y1, x0:x1].astype(np.int16)
        if self._cooldown > 0:
            self._cooldown -= 1
        if self._prev_patch is not None and self._cooldown == 0:
            diff = float(np.abs(patch - self._prev_patch).mean())
            if diff >= SCORE_DIFF_THRESHOLD:
                self._score += 1
                self._cooldown = SCORE_COOLDOWN_STEPS
        self._prev_patch = patch
        # Detect GAME OVER -> signal episode done so env.reset() can fire
        # the REMATCH tap and we don't keep stepping on a frozen frame.
        done = is_game_over(rgb_frame)
        return self._score, done
