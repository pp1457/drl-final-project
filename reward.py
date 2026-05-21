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


# Pinpointed 2026-05-21 on real q2_baseline.png: the CHI score is displayed in
# the red-bordered box at these coordinates within the 2340x1080 frame.
# The box contains both the "CHI" label (top half) and the score digits
# (bottom half); pixel diff against the previous frame's ROI fires whenever
# either changes — primarily when the score increments.
CHI_SCORE_ROI = (260, 360, 970, 1120)   # y0, y1, x0, x1

# Tuned by inspection: a quarter-clock tick changes ~10-15 mean-abs-diff;
# a score change is ~30-50. Threshold above the clock tick floor.
SCORE_DIFF_THRESHOLD = 25.0

# After a score change is detected, we suppress further detections for N
# frames to avoid double-counting the multi-frame scoreboard animation.
SCORE_COOLDOWN_STEPS = 8


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
        # Match-end detection deferred — assume episode ends via env's
        # max_episode_steps truncation.
        return self._score, False
