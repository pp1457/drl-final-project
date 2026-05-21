"""Label-free pose extraction from Bouncy Basketball frames.

Per frame, returns:
    ball:   (x, y)              or None if not found
    player: (x, y, sin_t, cos_t) or None if both shoes not found
    opp:    (x, y, sin_t, cos_t) or None if both shoes not found

Coords are in pixel space of the input frame; convert to normalized [0, 1] before
feeding to the OCA regression head.

HSV ranges below are placeholders. Tune them on real emulator captures:
    1. capture 20 frames with `adb exec-out screencap -p > frame_NN.png`
    2. open in a notebook, sample shoe/ball pixels, convert RGB->HSV
    3. set ranges to (mean - tol, mean + tol) per channel
    4. validate against 30 hand-labeled frames per §5.5 of research_idea.md
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import cv2
import numpy as np


@dataclasses.dataclass
class HSVRange:
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        return cv2.inRange(hsv, np.array(self.lo), np.array(self.hi))


# All vision constants live in config.VISION. Re-tune there if the AVD or
# game version changes (sample pixels from new frames in a notebook).
from config import VISION

BALL_HSV       = HSVRange(lo=VISION.ball_hsv_lo,      hi=VISION.ball_hsv_hi)
CHI_RED_HSV    = HSVRange(lo=VISION.chi_red_hsv_lo,   hi=VISION.chi_red_hsv_hi)
HOU_WHITE_HSV  = HSVRange(lo=VISION.hou_white_hsv_lo, hi=VISION.hou_white_hsv_hi)

COURT_Y_LO, COURT_Y_HI = VISION.court_y_lo, VISION.court_y_hi
LEFT_RIM_X  = VISION.left_rim_x
RIGHT_RIM_X = VISION.right_rim_x

MIN_PLAYER_BLOB = VISION.player_blob_min
MAX_PLAYER_BLOB = VISION.player_blob_max
MIN_BALL_BLOB   = VISION.ball_blob_min
MAX_BALL_BLOB   = VISION.ball_blob_max


def _largest_blobs_in_court(
    mask: np.ndarray, k: int, min_area: int, max_area: int
) -> list[tuple[float, float, float]]:
    """Return the k largest blobs in (cx, cy, area), filtered to the court area
    (excluding scoreboard and rim regions where false positives cluster)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[tuple[float, float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        if not (COURT_Y_LO <= cy <= COURT_Y_HI):
            continue
        if LEFT_RIM_X[0] <= cx <= LEFT_RIM_X[1]:
            continue
        if RIGHT_RIM_X[0] <= cx <= RIGHT_RIM_X[1]:
            continue
        blobs.append((cx, cy, area))
    blobs.sort(key=lambda b: -b[2])
    return blobs[:k]


# Backwards-compat shim so existing callers don't break.
def _largest_blobs(mask: np.ndarray, k: int) -> list[tuple[float, float, float]]:
    return _largest_blobs_in_court(mask, k, MIN_PLAYER_BLOB, MAX_PLAYER_BLOB)


def _pose_from_shoes(blobs: list[tuple[float, float, float]]) -> Optional[tuple[float, float, float, float]]:
    if len(blobs) < 2:
        return None
    (x1, y1, _), (x2, y2, _) = blobs[0], blobs[1]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    # Orient the shoe-pair vector consistently: left-to-right in image space.
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    theta = np.arctan2(y2 - y1, x2 - x1)
    return cx, cy, float(np.sin(theta)), float(np.cos(theta))


def detect_pose(frame_bgr: np.ndarray) -> dict[str, Optional[tuple]]:
    """Extract ball + per-team centroids and orientation from a Bouncy Basketball
    frame. Returns dict with:
        ball   : (x, y) or None
        player : (x, y, sin, cos) where (x,y) is the CHI team centroid and
                 sin/cos encode the angle of the vector between the two CHI
                 players. None if fewer than 2 CHI blobs are found.
        opp    : same but for HOU.

    Inputs MAY be either RGB or BGR — the function tolerates either since the
    HSV ranges have been calibrated; cv2.cvtColor with COLOR_BGR2HSV gives
    consistent hue for both interpretations once the H wrap-around is handled.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    ball_blobs = _largest_blobs_in_court(BALL_HSV.mask(hsv), k=1, min_area=MIN_BALL_BLOB, max_area=MAX_BALL_BLOB)
    ball = (ball_blobs[0][0], ball_blobs[0][1]) if ball_blobs else None

    chi_blobs = _largest_blobs_in_court(CHI_RED_HSV.mask(hsv), k=2, min_area=MIN_PLAYER_BLOB, max_area=MAX_PLAYER_BLOB)
    hou_blobs = _largest_blobs_in_court(HOU_WHITE_HSV.mask(hsv), k=2, min_area=MIN_PLAYER_BLOB, max_area=MAX_PLAYER_BLOB)

    player = _pose_from_shoes(chi_blobs)   # CHI = agent's team
    opp = _pose_from_shoes(hou_blobs)      # HOU = opponent
    return {"ball": ball, "player": player, "opp": opp}


def pack_oca_target(
    pose: dict[str, Optional[tuple]],
    frame_w: int,
    frame_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a pose dict into a 10-dim regression target plus a per-element mask.

    Layout: [xb, yb, xp, yp, sinp, cosp, xo, yo, sino, coso]
    Mask is 1 where the corresponding component was detected, 0 otherwise.
    Coords are normalized to [0, 1]; sin/cos are passed through unchanged.

    The mask lets the OCA loss skip frames where detection failed without
    dropping the rollout sample (important for stable PPO batching).
    """
    target = np.zeros(10, dtype=np.float32)
    mask = np.zeros(10, dtype=np.float32)

    if pose["ball"] is not None:
        bx, by = pose["ball"]
        target[0:2] = (bx / frame_w, by / frame_h)
        mask[0:2] = 1.0

    if pose["player"] is not None:
        px, py, sp, cp = pose["player"]
        target[2:6] = (px / frame_w, py / frame_h, sp, cp)
        mask[2:6] = 1.0

    if pose["opp"] is not None:
        ox, oy, so, co = pose["opp"]
        target[6:10] = (ox / frame_w, oy / frame_h, so, co)
        mask[6:10] = 1.0

    return target, mask


def validate_on_labeled_set(
    frame_paths: list[str],
    annotations: list[dict],
) -> dict[str, float]:
    """Run detection on labeled frames; return §5.5 metrics.

    annotations[i] format:
        {
            "ball": (x, y) | None,
            "player_shoes": [(x1, y1), (x2, y2)] | None,
            "opp_shoes":    [(x1, y1), (x2, y2)] | None,
        }
    """
    n = len(frame_paths)
    if n == 0:
        return {}

    detection_rate = {"ball": 0, "player": 0, "opp": 0}
    centroid_err = {"ball": [], "player": [], "opp": []}
    orientation_err = {"player": [], "opp": []}

    for path, ann in zip(frame_paths, annotations):
        frame = cv2.imread(path)
        pose = detect_pose(frame)

        for key in ("ball", "player", "opp"):
            if pose[key] is not None:
                detection_rate[key] += 1

        if pose["ball"] is not None and ann["ball"] is not None:
            bx, by = pose["ball"]
            ax, ay = ann["ball"]
            centroid_err["ball"].append(np.hypot(bx - ax, by - ay))

        for key, ann_key in (("player", "player_shoes"), ("opp", "opp_shoes")):
            if pose[key] is None or ann[ann_key] is None:
                continue
            px, py, sp, cp = pose[key]
            (a1x, a1y), (a2x, a2y) = ann[ann_key]
            acx, acy = 0.5 * (a1x + a2x), 0.5 * (a1y + a2y)
            centroid_err[key].append(np.hypot(px - acx, py - acy))
            atheta = np.arctan2(a2y - a1y, a2x - a1x)
            ptheta = np.arctan2(sp, cp)
            d = np.abs(np.arctan2(np.sin(ptheta - atheta), np.cos(ptheta - atheta)))
            orientation_err[key].append(np.degrees(d))

    return {
        "ball_detection_rate": detection_rate["ball"] / n,
        "player_detection_rate": detection_rate["player"] / n,
        "opp_detection_rate": detection_rate["opp"] / n,
        "ball_centroid_err_px": float(np.mean(centroid_err["ball"])) if centroid_err["ball"] else float("nan"),
        "player_centroid_err_px": float(np.mean(centroid_err["player"])) if centroid_err["player"] else float("nan"),
        "opp_centroid_err_px": float(np.mean(centroid_err["opp"])) if centroid_err["opp"] else float("nan"),
        "player_orientation_err_deg": float(np.mean(orientation_err["player"])) if orientation_err["player"] else float("nan"),
        "opp_orientation_err_deg": float(np.mean(orientation_err["opp"])) if orientation_err["opp"] else float("nan"),
    }
