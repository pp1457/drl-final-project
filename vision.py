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

# Cap cv2's internal thread pool. We run many worker processes; per-process
# parallelism in cv2 only inflates the global thread count and risks running
# out of pthreads on shared workstations like ws10.
cv2.setNumThreads(1)


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
SHOE_HSV       = HSVRange(lo=VISION.shoe_hsv_lo,      hi=VISION.shoe_hsv_hi)

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
    (excluding scoreboard and rim regions where false positives cluster).

    Uses VISION.exclusion_zones — a list of (x_min, x_max, y_min, y_max)
    rectangles. Any blob centroid inside any rectangle is dropped. The y filter
    [COURT_Y_LO, COURT_Y_HI] is still applied as a global bound to drop
    completely-off-court detections (e.g. crowd in upper bleachers).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[tuple[float, float, float]] = []
    zones = VISION.exclusion_zones
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
        in_zone = False
        for (zx0, zx1, zy0, zy1) in zones:
            if zx0 <= cx <= zx1 and zy0 <= cy <= zy1:
                in_zone = True
                break
        if in_zone:
            continue
        blobs.append((cx, cy, area))
    blobs.sort(key=lambda b: -b[2])
    return blobs[:k]


# Backwards-compat shim so existing callers don't break.
def _largest_blobs(mask: np.ndarray, k: int) -> list[tuple[float, float, float]]:
    return _largest_blobs_in_court(mask, k, MIN_PLAYER_BLOB, MAX_PLAYER_BLOB)


def _pose_from_shoes(blobs: list[tuple[float, float, float]]) -> Optional[tuple[float, float, float, float]]:
    """Legacy: team centroid from 2 blobs of the same team. Kept for backward
    compatibility with old callers — new per-player pipeline uses
    `_per_player_poses` below."""
    if len(blobs) < 2:
        return None
    (x1, y1, _), (x2, y2, _) = blobs[0], blobs[1]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    theta = np.arctan2(y2 - y1, x2 - x1)
    return cx, cy, float(np.sin(theta)), float(np.cos(theta))


def _shoe_blobs_in_feet_band(
    mask: np.ndarray
) -> list[tuple[float, float, float]]:
    """Return shoe-pair blobs whose centroid falls in the feet band
    [shoe_y_lo, shoe_y_hi]. Each player's two shoes typically merge into one
    blob (~30-50 px apart horizontally); we treat that merged blob as 'feet
    position' for one player."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[tuple[float, float, float]] = []
    y_lo, y_hi = VISION.shoe_y_lo, VISION.shoe_y_hi
    zones = VISION.exclusion_zones
    for c in contours:
        area = cv2.contourArea(c)
        if area < VISION.shoe_blob_min or area > VISION.shoe_blob_max:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        if not (y_lo <= cy <= y_hi):
            continue
        # Reuse the same exclusion zones as jerseys (mainly: rim regions).
        in_zone = False
        for (zx0, zx1, zy0, zy1) in zones:
            if zx0 <= cx <= zx1 and zy0 <= cy <= zy1:
                in_zone = True
                break
        if in_zone:
            continue
        blobs.append((cx, cy, area))
    blobs.sort(key=lambda b: -b[2])
    return blobs


def _per_player_pose(
    jersey_xy: tuple[float, float],
    shoe_blobs: list[tuple[float, float, float]],
) -> tuple[float, float, Optional[float], Optional[float]]:
    """Given a jersey centroid and a list of available shoe-pair blobs, find
    the nearest shoe blob below the jersey and compute the body-axis rotation.

    Returns (x, y, sin θ, cos θ). If no valid shoe found, returns sin/cos = None
    so the caller can mask out the rotation dims while keeping position.
    Body axis: vector from shoes → jersey. Standing upright → θ = -90° (up).
    """
    jx, jy = jersey_xy
    best = None
    best_dist = float("inf")
    for (sx, sy, sa) in shoe_blobs:
        dy = sy - jy
        if dy <= 0 or dy > VISION.jersey_to_shoe_max_dy:
            continue
        d = (sx - jx) ** 2 + (sy - jy) ** 2
        if d < best_dist:
            best_dist = d
            best = (sx, sy)
    if best is None:
        return jx, jy, None, None
    sx, sy = best
    # Body axis: shoes → jersey. dx, dy from shoes to jersey.
    dx = jx - sx
    dy_axis = jy - sy
    norm = (dx * dx + dy_axis * dy_axis) ** 0.5
    if norm < 1e-3:
        return jx, jy, None, None
    sin_t = dy_axis / norm   # for upright (jersey above shoes), jy<sy in image so dy<0 → sin negative (pointing up)
    cos_t = dx / norm
    return jx, jy, float(sin_t), float(cos_t)


def detect_pose(frame_rgb: np.ndarray) -> dict[str, Optional[tuple]]:
    """Extract ball + 4 per-player poses from a Bouncy Basketball frame.

    Returns dict with:
        ball:    (x, y) or None
        chi:     list of up to 2 (x, y, sin θ_or_None, cos θ_or_None) tuples
                 — agent's team players. sin/cos = None if body axis can't be
                 computed (no nearest shoe blob found).
        hou:     same but for opponent.
        # legacy aliases for backward compat:
        player:  team centroid of chi (xy + average-orientation), or None
        opp:     team centroid of hou, or None

    The new per-player fields (chi, hou) are the primary output. Their
    rotations (sin θ, cos θ) describe each player's body axis — the vector
    from feet to torso. Upright standing: θ ≈ -90° (sin≈-1, cos≈0).

    Input MUST be RGB (channel order [R, G, B], not BGR).
    """
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    ball_blobs = _largest_blobs_in_court(BALL_HSV.mask(hsv), k=1,
                                          min_area=MIN_BALL_BLOB, max_area=MAX_BALL_BLOB)
    ball = (ball_blobs[0][0], ball_blobs[0][1]) if ball_blobs else None

    chi_blobs = _largest_blobs_in_court(CHI_RED_HSV.mask(hsv), k=2,
                                         min_area=MIN_PLAYER_BLOB, max_area=MAX_PLAYER_BLOB)
    hou_blobs = _largest_blobs_in_court(HOU_WHITE_HSV.mask(hsv), k=2,
                                         min_area=MIN_PLAYER_BLOB, max_area=MAX_PLAYER_BLOB)
    shoe_blobs = _shoe_blobs_in_feet_band(SHOE_HSV.mask(hsv))

    # Per-player poses
    chi_players = [_per_player_pose((b[0], b[1]), shoe_blobs) for b in chi_blobs]
    hou_players = [_per_player_pose((b[0], b[1]), shoe_blobs) for b in hou_blobs]

    # Sort each team's players by x (leftmost first) so target dims are stable
    # across frames (otherwise player 1/2 swap would confuse the prediction head).
    chi_players.sort(key=lambda p: p[0])
    hou_players.sort(key=lambda p: p[0])

    # Legacy team-centroid (for any backward-compat path)
    player_legacy = _pose_from_shoes(chi_blobs)
    opp_legacy    = _pose_from_shoes(hou_blobs)

    return {
        "ball": ball,
        "chi": chi_players,    # list of (x, y, sin, cos) tuples; sin/cos may be None
        "hou": hou_players,
        "player": player_legacy,
        "opp":    opp_legacy,
    }


def pack_oca_target(
    pose: dict,
    frame_w: int,
    frame_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a pose dict into an 18-dim regression target + per-element mask.

    Layout:
        [0:2]    ball xy
        [2:6]    CHI player 1 (xy + sin θ + cos θ)
        [6:10]   CHI player 2
        [10:14]  HOU player 1
        [14:18]  HOU player 2
    Coords normalized to [0,1]; sin/cos pass-through.

    A player slot's mask bits:
      - position (xy) bits set whenever a jersey blob was detected for that slot
      - rotation (sin/cos) bits set additionally only if a shoe was paired
        (body axis computable). So partial-detection frames still contribute
        position learning even without the rotation.
    """
    target = np.zeros(18, dtype=np.float32)
    mask   = np.zeros(18, dtype=np.float32)

    if pose.get("ball") is not None:
        bx, by = pose["ball"]
        target[0:2] = (bx / frame_w, by / frame_h)
        mask[0:2] = 1.0

    # CHI players: slots [2:6] and [6:10]
    chi_players = pose.get("chi") or []
    for i, p in enumerate(chi_players[:2]):
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

    # HOU players: slots [10:14] and [14:18]
    hou_players = pose.get("hou") or []
    for i, p in enumerate(hou_players[:2]):
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
