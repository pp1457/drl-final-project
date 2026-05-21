"""Lightweight Python clone of Bouncy Basketball's physics regime.

Used ONLY for §5.3 robustness-to-perturbation evaluation, NOT for training.
The clone matches BouncyBasketballEnv's interface (Discrete(6) actions, 84x84
grayscale observation, same info payload schema) so a policy trained on the
Android env can be transferred zero-shot.

Physics constants (gravity, restitution, ball mass) are exposed as constructor
kwargs so §5.3 can sweep:
    g     in {0.75*g0, g0, 1.25*g0}
    e     in {0.80*e0, e0, 1.20*e0}
    m     in {0.70*m0, m0, 1.30*m0}

Game design (intentionally simplified):
    - 1-player shootaround (no opponent). Avoids modelling adversarial AI.
    - Side view, 480x270 world rendered to RGB then downsampled to 84x84 grayscale.
    - Player moves L/R, can jump, can charge+release a shot.
    - Ball is affected by gravity, elastic with ground/walls, scoring when it
      passes through the hoop (a small horizontal segment) with downward velocity.
    - Reward: +1 per goal, episode lasts max_steps frames.

The clone is NOT meant to be a perfect Bouncy Basketball replica — it is a
controlled physics testbed where we can vary the constants the closed APK won't
let us touch.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import cv2
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym  # type: ignore
    from gym import spaces  # type: ignore

import pygame
import pymunk

from env import (
    CHARGE,
    JUMP,
    LEFT,
    NOOP,
    N_ACTIONS,
    OBS_H,
    OBS_W,
    OCA_DIM,
    RELEASE,
    RIGHT,
)


# Render resolution (rendered RGB before downsample).
WORLD_W, WORLD_H = 480, 270


@dataclasses.dataclass
class PhysicsParams:
    gravity: float = 900.0           # px / s^2 downward
    restitution: float = 0.75        # ball bounciness
    ball_mass: float = 1.0
    ball_radius: float = 10.0
    player_speed: float = 200.0      # px / s when moving
    jump_velocity: float = 420.0     # px / s upward impulse
    charge_rate: float = 18.0        # added to shot impulse per step while charging
    max_charge: float = 900.0
    shot_angle_deg: float = 60.0     # upward angle from horizontal at release


class BouncyBasketballCloneEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        physics: Optional[PhysicsParams] = None,
        max_episode_steps: int = 1024,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.physics = physics or PhysicsParams()
        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_H, OBS_W), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._rng = np.random.default_rng(seed)

        pygame.init()
        # Off-screen surface; no display required (server use)
        self._surface = pygame.Surface((WORLD_W, WORLD_H))
        self._dt = 1.0 / 30.0
        self._step_count = 0
        self._score = 0

        # Hoop: horizontal segment 30 px wide, at (right side, ~upper third)
        self._hoop_x = WORLD_W - 80
        self._hoop_y = 90
        self._hoop_w = 40

        # Charge state
        self._charge = 0.0
        self._charging = False

        self._setup_world()

    # -----------------------------------------------------------------
    # World setup
    # -----------------------------------------------------------------
    def _setup_world(self) -> None:
        self._space = pymunk.Space()
        self._space.gravity = (0.0, self.physics.gravity)
        # Walls
        thickness = 4.0
        bounds = [
            ((0, 0), (WORLD_W, 0)),                       # top
            ((0, 0), (0, WORLD_H)),                       # left
            ((WORLD_W, 0), (WORLD_W, WORLD_H)),           # right
            ((0, WORLD_H), (WORLD_W, WORLD_H)),           # floor
        ]
        for a, b in bounds:
            seg = pymunk.Segment(self._space.static_body, a, b, thickness / 2)
            seg.elasticity = self.physics.restitution
            seg.friction = 0.5
            self._space.add(seg)

        # Player (a kinematic body we move ourselves)
        self._player_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._player_body.position = (80, WORLD_H - 30)
        self._player_w, self._player_h = 24, 36
        shape = pymunk.Poly.create_box(self._player_body, (self._player_w, self._player_h))
        shape.elasticity = 0.0
        shape.friction = 0.8
        self._space.add(self._player_body, shape)
        self._player_vy = 0.0       # we integrate vertical motion manually
        self._on_ground = True

        # Ball
        moment = pymunk.moment_for_circle(self.physics.ball_mass, 0, self.physics.ball_radius)
        self._ball_body = pymunk.Body(self.physics.ball_mass, moment)
        self._ball_body.position = (self._player_body.position[0], self._player_body.position[1] - 30)
        ball_shape = pymunk.Circle(self._ball_body, self.physics.ball_radius)
        ball_shape.elasticity = self.physics.restitution
        ball_shape.friction = 0.5
        self._space.add(self._ball_body, ball_shape)
        self._ball_held = True

    # -----------------------------------------------------------------
    # Gym API
    # -----------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._score = 0
        self._charge = 0.0
        self._charging = False
        self._setup_world()
        rgb = self._render_rgb()
        obs = self._to_obs(rgb)
        return obs, self._build_info(rgb)

    def step(self, action: int):
        self._apply_action(int(action))
        self._space.step(self._dt)
        self._update_player_vertical()

        # Hoop scoring: ball center within hoop segment & moving downward
        bx, by = self._ball_body.position
        vy = self._ball_body.velocity.y
        scored = (
            self._hoop_x - self._hoop_w / 2 < bx < self._hoop_x + self._hoop_w / 2
            and abs(by - self._hoop_y) < 6
            and vy > 30
        )
        reward = 1.0 if scored else 0.0
        if scored:
            self._score += 1
            self._respawn_ball()

        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        terminated = False
        rgb = self._render_rgb()
        obs = self._to_obs(rgb)
        return obs, reward, terminated, truncated, self._build_info(rgb)

    def close(self) -> None:
        pygame.quit()

    # -----------------------------------------------------------------
    # Action / physics helpers
    # -----------------------------------------------------------------
    def _apply_action(self, action: int) -> None:
        px, py = self._player_body.position
        if action == LEFT:
            px = max(20, px - self.physics.player_speed * self._dt)
        elif action == RIGHT:
            px = min(WORLD_W - 20, px + self.physics.player_speed * self._dt)
        elif action == JUMP and self._on_ground:
            self._player_vy = -self.physics.jump_velocity
            self._on_ground = False
        elif action == CHARGE:
            self._charging = True
            self._charge = min(self.physics.max_charge, self._charge + self.physics.charge_rate)
        elif action == RELEASE:
            if self._ball_held:
                self._release_ball()
            self._charging = False
            self._charge = 0.0
        self._player_body.position = (px, py)
        if self._ball_held:
            self._ball_body.position = (px, py - 30)
            self._ball_body.velocity = (0, 0)

    def _update_player_vertical(self) -> None:
        if self._on_ground:
            return
        self._player_vy += self.physics.gravity * self._dt
        px, py = self._player_body.position
        py += self._player_vy * self._dt
        ground_y = WORLD_H - 30
        if py >= ground_y:
            py = ground_y
            self._player_vy = 0.0
            self._on_ground = True
        self._player_body.position = (px, py)

    def _release_ball(self) -> None:
        angle = np.radians(self.physics.shot_angle_deg)
        # Shoot toward the hoop (right side)
        vx = self._charge * np.cos(angle)
        vy = -self._charge * np.sin(angle)
        self._ball_body.velocity = (vx, vy)
        self._ball_held = False

    def _respawn_ball(self) -> None:
        self._ball_body.position = (
            self._player_body.position[0],
            self._player_body.position[1] - 30,
        )
        self._ball_body.velocity = (0, 0)
        self._ball_held = True

    # -----------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------
    def _render_rgb(self) -> np.ndarray:
        s = self._surface
        s.fill((240, 240, 240))
        # Hoop
        pygame.draw.rect(
            s, (220, 80, 0),
            pygame.Rect(int(self._hoop_x - self._hoop_w / 2), int(self._hoop_y) - 2, self._hoop_w, 4),
        )
        # Player body
        px, py = self._player_body.position
        pygame.draw.rect(
            s, (30, 30, 200),
            pygame.Rect(int(px - self._player_w / 2), int(py - self._player_h / 2), self._player_w, self._player_h),
        )
        # Player shoes (so OCA-derived orientation is meaningful)
        shoe_w, shoe_h = 5, 4
        shoe_y = int(py + self._player_h / 2 - shoe_h)
        pygame.draw.rect(s, (20, 60, 220),
                         pygame.Rect(int(px - self._player_w / 2 + 1), shoe_y, shoe_w, shoe_h))
        pygame.draw.rect(s, (20, 60, 220),
                         pygame.Rect(int(px + self._player_w / 2 - 1 - shoe_w), shoe_y, shoe_w, shoe_h))
        # Ball
        bx, by = self._ball_body.position
        pygame.draw.circle(s, (255, 140, 0), (int(bx), int(by)), int(self.physics.ball_radius))
        # surfarray returns (W, H, 3); transpose to (H, W, 3) for cv2
        arr = pygame.surfarray.array3d(s)
        return np.transpose(arr, (1, 0, 2))

    @staticmethod
    def _to_obs(rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)

    def _build_info(self, rgb: np.ndarray) -> dict[str, Any]:
        # Engine-true coordinates: pack into the same 10-dim OCA target schema.
        H, W = rgb.shape[:2]
        bx, by = self._ball_body.position
        px, py = self._player_body.position
        # Player orientation in this clone is always upright (theta=0) so
        # (sin, cos) = (0, 1). Opp slot is unused -> zero-masked.
        target = np.zeros(OCA_DIM, dtype=np.float32)
        mask = np.zeros(OCA_DIM, dtype=np.float32)
        target[0:2] = (bx / W, by / H)
        mask[0:2] = 1.0
        target[2:6] = (px / W, py / H, 0.0, 1.0)
        mask[2:6] = 1.0
        # opp slot left zeroed and masked off
        return {
            "full_rgb": rgb,
            "oca_target": target,
            "oca_mask": mask,
            "raw_score": self._score,
            "score_delta": 0.0,
        }


def make_perturbation_envs() -> dict[str, BouncyBasketballCloneEnv]:
    """Build the 9 perturbation conditions for §5.3 evaluation."""
    base = PhysicsParams()
    out: dict[str, BouncyBasketballCloneEnv] = {}
    for label, mult in [("g_lo", 0.75), ("g_mid", 1.0), ("g_hi", 1.25)]:
        p = dataclasses.replace(base, gravity=base.gravity * mult)
        out[f"gravity_{label}"] = BouncyBasketballCloneEnv(physics=p)
    for label, mult in [("e_lo", 0.80), ("e_hi", 1.20)]:
        p = dataclasses.replace(base, restitution=base.restitution * mult)
        out[f"restitution_{label}"] = BouncyBasketballCloneEnv(physics=p)
    for label, mult in [("m_lo", 0.70), ("m_hi", 1.30)]:
        p = dataclasses.replace(base, ball_mass=base.ball_mass * mult)
        out[f"mass_{label}"] = BouncyBasketballCloneEnv(physics=p)
    return out
