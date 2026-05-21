"""§5.3 robustness-to-perturbation evaluation.

Loads a trained checkpoint, evaluates it on the Python clone under 9
perturbation conditions, and reports score retention vs. the nominal-physics
clone.

Usage:
    python eval_robustness.py --checkpoint checkpoints/<run>/step_1000000.pt \\
        --episodes 50 --output robustness_<run>.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Optional

import numpy as np
import torch
import gymnasium as gym

from clone_env import BouncyBasketballCloneEnv, PhysicsParams
from env import OBS_H, OBS_W
from eval import load_agent, rollout_episodes


def make_clone_wrapped(physics: PhysicsParams, frame_stack: int) -> gym.Env:
    env = BouncyBasketballCloneEnv(physics=physics)
    env = gym.wrappers.TransformObservation(
        env,
        lambda o: o[..., None],
        observation_space=gym.spaces.Box(
            low=0, high=255, shape=(OBS_H, OBS_W, 1), dtype=np.uint8
        ),
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


PERTURBATIONS: dict[str, dict[str, float]] = {
    "nominal":         {},
    "gravity_lo":      {"gravity_mult": 0.75},
    "gravity_hi":      {"gravity_mult": 1.25},
    "restitution_lo":  {"restitution_mult": 0.80},
    "restitution_hi":  {"restitution_mult": 1.20},
    "mass_lo":         {"ball_mass_mult": 0.70},
    "mass_hi":         {"ball_mass_mult": 1.30},
}


def physics_for(label: str) -> PhysicsParams:
    base = PhysicsParams()
    cfg = PERTURBATIONS[label]
    return dataclasses.replace(
        base,
        gravity=base.gravity * cfg.get("gravity_mult", 1.0),
        restitution=base.restitution * cfg.get("restitution_mult", 1.0),
        ball_mass=base.ball_mass * cfg.get("ball_mass_mult", 1.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, train_args = load_agent(args.checkpoint, device)

    results: dict[str, dict] = {}
    for label in PERTURBATIONS:
        env = make_clone_wrapped(physics_for(label), train_args["frame_stack"])
        res = rollout_episodes(agent, env, args.episodes, device, args.seed, args.deterministic)
        env.close()
        results[label] = {
            "mean": res.mean_return,
            "std": res.std_return,
            "median": res.median_return,
            "q05": res.q05,
            "q95": res.q95,
        }
        print(
            f"{label:18s}  mean {res.mean_return:7.3f}  std {res.std_return:6.3f}  "
            f"median {res.median_return:7.3f}"
        )

    nominal_mean = results["nominal"]["mean"]
    for label, r in results.items():
        r["retention"] = r["mean"] / nominal_mean if nominal_mean else float("nan")

    print()
    print("Score retention (mean / nominal_mean):")
    for label, r in results.items():
        print(f"  {label:18s}  {r['retention']:.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "results": results}, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
