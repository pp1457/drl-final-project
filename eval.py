"""Load a trained checkpoint, run N matches, report return statistics.

Used for:
    - §5.2 asymptotic-performance evaluation on the Android env
    - §5.3 robustness-to-perturbation evaluation on the Python clone (via
      eval_robustness.py)

Usage:
    python eval.py --checkpoint checkpoints/<run>/step_1000000.pt \\
        --env-id fake --episodes 100 --seed 42

    # robustness sweep across 9 perturbation conditions
    python eval_robustness.py --checkpoint <path> --episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import gymnasium as gym

from env import N_ACTIONS, OBS_H, OBS_W
from train import Agent, make_env_fn


@dataclass
class EvalResult:
    n_episodes: int
    mean_return: float
    std_return: float
    median_return: float
    q05: float
    q95: float
    mean_length: float
    raw_returns: list[float]


def load_agent(checkpoint_path: str, device: torch.device) -> tuple[Agent, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    agent = Agent(args["frame_stack"], N_ACTIONS, args["aux_mode"]).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent, args


@torch.no_grad()
def rollout_episodes(
    agent: Agent,
    env: gym.Env,
    n_episodes: int,
    device: torch.device,
    seed: int = 0,
    deterministic: bool = False,
) -> EvalResult:
    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(n_episodes):
        obs, _info = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_len = 0
        while True:
            o = torch.as_tensor(np.asarray(obs), device=device).squeeze(-1).unsqueeze(0).float()
            logits = agent.actor(agent.encode(o))
            if deterministic:
                action = int(logits.argmax(-1).item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
            obs, r, term, trunc, _info = env.step(action)
            ep_return += float(r)
            ep_len += 1
            if term or trunc:
                break
        returns.append(ep_return)
        lengths.append(ep_len)
    arr = np.asarray(returns, dtype=np.float64)
    return EvalResult(
        n_episodes=n_episodes,
        mean_return=float(arr.mean()),
        std_return=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        median_return=float(np.median(arr)),
        q05=float(np.quantile(arr, 0.05)),
        q95=float(np.quantile(arr, 0.95)),
        mean_length=float(np.mean(lengths)),
        raw_returns=list(returns),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env-id", default="fake", choices=["fake", "bouncy"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--output", default=None, help="optional JSON output file")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, train_args = load_agent(args.checkpoint, device)

    env_fn = make_env_fn(args.env_id, rank=0, frame_stack=train_args["frame_stack"])
    env = env_fn()
    res = rollout_episodes(agent, env, args.episodes, device, args.seed, args.deterministic)
    env.close()

    print(f"checkpoint: {args.checkpoint}")
    print(f"env_id:     {args.env_id}")
    print(f"episodes:   {res.n_episodes}")
    print(f"return:     mean {res.mean_return:.3f}  std {res.std_return:.3f}  median {res.median_return:.3f}")
    print(f"            q05 {res.q05:.3f}  q95 {res.q95:.3f}")
    print(f"length:     mean {res.mean_length:.1f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(res.__dict__, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
