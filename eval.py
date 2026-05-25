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

from env import N_ACTIONS, OBS_H, OBS_W, BouncyBasketballEnv
from train import Agent, make_env_fn


def _make_bouncy_env_for_serial(serial: str, backend_name: str, frame_skip: int,
                                max_episode_steps: int = 0):
    """Build a Bouncy env that connects to an explicit adb serial — bypasses
    orchestrate's endpoints.json so we can eval on a side emulator while the
    training farm holds the main ports.

    max_episode_steps: 0 = use env default (1024); >0 caps episodes earlier.
    Useful for eval to shorten per-episode wall-time (each step is ~500ms
    of adb overhead, so 1024-step episodes = 8 min).
    """
    from adb_backend import AdbBackend, AdbMotionEventBackend
    from orchestrate import EmulatorEndpoint
    from reward import ScoreboardDiffReward
    from vision import detect_pose
    from config import EMU
    endpoint = EmulatorEndpoint(
        adb_serial=serial,
        minicap_port=0,
        minitouch_port=0,
        snapshot_names=list(EMU.snapshot_pool),
    )
    if backend_name == "adb-motionevent":
        backend = AdbMotionEventBackend(endpoint)
        backend.setup()
    else:
        backend = AdbBackend(endpoint)
    reward_extractor = ScoreboardDiffReward()
    env_kwargs = {
        "backend": backend,
        "pose_extractor": detect_pose,
        "reward_extractor": reward_extractor,
    }
    if frame_skip > 0:
        env_kwargs["frame_skip"] = frame_skip
    if max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = max_episode_steps
    env = BouncyBasketballEnv(**env_kwargs)
    env._reward_state = reward_extractor
    return env


def _wrap_env_for_eval(env, frame_stack: int):
    """Apply the same observation transforms make_env_fn would (TransformObs +
    FrameStack + RecordEpisodeStatistics) so the trained policy's tensor shapes
    line up."""
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
    # Reconstruct Agent with the SAME flags it was trained with so the
    # state_dict matches (RND adds rnd_target/rnd_predictor submodules;
    # charge_dim changes actor/critic input size).
    agent = Agent(
        args["frame_stack"], N_ACTIONS, args["aux_mode"],
        use_rnd=args.get("use_rnd", False),
        use_charge_dim=args.get("use_charge_dim", False),
    ).to(device)
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
    use_charge = bool(getattr(agent, "use_charge_dim", False))
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_len = 0
        charge_dur = 0.0
        while True:
            o = torch.as_tensor(np.asarray(obs), device=device).squeeze(-1).unsqueeze(0).float()
            z = agent.encode(o)
            if use_charge:
                cd = torch.tensor([[charge_dur]], device=device, dtype=torch.float32)
                h = agent._head_input(z, cd)
            else:
                h = z
            logits = agent.actor(h)
            if deterministic:
                action = int(logits.argmax(-1).item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
            obs, r, term, trunc, info = env.step(action)
            if use_charge and isinstance(info, dict) and "charge_duration" in info:
                charge_dur = float(info["charge_duration"])
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
    ap.add_argument(
        "--serial",
        default=None,
        help="If set, bypass endpoints.json and connect directly to this adb "
             "serial (e.g. emulator-6558). Used when eval'ing on a side emulator "
             "while training holds the main farm.",
    )
    ap.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="If >0, override max_episode_steps. Each step is ~500ms wall time, "
             "so 256 ≈ ~2 min/episode (~1 quarter), 1024 ≈ ~8 min/episode (env default).",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, train_args = load_agent(args.checkpoint, device)

    # Pull backend + frame_skip from train_args so the eval matches training
    # conditions (variable-hold motionevent backend vs atomic adb taps changes
    # the policy's effective action semantics).
    backend = train_args.get("backend", "adb")
    frame_skip = train_args.get("frame_skip", 0)
    if args.env_id == "bouncy" and args.serial:
        env = _make_bouncy_env_for_serial(
            args.serial, backend_name=backend, frame_skip=frame_skip,
            max_episode_steps=args.max_episode_steps,
        )
        # Wrap with the same observation transforms make_env_fn applies so the
        # checkpoint's tensors line up.
        env_fn = lambda: env
        env = _wrap_env_for_eval(env, frame_stack=train_args["frame_stack"])
    else:
        env_fn = make_env_fn(
            args.env_id, rank=0, frame_stack=train_args["frame_stack"],
            backend=backend, frame_skip=frame_skip,
        )
        env = env_fn()
    res = rollout_episodes(agent, env, args.episodes, device, args.seed, args.deterministic)
    env.close()
    # Final cumulative score per episode — under opponent_score_weight=-1 this
    # is (CHI score − HOU score) i.e. the net match outcome.
    print("per-episode net scores:", [round(r, 2) for r in res.raw_returns])

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
