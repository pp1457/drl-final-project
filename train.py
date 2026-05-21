"""PPO trainer with three auxiliary-task configurations.

Forked from CleanRL's ppo_atari.py (https://github.com/vwxyzjn/cleanrl), kept
single-file. Encoder is split out so OCA and DPR heads can hang off the shared
latent z_t.

Three configurations, toggled by --aux-mode:

    baseline   λ = 0, encoder learns from RL gradients only
    oca        λ * MSE between aux MLP head and cached pose targets
    dpr        λ * MSE between transposed-conv decoder and next-frame pixels

Aux supervision pairs (s_t, target_{t+1}): we predict the *next* state's target
from the current observation, forcing the encoder to encode forward dynamics.

Smoke testing:
    python train.py --env-id fake --total-timesteps 50000 --num-envs 4 --aux-mode oca

Real run on the emulator farm (after Person A's backend lands):
    python train.py --env-id bouncy --total-timesteps 1000000 --num-envs 8 \\
        --aux-mode oca --aux-coef 0.5 --seed 0
"""

from __future__ import annotations

import os

# IMPORTANT: cap intra-op thread pools BEFORE importing cv2, torch, numpy.
# With 144 cores on ws10, every worker process would default to a 144-thread
# pool for OpenMP / BLAS / cv2. Across 6 AsyncVectorEnv workers + the main
# process, the total thread count overflows the per-user pthread/process
# limit (we hit `pthread_create: Resource temporarily unavailable`). The
# emulators themselves already give us all the parallelism we need; each
# worker just needs single-threaded numpy/cv2/torch.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENCV_NUM_THREADS"):
    os.environ[_k] = "1"   # force-override; subprocess workers inherit this

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass

import cv2  # noqa: F401  (used indirectly via env.py)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from env import (
    BouncyBasketballEnv,
    EmulatorBackend,
    FakeBackend,
    fake_reward_extractor,
    N_ACTIONS,
    OBS_H,
    OBS_W,
    OCA_DIM,
)
from vision import detect_pose


# ---------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------
# Defaults sourced from config.PPO / config.MODEL so tweaks live in one file.
from config import PPO, MODEL


@dataclass
class Args:
    env_id: str = "fake"               # 'fake' | 'bouncy' | 'clone'
    aux_mode: str = "baseline"         # 'baseline' | 'oca' | 'dpr'
    aux_coef: float = PPO.aux_coef
    seed: int = 0
    total_timesteps: int = PPO.total_timesteps
    num_envs: int = 8
    num_steps: int = PPO.num_steps
    learning_rate: float = PPO.learning_rate
    anneal_lr: bool = PPO.anneal_lr
    gamma: float = PPO.gamma
    gae_lambda: float = PPO.gae_lambda
    num_minibatches: int = PPO.num_minibatches
    update_epochs: int = PPO.update_epochs
    clip_coef: float = PPO.clip_coef
    vf_coef: float = PPO.vf_coef
    ent_coef: float = PPO.ent_coef
    max_grad_norm: float = PPO.max_grad_norm
    frame_stack: int = MODEL.frame_stack
    log_dir: str = "runs"
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 50          # save every N updates
    wandb: bool = False
    wandb_project: str = "drl-final-bouncy"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    for fld in Args.__dataclass_fields__.values():
        kw = {"default": fld.default, "type": type(fld.default)}
        if fld.type is bool or isinstance(fld.default, bool):
            kw["action"] = "store_true" if not fld.default else "store_false"
            kw.pop("type")
            kw.pop("default")
            p.add_argument(f"--{fld.name.replace('_', '-')}", **kw)
        else:
            p.add_argument(f"--{fld.name.replace('_', '-')}", **kw)
    return Args(**vars(p.parse_args()))


# ---------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------
def _make_bouncy_env(rank: int) -> gym.Env:
    """Real-emulator env. Reads endpoint allocations from the JSON written by
    `orchestrate.py launch --n N`. The trainer assumes the farm is already up
    (run `python orchestrate.py launch --n <num_envs>` before `train.py`)."""
    from adb_backend import AdbBackend
    from orchestrate import load_endpoints
    from reward import ScoreboardDiffReward

    endpoints = load_endpoints()
    if rank >= len(endpoints):
        raise RuntimeError(
            f"rank {rank} but only {len(endpoints)} endpoints; relaunch the farm with --n >= num_envs"
        )
    backend = AdbBackend(endpoints[rank])
    reward_extractor = ScoreboardDiffReward()
    env = BouncyBasketballEnv(
        backend=backend,
        pose_extractor=detect_pose,
        reward_extractor=reward_extractor,
        max_episode_steps=4096,
    )
    # Stash the reward extractor so reset() can reset its internal state.
    env._reward_state = reward_extractor  # type: ignore[attr-defined]
    return env


def _make_fake_env(rank: int) -> gym.Env:
    """Synthetic backend for smoke-testing the training loop end-to-end."""
    return BouncyBasketballEnv(
        backend=FakeBackend(seed=rank),
        pose_extractor=detect_pose,
        reward_extractor=fake_reward_extractor,
        max_episode_steps=512,
    )


def _make_clone_env(rank: int) -> gym.Env:
    """Python physics clone. Production use is §5.3 eval-only, but we also
    support training on it as a smoke test."""
    from clone_env import BouncyBasketballCloneEnv
    return BouncyBasketballCloneEnv(seed=rank)


def make_env_fn(env_id: str, rank: int, frame_stack: int):
    def thunk():
        if env_id == "fake":
            env = _make_fake_env(rank)
        elif env_id == "bouncy":
            env = _make_bouncy_env(rank)
        elif env_id == "clone":
            env = _make_clone_env(rank)
        else:
            raise ValueError(f"unknown env_id: {env_id}")
        # Expand obs to (H, W, 1) so FrameStack produces (k, H, W, 1)
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

    return thunk


# ---------------------------------------------------------------
# Network
# ---------------------------------------------------------------
def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """Standard Atari-style encoder. In: (B, k, H, W) uint8. Out: (B, 512)."""

    def __init__(self, frame_stack: int):
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            _layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x / 255.0)


class DPRDecoder(nn.Module):
    """Transposed-conv decoder: 512 -> (1, 84, 84). Predicts next frame as
    grayscale; comparison made against the next observation's last channel."""

    def __init__(self):
        super().__init__()
        self.fc = _layer_init(nn.Linear(512, 64 * 7 * 7))
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),  # 7 -> 9
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2),  # 9 -> 20
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2),   # 20 -> 42
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2),    # 42 -> 86
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 64, 7, 7)
        x = self.deconv(x)
        # crop to 84x84 (output is 86x86)
        return x[:, :, 1:85, 1:85]


class Agent(nn.Module):
    def __init__(self, frame_stack: int, n_actions: int, aux_mode: str):
        super().__init__()
        self.encoder = NatureCNN(frame_stack)
        self.actor = _layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = _layer_init(nn.Linear(512, 1), std=1.0)
        self.aux_mode = aux_mode
        if aux_mode == "oca":
            self.aux_head: nn.Module = nn.Sequential(
                _layer_init(nn.Linear(512, 128)),
                nn.ReLU(),
                _layer_init(nn.Linear(128, OCA_DIM), std=0.01),
            )
        elif aux_mode == "dpr":
            self.aux_head = DPRDecoder()
        elif aux_mode == "baseline":
            self.aux_head = nn.Identity()
        else:
            raise ValueError(f"unknown aux_mode: {aux_mode}")

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encode(obs))

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ):
        z = self.encode(obs)
        logits = self.actor(z)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(z), z

    def aux_predict(self, z: torch.Tensor) -> torch.Tensor:
        if self.aux_mode == "baseline":
            return z * 0  # placeholder, never used
        return self.aux_head(z)


# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------
def compute_aux_loss(
    aux_mode: str,
    z_t: torch.Tensor,
    agent: Agent,
    next_obs_chw: torch.Tensor,
    oca_target: torch.Tensor,
    oca_mask: torch.Tensor,
) -> torch.Tensor:
    if aux_mode == "baseline":
        return torch.zeros((), device=z_t.device)
    pred = agent.aux_predict(z_t)
    if aux_mode == "oca":
        sq = (pred - oca_target) ** 2
        denom = oca_mask.sum().clamp_min(1.0)
        return (sq * oca_mask).sum() / denom
    if aux_mode == "dpr":
        # Target: most recent grayscale frame of next_obs (channel index -1)
        target = next_obs_chw[:, -1:, :, :].float() / 255.0
        return ((pred - target) ** 2).mean()
    raise ValueError(aux_mode)


def main():
    args = parse_args()
    run_name = f"{args.env_id}_{args.aux_mode}_lam{args.aux_coef}_s{args.seed}_{int(time.time())}"
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args), sync_tensorboard=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # Cap intra-op torch threads too. The env-var prologue above covers OpenMP,
    # but torch's own pool needs an explicit call. Inter-op stays at 1 to
    # avoid cross-process thread contention.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    envs = gym.vector.AsyncVectorEnv(
        [make_env_fn(args.env_id, i, args.frame_stack) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    n_actions = int(envs.single_action_space.n)
    assert n_actions == N_ACTIONS, (n_actions, N_ACTIONS)

    agent = Agent(args.frame_stack, n_actions, args.aux_mode).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # FrameStack returns shape (k, H, W, 1); permute to (k, H, W) for the CNN
    def to_chw(o):
        # o: (B, k, H, W, 1) uint8 -> (B, k, H, W) uint8
        return torch.as_tensor(o, device=device).squeeze(-1)

    obs_shape = (args.frame_stack, OBS_H, OBS_W)
    obs_buf = torch.zeros((args.num_steps, args.num_envs) + obs_shape, dtype=torch.uint8, device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    oca_target_buf = torch.zeros((args.num_steps, args.num_envs, OCA_DIM), device=device)
    oca_mask_buf = torch.zeros((args.num_steps, args.num_envs, OCA_DIM), device=device)
    next_obs_buf = torch.zeros((args.num_steps, args.num_envs) + obs_shape, dtype=torch.uint8, device=device)

    ckpt_path = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(ckpt_path, exist_ok=True)

    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset(seed=args.seed)
    next_obs = to_chw(next_obs)
    next_done = torch.zeros(args.num_envs, device=device)

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    ep_returns = deque(maxlen=64)

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs.float())
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)
            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = to_chw(next_obs_np)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)
            next_obs_buf[step] = next_obs

            # OCA targets come from the *next* step's info (target_{t+1})
            if "oca_target" in info:
                oca_target_buf[step] = torch.as_tensor(np.stack(info["oca_target"]), device=device)
                oca_mask_buf[step] = torch.as_tensor(np.stack(info["oca_mask"]), device=device)

            if "episode" in info:
                # AsyncVectorEnv RecordEpisodeStatistics packs into _episode mask
                for r, mask in zip(info["episode"]["r"], info["episode"].get("_r", [True]*args.num_envs)):
                    if mask:
                        ep_returns.append(float(r))

        # Bootstrap value at the end of the rollout
        with torch.no_grad():
            next_value = agent.get_value(next_obs.float()).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nonterminal * lastgaelam
            returns = advantages + values_buf

        # Flatten the batch
        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_next_obs = next_obs_buf.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_oca_target = oca_target_buf.reshape(-1, OCA_DIM)
        b_oca_mask = oca_mask_buf.reshape(-1, OCA_DIM)

        idx = np.arange(batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, minibatch_size):
                mb = idx[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue, z_t = agent.get_action_and_value(
                    b_obs[mb].float(), b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb]) ** 2).mean()
                ent_loss = entropy.mean()

                aux_loss = compute_aux_loss(
                    args.aux_mode,
                    z_t,
                    agent,
                    b_next_obs[mb],
                    b_oca_target[mb],
                    b_oca_mask[mb],
                )

                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss + args.aux_coef * aux_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/sps", sps, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", ent_loss.item(), global_step)
        writer.add_scalar("losses/aux_loss", aux_loss.item(), global_step)
        writer.add_scalar("charts/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        if ep_returns:
            writer.add_scalar("charts/episode_return_mean", float(np.mean(ep_returns)), global_step)
        print(
            f"upd {update}/{num_updates} step {global_step} sps {sps} "
            f"pg {pg_loss.item():.3f} v {v_loss.item():.3f} "
            f"ent {ent_loss.item():.3f} aux {aux_loss.item():.4f} "
            f"ret {np.mean(ep_returns) if ep_returns else float('nan'):.2f}"
        )

        if update % args.ckpt_every == 0 or update == num_updates:
            torch.save(
                {
                    "agent": agent.state_dict(),
                    "args": vars(args),
                    "global_step": global_step,
                    "update": update,
                },
                os.path.join(ckpt_path, f"step_{global_step}.pt"),
            )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
