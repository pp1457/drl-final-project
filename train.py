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
    backend: str = "adb"               # 'adb' | 'adb-motionevent' | 'adb-minitouch' | 'minicap'
    frame_skip: int = 0                # 0 -> use config default (ACTIONS.frame_skip); >0 overrides
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
    # ---- Random Network Distillation (intrinsic motivation) ------------
    # When use_rnd=True, train a small predictor network to match a frozen
    # random target. Intrinsic reward per step = MSE between predictor and
    # target features, normalized by running std. Helps PPO commit to a
    # non-uniform policy when extrinsic reward is too sparse to provide a
    # gradient (the v8 failure mode).
    use_rnd: bool = False
    rnd_coef: float = 0.5         # weight of intrinsic reward in total reward
    rnd_loss_coef: float = 1.0    # weight of predictor MSE in total loss
    # ---- Charge-duration as extra obs dim -------------------------------
    # When use_charge_dim=True, concat info["charge_duration"]/30.0 (~[0,1])
    # to the encoder output before actor/critic heads. Lets the policy
    # condition on its own held-press history (Bouncy Basketball's charge
    # mechanic accumulates 1-25 hold-steps for shot power; the 4-frame stack
    # alone gives ~130ms history which is insufficient for charge timing).
    use_charge_dim: bool = False
    # ---- Resume training from checkpoint --------------------------------
    # Path to a checkpoint file saved by a prior run. Loads agent weights,
    # optimizer state, RND running stats, and continues from the saved
    # global_step. Used to extend training beyond the original
    # total_timesteps without throwing away learned progress.
    resume: str = ""
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
def _make_bouncy_env(rank: int, backend_name: str = "adb", frame_skip: int = 0) -> gym.Env:
    """Real-emulator env. Reads endpoint allocations from the JSON written by
    `orchestrate.py launch --n N`. The trainer assumes the farm is already up
    (run `python orchestrate.py launch --n <num_envs>` before `train.py`).

    backend_name:
      'adb'           - plain adb input swipe (atomic 132ms tap per PRESS).
      'adb-minitouch' - adb screencap for frames + minitouch for actions
                        (state-based touch, persists across env steps).
      'minicap'       - minicap + minitouch (broken on Android 12 right now;
                        see scripts/minicap_binaries notes).
    """
    from orchestrate import load_endpoints
    from reward import ScoreboardDiffReward

    endpoints = load_endpoints()
    if rank >= len(endpoints):
        raise RuntimeError(
            f"rank {rank} but only {len(endpoints)} endpoints; relaunch the farm with --n >= num_envs"
        )
    if backend_name == "minicap":
        from minicap_backend import MinicapMinitouchBackend
        backend = MinicapMinitouchBackend(endpoints[rank])
        backend.setup()
    elif backend_name == "adb-minitouch":
        from adb_backend import AdbMinitouchBackend
        backend = AdbMinitouchBackend(endpoints[rank])
        backend.setup()
    elif backend_name == "adb-motionevent":
        from adb_backend import AdbMotionEventBackend
        backend = AdbMotionEventBackend(endpoints[rank])
        backend.setup()
    else:
        from adb_backend import AdbBackend
        backend = AdbBackend(endpoints[rank])
    reward_extractor = ScoreboardDiffReward()
    env_kwargs = {
        "backend": backend,
        "pose_extractor": detect_pose,
        "reward_extractor": reward_extractor,
    }
    if frame_skip > 0:
        env_kwargs["frame_skip"] = frame_skip
    env = BouncyBasketballEnv(**env_kwargs)
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


def make_env_fn(env_id: str, rank: int, frame_stack: int, backend: str = "adb", frame_skip: int = 0):
    def thunk():
        if env_id == "fake":
            env = _make_fake_env(rank)
        elif env_id == "bouncy":
            env = _make_bouncy_env(rank, backend_name=backend, frame_skip=frame_skip)
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


class RNDNet(nn.Module):
    """Small CNN+MLP for RND target (frozen random) and predictor (trained).
    Both share architecture; target weights are frozen at init."""
    def __init__(self, frame_stack: int, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            _layer_init(nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)),
            nn.ELU(),
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ELU(),
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ELU(),
            nn.Flatten(),
            _layer_init(nn.Linear(64 * 7 * 7, latent_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x / 255.0)


class Agent(nn.Module):
    def __init__(self, frame_stack: int, n_actions: int, aux_mode: str,
                 use_rnd: bool = False, use_charge_dim: bool = False):
        super().__init__()
        self.encoder = NatureCNN(frame_stack)
        self.use_charge_dim = use_charge_dim
        # When use_charge_dim=True, the actor/critic see [encoder_features (512), charge_dur (1)]
        # = 513-dim input. Aux head still uses raw 512-dim encoder output (the
        # aux task is about visual features, charge state is a policy-side input).
        head_in_dim = 512 + (1 if use_charge_dim else 0)
        self.actor = _layer_init(nn.Linear(head_in_dim, n_actions), std=0.01)
        self.critic = _layer_init(nn.Linear(head_in_dim, 1), std=1.0)
        self.aux_mode = aux_mode
        self.use_rnd = use_rnd
        if use_rnd:
            self.rnd_target = RNDNet(frame_stack)
            self.rnd_predictor = RNDNet(frame_stack)
            for p in self.rnd_target.parameters():
                p.requires_grad_(False)
        if aux_mode == "oca":
            self.aux_head: nn.Module = nn.Sequential(
                _layer_init(nn.Linear(512, MODEL.oca_hidden_dim)),
                nn.ReLU(),
                _layer_init(nn.Linear(MODEL.oca_hidden_dim, OCA_DIM), std=0.01),
            )
        elif aux_mode == "dpr":
            self.aux_head = DPRDecoder()
        elif aux_mode == "baseline":
            self.aux_head = nn.Identity()
        else:
            raise ValueError(f"unknown aux_mode: {aux_mode}")

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def _head_input(self, z: torch.Tensor, charge_dur: torch.Tensor | None) -> torch.Tensor:
        """Concatenate the optional charge_dur input to encoder features for
        actor/critic. Normalize charge_dur by /30 so 0..30 charge maps to ~[0,1]
        (the charge cap in the in-game shot mechanic)."""
        if self.use_charge_dim:
            if charge_dur is None:
                charge_dur = torch.zeros(z.shape[0], 1, device=z.device)
            return torch.cat([z, charge_dur.view(-1, 1) / 30.0], dim=-1)
        return z

    def get_value(self, obs: torch.Tensor, charge_dur: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(self._head_input(self.encode(obs), charge_dur))

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None,
        charge_dur: torch.Tensor | None = None,
    ):
        z = self.encode(obs)
        h = self._head_input(z, charge_dur)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h), z

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

    # SyncVectorEnv (sequential in main process) instead of AsyncVectorEnv:
    # - sidesteps gym's TimeoutExpired-reconstruct bug (worker→parent pipe
    #   tried `exctype(value)` on subprocess.TimeoutExpired which needs 2
    #   positional args; killed ws1+ws6 mid-training)
    # - lets exceptions in env.step / env.reset bubble straight into the
    #   rollout loop where we can recover
    # At N_ENVS=2 with adb-bound latency, sequential vs parallel costs ~0.
    env_fns = [
        make_env_fn(args.env_id, i, args.frame_stack, backend=args.backend, frame_skip=args.frame_skip)
        for i in range(args.num_envs)
    ]
    envs = gym.vector.SyncVectorEnv(env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    n_actions = int(envs.single_action_space.n)
    assert n_actions == N_ACTIONS, (n_actions, N_ACTIONS)

    agent = Agent(
        args.frame_stack, n_actions, args.aux_mode,
        use_rnd=args.use_rnd, use_charge_dim=args.use_charge_dim,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # RND intrinsic-reward running stats: normalize intrinsic rewards by
    # their std over time (per the RND paper). Initialize lazily.
    rnd_running_mean = torch.tensor(0.0, device=device)
    rnd_running_var = torch.tensor(1.0, device=device)
    rnd_running_count = 1e-4

    # Resume from checkpoint: restores agent + optimizer + RND stats so a
    # training run can continue from where a prior run stopped. global_step
    # is also restored so PPO updates / global_step accounting stays correct.
    resume_from_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "rnd_running_mean" in ckpt:
            rnd_running_mean = ckpt["rnd_running_mean"].to(device)
            rnd_running_var = ckpt["rnd_running_var"].to(device)
            rnd_running_count = ckpt.get("rnd_running_count", 1e-4)
        resume_from_step = ckpt.get("global_step", 0)
        print(f"Resumed at global_step={resume_from_step}, update={ckpt.get('update', 0)}", flush=True)

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
    # Charge-duration tracking: shape (num_steps, num_envs). Each env tracks
    # consecutive PRESS-step count; provided to agent.get_action_and_value
    # when use_charge_dim=True. Tracked as a separate tensor so it's easy to
    # plumb through rollout, buffer, and PPO update without touching the obs.
    charge_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    next_charge = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    ckpt_path = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(ckpt_path, exist_ok=True)

    global_step = resume_from_step
    start_time = time.time()
    next_obs, info = envs.reset(seed=args.seed)
    next_obs = to_chw(next_obs)
    next_done = torch.zeros(args.num_envs, device=device)

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    ep_returns = deque(maxlen=64)

    def _recover_envs(reason: str):
        """Hard-restart the vec env after a step/reset failure.

        Designed to NEVER raise — every step is wrapped in a try/except. On
        failure to bring envs back up, returns a zero-tensor obs so the
        outer loop can increment consecutive_failures and (eventually) bail
        cleanly via the kill-switch at consecutive_failures >= N.

        Recovery sequence:
          1. Print the reason + traceback.
          2. For each endpoint: `adb emu kill` (fire-and-forget).
          3. WAIT for each endpoint to leave the adb device list — the kill
             is async, so checking too fast lets supervise_once see "device"
             and skip relaunch. This was the bug that crashed ws3 + ws4 the
             first time around.
          4. For each endpoint: directly call `_launch_emulator` +
             `_wait_boot` (bypass supervise_once's flaky alive-check entirely
             — we KNOW the emulator should be dead by now).
          5. close + remake SyncVectorEnv.
          6. envs.reset() (and tolerate failure here too — return zeros if so).
        """
        nonlocal envs
        import traceback, subprocess, time as _time
        print(f"!! recovering envs: {reason}", flush=True)
        try:
            traceback.print_exc()
        except Exception:
            pass

        # --- helper that absolutely cannot raise --------------------------
        def _safe(label, fn):
            try:
                return fn()
            except Exception as e:
                print(f"   recovery: {label} failed: {e!r}", flush=True)
                return None

        if args.env_id == "bouncy":
            endpoints = _safe("load_endpoints", lambda: __import__('orchestrate').load_endpoints()) or []
            # Step 1: kill every emulator (async).
            for ep in endpoints:
                _safe(
                    f"emu kill {ep.adb_serial}",
                    lambda ep=ep: subprocess.run(
                        ["adb", "-s", ep.adb_serial, "emu", "kill"],
                        capture_output=True, timeout=5.0,
                    ),
                )
            # Step 2: wait for each emulator to leave the adb device list.
            for ep in endpoints:
                deadline = _time.time() + 20.0
                while _time.time() < deadline:
                    r = subprocess.run(
                        ["adb", "-s", ep.adb_serial, "get-state"],
                        capture_output=True, timeout=3.0,
                    )
                    if r.stdout.strip() != b"device":
                        break
                    _time.sleep(0.5)
                else:
                    print(f"   recovery: {ep.adb_serial} still 'device' after kill — proceeding anyway", flush=True)
            # Step 3: relaunch each emulator directly (don't trust
            # supervise_once's alive-check after the kill).
            try:
                from orchestrate import _launch_emulator, _wait_boot, BASE_PORT, EMU
                snapshot = EMU.default_boot_snapshot
            except Exception as e:
                print(f"   recovery: orchestrate import failed: {e!r}", flush=True)
                _launch_emulator = None
            if _launch_emulator is not None:
                for ep in endpoints:
                    rank = (int(ep.adb_serial.split("-")[1]) - BASE_PORT) // 2
                    _safe(
                        f"relaunch {ep.adb_serial}",
                        lambda rank=rank: _launch_emulator(rank=rank, read_only=True, snapshot=snapshot),
                    )
                for ep in endpoints:
                    _safe(
                        f"wait_boot {ep.adb_serial}",
                        lambda ep=ep: _wait_boot(ep.adb_serial, timeout=180.0),
                    )

        # Step 4: remake the vec env.
        _safe("envs.close", lambda: envs.close())
        envs = gym.vector.SyncVectorEnv(env_fns)
        fresh = _safe("envs.reset", lambda: envs.reset(seed=args.seed + update))
        if fresh is None:
            # Couldn't reset. Return zeros so the outer loop can record this
            # as a consecutive failure and eventually bail. Never raise.
            print(f"   recovery: returning zero obs as fallback", flush=True)
            zeros = torch.zeros((args.num_envs,) + obs_shape, dtype=torch.uint8, device=device)
            return zeros
        return to_chw(fresh[0])

    consecutive_failures = 0
    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        try:
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs_buf[step] = next_obs
                dones_buf[step] = next_done
                charge_buf[step] = next_charge
                with torch.no_grad():
                    action, logprob, _, value, _ = agent.get_action_and_value(
                        next_obs.float(),
                        charge_dur=next_charge if args.use_charge_dim else None,
                    )
                    values_buf[step] = value.flatten()
                actions_buf[step] = action
                logprobs_buf[step] = logprob

                next_obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
                done = np.logical_or(term, trunc)
                rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
                next_obs = to_chw(next_obs_np)
                next_done = torch.as_tensor(done, dtype=torch.float32, device=device)
                next_obs_buf[step] = next_obs
                # Pull charge_duration for the NEXT step from this step's info.
                # If info doesn't have it (e.g., fake env, no charge tracking),
                # fall back to in-Python computation from this step's action.
                if "charge_duration" in info:
                    next_charge = torch.as_tensor(
                        np.asarray(info["charge_duration"], dtype=np.float32),
                        device=device,
                    )
                else:
                    a_t = action.float()
                    next_charge = (next_charge + a_t) * a_t  # +1 if PRESS, 0 if NO_PRESS

                # RND intrinsic reward — predictor's distance from the frozen
                # target tells us how "novel" this observation is. Added to
                # the extrinsic reward to break the uniform-random equilibrium
                # when extrinsic gradient is near zero (the v8 failure mode).
                if args.use_rnd:
                    with torch.no_grad():
                        tgt = agent.rnd_target(next_obs.float())
                        prd = agent.rnd_predictor(next_obs.float())
                        intrinsic = ((prd - tgt) ** 2).mean(dim=-1)
                        # Normalize by running std for stable scale
                        rnd_running_count += float(args.num_envs)
                        delta = intrinsic.mean() - rnd_running_mean
                        rnd_running_mean = rnd_running_mean + delta * args.num_envs / rnd_running_count
                        rnd_running_var = rnd_running_var * 0.99 + intrinsic.var() * 0.01
                        intrinsic_norm = intrinsic / (rnd_running_var.sqrt() + 1e-8)
                        rewards_buf[step] = rewards_buf[step] + args.rnd_coef * intrinsic_norm

                # OCA targets come from the *next* step's info (target_{t+1})
                if "oca_target" in info:
                    oca_target_buf[step] = torch.as_tensor(np.stack(info["oca_target"]), device=device)
                    oca_mask_buf[step] = torch.as_tensor(np.stack(info["oca_mask"]), device=device)

                if "episode" in info:
                    for r, mask in zip(info["episode"]["r"], info["episode"].get("_r", [True]*args.num_envs)):
                        if mask:
                            ep_returns.append(float(r))
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                raise RuntimeError(
                    f"rollout failed {consecutive_failures} times in a row; bailing"
                ) from e
            next_obs = _recover_envs(f"upd {update} step {step}: {e!r}")
            next_done = torch.zeros(args.num_envs, device=device)
            continue  # skip the PPO update on a partial/corrupt rollout

        # Bootstrap value at the end of the rollout
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs.float(),
                charge_dur=next_charge if args.use_charge_dim else None,
            ).reshape(1, -1)
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

        # Shift OCA targets by config.MODEL.oca_horizon_steps so the head
        # predicts pose K steps into the future, not just t+1 (which the
        # encoder solves trivially in <10 updates). Per-env shift, with the
        # last K positions masked since we don't have a target that far ahead.
        if args.aux_mode == "oca":
            K = MODEL.oca_horizon_steps
            if K > 1:
                # oca_target_buf shape: (num_steps, num_envs, OCA_DIM)
                # Want oca_target_buf[i] := original oca_target_buf[i+K], for i+K < num_steps.
                shifted_t = torch.zeros_like(oca_target_buf)
                shifted_m = torch.zeros_like(oca_mask_buf)
                shifted_t[: args.num_steps - K] = oca_target_buf[K:]
                shifted_m[: args.num_steps - K] = oca_mask_buf[K:]
                # Tail K positions stay zero in both target and mask (mask=0 -> ignored by loss)
                oca_target_buf = shifted_t
                oca_mask_buf = shifted_m

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
        b_charge = charge_buf.reshape(-1)

        idx = np.arange(batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, minibatch_size):
                mb = idx[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue, z_t = agent.get_action_and_value(
                    b_obs[mb].float(), b_actions[mb],
                    charge_dur=b_charge[mb] if args.use_charge_dim else None,
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

                # RND predictor loss: train predictor to match the frozen
                # random target. As predictor improves on visited states,
                # the intrinsic reward shrinks there, pushing exploration to
                # novel states.
                rnd_loss = torch.tensor(0.0, device=device)
                if args.use_rnd:
                    with torch.no_grad():
                        rnd_tgt = agent.rnd_target(b_obs[mb].float())
                    rnd_prd = agent.rnd_predictor(b_obs[mb].float())
                    rnd_loss = ((rnd_prd - rnd_tgt) ** 2).mean()

                loss = (
                    pg_loss
                    - args.ent_coef * ent_loss
                    + args.vf_coef * v_loss
                    + args.aux_coef * aux_loss
                    + args.rnd_loss_coef * rnd_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        # Frame-health stats over the just-collected rollout: what fraction
        # of frames had ANY object detected (i.e. weren't all-zero mask),
        # and the mean number of detected components per frame (out of 10).
        # If frame_health is low we're training on garbage / stuck frames.
        mask_any_detected = (oca_mask_buf.sum(dim=-1) > 0).float().mean().item()
        mask_mean_components = oca_mask_buf.sum(dim=-1).mean().item()
        writer.add_scalar("charts/sps", sps, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", ent_loss.item(), global_step)
        writer.add_scalar("losses/aux_loss", aux_loss.item(), global_step)
        if args.use_rnd:
            writer.add_scalar("losses/rnd_loss", rnd_loss.item(), global_step)
            writer.add_scalar("charts/rnd_running_std", rnd_running_var.sqrt().item(), global_step)
        writer.add_scalar("charts/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/frame_health_any_detected", mask_any_detected, global_step)
        writer.add_scalar("charts/frame_health_components_mean", mask_mean_components, global_step)
        if ep_returns:
            writer.add_scalar("charts/episode_return_mean", float(np.mean(ep_returns)), global_step)
        rnd_msg = f" rnd {rnd_loss.item():.4f}" if args.use_rnd else ""
        print(
            f"upd {update}/{num_updates} step {global_step} sps {sps} "
            f"pg {pg_loss.item():.3f} v {v_loss.item():.3f} "
            f"ent {ent_loss.item():.3f} aux {aux_loss.item():.4f}{rnd_msg} "
            f"ret {np.mean(ep_returns) if ep_returns else float('nan'):.2f} "
            f"fh {mask_any_detected:.2f}/{mask_mean_components:.1f}"
        )

        if update % args.ckpt_every == 0 or update == num_updates:
            torch.save(
                {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "global_step": global_step,
                    "update": update,
                    "rnd_running_mean": rnd_running_mean.cpu(),
                    "rnd_running_var": rnd_running_var.cpu(),
                    "rnd_running_count": rnd_running_count,
                },
                os.path.join(ckpt_path, f"step_{global_step}.pt"),
            )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
