"""Sequentially run all 9 configurations of the ablation matrix.

3 configs × 3 seeds = 9 runs, one after another. Each run writes its own
checkpoint directory + log file. If a run fails (emulator crash, etc.) we
log it and move on rather than aborting the whole matrix.

Skips configs that already have a final checkpoint, so re-running is safe.

Usage:
    # In tmux + nohup so an SSH drop doesn't kill it:
    tmux new -s overnight
    source /tmp2/$USER/DRL_final_project/android_env.sh
    cd /tmp2/$USER/DRL_final_project
    # Pre-launch the emulator farm (one-time, not per-config):
    EMU_LAUNCH_STAGGER_S=8 .venv/bin/python orchestrate.py launch --n 4
    # Then:
    nohup .venv/bin/python run_all.py --steps 50000 --num-envs 4 \\
        > /tmp2/$USER/DRL_final/run_all.log 2>&1 &
    # Ctrl-b d to detach. Reattach with: tmux attach -t overnight
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import config


CONFIGS = ["baseline", "oca", "dpr"]
SEEDS = [0, 1, 2]


def run_name(mode: str, seed: int, steps: int) -> str:
    return f"bouncy_{mode}_lam{config.PPO.aux_coef}_s{seed}_{steps}"


def already_done(mode: str, seed: int, steps: int, ckpt_root: Path) -> bool:
    """Check if a final checkpoint exists for this config."""
    name_prefix = f"bouncy_{mode}_"
    for run_dir in ckpt_root.glob(f"{name_prefix}*_s{seed}_*"):
        if not run_dir.is_dir():
            continue
        # final checkpoint filename is step_<steps>.pt
        final_ckpt = run_dir / f"step_{steps}.pt"
        if final_ckpt.exists():
            return True
    return False


def run_one(mode: str, seed: int, args) -> int:
    """Spawn one training run; return its exit code."""
    cmd = [
        sys.executable, "train.py",
        "--env-id", "bouncy",
        "--aux-mode", mode,
        "--seed", str(seed),
        "--total-timesteps", str(args.steps),
        "--num-envs", str(args.num_envs),
        "--num-steps", str(args.num_steps),
        "--num-minibatches", str(args.num_minibatches),
        "--update-epochs", str(args.update_epochs),
        "--ckpt-every", str(args.ckpt_every),
    ]
    if args.wandb:
        cmd.append("--wandb")
    log_path = Path(args.log_dir) / f"{mode}_s{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\n[run_all] {mode} seed={seed} -> {log_path}\n{'=' * 60}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dur = time.time() - t0
    print(f"[run_all] {mode} seed={seed} exited {result.returncode} after {dur/60:.1f} min", flush=True)
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000, help="total timesteps per run")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=128)
    ap.add_argument("--num-minibatches", type=int, default=4)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=50, help="save every N updates")
    ap.add_argument("--ckpt-root", default="checkpoints",
                    help="root directory under which train.py writes its run-named subdirs")
    ap.add_argument("--log-dir", default=f"/tmp2/{os.environ['USER']}/DRL_final/run_all_logs")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="optional subset of modes (e.g. --only oca dpr)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="optional subset of seeds (e.g. --seeds 0 1)")
    args = ap.parse_args()

    modes = args.only or CONFIGS
    seeds = args.seeds or SEEDS
    ckpt_root = Path(args.ckpt_root)

    print(f"[run_all] modes={modes}  seeds={seeds}  steps={args.steps}  num_envs={args.num_envs}")
    print(f"[run_all] est. wall time per run @ 4 SPS = {args.steps / 4 / 60:.1f} min")
    print(f"[run_all] est. total = {len(modes) * len(seeds) * args.steps / 4 / 3600:.1f} hours")

    summary = []
    for mode in modes:
        for seed in seeds:
            if already_done(mode, seed, args.steps, ckpt_root):
                print(f"[run_all] SKIP {mode} seed={seed} (final ckpt already exists)")
                summary.append((mode, seed, "skipped", 0.0))
                continue
            t0 = time.time()
            rc = run_one(mode, seed, args)
            dur = (time.time() - t0) / 60
            summary.append((mode, seed, "ok" if rc == 0 else f"failed(rc={rc})", dur))

    print("\n========== SUMMARY ==========")
    for mode, seed, status, dur in summary:
        print(f"  {mode:9s} seed={seed}  {status:18s}  {dur:5.1f} min")


if __name__ == "__main__":
    main()
