"""Record a short demo video of an agent playing Bouncy Basketball.

Captures full-res screencaps from the emulator at each step, runs the
agent in deterministic mode, and writes a numbered PNG per step. The
caller is expected to convert the PNG sequence to mp4 with ffmpeg.
"""
import argparse, os, subprocess, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from eval import load_agent, _make_bouncy_env_for_serial, _wrap_env_for_eval

N_ACTIONS = 2


def grab_fullres(serial: str) -> bytes:
    """Raw PNG bytes via `adb exec-out screencap -p`."""
    cmd = ["adb", "-s", serial, "exec-out", "screencap", "-p"]
    out = subprocess.run(cmd, capture_output=True, timeout=10.0, check=True)
    return out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--serial", required=True, help="emulator-XXXX")
    ap.add_argument("--steps", type=int, default=120,
                    help="Number of env steps to record (~0.5s each).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")
    agent, ckpt_args = load_agent(args.checkpoint, device)
    use_charge = bool(getattr(agent, "use_charge_dim", False))

    backend = ckpt_args.get("backend", "adb-motionevent")
    frame_skip = ckpt_args.get("frame_skip", 1)
    env = _make_bouncy_env_for_serial(
        args.serial,
        backend_name=backend,
        frame_skip=frame_skip,
        max_episode_steps=args.steps + 50,
    )
    env = _wrap_env_for_eval(env, frame_stack=ckpt_args.get("frame_stack", 4))

    obs, info = env.reset(seed=0)
    charge_dur = 0.0
    print(f"[record] reset done at {time.strftime('%H:%M:%S')}, "
          f"capturing {args.steps} steps...", flush=True)

    for t in range(args.steps):
        # Save full-res frame BEFORE the action so we see the state
        # the agent saw.
        try:
            png = grab_fullres(args.serial)
            with open(os.path.join(args.out_dir, f"frame_{t:05d}.png"), "wb") as f:
                f.write(png)
        except Exception as e:
            print(f"[record] frame {t}: capture failed: {e}", flush=True)

        o = (torch.as_tensor(np.asarray(obs), device=device)
                 .squeeze(-1).unsqueeze(0).float())
        with torch.no_grad():
            z = agent.encode(o)
            if use_charge:
                cd = torch.tensor([[charge_dur]], device=device,
                                  dtype=torch.float32)
                h = agent._head_input(z, cd)
            else:
                h = z
            logits = agent.actor(h)
            action = int(logits.argmax(-1).item())

        obs, r, term, trunc, info = env.step(action)
        if use_charge and isinstance(info, dict) and "charge_duration" in info:
            charge_dur = float(info["charge_duration"])

        if (t + 1) % 20 == 0:
            print(f"[record] step {t+1}/{args.steps} action={action} "
                  f"reward={r:+.1f} term={term} trunc={trunc}", flush=True)

        if term or trunc:
            print(f"[record] episode ended at step {t+1}", flush=True)
            obs, info = env.reset(seed=t + 1)
            charge_dur = 0.0

    env.close()
    print(f"[record] DONE, wrote {args.steps} PNGs to {args.out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
