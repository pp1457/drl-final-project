"""Step the env until termination, log overlay_streak progression + capture
frames around termination. Helps diagnose what red-text state is closing the
episode at ~83 steps."""
import cv2, numpy as np, time, sys
import gymnasium as gym
sys.path.insert(0, "/tmp2/b12902046/DRL_final_project")
from train import make_env_fn

# Use --serial style endpoint by importing eval helpers
from eval import _make_bouncy_env_for_serial, _wrap_env_for_eval

# Pick training args to match: backend=adb-motionevent, frame_skip=1
inner_env = _make_bouncy_env_for_serial("emulator-6558", "adb-motionevent", 1)
env = _wrap_env_for_eval(inner_env, frame_stack=4)
obs, info = env.reset(seed=11)
unwrapped = env.unwrapped
print(f"reset OK. max_episode_steps={unwrapped.max_episode_steps}, OVERLAY_LIMIT={unwrapped._OVERLAY_STREAK_LIMIT}")

# Snap a frame every step; save the last 8 to disk
frame_buf = []
streak_log = []

for step in range(150):
    action = np.random.randint(0, 2)  # random press/no-press
    obs, r, term, trunc, info = env.step(action)
    streak_log.append((step+1, unwrapped._overlay_streak, r))
    rgb = unwrapped.backend.grab_frame()
    frame_buf.append((step+1, rgb))
    if len(frame_buf) > 8:
        frame_buf.pop(0)
    if term or trunc:
        print(f"episode ended at step {step+1}: term={term} trunc={trunc} reward={r}")
        # Save the last 8 frames
        for s, fr in frame_buf:
            cv2.imwrite(f"/tmp/diag_short_step_{s:03d}.png", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        print("saved last 8 frames to /tmp/diag_short_step_*.png")
        break
else:
    print("ran 150 steps without termination")
    for s, fr in frame_buf:
        cv2.imwrite(f"/tmp/diag_short_step_{s:03d}.png", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

# Dump streak progression (last 50)
print("\nstreak progression (last 50):")
for s, streak, r in streak_log[-50:]:
    print(f"  step {s:>3}: streak={streak:>3}  reward={r:+.2f}")
