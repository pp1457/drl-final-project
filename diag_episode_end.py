"""Step the real env, watch when episode_done flips True, save the frame."""
import cv2, numpy as np, time, sys
import gymnasium as gym
sys.path.insert(0, "/tmp2/b12902046/DRL_final_project")
from train import make_env_fn

env_fn = make_env_fn("bouncy", rank=0, frame_stack=4, backend="adb-motionevent", frame_skip=1)
env = env_fn()
obs, info = env.reset(seed=7)
print("reset OK")
# Save the post-reset frame to see what state we're in
unwrapped = env.unwrapped
rgb = unwrapped.backend.grab_frame()
cv2.imwrite("/tmp/diag_post_reset.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print("saved post-reset frame: /tmp/diag_post_reset.png")

for step in range(30):
    action = 1  # always PRESS
    obs, r, term, trunc, info = env.step(action)
    if term or trunc:
        print(f"episode ended at step {step+1}: term={term} trunc={trunc} reward={r}")
        rgb_end = unwrapped.backend.grab_frame()
        cv2.imwrite(f"/tmp/diag_end_step_{step+1}.png", cv2.cvtColor(rgb_end, cv2.COLOR_RGB2BGR))
        # Check red pixels
        y0, y1 = 20, 130
        band = rgb_end[y0:y1]
        hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
        red1 = cv2.inRange(hsv, np.array([0, 180, 100]), np.array([8, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 180, 100]), np.array([180, 255, 255]))
        red = cv2.bitwise_or(red1, red2)
        print(f"  red pixels in band: {int(red.sum() // 255)} (threshold 1500)")
        break
else:
    print("ran 30 steps without termination")
