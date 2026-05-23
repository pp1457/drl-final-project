"""End-to-end verification: play random actions, confirm CHI + HOU both score,
vision detects players, and reward signal works correctly."""
import cv2, numpy as np, time
from adb_backend import AdbBackend
from orchestrate import EmulatorEndpoint
from reward import ScoreboardDiffReward, is_game_over
from vision import detect_pose

ep = EmulatorEndpoint(adb_serial="emulator-6558", minicap_port=0, minitouch_port=0,
                     snapshot_names=("clean_boot","clean_boot_lac"))
b = AdbBackend(ep)
b.load_snapshot("clean_boot")
time.sleep(3)
for _ in range(2):
    b.send_tap(1493, 918); time.sleep(0.5)
    b.send_tap(1170, 793); time.sleep(0.5)
    b.send_tap(1366, 793); time.sleep(2.0)

rwd = ScoreboardDiffReward()
rwd.reset()
chi_events = 0; hou_events = 0
last_score = 0.0
pose_detected = {"ball": 0, "player": 0, "opp": 0}
game_over_hits = 0
N = 100
print(f"playing {N} steps of random actions...")
for step in range(N):
    if step % 20 == 0:
        print(f"  step {step}: chi={chi_events} hou={hou_events} game_over_hits={game_over_hits}", flush=True)
    action = np.random.randint(0, 2)
    # Tap or release
    if action == 1:
        b.send_tap(1170, 793)
    time.sleep(0.05)
    rgb = b.grab_frame()
    score, done = rwd(rgb)
    delta = score - last_score
    if delta > 0.5: chi_events += 1
    elif delta < -0.5: hou_events += 1
    last_score = score
    if done: game_over_hits += 1
    pose = detect_pose(rgb)
    for k in pose_detected:
        if pose.get(k) is not None: pose_detected[k] += 1

print(f"\n=== RESULTS over {N} random steps ===")
print(f"CHI score events: {chi_events}")
print(f"HOU score events: {hou_events}")
print(f"  (Both should be >0 if ROIs correct. With opponent_score_weight=-1, net_score={last_score:+.1f})")
print(f"is_game_over fired: {game_over_hits}  (expected ~0-1 if no real game over; high if false positive)")
print(f"Pose detection (out of 250 frames):")
for k, n in pose_detected.items():
    print(f"  {k}: {n} / 250 ({100*n/250:.0f}%)")
# Final state
final = b.grab_frame()
cv2.imwrite("/tmp/diag_verify_final.png", cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
print("saved /tmp/diag_verify_final.png")
