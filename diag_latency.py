"""Measure per-step latency breakdown."""
import time, numpy as np
from adb_backend import AdbMotionEventBackend
from orchestrate import EmulatorEndpoint
from vision import detect_pose
from reward import ScoreboardDiffReward

ep = EmulatorEndpoint(adb_serial="emulator-6558", minicap_port=0, minitouch_port=0,
                     snapshot_names=("clean_boot","clean_boot_lac"))
b = AdbMotionEventBackend(ep)
b.setup()
b.load_snapshot("clean_boot")
time.sleep(2)
for _ in range(2):
    b.send_tap(1493, 918); time.sleep(0.5)
    b.send_tap(1170, 793); time.sleep(0.5)
    b.send_tap(1366, 793); time.sleep(2.0)

rwd = ScoreboardDiffReward(); rwd.reset()
times = {"send_action": [], "grab_frame": [], "pose": [], "reward": [], "total": []}
N = 40
print(f"timing {N} steps...")
for i in range(N):
    t0 = time.perf_counter()
    action = i % 2
    t1 = time.perf_counter()
    b.send_action(action, hold_ms=33)
    t2 = time.perf_counter()
    rgb = b.grab_frame()
    t3 = time.perf_counter()
    pose = detect_pose(rgb)
    t4 = time.perf_counter()
    _ = rwd(rgb)
    t5 = time.perf_counter()
    times["send_action"].append(t2-t1)
    times["grab_frame"].append(t3-t2)
    times["pose"].append(t4-t3)
    times["reward"].append(t5-t4)
    times["total"].append(t5-t1)
print(f"\n--- mean ± std (ms) over {N} steps ---")
for k, v in times.items():
    arr = np.array(v) * 1000
    print(f"  {k:>14}: {arr.mean():6.1f} ± {arr.std():5.1f}  (min {arr.min():.1f}, max {arr.max():.1f})")
print(f"\nSteps/sec: {N / sum(times['total']):.1f}")
b.teardown()
