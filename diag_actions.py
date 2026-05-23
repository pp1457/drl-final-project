"""Test if our PRESS action actually controls the game.
If always-press scores differently from never-press, our taps matter.
If they score the same, taps are ineffective."""
import cv2, numpy as np, time
from adb_backend import AdbMotionEventBackend
from orchestrate import EmulatorEndpoint
from reward import ScoreboardDiffReward

ep = EmulatorEndpoint(adb_serial="emulator-6558", minicap_port=0, minitouch_port=0,
                     snapshot_names=("clean_boot","clean_boot_lac"))

def run_trial(label, action_fn, N=200):
    b = AdbMotionEventBackend(ep)
    b.setup()
    b.load_snapshot("clean_boot")
    time.sleep(3)
    # Advance to gameplay
    for _ in range(2):
        b.send_tap(1493, 918); time.sleep(0.5)
        b.send_tap(1170, 793); time.sleep(0.5)
        b.send_tap(1366, 793); time.sleep(2.0)
    rwd = ScoreboardDiffReward(); rwd.reset()
    chi = hou = 0
    for i in range(N):
        action = action_fn(i)
        b.send_action(action, hold_ms=33)
        rgb = b.grab_frame()
        score, done = rwd(rgb)
        # delta
        if i == 0:
            prev = score
        d = score - prev
        if d > 0.5: chi += 1
        elif d < -0.5: hou += 1
        prev = score
    b.teardown()
    final = b.grab_frame()
    cv2.imwrite(f"/tmp/diag_act_{label}.png", cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"{label:>16}: CHI events={chi}  HOU events={hou}  net={chi-hou}  net_score={score:+.1f}")
    return chi, hou

# 200 steps × 33ms = 6.6s of game time per trial
print("=== always PRESS (action=1) ===")
chi_p, hou_p = run_trial("always_press", lambda i: 1, N=200)
print()
print("=== never PRESS (action=0) ===")
chi_n, hou_n = run_trial("never_press", lambda i: 0, N=200)
print()
print("=== alternating (PRESS every 4 steps) ===")
chi_a, hou_a = run_trial("alternating", lambda i: 1 if i % 4 == 0 else 0, N=200)
print()
print(f"=== SUMMARY ===")
print(f"always_press: chi={chi_p}, hou={hou_p}")
print(f"never_press:  chi={chi_n}, hou={hou_n}")
print(f"alternating:  chi={chi_a}, hou={hou_a}")
print(f"\nIf always_press chi >> never_press chi, our taps cause CHI to shoot.")
print(f"If similar, our taps don't matter — game plays autonomously.")
