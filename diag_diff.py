"""Capture frames before/after a known score change, measure mean-abs-diff
to calibrate the score-detection threshold."""
import cv2, numpy as np, time
from adb_backend import AdbBackend
from orchestrate import EmulatorEndpoint

ep = EmulatorEndpoint(adb_serial="emulator-6558", minicap_port=0, minitouch_port=0,
                     snapshot_names=("clean_boot","clean_boot_lac"))
b = AdbBackend(ep)
b.load_snapshot("clean_boot")
time.sleep(3)
# Advance to gameplay
for _ in range(2):
    b.send_tap(1493, 918); time.sleep(0.5)
    b.send_tap(1170, 793); time.sleep(0.5)
    b.send_tap(1366, 793); time.sleep(2.0)

# Use a tight digit-only ROI
chi_roi = (320, 380, 1240, 1360)
hou_roi = (320, 380, 980, 1100)

# Capture 50 consecutive frames at 100ms intervals; report frame-to-frame diffs
prev_chi = None
prev_hou = None
print("frame | chi_diff | hou_diff | screen_state")
print("-" * 70)
diffs_chi = []
diffs_hou = []
for i in range(60):
    if i % 3 == 0:
        b.send_tap(1170, 793)  # press occasionally
    time.sleep(0.08)
    rgb = b.grab_frame()
    chi = rgb[chi_roi[0]:chi_roi[1], chi_roi[2]:chi_roi[3]].astype(np.int16)
    hou = rgb[hou_roi[0]:hou_roi[1], hou_roi[2]:hou_roi[3]].astype(np.int16)
    if prev_chi is not None:
        cd = float(np.abs(chi - prev_chi).mean())
        hd = float(np.abs(hou - prev_hou).mean())
        diffs_chi.append(cd)
        diffs_hou.append(hd)
        mark = ""
        if cd > 5: mark += " CHI?"
        if hd > 5: mark += " HOU?"
        print(f"{i:>3} | {cd:>7.2f} | {hd:>7.2f} |{mark}")
    prev_chi = chi; prev_hou = hou

import statistics
print()
print(f"chi: mean={statistics.mean(diffs_chi):.2f} max={max(diffs_chi):.2f} std={statistics.pstdev(diffs_chi):.2f}")
print(f"hou: mean={statistics.mean(diffs_hou):.2f} max={max(diffs_hou):.2f} std={statistics.pstdev(diffs_hou):.2f}")
final = b.grab_frame()
cv2.imwrite("/tmp/diag_diff_final.png", cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
print("saved /tmp/diag_diff_final.png")
