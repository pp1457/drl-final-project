"""Load snapshot, advance to gameplay, capture frame + measure score-region pixels.
Then crop scoreboard area and save zoomed copy so we can read off ROI coords."""
import cv2, numpy as np, time
from adb_backend import AdbBackend
from orchestrate import load_endpoints, EmulatorEndpoint

# Use emulator-6558 directly
ep = EmulatorEndpoint(adb_serial="emulator-6558", minicap_port=0, minitouch_port=0,
                     snapshot_names=("clean_boot","clean_boot_lac"))
b = AdbBackend(ep)
b.load_snapshot("clean_boot")
time.sleep(3)

# Advance through PLAY/NEXT_QUARTER/REMATCH to get into gameplay
for _ in range(2):
    b.send_tap(1493, 918); time.sleep(0.5)
    b.send_tap(1170, 793); time.sleep(0.5)
    b.send_tap(1366, 793); time.sleep(2.0)

# Capture
rgb = b.grab_frame()
print(f"frame shape: {rgb.shape}")
cv2.imwrite("/tmp/roi_gameplay.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# Crop top scoreboard region for zoomed inspection
top = rgb[0:200, 700:1700]  # estimated scoreboard area
cv2.imwrite("/tmp/roi_top_zoom.png", cv2.cvtColor(top, cv2.COLOR_RGB2BGR))
print(f"saved roi_gameplay.png ({rgb.shape}) and roi_top_zoom.png ({top.shape})")

# Print HSV stats for current chi/hou ROIs to verify they're NOT on the scoreboard
roi_chi = rgb[260:360, 970:1120]
roi_hou = rgb[260:360, 800:950]
print(f"\ncurrent chi_score_roi (y=260-360, x=970-1120) shape={roi_chi.shape}")
print(f"  mean RGB: {roi_chi.mean(axis=(0,1))}")
print(f"current hou_score_roi (y=260-360, x=800-950) shape={roi_hou.shape}")
print(f"  mean RGB: {roi_hou.mean(axis=(0,1))}")
