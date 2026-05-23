"""Capture frame, crop the current ROIs and a wider scan band to find true scoreboard."""
import cv2, numpy as np, time
from adb_backend import AdbBackend
from orchestrate import EmulatorEndpoint

ep = EmulatorEndpoint(adb_serial="emulator-6558", minicap_port=0, minitouch_port=0,
                     snapshot_names=("clean_boot","clean_boot_lac"))
b = AdbBackend(ep)
b.load_snapshot("clean_boot")
time.sleep(3)
for _ in range(2):
    b.send_tap(1493, 918); time.sleep(0.5)
    b.send_tap(1170, 793); time.sleep(0.5)
    b.send_tap(1366, 793); time.sleep(2.0)

rgb = b.grab_frame()
print(f"frame: {rgb.shape}")
cv2.imwrite("/tmp/roi_g.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# Save the current "CHI" and "HOU" ROIs separately
chi_roi = rgb[260:360, 970:1120]
hou_roi = rgb[260:360, 800:950]
cv2.imwrite("/tmp/roi_chi_current.png", cv2.cvtColor(chi_roi, cv2.COLOR_RGB2BGR))
cv2.imwrite("/tmp/roi_hou_current.png", cv2.cvtColor(hou_roi, cv2.COLOR_RGB2BGR))

# Also save a wider band y=200-500 to see what's there
cv2.imwrite("/tmp/roi_band_200_500.png", cv2.cvtColor(rgb[200:500, 600:1700], cv2.COLOR_RGB2BGR))

# Save full top half
cv2.imwrite("/tmp/roi_top_half.png", cv2.cvtColor(rgb[0:540, :], cv2.COLOR_RGB2BGR))
print("saved /tmp/roi_g.png, roi_chi_current.png, roi_hou_current.png, roi_band_200_500.png, roi_top_half.png")
