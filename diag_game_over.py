"""One-shot diagnostic: load snapshot, advance into gameplay, measure red-pixel
count in the GAME OVER y-band on the resulting frame. Should be ~0 in true
gameplay; if >1500, our is_game_over false-positives."""
import cv2, numpy as np, time
from adb_backend import AdbBackend
from orchestrate import load_endpoints

ep = load_endpoints()[0]
b = AdbBackend(ep)
b.load_snapshot(ep.snapshot_names[0])
time.sleep(2.0)

# Advance into gameplay
for _ in range(2):
    b.send_tap(1493, 918); time.sleep(0.5)
    b.send_tap(1170, 793); time.sleep(0.5)
    b.send_tap(1366, 793); time.sleep(2.0)

rgb = b.grab_frame()
print("frame shape:", rgb.shape)
y0, y1 = 20, 130
band = rgb[y0:y1]
hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
red1 = cv2.inRange(hsv, np.array([0, 180, 100]), np.array([8, 255, 255]))
red2 = cv2.inRange(hsv, np.array([170, 180, 100]), np.array([180, 255, 255]))
red = cv2.bitwise_or(red1, red2)
n_red = int(red.sum() // 255)
print(f"red pixels in band (y={y0}-{y1}):", n_red, "  threshold: 1500")
print(f"is_game_over would return:", n_red > 1500)
cv2.imwrite("/tmp/eval_frame_in_game.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite("/tmp/eval_frame_band.png", cv2.cvtColor(band, cv2.COLOR_RGB2BGR))
print("saved /tmp/eval_frame_in_game.png and /tmp/eval_frame_band.png")
