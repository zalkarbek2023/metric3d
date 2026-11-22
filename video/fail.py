"""The failure shot for the video: 0084 frame 381, track GT #27 (worst M1 case of the test). Photo crop with our box
(orange), the human-labelled 3D box (white) and LiDAR points in it (white), plus the YOLO26-depth crop (turbo, 0-60 m).
usage: python fail.py data/test_letterbox.csv vid"""
import os, sys
import numpy as np, cv2
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitti
from run3d import depth_map

csv, out = sys.argv[1], sys.argv[2]
n, f, tid = "0084", 381, 27
import csv as C
row = next(r for r in C.DictReader(open(csv)) if r["drive"] == n and int(r["frame"]) == f and r["gt_tid"] == str(tid))
cal = kitti.calib(); K = cal["K"]
img = cv2.imread(os.path.join(kitti.drive_dir(n), "image_02", "data", f"{f:010d}.png"))
H, W = img.shape[:2]
d = depth_map(YOLO(os.path.expanduser("~/flatreal/engines/yolo26x-depth-fp16.engine"), task="depth"), img, "letterbox", 768)
o = next(o for o in kitti.tracklets(n, cal)[f] if o["tid"] == tid)
pc, pv = kitti.velo(n, f, cal)
L = kitti.lidar_in_box(o, pc, pv, K, (W, H))
S = 4
X0, Y0, X1, Y1 = 452, 112, 752, 312  # 300 x 200 px around the car -> 1200 x 800
up = lambda p: (int((p[0] - X0) * S), int((p[1] - Y0) * S))
vis = cv2.resize(img[Y0:Y1, X0:X1], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
q = kitti.project(o["corners"], K)
for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
    cv2.line(vis, up(q[i]), up(q[j]), (255, 255, 255), 3, cv2.LINE_AA)
for p in kitti.project(L, K):
    cv2.circle(vis, up(p), 5, (255, 255, 255), -1, cv2.LINE_AA)
b = [float(row[k]) for k in ("x0", "y0", "x1", "y1")]
cv2.rectangle(vis, up(b[:2]), up(b[2:]), (76, 138, 255), 4, cv2.LINE_AA)
cv2.imwrite(os.path.join(out, "fail_photo.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
dc = cv2.applyColorMap((255 * (1 - np.clip(d / 60, 0, 1))).astype(np.uint8), cv2.COLORMAP_TURBO)
dv = cv2.resize(dc[Y0:Y1, X0:X1], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
cv2.rectangle(dv, up(b[:2]), up(b[2:]), (255, 255, 255), 4, cv2.LINE_AA)
cv2.imwrite(os.path.join(out, "fail_depth.jpg"), dv, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("ours z_2", row["z_2"], "lidar", row["lidar_z"], "gt centre z", row["gt_z"], "lidar pts", len(L))
