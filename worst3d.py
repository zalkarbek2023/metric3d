"""Sheet of the worst M1 cases (car/person, matched, enough LiDAR): image crop with our box (green), projected GT box
(white), visible LiDAR points in the GT box, and the depth map crop. usage: python worst3d.py rows.csv out.jpg [--n 12]"""
import argparse, os
import numpy as np, cv2
from ultralytics import YOLO
import kitti, score3d
from run3d import depth_map

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=12); ap.add_argument("--e", type=int, default=2)
ap.add_argument("--variant", default="letterbox")
ap.add_argument("--min-z", type=float, default=0)
a = ap.parse_args()
cal = kitti.calib(); K = cal["K"]
rows = [r for r in score3d.load(a.csv) if r["gt_cls"] in score3d.CLASSES and r["cls"] in score3d.CLASSES[r["gt_cls"]]
        and r["lidar_z"] and int(r["lidar_n"]) >= 10 and r[f"z_{a.e}"] and float(r["lidar_z"]) >= a.min_z]
rows.sort(key=lambda r: -abs(float(r[f"z_{a.e}"]) / float(r["lidar_z"]) - 1))
dep = YOLO(os.path.expanduser("~/flatreal/engines/yolo26x-depth-fp16.engine"), task="depth")
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
tiles, cache = [], {}
for r in rows[: a.n]:
    n, f = r["drive"], int(r["frame"])
    img = cv2.imread(os.path.join(kitti.drive_dir(n), "image_02", "data", f"{f:010d}.png"))
    H, W = img.shape[:2]
    d = depth_map(dep, img, a.variant, 768)
    o = next(o for o in kitti.tracklets(n, cal)[f] if o["tid"] == int(r["gt_tid"]))
    pc, pv = kitti.velo(n, f, cal)
    L = kitti.lidar_in_box(o, pc, pv, K, (W, H))
    vis = img.copy()
    for (u, v) in kitti.project(L, K).astype(int):
        cv2.circle(vis, (u, v), 1, (0, 0, 255), -1)
    q = kitti.project(o["corners"], K).astype(int)
    for i, j in EDGES:
        cv2.line(vis, tuple(q[i]), tuple(q[j]), (255, 255, 255), 1)
    x0, y0, x1, y1 = (int(float(r[k])) for k in ("x0", "y0", "x1", "y1"))
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
    v = d[np.isfinite(d)]
    dc = cv2.applyColorMap((255 * (1 - np.clip(d / 60, 0, 1))).astype(np.uint8), cv2.COLORMAP_TURBO)
    pad = 40
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    pane = np.hstack([vis[Y0:Y1, X0:X1], dc[Y0:Y1, X0:X1]])
    s = 200 / pane.shape[0]
    pane = cv2.resize(pane, None, fx=s, fy=s)
    pane = cv2.copyMakeBorder(pane, 0, 0, 0, max(0, 900 - pane.shape[1]), cv2.BORDER_CONSTANT)[:, :900]
    txt = f"{n}#{f} {r['cls']} ours {float(r[f'z_{a.e}']):.1f} lidar {float(r['lidar_z']):.1f} gtc {float(r['gt_z']):.1f} occ{r['occ']} n{r['lidar_n']}"
    cv2.putText(pane, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    tiles.append(pane)
cv2.imwrite(a.out, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("wrote", a.out, len(tiles))
