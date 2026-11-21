"""What Ultralytics model.calibrate() would choose on the val drives, without building a depth dataset.

calibrate() (ultralytics 8.4.193) fits only a global log scale d' = exp(a*log d + b) against labelled depth and keeps it
only if it beats the raw output on held-out images (select_calibration_cv: 2 folds, margin 0.002, GT < max_depth).
Here the labelled depth is the raw LiDAR scan projected into image_02 (sparser and noisier than the KITTI depth
benchmark maps the model was validated on). Pairs come from our TensorRT engine, which already carries the released
calibration (a, b) = CURRENT, so raw log depth = log(engine) - b. The library's own select_calibration_cv makes the choice.
Output: the factor to multiply our engine depth by (1.0 = keep as released).
usage: python calib_scale.py --drives 0002 0020 0029 0048 0059 0086 [--images 200]
"""
import argparse, json, os
import numpy as np, cv2
from ultralytics import YOLO
from ultralytics.models.yolo.depth.calibrate import select_calibration_cv
import kitti
from run3d import depth_map

CURRENT = (1.0, -0.316650390625)  # cal_a, cal_b inside yolo26x-depth.pt (read from the head buffers)
MAX_DEPTH = 80.0  # depth-kitti.yaml max_depth

ap = argparse.ArgumentParser()
ap.add_argument("--drives", nargs="+", required=True)
ap.add_argument("--images", type=int, default=200)
ap.add_argument("--variant", default="letterbox")
ap.add_argument("--depth", default=os.path.expanduser("~/flatreal/engines/yolo26x-depth-fp16.engine"))
ap.add_argument("--out", default="data/calib_scale.json")
a = ap.parse_args()
cal = kitti.calib()
K = cal["K"]
dep = YOLO(a.depth, task="depth")
frames = [(n, f) for n in a.drives for f in range(len(os.listdir(os.path.join(kitti.drive_dir(n), "image_02", "data"))))]
pick = [frames[i] for i in np.linspace(0, len(frames) - 1, min(a.images, len(frames))).astype(int)]  # even over val
rng = np.random.default_rng(0)
pairs = []
for n, f in pick:
    img = cv2.imread(os.path.join(kitti.drive_dir(n), "image_02", "data", f"{f:010d}.png"))
    H, W = img.shape[:2]
    d = depth_map(dep, img, a.variant, 768)
    P, _ = kitti.velo(n, f, cal)
    uv = kitti.project(P, K)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (P[:, 2] > 1e-3) & (P[:, 2] < MAX_DEPTH)
    pred, gt = d[v[ok], u[ok]], P[ok, 2]
    ok = np.isfinite(pred) & (pred > 1e-3)
    lp, lg = np.log(pred[ok]) - CURRENT[1], np.log(gt[ok])  # raw = engine / exp(b_current)  (a_current = 1)
    if lp.size > 20_000:
        k = rng.choice(lp.size, 20_000, replace=False)
        lp, lg = lp[k], lg[k]
    if lp.size:
        pairs.append((lp, lg))
res = select_calibration_cv(pairs, margin=0.002, current=CURRENT)
factor = float(np.exp(res["b"] - CURRENT[1])) if res["a"] == 1.0 else None
out = {"drives": a.drives, "images": len(pairs), "pixels": int(sum(p[0].size for p in pairs)), "chosen": res["name"],
       "a": res["a"], "b": res["b"], "cv_delta1": res["cv_scores"], "engine_factor": factor,
       "median_ratio_lidar_over_engine": float(np.exp(np.median(np.concatenate([g - (p + CURRENT[1]) for p, g in pairs]))))}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1))
