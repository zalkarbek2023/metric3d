"""Detector + tracker + depth on KITTI raw drives; one CSV row per detection, with the matched 3D tracklet and LiDAR.

usage: python run3d.py --drives 0020 0002 --variant letterbox --out data/rows_val_letterbox.csv
       python run3d.py --drives ... --depth-dir data/dp2   # precomputed <drive>/<frame>.npy maps (Depth Pro 2 reference)
Depth variants: letterbox = Ultralytics default predict on the 1242x375 frame; stretch = frame resized to a 768x768
square first (how depth val feeds images), the map resized back.
Per detection and mask erosion e (px): n_e, z_e = median depth over the eroded mask (m), x_e / y_e = medians of the
back-projected points (camera-2 frame). Matching to tracklets: greedy 2D IoU >= 0.5 against the projected 3D boxes.
"""
import argparse, csv, os, time
import numpy as np, cv2
from ultralytics import YOLO
import kitti

COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
EROSIONS = (0, 2, 5)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def match(dets, gts):
    pairs = sorted(((iou(d, g), i, j) for i, d in enumerate(dets) for j, g in enumerate(gts)), reverse=True)
    out, ud, ug = {}, set(), set()
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i not in ud and j not in ug:
            ud.add(i), ug.add(j)
            out[i] = (j, v)
    return out


def road_plane(d, union, K, rng, n=3000, iters=200, tol=0.08, max_z=30.0):
    """Road plane from the depth map: pixels below the horizon outside all detection masks, RANSAC with the normal
    within 25 deg of the camera's vertical, least squares on inliers. Returns (n, d0) with n pointing up (n_y < 0)
    and d0 = camera height above the road (m), or None."""
    H, W = d.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.nonzero(~union[int(cy) + 30:, :])
    ys = ys + int(cy) + 30
    if len(ys) < 100:
        return None
    k = rng.choice(len(ys), min(n, len(ys)), replace=False)
    ys, xs = ys[k], xs[k]
    z = d[ys, xs]
    ok = np.isfinite(z) & (z > 0.5) & (z < max_z)
    P = np.stack([(xs[ok] - cx) * z[ok] / fx, (ys[ok] - cy) * z[ok] / fy, z[ok]], 1)
    if len(P) < 100:
        return None
    best, bn = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        nv = np.cross(b - a, c - a)
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv = nv / nn
        if abs(nv[1]) < np.cos(np.radians(25)):
            continue
        inl = np.abs(P @ nv - nv @ a) < tol
        if inl.sum() > bn:
            bn, best = inl.sum(), inl
    if best is None or bn < 50:
        return None
    Q = P[best]
    c0 = Q.mean(0)
    nv = np.linalg.svd(Q - c0, full_matrices=False)[2][-1]
    if nv[1] > 0:
        nv = -nv
    return nv, -nv @ c0


def contact(plane, u, v, K):
    """Where the ray through pixel (u, v) hits the road plane: (x, z) in camera-2 frame, or None."""
    nv, d0 = plane
    r = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    den = nv @ r
    if den >= -1e-6:
        return None
    t = -d0 / den
    return t * r[0], t * r[2]


def depth_map(dep, img, variant, imgsz):
    H, W = img.shape[:2]
    src = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA) if variant == "stretch" else img
    d = dep.predict(src, imgsz=imgsz, verbose=False)[0].depth.data.float().cpu().numpy()
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drives", nargs="+", required=True)
    ap.add_argument("--variant", default="letterbox", choices=["letterbox", "stretch"])
    ap.add_argument("--depth-dir", default=None)
    ap.add_argument("--det", default=os.path.expanduser("~/flatreal/engines/yolo26x-seg-fp16.engine"))
    ap.add_argument("--depth", default=os.path.expanduser("~/flatreal/engines/yolo26x-depth-fp16.engine"))
    ap.add_argument("--imgsz-depth", type=int, default=768)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cal = kitti.calib()
    K = cal["K"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dep = None if a.depth_dir else YOLO(a.depth, task="depth")
    cols = ["drive", "frame", "det", "track", "cls", "conf", "x0", "y0", "x1", "y1"]
    cols += [f"{q}_{e}" for e in EROSIONS for q in ("n", "z", "x", "y")]
    cols += ["cam_h", "g_x", "g_z", "cut_bottom"]
    cols += ["gt_tid", "gt_cls", "gt_type", "iou", "occ", "trunc", "gt_x", "gt_y", "gt_z", "gt_h", "gt_w", "gt_l",
             "lidar_n", "lidar_z"]
    t0, nrow = time.time(), 0
    rng = np.random.default_rng(0)
    kernels = {e: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)) for e in EROSIONS if e}
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, cols)
        w.writeheader()
        for n in a.drives:
            det = YOLO(a.det, task="segment")  # fresh tracker state per drive
            tr = kitti.tracklets(n, cal)
            img_dir = os.path.join(kitti.drive_dir(n), "image_02", "data")
            frames = sorted(int(f[:-4]) for f in os.listdir(img_dir))
            if a.max_frames:
                frames = frames[: a.max_frames]
            for f in frames:
                img = cv2.imread(os.path.join(img_dir, f"{f:010d}.png"))
                H, W = img.shape[:2]
                r = det.track(img, persist=True, tracker="bytetrack.yaml", classes=list(COCO), conf=0.25, imgsz=640,
                              retina_masks=True, verbose=False)[0]
                if r.masks is None or len(r.boxes) == 0:
                    continue
                if a.depth_dir:
                    d = np.load(os.path.join(a.depth_dir, n, f"{f:010d}.npy")).astype(np.float32)
                    if d.shape != (H, W):
                        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
                else:
                    d = depth_map(dep, img, a.variant, a.imgsz_depth)
                pc, pv = kitti.velo(n, f, cal)
                gts = [o for o in tr.get(f, []) if o["trunc"] not in (2, 3)]
                gb = [kitti.box2d(o, K, (W, H)) for o in gts]
                gts = [o for o, b in zip(gts, gb) if b is not None]
                gb = [b for b in gb if b is not None]
                boxes = r.boxes.xyxy.cpu().numpy()
                masks = r.masks.data.cpu().numpy().astype(np.uint8)
                ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(boxes)
                m = match([tuple(b) for b in boxes], gb)
                union = cv2.dilate(masks.max(0), np.ones((9, 9), np.uint8)).astype(bool)
                plane = road_plane(d, union, K, rng)
                for i, b in enumerate(boxes):
                    row = {"drive": n, "frame": f, "det": i, "track": int(ids[i]), "cls": COCO[int(r.boxes.cls[i])],
                           "conf": round(float(r.boxes.conf[i]), 3),
                           "x0": round(float(b[0]), 1), "y0": round(float(b[1]), 1),
                           "x1": round(float(b[2]), 1), "y1": round(float(b[3]), 1)}
                    for e in EROSIONS:
                        mk = cv2.erode(masks[i], kernels[e]) if e else masks[i]
                        ys, xs = np.nonzero(mk)
                        z = d[ys, xs]
                        ok = np.isfinite(z) & (z > 0)
                        ys, xs, z = ys[ok], xs[ok], z[ok]
                        if len(z) < 5:
                            row.update({f"n_{e}": len(z), f"z_{e}": "", f"x_{e}": "", f"y_{e}": ""})
                            continue
                        row.update({f"n_{e}": len(z), f"z_{e}": round(float(np.median(z)), 3),
                                    f"x_{e}": round(float(np.median((xs - cx) * z / fx)), 3),
                                    f"y_{e}": round(float(np.median((ys - cy) * z / fy)), 3)})
                    if plane is not None:
                        g = contact(plane, (b[0] + b[2]) / 2, b[3], K)
                        row.update({"cam_h": round(float(plane[1]), 3)})
                        if g is not None:
                            row.update({"g_x": round(float(g[0]), 3), "g_z": round(float(g[1]), 3)})
                    row["cut_bottom"] = int(b[3] >= H - 3)
                    if i in m:
                        j, v = m[i]
                        o = gts[j]
                        L = kitti.lidar_in_box(o, pc, pv, K, (W, H))
                        row.update({"gt_tid": o["tid"], "gt_cls": o["cls"], "gt_type": o["type"], "iou": round(v, 3),
                                    "occ": o["occ"], "trunc": o["trunc"],
                                    "gt_x": round(float(o["centre"][0]), 3), "gt_y": round(float(o["centre"][1]), 3),
                                    "gt_z": round(float(o["centre"][2]), 3), "gt_h": o["h"], "gt_w": o["w"], "gt_l": o["l"],
                                    "lidar_n": len(L), "lidar_z": round(float(np.median(L[:, 2])), 3) if len(L) else ""})
                    w.writerow(row)
                    nrow += 1
            print(f"{n}: {len(frames)} frames, {nrow} rows so far, {time.time() - t0:.0f} s", flush=True)
    print(f"done -> {a.out} ({nrow} rows, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
