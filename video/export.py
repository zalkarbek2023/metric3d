"""Export one test-drive segment for the video. Replays the drive from frame 0 exactly like run3d.py (same engines,
tracker, RANSAC seed), so track IDs, boxes and road planes equal the test CSV - checked against it row by row.
Centres and speeds come from the same score3d functions that produced the test numbers.

usage: python export.py --drive 0009 --f0 300 --f1 425 --csv data/test_letterbox.csv --out vid
writes vid/img/NNN.jpg (photo), vid/seg/NNN.jpg (photo + masks), vid/depth.bin (uint16 cm, grid per frame),
       vid/lidar.bin (int16 cm xyz, camera-2 frame), vid/scene.json (objects, GT boxes, planes, speeds, trails)
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np, cv2
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitti, score3d as S
from run3d import COCO, match, road_plane, depth_map

ap = argparse.ArgumentParser()
ap.add_argument("--drive", default="0009")
ap.add_argument("--f0", type=int, default=300)
ap.add_argument("--f1", type=int, default=425)
ap.add_argument("--csv", default="data/test_letterbox.csv")
ap.add_argument("--out", default="vid")
ap.add_argument("--grid", type=int, default=4)  # depth grid step (px)
ap.add_argument("--lidar-max", type=int, default=12000)
a = ap.parse_args()
n = a.drive
for d in ("img", "seg"):
    os.makedirs(os.path.join(a.out, d), exist_ok=True)

cal = kitti.calib()
K = cal["K"]
rows = S.load(a.csv)
csv_f = defaultdict(list)
for r in rows:
    if r["drive"] == n:
        csv_f[int(r["frame"])].append(r)
rc = S.R_CLASS
rec = S.m3_windows([r for r in rows if r["drive"] == n], 2, K, cal, rc)
speed = {(x["track"], x["frame"]): x for x in rec}
poses = kitti.poses(n, cal)
tr = kitti.tracklets(n, cal)
cam_h_all = [float(r["cam_h"]) for r in rows if r["cam_h"]]
seen = set()
cam_h_frames = []  # one value per (drive, frame): the plane is per frame
for r in rows:
    k = (r["drive"], r["frame"])
    if r["cam_h"] and k not in seen:
        seen.add(k)
        cam_h_frames.append(float(r["cam_h"]))

det = YOLO(os.path.expanduser("~/flatreal/engines/yolo26x-seg-fp16.engine"), task="segment")
dep = YOLO(os.path.expanduser("~/flatreal/engines/yolo26x-depth-fp16.engine"), task="depth")
rng = np.random.default_rng(0)
img_dir = os.path.join(kitti.drive_dir(n), "image_02", "data")
COL = {"car": (76, 138, 255), "truck": (76, 138, 255), "bus": (76, 138, 255), "person": (113, 204, 46),
       "bicycle": (102, 209, 255), "motorcycle": (102, 209, 255)}  # BGR
frames_out, depth_out, lidar_out, lidar_idx = [], [], [], []
mism = 0
for f in range(0, a.f1 + 1):
    img = cv2.imread(os.path.join(img_dir, f"{f:010d}.png"))
    H, W = img.shape[:2]
    r = det.track(img, persist=True, tracker="bytetrack.yaml", classes=list(COCO), conf=0.25, imgsz=640,
                  retina_masks=True, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        if f >= a.f0:
            print("frame without detections", f)
        continue
    d = depth_map(dep, img, "letterbox", 768)
    boxes = r.boxes.xyxy.cpu().numpy()
    masks = r.masks.data.cpu().numpy().astype(np.uint8)
    union = cv2.dilate(masks.max(0), np.ones((9, 9), np.uint8)).astype(bool)
    plane = road_plane(d, union, K, rng)
    if f < a.f0:
        continue
    k = f - a.f0
    ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(boxes)
    # --- identity check against the test CSV
    cr = sorted(csv_f[f], key=lambda x: int(x["det"]))
    if len(cr) != len(boxes):
        mism += 1
    for i, row in enumerate(cr[: len(boxes)]):
        if int(row["track"]) != int(ids[i]) or abs(float(row["x0"]) - round(float(boxes[i][0]), 1)) > 0.05 or \
                (row["cam_h"] and plane is not None and abs(float(row["cam_h"]) - round(float(plane[1]), 3)) > 1e-3):
            mism += 1
    cv2.imwrite(os.path.join(a.out, "img", f"{k:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    seg = img.copy()
    for i in range(len(boxes)):
        c = COL[COCO[int(r.boxes.cls[i])]]
        m = masks[i].astype(bool)
        seg[m] = (0.45 * seg[m] + 0.55 * np.array(c)).astype(np.uint8)
        cnt, _ = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg, cnt, -1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(a.out, "seg", f"{k:03d}.jpg"), seg, [cv2.IMWRITE_JPEG_QUALITY, 90])
    g = a.grid
    dg = d[g // 2::g, g // 2::g]
    depth_out.append(np.clip(np.nan_to_num(dg, nan=0) * 100, 0, 65535).astype(np.uint16))
    pc, _ = kitti.velo(n, f, cal)
    uv = kitti.project(pc, K) if len(pc) else np.zeros((0, 2))
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (pc[:, 2] < 80) if len(pc) else []
    P = pc[keep] if len(pc) else pc
    if len(P) > a.lidar_max:
        P = P[np.random.default_rng(f).choice(len(P), a.lidar_max, replace=False)]
    lidar_idx.append([sum(len(x) for x in lidar_out), len(P)])
    lidar_out.append(np.round(P * 100).astype(np.int16))
    objs = []
    for row in cr:
        c = S.our_cls(row)
        if not row["z_2"] or c is None:
            continue
        s = S.surface(row, 2, K)
        cen = {m: [round(float(v), 2) for v in S.push(s[m], rc[m].get(c, 0.0))] for m in S.METHODS}
        o = {"det": int(row["det"]), "track": int(row["track"]), "cls": row["cls"], "conf": float(row["conf"]),
             "box": [float(row[q]) for q in ("x0", "y0", "x1", "y1")], "c": cen,
             "gt_tid": int(row["gt_tid"]) if row["gt_tid"] else None}
        if row["gt_tid"] and S.consistent(row):
            o["err"] = {m: round(float(np.hypot(cen[m][0] - float(row["gt_x"]), cen[m][1] - float(row["gt_z"]))), 2)
                        for m in S.METHODS}
        sp = speed.get((int(row["track"]), f))
        if sp:
            o["kmh"] = {m: round(float(sp[m]), 1) for m in ("gt",) + S.METHODS if np.isfinite(sp[m])}
        objs.append(o)
    gts = []
    for ob in tr.get(f, []):
        if ob["cls"] not in ("car", "person") or ob["trunc"] in (2, 3) or kitti.box2d(ob, K, (W, H)) is None:
            continue
        gts.append({"tid": ob["tid"], "type": ob["type"], "corners": np.round(ob["corners"], 2).tolist(),
                    "centre": np.round(ob["centre"], 2).tolist(), "occ": ob["occ"]})
    frames_out.append({"k": k, "frame": f, "objects": objs, "gt": gts,
                       "plane": None if plane is None else [round(float(v), 4) for v in plane[0]] + [round(float(plane[1]), 3)]})

# trails: each track's hybrid centre over the last 2 s, expressed in the current camera frame through the OXTS poses
hist = defaultdict(dict)
for fr in frames_out:
    for o in fr["objects"]:
        x, z = o["c"]["hybrid"]
        hist[o["track"]][fr["frame"]] = poses[fr["frame"]] @ np.array([x, 0.0, z, 1.0])
for fr in frames_out:
    inv = np.linalg.inv(poses[fr["frame"]])
    for o in fr["objects"]:
        pts = [inv @ hist[o["track"]][j] for j in range(fr["frame"] - 20, fr["frame"] + 1) if j in hist[o["track"]]]
        o["trail"] = [[round(float(p[0]), 2), round(float(p[2]), 2)] for p in pts]
# ego motion in the first frame's camera coordinates (for a world-fixed view if wanted)
inv0 = np.linalg.inv(poses[a.f0])
ego = [np.round((inv0 @ poses[f])[:3, 3], 2).tolist() for f in range(a.f0, a.f1 + 1)]

D = np.stack(depth_out)
D.tofile(os.path.join(a.out, "depth.bin"))
np.concatenate(lidar_out).tofile(os.path.join(a.out, "lidar.bin"))
meta = {"drive": n, "f0": a.f0, "f1": a.f1, "fps": 10, "size": [W, H], "K": np.round(K, 3).tolist(),
        "grid": [int(D.shape[2]), int(D.shape[1])], "grid_step": a.grid, "lidar_idx": lidar_idx, "ego": ego,
        "cam_h_true": S.CAM_H, "cam_h_test_median": round(float(np.median(cam_h_frames)), 3),
        "cam_h_test_p10_p90": [round(float(np.percentile(cam_h_frames, q)), 3) for q in (10, 90)],
        "cam_h_frames": len(cam_h_frames), "frames": frames_out, "csv_mismatches": mism}
json.dump(meta, open(os.path.join(a.out, "scene.json"), "w"), separators=(",", ":"))
print(f"frames {len(frames_out)}  depth {D.shape}  lidar {sum(x[1] for x in lidar_idx)} pts  mismatches vs CSV {mism}")
print("cam_h test median", meta["cam_h_test_median"], meta["cam_h_test_p10_p90"], "over", len(cam_h_frames), "frames")
