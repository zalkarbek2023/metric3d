"""TensorRT FP16 latency of the whole 3D pipeline on val frames (never on test, see PREREG.md):
YOLO26x-seg + bytetrack, YOLO26x-depth (768 letterbox), per-detection mask depth + back-projection, road plane
(RANSAC) and box-bottom contact. The same engines and settings run3d.py uses; no LiDAR / GT work is timed.
usage: python latency3d.py [--drives 0086 0029] [--n 300]
"""
import argparse, os, time
import numpy as np, cv2, torch
from ultralytics import YOLO
import kitti
from run3d import COCO, road_plane, contact, depth_map

ap = argparse.ArgumentParser()
ap.add_argument("--drives", nargs="+", default=["0086", "0029"])
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
eng = os.path.expanduser("~/flatreal/engines")
dep = YOLO(f"{eng}/yolo26x-depth-fp16.engine", task="depth")
K = kitti.calib()["K"]
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
T = {"seg+track": [], "depth": [], "3d": [], "total": []}
ndet = []
for n in a.drives:
    det = YOLO(f"{eng}/yolo26x-seg-fp16.engine", task="segment")
    rng = np.random.default_rng(0)
    img_dir = os.path.join(kitti.drive_dir(n), "image_02", "data")
    for k, fn in enumerate(sorted(os.listdir(img_dir))):
        if k >= 20 + a.n // len(a.drives):  # the same number of timed frames from every drive
            break
        img = cv2.imread(os.path.join(img_dir, fn))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = det.track(img, persist=True, tracker="bytetrack.yaml", classes=list(COCO), conf=0.25, imgsz=640,
                      retina_masks=True, verbose=False)[0]
        torch.cuda.synchronize(); t1 = time.perf_counter()
        d = depth_map(dep, img, "letterbox", 768)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        m = 0
        if r.masks is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            masks = r.masks.data.cpu().numpy().astype(np.uint8)
            m = len(boxes)
            plane = road_plane(d, cv2.dilate(masks.max(0), np.ones((9, 9), np.uint8)).astype(bool), K, rng)
            for i, b in enumerate(boxes):
                ys, xs = np.nonzero(cv2.erode(masks[i], kern))
                z = d[ys, xs]
                if len(z) >= 5:
                    np.median(z), np.median((xs - cx) * z / fx)
                if plane is not None:
                    contact(plane, (b[0] + b[2]) / 2, b[3], K)
        t3 = time.perf_counter()
        if k >= 20:  # warm-up per drive (new tracker)
            for key, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)):
                T[key].append(1000 * v)
            ndet.append(m)
print(f"{torch.cuda.get_device_name()}  {len(ndet)} KITTI val frames 1242x375 ({' '.join(a.drives)}), seg 640 / depth 768, "
      f"FP16 engines, objects per frame median {np.median(ndet):.0f} (max {max(ndet)})")
has = np.array(ndet) > 0
for key, v in T.items():
    v = np.array(v)
    print(f"{key:9s} all frames median {np.median(v):6.1f} ms  p95 {np.percentile(v, 95):6.1f} ms  |  "
          f"frames with objects ({has.sum()}) median {np.median(v[has]):6.1f} ms  p95 {np.percentile(v[has], 95):6.1f} ms")
