"""Score run3d.py rows against LiDAR / 3D tracklets.

M1 - distance to the visible surface: our median mask depth vs the median depth of the LiDAR points inside the
     matched 3D box (no shape assumption, tests only the depth). Abs error (m), abs rel, signed rel (bias), delta<1.25.
B0 - classic baseline with no depth network: box bottom on a flat road, Z = fy * CAM_H / (y1 - cy).
usage: python score3d.py data/rows_val_letterbox.csv [more.csv ...] [--e 2]
"""
import argparse, csv
import numpy as np
import kitti

CAM_H = 1.65  # KITTI camera height above the road (m), from the sensor setup
BINS = [(0, 10), (10, 20), (20, 40), (40, 80), (0, 80)]
CLASSES = {"car": {"car", "truck", "bus"}, "person": {"person"}}


def load(path):
    return list(csv.DictReader(open(path)))


def m1_rows(rows, e, min_lidar=10):
    """(class, lidar_z, our_z, b0_z) for matched detections with enough LiDAR points and a class-consistent match."""
    K = kitti.calib()["K"]
    fy, cy = K[1, 1], K[1, 2]
    out = []
    for r in rows:
        if not r["gt_cls"] or r["gt_cls"] not in CLASSES or r["cls"] not in CLASSES[r["gt_cls"]]:
            continue
        if not r["lidar_z"] or int(r["lidar_n"]) < min_lidar or not r[f"z_{e}"]:
            continue
        y1 = float(r["y1"])
        b0 = fy * CAM_H / (y1 - cy) if y1 - cy > 5 else np.nan
        out.append((r["gt_cls"], float(r["lidar_z"]), float(r[f"z_{e}"]), b0))
    return out


def table(name, data):
    print(f"\n{name}")
    print(f"{'class':7s} {'range m':8s} {'n':>5s} | {'abs err m':>9s} {'abs rel':>8s} {'bias':>7s} {'d<1.25':>7s} | B0: {'abs rel':>8s} {'bias':>7s}")
    for c in ("car", "person"):
        for lo, hi in BINS:
            s = [x for x in data if x[0] == c and lo <= x[1] < hi]
            if len(s) < 5:
                continue
            L = np.array([x[1] for x in s]); Z = np.array([x[2] for x in s]); B = np.array([x[3] for x in s])
            rel = (Z - L) / L
            ok = np.isfinite(B)
            brel = (B[ok] - L[ok]) / L[ok]
            d1 = np.mean(np.maximum(Z / L, L / Z) < 1.25)
            print(f"{c:7s} {lo:>3d}-{hi:<4d} {len(s):5d} | {np.median(np.abs(Z - L)):9.2f} {np.median(np.abs(rel)):8.1%} "
                  f"{np.median(rel):+7.1%} {d1:7.1%} | {np.median(np.abs(brel)) if ok.any() else np.nan:8.1%} "
                  f"{np.median(brel) if ok.any() else np.nan:+7.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--e", type=int, default=2, help="mask erosion (px) column to score")
    a = ap.parse_args()
    for p in a.csv:
        table(f"{p}  (erosion {a.e} px)", m1_rows(load(p), a.e))
