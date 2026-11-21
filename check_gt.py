"""Sanity sheet: LiDAR points (coloured by depth) + projected 3D tracklet boxes on image_02, a few frames of one drive.
usage: python check_gt.py 0020 [out.jpg]"""
import sys, os
import numpy as np, cv2
import kitti

n = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else f"out/check_gt_{n}.jpg"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
cal = kitti.calib()
K, size = cal["K"], cal["size"]
tr = kitti.tracklets(n, cal)
nf = len(os.listdir(os.path.join(kitti.drive_dir(n), "image_02", "data")))
frames = [int(f) for f in np.linspace(0, nf - 1, 4)]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
tiles = []
for f in frames:
    img = cv2.imread(os.path.join(kitti.drive_dir(n), "image_02", "data", f"{f:010d}.png"))
    pc, pv = kitti.velo(n, f, cal)
    uv = kitti.project(pc, K)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
    col = cv2.applyColorMap(np.clip(255 - pc[ok, 2] * 255 / 60, 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
    for (u, v), c in zip(uv[ok].astype(int), col[:, 0]):
        cv2.circle(img, (u, v), 1, tuple(int(x) for x in c), -1)
    for o in tr.get(f, []):
        if o["corners"][:, 2].min() < 0.5:
            continue
        q = kitti.project(o["corners"], K).astype(int)
        colr = (0, 255, 0) if o["occ"] == 0 else (0, 255, 255) if o["occ"] == 1 else (0, 0, 255)
        for a, b in EDGES:
            cv2.line(img, tuple(q[a]), tuple(q[b]), colr, 1)
        nl = len(kitti.lidar_in_box(o, pc, pv))
        cv2.putText(img, f"{o['cls']} z{o['centre'][2]:.1f} L{nl}", tuple(q[4]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colr, 1)
    cv2.putText(img, f"{n} #{f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(img)
cv2.imwrite(out, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("wrote", out, "frames", frames, "objects", sum(len(tr.get(f, [])) for f in frames))
