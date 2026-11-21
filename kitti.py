"""KITTI raw helpers: calibration, 3D tracklets, OXTS poses, Velodyne points. Everything is returned in the
camera-2 frame (left colour camera, rectified): x right, y down, z forward, metres. Image_02 pixels map to it with K.

Layout (fetch_kitti.sh): kitti/2011_09_26/{calib_*.txt, <drive>_sync/{image_02,velodyne_points,oxts}/data, tracklet_labels.xml}
"""
import os
import xml.etree.ElementTree as ET
import numpy as np

ROOT = os.path.expanduser("~/metric3d/kitti/2011_09_26")
VAL = ["0002", "0020", "0029", "0048", "0059", "0086"]  # every third drive in sorted order (fixed before any run)
TEST = ["0009", "0013", "0023", "0027", "0036", "0046", "0052", "0056", "0064", "0084", "0093"]
CLASS_MAP = {"Car": "car", "Van": "car", "Pedestrian": "person", "Person_sitting": "person", "Cyclist": "cyclist",
             "Truck": "truck", "Tram": "tram", "Misc": "misc"}


def drive_dir(n):
    return os.path.join(ROOT, f"2011_09_26_drive_{n}_sync")


def _read(path):
    out = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k] = np.array([float(x) for x in v.split()])
            except ValueError:
                pass
    return out


def _T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def calib(root=ROOT):
    """K (3x3) of image_02 and 4x4 transforms velo->cam2 and imu->cam2."""
    c = _read(os.path.join(root, "calib_cam_to_cam.txt"))
    v = _read(os.path.join(root, "calib_velo_to_cam.txt"))
    i = _read(os.path.join(root, "calib_imu_to_velo.txt"))
    P2 = c["P_rect_02"].reshape(3, 4)
    K = P2[:, :3].copy()
    R_rect = np.eye(4)
    R_rect[:3, :3] = c["R_rect_00"].reshape(3, 3)
    T2 = np.eye(4)  # rectified cam0 -> rectified cam2 (baseline offset from P_rect_02)
    T2[0, 3] = P2[0, 3] / P2[0, 0]
    T2[1, 3] = P2[1, 3] / P2[1, 1]
    T2[2, 3] = P2[2, 3]
    velo_cam2 = T2 @ R_rect @ _T(v["R"].reshape(3, 3), v["T"])
    imu_cam2 = velo_cam2 @ _T(i["R"].reshape(3, 3), i["T"])
    return {"K": K, "velo_cam2": velo_cam2, "imu_cam2": imu_cam2, "size": c["S_rect_02"].astype(int)}  # size = (w, h)


def velo(n, frame, cal):
    """Velodyne points of one frame in camera-2 coordinates, only in front of the camera. (N, 3)"""
    p = np.fromfile(os.path.join(drive_dir(n), "velodyne_points", "data", f"{frame:010d}.bin"), np.float32).reshape(-1, 4)
    q = (cal["velo_cam2"] @ np.c_[p[:, :3], np.ones(len(p))].T).T[:, :3]
    return q[q[:, 2] > 0.5], p[q[:, 2] > 0.5, :3]  # (cam2 points, same points in velo frame)


def poses(n, cal):
    """Camera-2 -> world 4x4 pose per frame from OXTS (standard KITTI Mercator conversion, first frame's latitude)."""
    files = sorted(os.listdir(os.path.join(drive_dir(n), "oxts", "data")))
    rows = [np.array(open(os.path.join(drive_dir(n), "oxts", "data", f)).read().split(), float) for f in files]
    er, scale = 6378137.0, np.cos(rows[0][0] * np.pi / 180)
    out = []
    for r in rows:
        lat, lon, alt, roll, pitch, yaw = r[:6]
        t = [scale * er * lon * np.pi / 180, scale * er * np.log(np.tan((90 + lat) * np.pi / 360)), alt]
        cx, sx, cy, sy, cz, sz = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        out.append(_T(Rz @ Ry @ Rx, t) @ np.linalg.inv(cal["imu_cam2"]))
    return out


def times(n):
    """Capture time (s from the drive's first frame) of every image_02 frame."""
    out = []
    for line in open(os.path.join(drive_dir(n), "image_02", "timestamps.txt")):
        hh, mm, ss = line.split()[1].split(":")
        out.append(int(hh) * 3600 + int(mm) * 60 + float(ss))
    return np.array(out) - out[0]


def _box_corners(h, w, l):
    x = np.array([-l, -l, l, l, -l, -l, l, l]) / 2
    y = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    z = np.array([0, 0, 0, 0, h, h, h, h], float)
    return np.stack([x, y, z], 1)


def tracklets(n, cal):
    """{frame: [obj, ...]}; obj has track id, class, dims, box corners (cam2), centre (cam2), occlusion, truncation,
    and (R, t) of the box in the velo frame so LiDAR points inside it can be selected."""
    root = ET.parse(os.path.join(drive_dir(n), "tracklet_labels.xml")).getroot().find("tracklets")
    out = {}
    tid = 0
    for it in root.findall("item"):
        typ = it.find("objectType").text
        h, w, l = (float(it.find(k).text) for k in ("h", "w", "l"))
        f0 = int(it.find("first_frame").text)
        corners0 = _box_corners(h, w, l)
        for k, p in enumerate(it.find("poses").findall("item")):
            tx, ty, tz, rz = (float(p.find(q).text) for q in ("tx", "ty", "tz", "rz"))
            occ, trunc = int(p.find("occlusion").text), int(p.find("truncation").text)
            c, s = np.cos(rz), np.sin(rz)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            t = np.array([tx, ty, tz])
            cv = corners0 @ R.T + t  # velo frame, bottom at tz
            cc = (cal["velo_cam2"] @ np.c_[cv, np.ones(8)].T).T[:, :3]
            centre = (cal["velo_cam2"] @ np.r_[t + [0, 0, h / 2], 1])[:3]
            out.setdefault(f0 + k, []).append({"tid": tid, "type": typ, "cls": CLASS_MAP.get(typ, "misc"),
                                               "h": h, "w": w, "l": l, "corners": cc, "centre": centre,
                                               "occ": occ, "trunc": trunc, "R": R, "t": t})
        tid += 1
    return out


def project(P, K):
    """(N, 3) camera points -> (N, 2) pixels."""
    q = P @ K.T
    return q[:, :2] / q[:, 2:3]


def box2d(obj, K, size):
    """2D box of the projected 3D box, clipped to the image; None if any corner is behind the camera."""
    c = obj["corners"]
    if (c[:, 2] < 0.5).any():
        return None
    uv = project(c, K)
    w, h = size
    x0, y0 = max(0, uv[:, 0].min()), max(0, uv[:, 1].min())
    x1, y1 = min(w - 1, uv[:, 0].max()), min(h - 1, uv[:, 1].max())
    return (x0, y0, x1, y1) if x1 - x0 > 2 and y1 - y0 > 2 else None


def lidar_in_box(obj, pts_cam, pts_velo, K=None, size=None):
    """LiDAR points (camera-2 frame) that fall inside the object's 3D box; with K and size, only those the camera
    can see (projecting inside the image), since the LiDAR also sees the parts of a truncated object outside the frame."""
    q = (pts_velo - obj["t"]) @ obj["R"]  # into the box frame
    m = (np.abs(q[:, 0]) <= obj["l"] / 2) & (np.abs(q[:, 1]) <= obj["w"] / 2) & (q[:, 2] >= 0) & (q[:, 2] <= obj["h"])
    P = pts_cam[m]
    if K is not None and len(P):
        uv = project(P, K)
        P = P[(uv[:, 0] >= 0) & (uv[:, 0] < size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[1])]
    return P
