"""Smoke test of every loader run3d.py touches, on every test frame, without models or metrics."""
import os, kitti
cal = kitti.calib()
for n in "0009 0013 0023 0027 0036 0046 0052 0056 0064 0084 0093".split():
    frames = sorted(int(f[:-4]) for f in os.listdir(os.path.join(kitti.drive_dir(n), "image_02", "data")))
    tr = kitti.tracklets(n, cal); po = kitti.poses(n, cal); ts = kitti.times(n)
    empty = sum(len(kitti.velo(n, f, cal)[0]) == 0 for f in frames)
    print(n, len(frames), "frames, poses", len(po), "times", len(ts), "frames with tracklets", len(tr), "empty lidar", empty, flush=True)
print("LOADCHECK OK")
