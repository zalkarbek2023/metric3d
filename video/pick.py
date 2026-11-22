"""Per test drive: how many cars, how good hybrid/B0 centre is there, how many moving cars - to choose the drive for
the video honestly (near the test median, not the best). usage: python video/pick.py data/test_letterbox.csv"""
import sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, ".")
import kitti, score3d as S

rows = S.load(sys.argv[1])
cal = kitti.calib(); K = cal["K"]
d2 = S.m2_rows(rows, 2, K)
rc = S.R_CLASS
rec = S.m3_windows(rows, 2, K, cal, rc)
print("drive frames rows | car M2 n  hybrid  b0 | M3 windows moving(gt>5) tracks_moving | cars/frame")
for n in kitti.TEST:
    c = [x for x in d2 if x[4][0] == n and x[0] == "car" and x[1] < 80]
    hy = S.m2_err(c, "hybrid", rc["hybrid"]["car"]) if c else np.array([np.nan])
    b0 = S.m2_err(c, "b0", rc["b0"]["car"]) if c else np.array([np.nan])
    r3 = [x for x in rec if x["drive"] == n and x["cls"] == "car"]
    mov = [x for x in r3 if x["gt"] > 5]
    fr = {int(r["frame"]) for r in rows if r["drive"] == n}
    cars = sum(1 for r in rows if r["drive"] == n and r["cls"] in ("car", "truck", "bus"))
    print(f"{n}  {len(fr):4d} {sum(r['drive']==n for r in rows):5d} | {len(c):5d} {np.nanmedian(hy):6.2f} {np.nanmedian(b0):6.2f} | "
          f"{len(r3):5d} {len(mov):5d} {len({x['track'] for x in mov}):3d} | {cars/max(1,len(fr)):.1f}")
