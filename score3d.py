"""Score run3d.py rows against LiDAR / 3D tracklets.

M1 - distance to the visible surface: our median mask depth vs the median depth of the LiDAR points inside the
     matched 3D box (no shape assumption, tests only the depth). Abs error (m), abs rel, signed rel (bias), delta<1.25.
M2 - object centre on the ground plane (x, z): a surface point pushed back along the horizontal viewing ray by
     r_class metres, vs the tracklet box centre. Median / p90 error (m) by GT range.
M3 - speed: per tracker track, centres -> world (OXTS poses) -> Theil-Sen line over a 3 s window -> km/h, vs a
     least-squares fit on the matched tracklet's centres at the same frames. Windows ending within 40 m;
     moving/stopped at 5 km/h. 95 % CI: bootstrap over tracks.
Methods: depth = median of mask depth (YOLO26-depth); hybrid = box-bottom ray on the road plane fitted to the depth
map (falls back to depth when the box bottom touches the frame edge, the contact is beyond HY_MAX or disagrees with the
mask depth by more than HY_DIS); B0 = classic box bottom on a flat road at the
known camera height (no depth network).
usage: python score3d.py data/val.csv [more.csv ...] [--join] [--e 2] [--scale 1.0] [--tune --win 1 2 3 --fit lsq theilsen]
"""
import argparse, csv
from collections import Counter, defaultdict
import numpy as np
import kitti

CAM_H = 1.65  # KITTI camera height above the road (m), from the sensor setup
BINS = [(0, 10), (10, 20), (20, 40), (40, 80), (0, 80)]
CLASSES = {"car": {"car", "truck", "bus"}, "person": {"person"}}
METHODS = ("depth", "hybrid", "b0")
R_CLASS = {"depth": {"car": 0.5, "person": 0.0}, "hybrid": {"car": 1.25, "person": 0.0},
           "b0": {"car": 2.0, "person": 0.0}}  # tuned on val6 letterbox (--tune, 22.11), frozen in PREREG.md
WIN_S, FIT, MAX_RANGE, MOVING = 3.0, "theilsen", 40.0, 5.0  # speed window (s), line fit, range (m), km/h (val6)
FILL = 0.8  # a window needs detections in >= 80 % of its frames (10 Hz)


def load(path):
    return list(csv.DictReader(open(path)))


def f_(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else np.nan


SCALE = 1.0  # global depth factor (--scale): 1.0 = model as released; calib_scale.py gives what calibrate() would pick
HY_MAX, HY_DIS = 40.0, 0.5  # hybrid uses the road contact only if it is nearer than HY_MAX m and within HY_DIS
#                            (relative) of the mask depth; otherwise the depth point (tuned on val6 with --tune)


def surface(r, e, K):
    """Surface point (x, z) of each method in the camera-2 frame (NaN where the method has no answer). A global depth
    scale multiplies the depth and hybrid points alike (the road plane scales with the depth map)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dz = (SCALE * f_(r, f"x_{e}"), SCALE * f_(r, f"z_{e}"))
    gz = SCALE * f_(r, "g_z")
    use = r.get("cut_bottom") == "0" and gz <= HY_MAX and abs(gz / dz[1] - 1) <= HY_DIS  # NaN compares False
    hy = (SCALE * f_(r, "g_x"), gz) if use else dz
    y1, u = f_(r, "y1"), (f_(r, "x0") + f_(r, "x1")) / 2
    zb = fy * CAM_H / (y1 - cy) if y1 - cy > 5 else np.nan
    return {"depth": dz, "hybrid": hy, "b0": ((u - cx) * zb / fx, zb)}


def push(p, rr):
    """Move a ground point rr metres further along its horizontal viewing ray."""
    x, z = p
    d = np.hypot(x, z)
    return (x + rr * x / d, z + rr * z / d) if d > 0 else (np.nan, np.nan)


def our_cls(r):
    return next((c for c, s in CLASSES.items() if r["cls"] in s), None)


def consistent(r):
    return r["gt_cls"] in CLASSES and r["cls"] in CLASSES[r["gt_cls"]]


def m1_rows(rows, e, K, min_lidar=10):
    """(class, lidar_z, depth_z, b0_z, hybrid_z) for matched detections with enough LiDAR points."""
    out = []
    for r in rows:
        if not consistent(r) or not r["lidar_z"] or int(r["lidar_n"]) < min_lidar or not r[f"z_{e}"]:
            continue
        s = surface(r, e, K)
        out.append((r["gt_cls"], float(r["lidar_z"]), s["depth"][1], s["b0"][1], s["hybrid"][1]))
    return out


def m1_table(name, data):
    print(f"\nM1 surface distance - {name}")
    print(f"{'class':7s} {'range m':8s} {'n':>5s} | depth: {'abs err m':>9s} {'abs rel':>8s} {'bias':>7s} {'d<1.25':>7s} | B0: {'abs rel':>8s} {'bias':>7s} | hybrid: {'abs rel':>8s} {'bias':>7s}")
    for c in CLASSES:
        for lo, hi in BINS:
            s = [x for x in data if x[0] == c and lo <= x[1] < hi]
            if len(s) < 5:
                continue
            L, Z, B, Hy = (np.array([x[k] for x in s]) for k in (1, 2, 3, 4))
            rel, hrel = (Z - L) / L, (Hy - L) / L
            ok = np.isfinite(B)
            brel = (B[ok] - L[ok]) / L[ok]
            d1 = np.mean(np.maximum(Z / L, L / Z) < 1.25)
            print(f"{c:7s} {lo:>3d}-{hi:<4d} {len(s):5d} | {np.median(np.abs(Z - L)):9.2f} {np.median(np.abs(rel)):8.1%} "
                  f"{np.median(rel):+7.1%} {d1:7.1%} | {np.median(np.abs(brel)) if ok.any() else np.nan:8.1%} "
                  f"{np.median(brel) if ok.any() else np.nan:+7.1%} | {np.median(np.abs(hrel)):8.1%} {np.median(hrel):+7.1%}")


def m2_rows(rows, e, K):
    """(class, gt range, gt (x, z), {method: surface point}, (drive, gt track)) for matched, class-consistent rows."""
    out = []
    for r in rows:
        if not consistent(r) or not r[f"z_{e}"] or not r["gt_z"]:
            continue
        g = (float(r["gt_x"]), float(r["gt_z"]))
        out.append((r["gt_cls"], np.hypot(*g), g, surface(r, e, K), (r["drive"], r["gt_tid"])))
    return out


def m2_err(data, method, rr):
    return np.array([np.hypot(*np.subtract(push(x[3][method], rr), x[2])) for x in data])


def m2_tune(data):
    """Median centre error vs r for each method and class (val only); returns the best r per method/class."""
    grid = np.round(np.arange(0, 3.01, 0.25), 2)
    best = {}
    print("\nM2 tuning: median centre error (m) over 0-80 m vs r_class")
    print(f"{'method':7s} {'class':7s} {'n':>5s} | " + " ".join(f"{g:5.2f}" for g in grid))
    for m in METHODS:
        best[m] = {}
        for c in CLASSES:
            s = [x for x in data if x[0] == c and x[1] < 80]
            if len(s) < 5:
                continue
            med = [np.nanmedian(m2_err(s, m, g)) for g in grid]
            best[m][c] = float(grid[int(np.nanargmin(med))])
            print(f"{m:7s} {c:7s} {len(s):5d} | " + " ".join(f"{v:5.2f}" for v in med) + f"  -> r = {best[m][c]}")
    return best


def hybrid_tune(rows, e, K):
    """Grid over the hybrid rule (HY_MAX, HY_DIS) with r re-picked per cell; criterion: median car centre error
    0-80 m (fixed before looking). Sets the module globals to the winner and returns it."""
    global HY_MAX, HY_DIS
    grid = np.round(np.arange(0, 3.01, 0.25), 2)
    res = {}
    print("\nHybrid rule tuning: car centre error median (p90) at the best r, 0-80 m")
    for hm in (20, 30, 40, 60, np.inf):
        cells = []
        for hd in (0.1, 0.2, 0.3, 0.5, np.inf):
            HY_MAX, HY_DIS = hm, hd
            d = [x for x in m2_rows(rows, e, K) if x[0] == "car" and x[1] < 80]
            med, rr = min((np.nanmedian(m2_err(d, "hybrid", g)), g) for g in grid)
            res[(hm, hd)] = (med, rr)
            cells.append(f"dis {hd:>4}: {med:.3f} r={rr:<4} p90 {np.nanpercentile(m2_err(d, 'hybrid', rr), 90):5.2f}")
        print(f"max {hm:>4}: " + " | ".join(cells))
    (HY_MAX, HY_DIS), (med, rr) = min(res.items(), key=lambda kv: kv[1][0])
    print(f"-> HY_MAX = {HY_MAX}, HY_DIS = {HY_DIS} (median {med:.3f} m at r = {rr})")
    return HY_MAX, HY_DIS


def m2_table(name, data, rc):
    print(f"\nM2 ground-plane centre error (m) - {name}; r_class = {rc}")
    print(f"{'class':7s} {'range m':8s} {'n':>5s} | " + " | ".join(f"{m}: {'med':>5s} {'p90':>5s}" for m in METHODS))
    for c in CLASSES:
        for lo, hi in BINS:
            s = [x for x in data if x[0] == c and lo <= x[1] < hi]
            if len(s) < 5:
                continue
            E = {m: m2_err(s, m, rc[m].get(c, 0.0)) for m in METHODS}
            print(f"{c:7s} {lo:>3d}-{hi:<4d} {len(s):5d} | " + " | ".join(
                f"{m}: {np.nanmedian(E[m]):5.2f} {np.nanpercentile(E[m], 90):5.2f}" for m in METHODS))
            if (lo, hi) == (0, 80):
                med = lambda m: lambda i: np.nanmedian(E[m][i])
                st = {m: med(m) for m in METHODS}
                st.update({f"{m}-b0": (lambda m: lambda i: med(m)(i) - med("b0")(i))(m) for m in ("depth", "hybrid")})
                ci = boot_ci([x[4] for x in s], st)
                print(f"{'':22s}  95% CI by GT track ({len({x[4] for x in s})} tracks): " + ", ".join(
                    f"{k} [{v[0]:+.2f}, {v[1]:+.2f}]" if "-" in k else f"{k} [{v[0]:.2f}, {v[1]:.2f}]" for k, v in ci.items()))


def fit_speed(t, P, fit="lsq"):
    """km/h from a line through horizontal world positions P (N, 2) at times t: least squares or Theil-Sen
    (median of pairwise slopes per axis, robust to single-frame depth jumps)."""
    if fit == "theilsen":
        i, j = np.triu_indices(len(t), 1)
        v = np.median((P[j] - P[i]) / (t[j] - t[i])[:, None], 0)
    else:
        A = np.c_[t - t.mean(), np.ones(len(t))]
        v = np.linalg.lstsq(A, P, rcond=None)[0][0]
    return float(np.hypot(*v) * 3.6)


def m3_windows(rows, e, K, cal, rc, win=WIN_S, fit=FIT):
    """One record per (track, window end frame): our speed per method and the tracklet's speed at the same frames."""
    min_pts = int(round(FILL * win * 10))
    by_track = defaultdict(dict)
    for r in rows:
        c = our_cls(r)
        if c and r["track"] not in ("", "-1") and r[f"z_{e}"]:
            by_track[(r["drive"], int(r["track"]))][int(r["frame"])] = r
    out = []
    for n in sorted({k[0] for k in by_track}):
        poses, ts, tr = kitti.poses(n, cal), kitti.times(n), kitti.tracklets(n, cal)
        gt = defaultdict(dict)  # tid -> frame -> camera-2 centre
        for f, objs in tr.items():
            for o in objs:
                gt[o["tid"]][f] = (o["centre"], o["cls"])
        world = lambda f, x, y, z: (poses[f] @ np.array([x, y, z, 1.0]))[:2]
        for (dn, tid), fr in by_track.items():
            if dn != n:
                continue
            for f in sorted(fr):
                F = [g for g in fr if ts[f] - win < ts[g] <= ts[f] + 1e-6]
                if len(F) < min_pts:
                    continue
                F.sort()
                ids = Counter(fr[g]["gt_tid"] for g in F if fr[g]["gt_tid"] and consistent(fr[g]))
                if not ids:
                    continue
                gid, cnt = ids.most_common(1)[0]
                gid = int(gid)
                if cnt < min_pts // 2 or f not in gt[gid] or np.linalg.norm(gt[gid][f][0][[0, 2]]) > MAX_RANGE:
                    continue
                Fg = [g for g in F if g in gt[gid]]
                if len(Fg) < min_pts:
                    continue
                t = ts[F]
                gs = fit_speed(ts[Fg], np.array([world(g, *gt[gid][g][0]) for g in Fg]))  # GT always least squares
                rec = {"drive": n, "track": tid, "frame": f, "cls": gt[gid][f][1], "gt": gs,
                       "range": float(np.linalg.norm(gt[gid][f][0][[0, 2]]))}
                for m in METHODS:
                    P = []
                    for g in F:
                        r = fr[g]
                        x, z = push(surface(r, e, K)[m], rc[m].get(our_cls(r), 0.0))
                        P.append(world(g, x, SCALE * f_(r, f"y_{e}") if r[f"y_{e}"] else 0.0, z))
                    P = np.array(P)
                    ok = np.isfinite(P).all(1)
                    rec[m] = fit_speed(t[ok], P[ok], fit) if ok.sum() >= min_pts else np.nan
                out.append(rec)
    return out


def boot_ci(keys, stats, n=1000, seed=0):
    """keys: track label per item; stats: {name: fn(index array) -> value}. 95 % CI of each, resampling whole tracks
    with replacement (windows / frames of one track are not independent)."""
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    G = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    vals = defaultdict(list)
    for _ in range(n):
        idx = np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])
        for name, fn in stats.items():
            vals[name].append(fn(idx))
    return {name: np.nanpercentile(v, [2.5, 97.5]) for name, v in vals.items()}


def m3_table(name, recs, win=WIN_S, fit=FIT):
    print(f"\nM3 speed (window {win:g} s, fit {fit}, detections in >= {FILL:.0%} of frames, GT within {MAX_RANGE:.0f} m) - {name}")
    for c in CLASSES:
        s = [w for w in recs if w["cls"] == c]
        if len(s) < 5:
            continue
        nt = len({(w["drive"], w["track"]) for w in s})
        g = np.array([w["gt"] for w in s])
        print(f"{c}: {len(s)} windows, {nt} tracks, GT speed median {np.median(g):.1f} km/h, moving {np.mean(g > MOVING):.0%}"
              f" (answering 'stopped' always: acc {np.mean(g <= MOVING):.1%})")
        est = {m: np.array([w[m] for w in s]) for m in METHODS}
        err = {m: np.abs(est[m] - g) for m in METHODS}
        hit = {m: np.where(np.isfinite(est[m]), (est[m] > MOVING) == (g > MOVING), np.nan) for m in METHODS}
        st = {}
        for m in METHODS:
            st[f"err {m}"] = (lambda m: lambda i: np.nanmedian(err[m][i]))(m)
            st[f"acc {m}"] = (lambda m: lambda i: np.nanmean(hit[m][i]))(m)
        for m in ("depth", "hybrid"):
            st[f"err {m}-b0"] = (lambda m: lambda i: st[f"err {m}"](i) - st["err b0"](i))(m)
            st[f"acc {m}-b0"] = (lambda m: lambda i: st[f"acc {m}"](i) - st["acc b0"](i))(m)
        ci = boot_ci([(w["drive"], w["track"]) for w in s], st)
        for m in METHODS:
            lo, hi = ci[f"err {m}"]
            alo, ahi = ci[f"acc {m}"]
            print(f"  {m:7s} abs err median {np.nanmedian(err[m]):5.1f} km/h [95% CI {lo:.1f}-{hi:.1f}], p90 "
                  f"{np.nanpercentile(err[m], 90):5.1f}, bias {np.nanmedian(est[m] - g):+5.1f}, "
                  f"moving/stopped acc {np.nanmean(hit[m]):.1%} [{alo:.1%}-{ahi:.1%}]")
        print("  paired vs B0 (95% CI): " + ", ".join(f"{k} [{ci[k][0]:+.2f}, {ci[k][1]:+.2f}]" for k in st if "-b0" in k))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--join", action="store_true", help="score all CSVs as one set")
    ap.add_argument("--e", type=int, default=2, help="mask erosion (px) column to score")
    ap.add_argument("--tune", action="store_true", help="pick r_class on these rows (val only) and use it")
    ap.add_argument("--win", type=float, nargs="+", default=[WIN_S], help="speed window(s), s")
    ap.add_argument("--fit", nargs="+", default=[FIT], choices=["lsq", "theilsen"])
    ap.add_argument("--scale", type=float, default=SCALE, help="global depth factor (calibrate() equivalent)")
    a = ap.parse_args()
    SCALE = a.scale
    cal = kitti.calib()
    K = cal["K"]
    sets = [(" + ".join(a.csv), sum((load(p) for p in a.csv), []))] if a.join else [(p, load(p)) for p in a.csv]
    for name, rows in sets:
        name = f"{name} (erosion {a.e} px, depth x{SCALE:g})"
        if a.tune:
            hybrid_tune(rows, a.e, K)
        m1_table(name, m1_rows(rows, a.e, K))
        d2 = m2_rows(rows, a.e, K)
        rc = m2_tune(d2) if a.tune else R_CLASS
        m2_table(name, d2, rc)
        for win in a.win:
            for fit in a.fit:
                m3_table(name, m3_windows(rows, a.e, K, cal, rc, win, fit), win, fit)
