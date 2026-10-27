#!/usr/bin/env python
"""
Fully automatic pick & place: the arm LOOKS AROUND by itself until one camera frame
contains both the red cube and the blue tray, locks their positions from that view,
then plans and executes exactly like pick_place.py (same perception, hand-eye, IK,
one-moving-jaw grasp, stop / hold / resume behaviour). No leader arm needed.

How the search works
  1. View library: thousands of random arm poses are checked with FK + hand-eye for
     "a usable camera view": camera 15-40 cm above the table and looking down,
     gripper well clear of the table, inside joint limits.
  2. Scan plan: a small set of those views (greedy set cover) that together see the
     whole reachable table area, ordered by pan so the arm sweeps smoothly.
  3. At each view: settle, detect. Every sighting (even a clipped one at the image
     edge) is remembered as a rough base-frame position + size.
       - both detected cleanly in one frame -> average 12 frames -> LOCK
       - otherwise, once both are known roughly, jump straight to the library view
         that best frames BOTH (centred, with margin) and try to lock there.
  4. All moves are joint paths checked against the table; they are stoppable and hold.

Stop any time: any key in the camera window, Enter in the terminal, or Ctrl+C.
The arm holds where it is; then r = resume, h = return to rest, x = exit still holding.

Example:
  python pick_place_search.py --camera /dev/video4 --port /dev/ttyACM0 --id so101_follower_vasco \\
      --table-z-mm <measured> --cube-size-mm 25 --tray-top-mm 16 \\
      --gripper-open <open> --gripper-closed <closed> --open-gap-mm <gap>
  Add --dry-run to search + plan without executing, --teach to (re)teach colors first,
  --confirm-plan to wait for Enter between planning and execution.
"""
import time
import xml.etree.ElementTree as ET

import cv2 as cv
import numpy as np

import pick_place as PP
from so101_kinematics import ARM_JOINTS, DEFAULT_URDF, JOINT_OFFSETS_DEG

IMG_W, IMG_H = 640, 480
SEARCH_WIN = "pick & place: search"
LIB_SAMPLES = 40000
CAM_H = (0.15, 0.40)            # camera height above table (m)
MIN_LOOK_DOWN = 0.5             # optical axis z component <= -this (30 deg below horizontal)
TCP_CLEAR = 0.08                # TCP at least this far above the table in search views (m)
PATH_CLEAR = 0.06               # ...and along every path between views
JOINT_MARGIN_DEG = 6.0
SCAN_R = (0.13, 0.33)           # table region to cover (radius from base, m)
SCAN_AZ_DEG = 70.0              # +- azimuth
VIEW_MARGIN_PX = 30
MAX_LOCK_TRIES = 20



# ============================================================ geometry helpers
def joint_limits_lerobot():
    """URDF joint limits converted to LeRobot degrees (lerobot = urdf + offset)."""
    root = ET.parse(DEFAULT_URDF).getroot()
    lim = {}
    for j in root.findall("joint"):
        if j.get("name") in ARM_JOINTS and j.find("limit") is not None:
            l = j.find("limit")
            lim[j.get("name")] = (np.degrees(float(l.get("lower"))), np.degrees(float(l.get("upper"))))
    lo = np.array([lim[n][0] for n in ARM_JOINTS]) + JOINT_OFFSETS_DEG + JOINT_MARGIN_DEG
    hi = np.array([lim[n][1] for n in ARM_JOINTS]) + JOINT_OFFSETS_DEG - JOINT_MARGIN_DEG
    return lo, hi


class ViewLibrary:
    def __init__(self, kin0, X, K, D, table_z, rng=None):
        self.kin0, self.X, self.K, self.D, self.table_z = kin0, X, K, D, table_z
        rng = rng or np.random.default_rng(0)
        lo, hi = joint_limits_lerobot()
        # keep wrist roll moderate so the image isn't upside down and paths stay calm
        lo[4], hi[4] = max(lo[4], JOINT_OFFSETS_DEG[4] - 60), min(hi[4], JOINT_OFFSETS_DEG[4] + 60)
        self.lo, self.hi = lo, hi
        qs, cams, hits = [], [], []

        def consider(q):
            T = kin0.fk(q)
            C = T @ X
            h = C[2, 3] - table_z
            d = C[:3, 2]
            if not (CAM_H[0] <= h <= CAM_H[1]) or d[2] > -MIN_LOOK_DOWN:
                return False
            if T[2, 3] - table_z < TCP_CLEAR or not 0.06 <= np.linalg.norm(T[:2, 3]) <= 0.36:
                return False
            s = (table_z - C[2, 3]) / d[2]
            hit = C[:3, 3] + s * d
            if not 0.08 <= np.linalg.norm(hit[:2]) <= 0.42:
                return False
            qs.append(q); cams.append(C); hits.append(hit[:2])
            return True

        n = 0
        for q in rng.uniform(lo, hi, size=(LIB_SAMPLES, 5)):
            consider(q); n += 1
        # grow the library around what was found, biased toward HIGHER views (wider footprint)
        base_q = np.array(qs)
        heights = np.array([c[2, 3] for c in cams])
        seeds = base_q[np.argsort(heights)[::-1][:max(20, len(base_q) // 3)]]
        for _ in range(3):
            new = np.clip(seeds[rng.integers(len(seeds), size=LIB_SAMPLES // 4)] +
                          rng.normal(0, 8.0, size=(LIB_SAMPLES // 4, 5)), lo, hi)
            for q in new:
                consider(q); n += 1
            heights = np.array([c[2, 3] for c in cams])
            seeds = np.array(qs)[np.argsort(heights)[::-1][:max(20, len(qs) // 3)]]
        self.q, self.cam, self.hit = np.array(qs), np.array(cams), np.array(hits)
        h = self.cam[:, 2, 3] - table_z
        print(f"View library: {len(self.q)} usable camera views from {n} sampled poses "
              f"(camera height {h.min()*100:.0f}-{h.max()*100:.0f} cm)")
        if len(self.q) < 50:
            raise RuntimeError("Too few usable views; check --table-z-mm and the hand-eye file")

    def project(self, i, pts_base):
        """Pixel coords of base-frame points in library view i (None for points behind camera)."""
        Tcb = np.linalg.inv(self.cam[i])
        P = (Tcb[:3, :3] @ np.asarray(pts_base, float).T).T + Tcb[:3, 3]
        if np.any(P[:, 2] <= 0.02):
            return None
        uv, _ = cv.projectPoints(P, np.zeros(3), np.zeros(3), self.K, self.D)
        return uv.reshape(-1, 2)

    def sees_all(self, pts_base, margin=VIEW_MARGIN_PX):
        uv, front = self.project_all(np.asarray(pts_base, float))
        return (front & (uv[..., 0] >= margin) & (uv[..., 0] <= IMG_W - margin) &
                (uv[..., 1] >= margin) & (uv[..., 1] <= IMG_H - margin))

    def scan_plan(self, max_views=10, q_start=None):
        """Greedy set cover of the reachable table region, ordered by pan for a smooth sweep."""
        rs = np.arange(SCAN_R[0], SCAN_R[1] + 1e-9, 0.02)
        azs = np.radians(np.arange(-SCAN_AZ_DEG, SCAN_AZ_DEG + 1e-9, 7.0))
        grid = np.array([[r * np.cos(a), r * np.sin(a), self.table_z + 0.01] for r in rs for a in azs])
        cover = self.sees_all(grid)                              # (n_views, n_grid)
        covered = np.zeros(len(grid), bool)
        chosen = []
        while len(chosen) < max_views and covered.mean() < 0.97:
            gain = (cover & ~covered).sum(1)
            gain[chosen] = -1
            best = int(np.argmax(gain))
            if gain[best] <= 0:
                break
            chosen.append(best)
            covered |= cover[best]
        pan = self.q[chosen, 0]
        order = list(np.array(chosen)[np.argsort(pan)])
        if q_start is not None and abs(self.q[order[-1], 0] - q_start[0]) < abs(self.q[order[0], 0] - q_start[0]):
            order = order[::-1]
        print(f"Scan plan: {len(order)} views covering {covered.mean()*100:.0f}% of the table region "
              f"(r {SCAN_R[0]*100:.0f}-{SCAN_R[1]*100:.0f} cm, +-{SCAN_AZ_DEG:.0f} deg)")
        return order

    def project_all(self, pts_base):
        """Project points into EVERY library view at once (OpenCV 5-coefficient model).
        Returns uv (n_views, n_pts, 2) and a mask of points in front of the camera."""
        R = self.cam[:, :3, :3]                                 # cam -> base
        t = self.cam[:, :3, 3]
        P = np.einsum("vji,vpj->vpi", R, pts_base[None, :, :] - t[:, None, :])   # base -> cam
        front = P[..., 2] > 0.02
        z = np.where(front, P[..., 2], 1.0)
        x, y = P[..., 0] / z, P[..., 1] / z
        k1, k2, p1, p2, k3 = (list(np.ravel(self.D)) + [0] * 5)[:5]
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        u = self.K[0, 0] * xd + self.K[0, 2]
        v = self.K[1, 1] * yd + self.K[1, 2]
        # strong barrel distortion folds far-off-axis points back into the image: reject them
        front &= r2 < 1.2
        return np.stack([u, v], -1), front

    def best_view_for(self, objects, q_now, exclude=(), margin=45):
        """Library view that frames every object (centre + rim points) with margin,
        preferring centred framing and little joint motion."""
        pts = []
        for o in objects:
            c, r = o["center"], o["radius"]
            pts.append(c)
            for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                pts.append(c + [r * np.cos(ang), r * np.sin(ang), 0])
        pts = np.array(pts)
        uv, front = self.project_all(pts)
        inside = (front & (uv[..., 0] >= margin) & (uv[..., 0] <= IMG_W - margin) &
                  (uv[..., 1] >= margin) & (uv[..., 1] <= IMG_H - margin)).all(1)
        for i in exclude:
            inside[i] = False
        if not inside.any():
            return None
        centre_off = np.linalg.norm(uv.mean(1) - [IMG_W / 2, IMG_H / 2], axis=1)
        cost = np.max(np.abs(self.q - q_now), axis=1) / 30.0 + centre_off / 80.0
        cost[~inside] = np.inf
        return int(np.argmin(cost))


# ============================================================ search
def safe_path(kin0, table_z, q_from6, q_to6, lib):
    """Joint path from q_from to q_to that keeps the TCP above the table; inserts a raised
    via-pose (the library's highest view, at the from/to pan) if the direct path dips."""
    def ok(qa, qb):
        za, zb = kin0.fk(qa[:5])[2, 3], kin0.fk(qb[:5])[2, 3]
        floor = min(table_z + PATH_CLEAR, za, zb) - 0.005
        return all(kin0.fk(qa[:5] + (qb[:5] - qa[:5]) * t)[2, 3] >= floor for t in np.linspace(0, 1, 15))
    if ok(q_from6, q_to6):
        return [q_to6]
    hi_i = int(np.argmax([kin0.fk(q)[2, 3] for q in lib.q[:2000]]))
    via = lib.q[hi_i].copy()
    v1 = np.r_[np.r_[q_from6[0], via[1:]], q_from6[5]]
    v2 = np.r_[np.r_[q_to6[0], via[1:]], q_to6[5]]
    path = [v1, v2, q_to6]
    prev = q_from6
    for p in path:
        if not ok(prev, p):
            raise RuntimeError("no safe joint path between views (would dip toward the table)")
        prev = p
    return path


def rough(perc, det_result, T_base_cam, name):
    """Rough base-frame centre + radius of a detected (possibly clipped) region."""
    contour = det_result[name].get("contour")
    if contour is None:
        return None
    pts = [perc.ray_to_plane(c, T_base_cam, perc.h[name]) for c in contour.reshape(-1, 2).astype(float)]
    pts = np.array([p for p in pts if p is not None])
    if len(pts) < 3:
        return None
    centre = pts.mean(0)
    radius = np.max(np.linalg.norm(pts[:, :2] - centre[:2], axis=1))
    return {"center": centre, "radius": radius * 1.2 + 0.01,
            "clean": det_result[name]["status"] == "OK", "time": time.time()}


def search_observe_factory(lib, args):
    def search_observe(ctx):
        a, arm, cap, perc, stop, kin0 = ctx.a, ctx.arm, ctx.cap, ctx.perc, ctx.stop, ctx.kin0
        grip = arm.q6()[5]
        known = {}                                   # name -> rough sighting
        tried_targeted = set()
        lock_tries = 0

        def go(i, label):
            q_to = np.r_[lib.q[i], grip]
            path = safe_path(kin0, perc.table_z, arm.q6(), q_to, lib)
            ctx.moved()
            arm.move(path, a.speed_deg_s, stop)
            t_end = time.time() + 0.5                 # settle + flush stale camera frames
            while time.time() < t_end:
                frame = PP.read_frame(cap)
                show(frame, None, label)
                if stop.requested():
                    arm.hold()
                    raise PP.Stopped(stop.reason, 0)

        def show(frame, res, label):
            view = frame if res is None else PP.draw(frame, res, label, True)
            if res is None:
                view = frame.copy()
                cv.putText(view, label, (10, IMG_H - 14), cv.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv.LINE_AA)
            cv.putText(view, "cube: " + ("seen" if "cube" in known else "?") + "   tray: " +
                       ("seen" if "plate" in known else "?"), (10, 94), cv.FONT_HERSHEY_SIMPLEX, 0.55,
                       (0, 230, 230), 2, cv.LINE_AA)
            cv.imshow(SEARCH_WIN, view)

        def look(label):
            """Detect at the current pose for a few frames; update `known`; True if both clean."""
            both = 0
            for _ in range(5):
                frame = PP.read_frame(cap)
                q = arm.q6()
                res = perc.detect(frame, q[:5])
                T_bc = kin0.fk(q[:5]) @ perc.X
                for n in ("cube", "plate"):
                    r = rough(perc, res, T_bc, n)
                    if r is not None and (n not in known or r["clean"] or not known[n]["clean"]):
                        known[n] = r
                both += res["cube"]["base"] is not None and res["plate"]["base"] is not None
                show(frame, res, label)
                if stop.requested():
                    arm.hold()
                    raise PP.Stopped(stop.reason, 0)
            return both >= 3

        def try_lock(label):
            nonlocal lock_tries
            lock_tries += 1
            locked, msg = PP.lock_targets(ctx, lambda: None)
            print(f"  lock at {label}: {msg}")
            return locked

        def lock_one(name):
            pts, yaws = [], []
            for _ in range(PP.DETECT_FRAMES):
                frame = PP.read_frame(cap)
                q = arm.q6()
                r = perc.detect(frame, q[:5])
                if r[name]["base"] is not None:
                    pts.append(r[name]["base"])
                    if name == "cube" and r["cube"].get("yaw") is not None:
                        yaws.append(r["cube"]["yaw"])
                cv.waitKey(1)
            if len(pts) < PP.DETECT_FRAMES * 0.7:
                return None
            if np.max(np.linalg.norm(np.array(pts) - np.median(pts, 0), axis=1)) * 1000 > PP.MAX_SPREAD_MM:
                return None
            o = {"base": np.median(pts, 0)}
            if yaws:
                o["yaw"] = float(np.angle(np.mean(np.exp(4j * np.array(yaws)))) / 4) % (np.pi / 2)
            return o

        def separate_views():
            """Fallback: lock each object from its own best view (positions are in the base
            frame, so they combine). Only used with --allow-separate-views."""
            out = {}
            for n in ("cube", "plate"):
                i = lib.best_view_for([known[n]], arm.q6()[:5])
                if i is None:
                    print(f"No view frames the {'cube' if n == 'cube' else 'tray'} cleanly.")
                    return None
                go(i, f"locking {'cube' if n == 'cube' else 'tray'} alone")
                o = lock_one(n)
                if o is None:
                    print(f"Could not lock the {'cube' if n == 'cube' else 'tray'} reliably.")
                    return None
                print(f"  locked {'cube' if n == 'cube' else 'tray'} at {np.round(o['base'] * 1000, 1)} mm")
                out[n] = o
            return out["cube"], out["plate"], arm.q6()

        # ---- optional color teaching at the first view
        scan = lib.scan_plan(args.scan_views, arm.q6())
        missing = [n for n in ("cube", "plate") if perc.profiles[n]["lab"] is None]
        if args.teach or missing:
            if missing:
                print(f"No color taught yet for: {', '.join('cube' if m == 'cube' else 'tray' for m in missing)}.")
            if stop.prompt("Enter = move to the first view to teach colors, q = quit: ", ["", "q"]) != "":
                return None
            go(scan[0], "teach")
            print("TEACH: c = cube color, p = tray color (drag over the TOP face), SPACE = start search")
            while True:
                frame = PP.read_frame(cap)
                q = arm.q6()
                res = perc.detect(frame, q[:5])
                cv.imshow(SEARCH_WIN, PP.draw(frame, res, "c cube | p tray | SPACE start search", True))
                k = cv.waitKey(30) & 0xFF
                if k in (ord("c"), ord("p")):
                    perc.teach(frame, "cube" if k == ord("c") else "plate")
                elif k == ord(" "):
                    break
                elif k in (ord("q"), 27):
                    return None
        elif not args.yes:
            if stop.prompt("\nThe arm will now move ON ITS OWN to look for the cube and tray.\n"
                           "Enter = start search, q = quit: ", ["", "q"]) != "":
                return None
        stop.clear()

        # ---- search loop
        queue_ = list(scan)
        step = 0
        while True:
            # targeted view as soon as both are roughly known
            target = None
            if "cube" in known and "plate" in known and lock_tries < MAX_LOCK_TRIES:
                target = lib.best_view_for([known["cube"], known["plate"]], arm.q6()[:5], tried_targeted)
                if target is None:
                    d = np.linalg.norm(known["cube"]["center"][:2] - known["plate"]["center"][:2]) * 100
                    if args.allow_separate_views:
                        print(f"No single view frames both ({d:.0f} cm apart): locking them from separate views.")
                        return separate_views()
                    print(f"No single view frames both objects ({d:.0f} cm apart). Move them closer, "
                          "or use --allow-separate-views.")
                    return None
            if target is not None:
                tried_targeted.add(target)
                label = "framing both"
                print(f"Both objects roughly located -> moving to a view that frames both")
                go(target, label)
                if look(label):
                    locked = try_lock(label)
                    if locked is not None:
                        return locked
                continue
            if not queue_:
                if args.allow_separate_views and "cube" in known and "plate" in known:
                    return separate_views()
                seen = [("cube" if n == "cube" else "tray") for n in known]
                print(f"Search finished without a frame containing both. Seen: {seen or 'nothing'}. "
                      "Check colors (--teach), lighting, and that both objects are within reach.")
                return None
            i = queue_.pop(0)
            step += 1
            label = f"scan view {step}/{len(scan)}"
            print(f"{label}")
            go(i, label)
            if look(label):
                locked = try_lock(label)
                if locked is not None:
                    return locked

    return search_observe


def main():
    p = PP.build_parser(leader_required=False, description=__doc__)
    p.add_argument("--scan-views", type=int, default=10, help="max views in the coverage scan")
    p.add_argument("--teach", action="store_true", help="teach cube/tray colors at the first view")
    p.add_argument("--confirm-plan", action="store_true", help="wait for Enter before executing the plan")
    p.add_argument("--yes", action="store_true", help="don't ask before the search starts moving the arm")
    p.add_argument("--allow-separate-views", action="store_true",
                   help="if no single view frames both, lock cube and tray from different views")
    a = PP.finish_args(p.parse_args())
    a.go = not a.confirm_plan

    # build the view library before touching the robot
    K, D = np.load(PP.config.CAMERA_MATRIX_PATH), np.load(PP.config.DIST_COEFFS_PATH)
    X = np.load(a.hand_eye)
    kin0 = PP.SO101Kinematics(tip_offset_m=0.0)
    t0 = time.time()
    lib = ViewLibrary(kin0, X, K, D, a.table_z_mm / 1000.0)
    print(f"  (built in {time.time() - t0:.1f} s)")
    PP.run(a, search_observe_factory(lib, a))


if __name__ == "__main__":
    main()