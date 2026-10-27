#!/usr/bin/env python
"""
FK check on the ChArUco board: put the pencil tip on board corners, compare FK with the
board's exact geometry. No ruler needed, the board IS the ruler.

Corners are given as (col,row) on the board's INNER corner grid: the points where two
black squares touch. (0,0) is any corner you choose; just count consistently.
For a 7x6-square board the inner grid is cols 0..5, rows 0..4.

Live (teleop with the leader; Enter = record, r = redo, q = done early):
    python fk_check_board.py --port /dev/ttyACM0 --id so101_follower_vasco \
        --leader-port /dev/ttyACM1 --leader-id so101_leader_vasco \
        --corners 0,0 5,0 5,4 0,4 2,2 4,1 --tip-offset-cm 3

Re-evaluate a saved run (e.g. after changing JOINT_OFFSETS_DEG), no robot needed:
    python fk_check_board.py --offline fk_board_YYYYmmdd_HHMMSS.json --tip-offset-cm 3

Reports:
  * every pairwise horizontal distance: FK vs board, error
  * best-fit of FK points onto the board grid: SCALE (should be 1.000) and residual per point
  * height of every point (all on the same flat board, so they should match)
  * pencil tilt from vertical at each point (keep < ~5 deg)
"""
import argparse
import json
import time
from datetime import datetime
from itertools import combinations

import numpy as np

from so101_kinematics import ARM_JOINTS, DEFAULT_URDF, JOINT_OFFSETS_DEG, SO101Kinematics

RATE_HZ = 30
SAMPLES = 15
SAMPLE_PERIOD_S = 0.03
SYNC_TIME_S = 2.0


def default_square_mm():
    try:
        import config
        return config.SQUARE_LENGTH_M * 1000.0
    except Exception:
        return None


def parse_corner(s):
    c, r = s.split(",")
    return (int(c), int(r))


# --------------------------------------------------------------------------- recording
def record_live(args, corners):
    """Teleop with the leader; Enter records the follower's joints. Returns list of joints."""
    import queue
    import threading
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    try:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    except ImportError:
        from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

    kin = SO101Kinematics(args.urdf, tip_offset_m=args.tip_offset_cm / 100.0)
    all_joints = ARM_JOINTS + ["gripper"]
    get = lambda d, names: np.array([d[f"{j}.pos"] for j in names], dtype=float)

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id, use_degrees=True))
    # calibrate=False: never start an interactive recalibration of the arms
    robot.connect(calibrate=False)
    leader.connect(calibrate=False)
    recorded = []
    try:
        start = get(robot.get_observation(), all_joints)
        goal = get(leader.get_action(), all_joints)
        steps = int(SYNC_TIME_S * RATE_HZ)
        for i in range(1, steps + 1):
            q = start + (goal - start) * i / steps
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(all_joints, q)})
            time.sleep(1.0 / RATE_HZ)

        cmds = queue.Queue()

        def reader():
            while True:
                line = input().strip().lower()
                cmds.put(line)
                if line == "q":
                    return

        threading.Thread(target=reader, daemon=True).start()
        print(f"\nTeleop running. Pencil on corner {corners[0]}, hold still, press Enter.")
        while True:
            t0 = time.perf_counter()
            robot.send_action(leader.get_action())
            try:
                cmd = cmds.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd == "q":
                break
            if cmd == "r" and recorded:
                recorded.pop()
                print(f"Redo corner {corners[len(recorded)]}.")
            elif cmd is not None and cmd != "r" and len(recorded) < len(corners):
                samples = []
                for _ in range(SAMPLES):
                    robot.send_action(leader.get_action())
                    samples.append(get(robot.get_observation(), ARM_JOINTS))
                    time.sleep(SAMPLE_PERIOD_S)
                samples = np.array(samples)
                q = samples.mean(axis=0)
                jitter = np.max(samples.max(axis=0) - samples.min(axis=0))
                T = kin.fk(q)
                tilt = np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))
                recorded.append(q)
                warn = []
                if jitter > 0.5:
                    warn.append(f"arm moved {jitter:.1f} deg while recording, press r to redo")
                if tilt > 5:
                    warn.append(f"pencil tilted {tilt:.1f} deg, straighten it and press r to redo")
                print(f"[{corners[len(recorded)-1]}] tip (mm) {np.round(T[:3, 3]*1000, 1)}  tilt {tilt:.1f} deg"
                      + ("   !! " + "; ".join(warn) if warn else ""))
                if len(recorded) < len(corners):
                    print(f"Next: corner {corners[len(recorded)]}")
                else:
                    print("All corners done. Park the follower on the table, then type q + Enter.")
            time.sleep(max(0.0, 1.0 / RATE_HZ - (time.perf_counter() - t0)))
        input("Torque turns OFF on disconnect. Make sure the arm is resting, then press Enter.")
    finally:
        robot.disconnect()
        leader.disconnect()
    return recorded


# --------------------------------------------------------------------------- evaluation
def fit_similarity_2d(src, dst):
    """Umeyama: find s, R, t minimising |dst - (s R src + t)|. Returns s, R, t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    E = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        E[1, 1] = -1  # allow mirrored corner numbering
    R = U @ E @ Vt
    s = np.trace(np.diag(S) @ E) / (xs ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def evaluate(corners, tips_mm, tilts_deg, square_mm):
    corners = list(corners)
    grid = np.array(corners, dtype=float) * square_mm      # board coords (mm)
    xy = tips_mm[:, :2]

    print(f"\nSquare size: {square_mm:.2f} mm")
    print("\n corner      FK tip x, y, z (mm)          tilt")
    for c, p, t in zip(corners, tips_mm, tilts_deg):
        print(f" {str(c):8s} {p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f}   {t:5.1f} deg")

    print("\n pair                 FK (mm)   board (mm)   error (mm)")
    errs = []
    for i, j in combinations(range(len(corners)), 2):
        d_fk = np.linalg.norm(xy[i] - xy[j])
        d_gt = np.linalg.norm(grid[i] - grid[j])
        if d_gt < 1e-9:
            continue
        errs.append(d_fk - d_gt)
        print(f" {str(corners[i]):8s}-{str(corners[j]):8s}  {d_fk:8.1f}   {d_gt:9.1f}   {d_fk - d_gt:+9.1f}")
    errs = np.array(errs)

    print("\n================ SUMMARY ================")
    print(f" Pairwise distance error: RMS {np.sqrt(np.mean(errs**2)):.1f} mm, max {np.max(np.abs(errs)):.1f} mm")
    if len(corners) >= 3:
        s, R, t = fit_similarity_2d(grid, xy)
        resid = np.linalg.norm(xy - (s * (R @ grid.T).T + t), axis=1)
        print(f" Best-fit scale FK/board: {s:.4f}  ({(s-1)*100:+.1f} %)")
        print(f" Best-fit residuals (mm): " + "  ".join(f"{c}:{r:.1f}" for c, r in zip(corners, resid)))
    h = tips_mm[:, 2]
    print(f" Heights: mean {h.mean():.1f} mm, spread (max-min) {h.max()-h.min():.1f} mm")
    print(f" Max pencil tilt: {np.max(tilts_deg):.1f} deg")
    print("\n Verdict guide:")
    print("   distance RMS < ~3 mm, max < ~5 mm, |scale| < 1 %, height spread < ~5 mm  -> FK good")
    print("   scale off by several %                -> joint offset (lift/elbow) or wrong square size")
    print("   heights drift with reach             -> shoulder_lift / elbow_flex offset")
    print("   one point much worse than the others -> pencil slipped / bad touch, redo it")
    return errs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offline", metavar="JSON", help="re-evaluate a saved run, no robot")
    p.add_argument("--port"); p.add_argument("--id")
    p.add_argument("--leader-port"); p.add_argument("--leader-id")
    p.add_argument("--corners", nargs="+", default=["0,0", "5,0", "5,4", "0,4", "2,2", "4,1"],
                   help="inner-corner grid coords col,row (default spread for a 7x6-square board)")
    p.add_argument("--square-mm", type=float, default=default_square_mm(),
                   help="board square size in mm (default: SQUARE_LENGTH_M from config.py)")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--tip-offset-cm", type=float, default=0.0)
    args = p.parse_args()
    if args.square_mm is None:
        p.error("no config.py found: pass --square-mm")

    if args.offline:
        with open(args.offline) as f:
            run = json.load(f)
        corners = [tuple(c) for c in run["corners"]]
        joints = [np.array(q, dtype=float) for q in run["joints_lerobot_deg"]]
        print(f"Loaded {len(joints)} points from {args.offline} (recorded {run.get('time', '?')})")
    else:
        if not all([args.port, args.id, args.leader_port, args.leader_id]):
            p.error("live mode needs --port --id --leader-port --leader-id (or use --offline)")
        corners = [parse_corner(c) for c in args.corners]
        joints = record_live(args, corners)
        corners = corners[:len(joints)]
        out = f"fk_board_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(out, "w") as f:
            json.dump({"time": datetime.now().isoformat(), "corners": corners,
                       "joints_lerobot_deg": [q.tolist() for q in joints],
                       "offsets_deg_at_record": JOINT_OFFSETS_DEG.tolist(),
                       "tip_offset_cm": args.tip_offset_cm, "square_mm": args.square_mm}, f, indent=1)
        print(f"Saved raw joints to {out} (re-evaluate later with --offline {out})")

    if len(joints) < 2:
        print("Need at least 2 points.")
        return

    kin = SO101Kinematics(args.urdf, tip_offset_m=args.tip_offset_cm / 100.0)
    Ts = [kin.fk(q) for q in joints]
    tips_mm = np.array([T[:3, 3] * 1000 for T in Ts])
    tilts = np.array([np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))) for T in Ts])
    print(f"\nOffsets (deg): {np.round(JOINT_OFFSETS_DEG, 2)}   pencil offset: {args.tip_offset_cm} cm")
    evaluate(corners, tips_mm, tilts, args.square_mm)


if __name__ == "__main__":
    main()
