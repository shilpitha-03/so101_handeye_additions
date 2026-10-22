#!/usr/bin/env python
"""
FK check with DISTANCES only (offset-corrected FK from so101_kinematics.py).

Two ways to run:
  1) Offline, using joint angles you already recorded (no robot needed):
       python fk_check_distances.py --offline --tip-offset-cm 3
     Joint angles come from RECORDED_JOINTS below (edit/paste new ones there).

  2) Live, recording new points with the leader arm (Enter = record, r = redo, q = done):
       python fk_check_distances.py --port /dev/ttyACM0 --id so101_follower_vasco \
           --leader-port /dev/ttyACM1 --leader-id so101_leader_vasco \
           --points A B C D E F --tip-offset-cm 3

Either way, it then asks for the ruler distance of each pair in PAIRS (blank = skip)
and prints FK vs measured, error, RMS and max. It also prints each point's FK height
(table points should be ~0 cm, box points ~ the box height).
"""
import argparse
import csv
import time
from datetime import datetime

import numpy as np

from so101_kinematics import ARM_JOINTS, DEFAULT_URDF, JOINT_OFFSETS_DEG, SO101Kinematics

# Pairs you measure with a ruler. Edit if you use different points.
PAIRS = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"), ("E", "F")]

# Raw LeRobot joint angles (deg) from the 2026-10-22 run. Paste new runs here.
RECORDED_JOINTS = {
    "A": [14.02, 87.43, -49.93, 68.97, 7.78],
    "B": [11.65, 154.07, -140.40, 98.59, 1.19],
    "C": [-4.26, 132.53, -106.73, 81.54, -13.41],
    "D": [-27.21, 133.41, -108.31, 77.93, -13.58],
    "E": [-27.30, 99.47, -104.09, 91.82, -12.00],
    "F": [-1.36, 112.84, -119.47, 94.46, -11.91],
}

RATE_HZ = 30
SAMPLES = 15
SAMPLE_PERIOD_S = 0.03
SYNC_TIME_S = 2.0


def record_live(args, points):
    """Teleop with the leader; Enter records the follower's joints. Returns {label: joints}."""
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

    all_joints = ARM_JOINTS + ["gripper"]
    get = lambda d, names: np.array([d[f"{j}.pos"] for j in names], dtype=float)

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id,
                                           use_degrees=True))
    robot.connect()
    leader.connect()
    recorded = {}
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
        idx = 0
        print(f"\nTeleop running. Put the pencil on {points[0]}, hold still, press Enter.")
        while True:
            t0 = time.perf_counter()
            robot.send_action(leader.get_action())
            try:
                cmd = cmds.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd == "q":
                break
            if cmd == "r" and idx > 0:
                idx -= 1
                recorded.pop(points[idx], None)
                print(f"Redo {points[idx]}.")
            elif cmd is not None and cmd != "r" and idx < len(points):
                samples = []
                for _ in range(SAMPLES):
                    robot.send_action(leader.get_action())
                    samples.append(get(robot.get_observation(), ARM_JOINTS))
                    time.sleep(SAMPLE_PERIOD_S)
                q = np.mean(samples, axis=0)
                recorded[points[idx]] = q
                print(f"[{points[idx]}] joints (deg): {np.round(q, 2)}")
                idx += 1
                print(f"Next: {points[idx]}" if idx < len(points)
                      else "Done. Park the follower on the table, then type q.")
            time.sleep(max(0.0, 1.0 / RATE_HZ - (time.perf_counter() - t0)))
        input("Torque turns OFF on disconnect. Make sure the arm is resting, then press Enter.")
    finally:
        robot.disconnect()
        leader.disconnect()
    return recorded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true", help="use RECORDED_JOINTS, no robot")
    p.add_argument("--port")
    p.add_argument("--id")
    p.add_argument("--leader-port")
    p.add_argument("--leader-id")
    p.add_argument("--points", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--tip-offset-cm", type=float, default=0.0)
    args = p.parse_args()

    if args.offline:
        joints = {k: np.array(v, dtype=float) for k, v in RECORDED_JOINTS.items()}
    else:
        if not all([args.port, args.id, args.leader_port, args.leader_id]):
            p.error("live mode needs --port --id --leader-port --leader-id (or use --offline)")
        joints = record_live(args, args.points)

    kin = SO101Kinematics(args.urdf, tip_offset_m=args.tip_offset_cm / 100.0)
    pos_cm = {k: kin.position(q) * 100 for k, q in joints.items()}

    np.set_printoptions(precision=2, suppress=True)
    print(f"\nOffsets (deg): {JOINT_OFFSETS_DEG}   pencil offset: {args.tip_offset_cm} cm")
    print("FK tip positions (cm) and height above the base:")
    for k, v in pos_cm.items():
        print(f"  {k}: {np.round(v, 2)}   height {v[2]:.2f}")

    print("\nType the ruler distance (cm) for each pair. Blank = skip.")
    rows = []
    for a, b in PAIRS:
        if a not in pos_cm or b not in pos_cm:
            continue
        raw = input(f"  {a}-{b}: ").strip()
        if not raw:
            continue
        d_fk = float(np.linalg.norm(pos_cm[b] - pos_cm[a]))
        rows.append((f"{a}-{b}", d_fk, float(raw), d_fk - float(raw)))

    if rows:
        print("\n pair   FK (cm)  ruler (cm)  error (cm)")
        for name, d_fk, d_ms, e in rows:
            print(f" {name:5s} {d_fk:8.2f} {d_ms:10.2f} {e:+10.2f}")
        errs = np.array([r[3] for r in rows])
        print(f"\n RMS error: {np.sqrt(np.mean(errs ** 2)):.2f} cm   "
              f"max: {np.max(np.abs(errs)):.2f} cm   ({len(rows)} pairs)")

        out = f"fk_dist_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "fk_cm", "ruler_cm", "error_cm", "tip_offset_cm", "offsets_deg"])
            for r in rows:
                w.writerow([r[0], round(r[1], 3), r[2], round(r[3], 3), args.tip_offset_cm,
                            " ".join(map(str, JOINT_OFFSETS_DEG))])
        print(f" Saved {out}")


if __name__ == "__main__":
    main()