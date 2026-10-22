#!/usr/bin/env python
"""
FK distance test for the SO-101 follower (solo version: teleop is built in).

What it tests:
    Does the URDF model (link lengths + joint zero offsets) match the real arm?
    You put the pencil tip on marked points A, B, C, ... using the leader arm.
    The script reads the follower's ENCODER joint angles, runs FK, and compares the
    distances between FK positions with the distances you measured with a ruler.
    Distances don't depend on where the base origin is, so no axis alignment needed.

Pencil offset:
    FK predicts gripper_frame_link (between the fingertips; its z axis points out of
    the jaws). If the pencil tip sticks out d cm past that point, pass
    --tip-offset-cm d so the script predicts the pencil tip instead.

Keys (type in the terminal, then press Enter):
    Enter   record the next point at the current pose
    r       redo the previous point
    q       finish (park the follower on the table with the leader FIRST)

Run (lerobot conda env):
    python fk_distance_test.py \
        --port /dev/ttyACM0 --id my_follower \
        --leader-port /dev/ttyACM1 --leader-id my_leader \
        --urdf ~/Documents/so101/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
        --points A B C D --tip-offset-cm 0
"""
import argparse
import csv
import itertools
import os
import queue
import threading
import time
from datetime import datetime

import numpy as np

from lerobot.model.kinematics import RobotKinematics

try:  # import paths differ between LeRobot versions
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
except ImportError:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
try:
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
except ImportError:
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = ARM_JOINTS + ["gripper"]

RATE_HZ = 30            # teleop loop rate
SAMPLES = 15            # encoder readings averaged per recorded point
SAMPLE_PERIOD_S = 0.03
MOVING_STD_DEG = 0.3    # if joints jitter more than this while recording, warn
SYNC_TIME_S = 2.0       # slow move from follower's pose to leader's pose at start


def joints_from(d, names):
    return np.array([d[f"{j}.pos"] for j in names], dtype=float)


def to_action(q):
    return {f"{j}.pos": float(v) for j, v in zip(ALL_JOINTS, q)}


def keyboard_reader(cmds):
    """Background thread: forwards each typed line to the main loop."""
    while True:
        line = input().strip().lower()
        cmds.put(line)
        if line == "q":
            return


def sync_to_leader(robot, leader):
    """Interpolate the follower to the leader's pose so it doesn't snap at startup."""
    start = joints_from(robot.get_observation(), ALL_JOINTS)
    goal = joints_from(leader.get_action(), ALL_JOINTS)
    steps = int(SYNC_TIME_S * RATE_HZ)
    for i in range(1, steps + 1):
        robot.send_action(to_action(start + (goal - start) * i / steps))
        time.sleep(1.0 / RATE_HZ)


def record_point(robot, leader):
    """Average several encoder readings while still holding the leader's pose."""
    samples = []
    for _ in range(SAMPLES):
        robot.send_action(leader.get_action())
        samples.append(joints_from(robot.get_observation(), ARM_JOINTS))
        time.sleep(SAMPLE_PERIOD_S)
    samples = np.array(samples)
    return samples.mean(axis=0), samples.std(axis=0)


def tip_position(kin, q_deg, tip_offset_m):
    """FK to gripper_frame_link, then shift along its z axis to the pencil tip."""
    T = kin.forward_kinematics(q_deg)
    return (T @ np.array([0.0, 0.0, tip_offset_m, 1.0]))[:3]


def make_leader(port, leader_id):
    try:
        return SO101Leader(SO101LeaderConfig(port=port, id=leader_id, use_degrees=True))
    except TypeError:
        print("WARNING: this LeRobot version's leader config has no use_degrees. "
              "Check that leader and follower report joints in the same units.")
        return SO101Leader(SO101LeaderConfig(port=port, id=leader_id))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="follower port")
    p.add_argument("--id", required=True, help="follower calibration id")
    p.add_argument("--leader-port", required=True)
    p.add_argument("--leader-id", required=True)
    p.add_argument("--urdf", required=True, help="path to so101_new_calib.urdf")
    p.add_argument("--points", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--tip-offset-cm", type=float, default=0.0,
                   help="pencil tip distance past gripper_frame_link, along its z axis")
    p.add_argument("--out", default=None, help="CSV filename prefix")
    args = p.parse_args()
    out = args.out or f"fk_test_{datetime.now():%Y%m%d_%H%M%S}"
    tip_offset_m = args.tip_offset_cm / 100.0

    np.set_printoptions(precision=2, suppress=True)
    kin = RobotKinematics(urdf_path=os.path.expanduser(args.urdf),
                          target_frame_name="gripper_frame_link", joint_names=ARM_JOINTS)
    # use_degrees=True: kinematics expects degrees, not the -100..100 range
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    leader = make_leader(args.leader_port, args.leader_id)
    robot.connect()
    leader.connect()

    records = {}  # label -> (joints_deg, position_m)
    order = list(args.points)
    idx = 0
    try:
        print(f"Pencil tip offset: {args.tip_offset_cm:.2f} cm along gripper_frame_link z")
        print("Syncing follower to leader pose...")
        sync_to_leader(robot, leader)

        cmds = queue.Queue()
        threading.Thread(target=keyboard_reader, args=(cmds,), daemon=True).start()
        print("\nTeleop running. Keys: Enter = record, r = redo previous, q = finish.")
        print(f"Put the pencil tip on {order[0]}, hold the leader still, press Enter.")

        while True:
            t0 = time.perf_counter()
            robot.send_action(leader.get_action())

            try:
                cmd = cmds.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd == "q":
                break
            elif cmd == "r":
                if idx == 0:
                    print("Nothing to redo yet.")
                else:
                    idx -= 1
                    records.pop(order[idx], None)
                    print(f"Redo {order[idx]}: put the pencil tip on it and press Enter.")
            elif cmd is not None:
                if idx >= len(order):
                    print("All points recorded. Park the follower on the table, then type q.")
                else:
                    label = order[idx]
                    q_mean, q_std = record_point(robot, leader)
                    pos = tip_position(kin, q_mean, tip_offset_m)
                    records[label] = (q_mean, pos)
                    print(f"\n[{label}] joint angles (deg): {q_mean}")
                    print(f"[{label}] FK pencil-tip position (m): {np.round(pos, 4)}")
                    if np.any(q_std > MOVING_STD_DEG):
                        print(f"  WARNING: arm was moving while recording "
                              f"(std deg {q_std}). Consider r to redo.")
                    idx += 1
                    if idx < len(order):
                        print(f"Next: put the pencil tip on {order[idx]} and press Enter.")
                    else:
                        print("All points recorded. Park the follower on the table, then type q.")

            time.sleep(max(0.0, 1.0 / RATE_HZ - (time.perf_counter() - t0)))

        # ---------- Compare FK distances with ruler distances ----------
        labels = [l for l in order if l in records]
        pair_rows = []
        if len(labels) >= 2:
            print("\nEnter the TRUE distances you measured (cm). Blank to skip a pair.")
            for a, b in itertools.combinations(labels, 2):
                delta = records[b][1] - records[a][1]
                fk_cm = np.linalg.norm(delta) * 100
                raw = input(f"  True {a}-{b} distance in cm (FK says {fk_cm:.2f}): ").strip()
                true_cm = float(raw) if raw else None
                err_cm = fk_cm - true_cm if true_cm is not None else None
                pair_rows.append((f"{a}-{b}", fk_cm, true_cm, err_cm, *(delta * 100)))

            print("\n pair   FK (cm)  true (cm)  error (cm)  error %   FK delta xyz (cm)")
            for name, fk_cm, true_cm, err_cm, dx, dy, dz in pair_rows:
                t = f"{true_cm:9.2f}" if true_cm is not None else "        -"
                e = f"{err_cm:+10.2f}" if err_cm is not None else "         -"
                pct = f"{100 * err_cm / true_cm:+7.1f}%" if true_cm else "       -"
                print(f" {name:5s} {fk_cm:8.2f} {t} {e}  {pct}   ({dx:+.2f}, {dy:+.2f}, {dz:+.2f})")

        # ---------- Save ----------
        with open(f"{out}_points.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", *[f"{j}_deg" for j in ARM_JOINTS], "x_m", "y_m", "z_m",
                        "tip_offset_cm"])
            for l in labels:
                q, pos = records[l]
                w.writerow([l, *np.round(q, 3), *np.round(pos, 5), args.tip_offset_cm])
        with open(f"{out}_pairs.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "fk_cm", "true_cm", "error_cm", "dx_cm", "dy_cm", "dz_cm"])
            for row in pair_rows:
                w.writerow([row[0], *[None if v is None else round(v, 3) for v in row[1:]]])
        print(f"\nSaved {out}_points.csv and {out}_pairs.csv")

        input("\nTorque turns OFF on disconnect. Make sure the arm is resting, then press Enter.")
    except KeyboardInterrupt:
        print("\nInterrupted. Torque turns OFF on disconnect - support the arm!")
    finally:
        robot.disconnect()
        leader.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()