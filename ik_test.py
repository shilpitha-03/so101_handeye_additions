#!/usr/bin/env python
"""
IK + low-level control test (distances only, YOU mark each target point).

No contact: at each target the tip stops TOUCH_GAP_M above the paper and the script
pauses so you can mark the paper directly under the tip with a fine pen. Distances are
measured from mark 1 (the script's own first 0,0 visit), not from the teleop start.

1. Teleop the pencil tip onto a start mark S (pointing straight down), press Enter.
   From then on the script controls the follower; the leader is released.
2. For each move (dx, dy) in cm relative to S, in base-frame axes:
     lift -> joint-space travel (min-jerk) -> straight vertical descent -> dot -> lift
   At each touchdown it prints:
     IK error        : FK(IK solution) vs goal         -> solver
     tracking error  : commanded vs encoder joints     -> servos / gravity sag
     reached error   : FK(encoder joints) vs goal      -> what the tracking error costs at the tip
3. You measure ruler distances from mark 1 to each mark; it compares them to the commands.

Run (lerobot env, next to so101_kinematics.py):
    python ik_test.py --port /dev/ttyACM0 --id so101_follower_vasco \
        --leader-port /dev/ttyACM1 --leader-id so101_leader_vasco --tip-offset-cm 3
Options:
    --moves 0,0 10,0 0,10 0,0 0,0     (cm; first 0,0 is the reference mark,
                                       later 0,0 visits test repeatability)
    --no-pause                        (don't stop at targets for marking)
    --correct 1                       (one software sag-correction step per touchdown)
    --dry-run                         (NO hardware: solve IK from --start-joints and print)

Safety: nothing moves unless you type 'go'. A bare Enter never starts motion.
If LeRobot asks about calibration at connect: press ENTER only (uses the saved file).
Never type 'c' there unless you intend to recalibrate.
"""
import argparse
import queue
import threading
import time
from datetime import datetime

import numpy as np

from so101_kinematics import ARM_JOINTS, DEFAULT_URDF, SO101Kinematics, tilt_from_vertical_deg

try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
except ImportError:
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
try:
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
except ImportError:
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

ALL_JOINTS = ARM_JOINTS + ["gripper"]
RATE_HZ = 50
HOVER_M = 0.03            # lift height between dots
TOUCH_GAP_M = 0.002       # stop this far ABOVE the paper at each target (no contact)
DESCENT_STEPS = 15        # straight-line descent/lift resolution
MAX_JOINT_SPEED_DEG = 45  # sets move durations
MIN_MOVE_S = 1.0
MAX_JUMP_DEG = 70         # refuse IK solutions that jump more than this from the seed
IK_MAX_ERR_M = 0.003
FLOOR_MARGIN_M = 0.005    # never command more than this below the start surface
SETTLE_S = 0.6


class Arm:
    def __init__(self, robot, gripper):
        self.robot, self.gripper = robot, gripper

    def read(self):
        obs = self.robot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in ARM_JOINTS], dtype=float)

    def send(self, q):
        a = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, q)}
        a["gripper.pos"] = float(self.gripper)
        self.robot.send_action(a)

    def move_joints(self, q_from, q_to, duration=None):
        """Min-jerk interpolation in joint space."""
        span = np.max(np.abs(q_to - q_from))
        duration = duration or max(MIN_MOVE_S, span / MAX_JOINT_SPEED_DEG)
        n = max(2, int(duration * RATE_HZ))
        for i in range(1, n + 1):
            s = i / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5
            self.send(q_from + (q_to - q_from) * s)
            time.sleep(1.0 / RATE_HZ)


def solve(kin, q_seed, T_goal, floor_z):
    if T_goal[2, 3] < floor_z:
        raise RuntimeError(f"goal z {T_goal[2, 3]:.4f} below floor {floor_z:.4f}")
    q, err = kin.ik(q_seed, T_goal)
    if err > IK_MAX_ERR_M:
        raise RuntimeError(f"IK did not converge ({err * 1000:.1f} mm)")
    jump = np.abs(q - q_seed)
    if np.any(jump > MAX_JUMP_DEG):
        raise RuntimeError(f"IK jump too large (deg): {np.round(jump, 1)}")
    return q, err


def line(kin, arm, q_start, T_from, T_to, floor_z, steps=DESCENT_STEPS, step_s=0.08):
    """Straight Cartesian line: IK at each step, stream the joint solutions."""
    q = q_start
    for i in range(1, steps + 1):
        T = T_from.copy()
        T[:3, 3] = T_from[:3, 3] + (T_to[:3, 3] - T_from[:3, 3]) * i / steps
        q_next, _ = solve(kin, q, T, floor_z)
        arm.move_joints(q, q_next, duration=step_s)
        q = q_next
    return q


def measure(kin, arm, q_cmd, T_goal, ik_err, label):
    time.sleep(SETTLE_S)
    q_meas = arm.read()
    reached = kin.fk(q_meas)
    miss = reached[:3, 3] - T_goal[:3, 3]
    print(f"  [{label}] IK err {ik_err * 1000:5.2f} mm | tracking err (deg) "
          f"{np.round(q_meas - q_cmd, 2)} | reached err {np.linalg.norm(miss) * 1000:5.1f} mm "
          f"(dx {miss[0] * 1000:+.1f}, dy {miss[1] * 1000:+.1f}, dz {miss[2] * 1000:+.1f}) | "
          f"tilt {tilt_from_vertical_deg(reached):.1f} deg")
    return q_meas, miss


SYNC_MAX_DIFF_DEG = 20.0


def ask_go(msg):
    """Only the word 'go' continues; anything else re-asks. 'quit' aborts."""
    while True:
        r = input(f"{msg} Type 'go' to continue (or 'quit'): ").strip().lower()
        if r == "go":
            return
        if r == "quit":
            raise KeyboardInterrupt


def teleop_to_start(robot, leader):
    get = lambda d, names: np.array([d[f"{j}.pos"] for j in names], dtype=float)
    while True:
        start = get(robot.get_observation(), ALL_JOINTS)
        goal = get(leader.get_action(), ALL_JOINTS)
        diff = goal - start
        print("\nLeader minus follower (deg):",
              {j: round(float(d), 1) for j, d in zip(ALL_JOINTS, diff)})
        if np.max(np.abs(diff[:5])) <= SYNC_MAX_DIFF_DEG:
            break
        input(f"Some joints differ by more than {SYNC_MAX_DIFF_DEG:.0f} deg. Move the LEADER "
              f"to roughly match the follower's pose, then press Enter to re-check.")
    ask_go("The follower will move to the leader's pose over 3 s.")
    for i in range(1, 3 * 30 + 1):
        q = start + diff * i / 90
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(ALL_JOINTS, q)})
        time.sleep(1 / 30)
    cmds = queue.Queue()
    threading.Thread(target=lambda: cmds.put(input()), daemon=True).start()
    print("\nTeleop: put the pencil on start mark S, pointing straight down, hold still,"
          " press Enter.")
    while cmds.empty():
        robot.send_action(leader.get_action())
        time.sleep(1 / 30)
    time.sleep(0.5)


def dry_run(kin, q_start, moves):
    """No hardware: IK for every target from a recorded start pose."""
    T0 = kin.fk(q_start)
    floor_z = T0[2, 3] - FLOOR_MARGIN_M
    print(f"DRY RUN (no hardware). Start tip {np.round(T0[:3, 3], 4)} m, "
          f"tilt {tilt_from_vertical_deg(T0):.1f} deg")
    q = q_start
    for dx, dy in moves:
        T = T0.copy()
        T[:3, 3] += [dx, dy, 0]
        try:
            qs, e = solve(kin, q, T, floor_z)
            print(f"  ({dx * 100:+.0f},{dy * 100:+.0f}) cm -> joints {qs}  IK err {e * 1000:.2f} mm  "
                  f"max joint change {np.max(np.abs(qs - q)):.1f} deg")
        except RuntimeError as err:
            print(f"  ({dx * 100:+.0f},{dy * 100:+.0f}) cm -> REFUSED: {err}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port")
    p.add_argument("--id")
    p.add_argument("--leader-port")
    p.add_argument("--leader-id")
    p.add_argument("--start-joints", type=float, nargs=5,
                   default=[-4.26, 132.53, -106.73, 81.54, -13.41],
                   help="dry run start pose, raw LeRobot deg (default: point C)")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--tip-offset-cm", type=float, default=0.0)
    p.add_argument("--moves", nargs="+", default=["0,0", "10,0", "0,10", "0,0", "0,0"])
    p.add_argument("--no-pause", action="store_true", help="don't pause at targets")
    p.add_argument("--correct", type=int, default=0, help="software correction steps")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    moves = [tuple(float(v) / 100 for v in m.split(",")) for m in args.moves]

    np.set_printoptions(precision=2, suppress=True)
    kin = SO101Kinematics(args.urdf, tip_offset_m=args.tip_offset_cm / 100.0)
    if args.dry_run:
        dry_run(kin, np.array(args.start_joints), moves)
        return
    if not all([args.port, args.id, args.leader_port, args.leader_id]):
        p.error("real runs need --port --id --leader-port --leader-id")
    print("Connecting. If LeRobot asks about calibration: press ENTER only, never 'c'.")
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id,
                                           use_degrees=True))
    robot.connect()
    leader.connect()
    results = []
    try:
        teleop_to_start(robot, leader)
        obs = robot.get_observation()
        arm = Arm(robot, obs["gripper.pos"])
        q = arm.read()
        T0 = kin.fk(q)
        floor_z = T0[2, 3] - FLOOR_MARGIN_M
        arm.move_joints(q, q, duration=0.3)  # hold here; leader no longer used
        # Goal orientation: tool z straight down, keep the start yaw.
        x_axis = T0[:3, 0].copy()
        x_axis[2] = 0.0
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.array([0.0, 0.0, -1.0])
        T_ref = T0.copy()
        T_ref[:3, :3] = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
        print(f"Start S: tip {np.round(T0[:3, 3], 4)} m, tilt {tilt_from_vertical_deg(T0):.1f} deg")
        print(f"Moves (cm): {[(m[0] * 100, m[1] * 100) for m in moves]}   "
              f"correction steps: {args.correct}")
        ask_go("The script will now move the arm on its own (lift, travel, dots).")
        T_up0 = T_ref.copy()
        T_up0[2, 3] += HOVER_M
        T_start = T0.copy()
        T_start[2, 3] += 0.005  # rise 5 mm before reorienting, so the tip doesn't drag
        q_s, _ = solve(kin, q, T_start, floor_z)
        arm.move_joints(q, q_s, duration=0.8)
        q_up, _ = solve(kin, q_s, T_up0, floor_z)
        arm.move_joints(q_s, q_up)
        q = q_up

        for i, (dx, dy) in enumerate(moves, 1):
            label = f"{i}: {dx * 100:+.0f},{dy * 100:+.0f} cm"
            T_touch = T_ref.copy()
            T_touch[:3, 3] += [dx, dy, TOUCH_GAP_M]
            T_hover = T_touch.copy()
            T_hover[2, 3] = T0[2, 3] + HOVER_M
            q_hover, _ = solve(kin, q, T_hover, floor_z)
            arm.move_joints(q, q_hover)
            q = line(kin, arm, q_hover, T_hover, T_touch, floor_z)
            q_cmd = q
            _, ik_err = kin.ik(q_hover, T_touch)
            q_meas, miss = measure(kin, arm, q_cmd, T_touch, ik_err, label)
            for k in range(args.correct):
                T_corr = T_touch.copy()
                T_corr[:3, 3] -= miss
                q_cmd, ik_err = solve(kin, q_meas, T_corr, floor_z)
                arm.move_joints(q_meas, q_cmd, duration=0.5)
                q_meas, miss = measure(kin, arm, q_cmd, T_touch, ik_err, f"{label} corr{k + 1}")
                q = q_cmd
            results.append((label, (dx, dy), np.linalg.norm(miss) * 1000))
            if not args.no_pause:
                ask_go(f"Target {label}: mark the paper directly under the pencil tip "
                       f"(fine pen). Then the arm lifts.")
            q = line(kin, arm, q, T_touch, T_hover, floor_z)

        q_rest, _ = solve(kin, q, T_up0, floor_z)
        arm.move_joints(q, q_rest)
        print("\nDone. Arm is hovering above the start.")

        ref_label, ref_xy, _ = results[0]
        print(f"\nRuler distance (cm) from mark [{ref_label}] to each other mark. Blank = skip.")
        rows = []
        for label, xy, reached_mm in results[1:]:
            cmd_cm = np.hypot(xy[0] - ref_xy[0], xy[1] - ref_xy[1]) * 100
            raw = input(f"  mark {label} (commanded {cmd_cm:.1f}): ").strip()
            if raw:
                rows.append((label, cmd_cm, float(raw), reached_mm))
        if rows:
            print("\n move              commanded  ruler   error   (FK-reached err)")
            for label, c, r, m in rows:
                print(f" {label:16s} {c:8.2f} {r:7.2f} {r - c:+7.2f} cm   ({m:.1f} mm)")
            e = np.array([r - c for _, c, r, _ in rows])
            print(f" RMS {np.sqrt(np.mean(e ** 2)):.2f} cm, max {np.max(np.abs(e)):.2f} cm")
            with open(f"ik_test_{datetime.now():%Y%m%d_%H%M%S}.csv", "w") as f:
                f.write("move,commanded_cm,ruler_cm,error_cm,fk_reached_err_mm,correct_steps\n")
                for label, c, r, m in rows:
                    f.write(f"\"{label}\",{c:.2f},{r:.2f},{r - c:.2f},{m:.1f},{args.correct}\n")
        input("\nSupport the arm (torque turns OFF on disconnect), then press Enter.")
    except KeyboardInterrupt:
        print("\nInterrupted. Torque turns OFF on disconnect - support the arm!")
    except RuntimeError as e:
        print(f"\nSTOPPED for safety: {e}")
        input("Support the arm (torque turns OFF on disconnect), then press Enter.")
    finally:
        robot.disconnect()
        leader.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()