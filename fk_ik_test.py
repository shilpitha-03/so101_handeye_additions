#!/usr/bin/env python
"""
FK / IK sanity check for the SO-101 follower using LeRobot's RobotKinematics.

Features:
  - FK test: move to a joint-angle target and report the resulting FK pose.
  - IK test: solve an end-effector target and check singularities along the path.
  - Base-frame and starting gripper-frame displacement diagnostics.
  - Post-move verification using FK of measured joint positions.

Examples:
  python fk_ik_test.py --port /dev/ttyACM0 --id my_follower \
      --urdf /absolute/path/to/so101_new_calib_camera.urdf

  python fk_ik_test.py --port /dev/ttyACM0 --id my_follower \
      --urdf /absolute/path/to/so101_new_calib_camera.urdf --dry-run

  python fk_ik_test.py --port /dev/ttyACM0 --id my_follower \
      --urdf /absolute/path/to/so101_new_calib_camera.urdf \
      --skip-fk --offset 0 0 0.03

  python fk_ik_test.py --port /dev/ttyACM0 --id my_follower \
      --urdf /absolute/path/to/so101_new_calib_camera.urdf \
      --skip-fk --target 0.20 0.0 0.10
"""

import argparse
import time

import numpy as np

from lerobot.model.kinematics import RobotKinematics

try:
    from lerobot.robots.so_follower import (
        SO101Follower,
        SO101FollowerConfig,
    )
except ImportError:
    from lerobot.robots.so101_follower import (
        SO101Follower,
        SO101FollowerConfig,
    )


ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# ---- Test targets -----------------------------------------------------------

# FK test: joint offsets added to CURRENT joints, in degrees.
FK_JOINT_OFFSET_DEG = np.array([10.0, -10.0, 10.0, 0.0, 0.0])

# Default IK target: offset in BASE-FRAME axes, in meters.
IK_EE_OFFSET_M = np.array([0.0, 0.0, 0.03])

# ---- Motion safety ----------------------------------------------------------

MAX_JOINT_JUMP_DEG = 30.0
MOVE_TIME_S = 2.0
RATE_HZ = 30

IK_MAX_ERR_M = 0.01
IK_TOL_M = 0.001
IK_MAX_ITERS = 200

# Post-move position error warning threshold.
POST_MOVE_WARN_ERR_M = 0.010  # 10 mm

# ---- Singularity guardrails -------------------------------------------------

SIGMA_MIN_THRESH = 0.02
COND_MAX = 25.0
PATH_CHECK_POINTS = 20
FD_EPS_DEG = 0.1


def read_arm_joints(robot):
    """Return measured arm joints (degrees) and gripper position."""
    obs = robot.get_observation()
    arm = np.array(
        [obs[f"{j}.pos"] for j in ARM_JOINTS],
        dtype=float,
    )
    return arm, obs["gripper.pos"]


def position_jacobian(kin, q_deg):
    """Numerical 3x5 position Jacobian, units m/rad."""
    J = np.zeros((3, len(q_deg)))

    for i in range(len(q_deg)):
        dq = np.zeros_like(q_deg)
        dq[i] = FD_EPS_DEG

        p_plus = kin.forward_kinematics(q_deg + dq)[:3, 3]
        p_minus = kin.forward_kinematics(q_deg - dq)[:3, 3]

        J[:, i] = (
            (p_plus - p_minus)
            / (2 * np.deg2rad(FD_EPS_DEG))
        )

    return J


def singularity_metrics(kin, q_deg):
    """Return minimum singular value, condition number, manipulability."""
    s = np.linalg.svd(position_jacobian(kin, q_deg), compute_uv=False)

    sigma_min = s[-1]
    cond = s[0] / max(sigma_min, 1e-12)
    manip = float(np.prod(s))

    return sigma_min, cond, manip


def check_singularity(kin, q_deg, label):
    sigma_min, cond, manip = singularity_metrics(kin, q_deg)

    ok = (
        sigma_min >= SIGMA_MIN_THRESH
        and cond <= COND_MAX
    )

    tag = "OK " if ok else "BAD"

    print(
        f"  [{tag}] {label}: "
        f"sigma_min={sigma_min:.4f} m/rad  "
        f"cond={cond:6.1f}  "
        f"manipulability={manip:.2e}"
    )

    return ok


def check_path(kin, q_start, q_goal):
    """Check singularity metrics along an interpolated joint-space path."""
    worst_sigma = np.inf
    worst_cond = 0.0
    worst_t = 0.0
    all_ok = True

    for t in np.linspace(0.0, 1.0, PATH_CHECK_POINTS + 1):
        q = q_start + (q_goal - q_start) * t
        sigma_min, cond, _ = singularity_metrics(kin, q)

        if sigma_min < worst_sigma:
            worst_sigma = sigma_min
            worst_cond = cond
            worst_t = t

        if sigma_min < SIGMA_MIN_THRESH or cond > COND_MAX:
            all_ok = False

    tag = "OK " if all_ok else "BAD"

    print(
        f"  [{tag}] path worst point (t={worst_t:.2f}): "
        f"sigma_min={worst_sigma:.4f} m/rad  "
        f"cond={worst_cond:6.1f}"
    )

    return all_ok


def move_joints(robot, start, goal, gripper):
    """Interpolate a joint-space move and preserve the gripper command."""
    jump = np.abs(goal - start)

    if np.any(jump > MAX_JOINT_JUMP_DEG):
        raise RuntimeError(
            f"Move too large (deg per joint): {np.round(jump, 1)}"
        )

    steps = int(MOVE_TIME_S * RATE_HZ)

    for i in range(1, steps + 1):
        q = start + (goal - start) * i / steps

        action = {
            f"{j}.pos": float(q[k])
            for k, j in enumerate(ARM_JOINTS)
        }
        action["gripper.pos"] = float(gripper)

        robot.send_action(action)
        time.sleep(1.0 / RATE_HZ)

    time.sleep(0.5)


def print_pose(label, T):
    np.set_printoptions(precision=4, suppress=True)

    print(f"{label} position (m): {T[:3, 3]}")
    print(f"{label} rotation:\n{T[:3, :3]}")


def print_displacement_diagnostics(T_start, T_goal):
    """
    Print the desired displacement in base coordinates and in the
    starting end-effector frame.

    Both transforms must describe the same end-effector frame relative
    to the robot base.

    delta_base = p_goal - p_start
    delta_ee = R_base_ee.T @ delta_base
    """
    p_start = T_start[:3, 3]
    p_goal = T_goal[:3, 3]

    delta_base = p_goal - p_start

    # Express the same displacement in the STARTING EE frame.
    R_base_ee = T_start[:3, :3]
    delta_ee = R_base_ee.T @ delta_base

    print("\n--- EE displacement diagnostics ---")
    print("Start position (base, m):        ", np.round(p_start, 5))
    print("Goal position (base, m):         ", np.round(p_goal, 5))
    print("Requested offset (base, m):      ", np.round(delta_base, 5))
    print("Requested offset (gripper, m):   ", np.round(delta_ee, 5))
    print(
        "Requested distance (mm):         ",
        round(np.linalg.norm(delta_base) * 1000, 2),
    )

    return delta_base, delta_ee


def verify_actual_motion(T_start, T_goal, T_actual):
    """
    Compare actual EE displacement with the requested displacement.

    T_actual is FK evaluated using the measured post-move joint angles.
    """
    requested_delta_base = T_goal[:3, 3] - T_start[:3, 3]
    actual_delta_base = T_actual[:3, 3] - T_start[:3, 3]

    displacement_error = actual_delta_base - requested_delta_base

    # Position error between the actual and requested final EE positions.
    final_position_error = np.linalg.norm(
        T_actual[:3, 3] - T_goal[:3, 3]
    )

    # Express actual displacement in the STARTING gripper frame.
    actual_delta_ee = (
        T_start[:3, :3].T @ actual_delta_base
    )

    # Requested displacement in the same starting gripper frame.
    requested_delta_ee = (
        T_start[:3, :3].T @ requested_delta_base
    )

    print("\n--- IK motion verification ---")
    print("Requested displacement (base, m):    ",
          np.round(requested_delta_base, 5))
    print("Actual displacement (base, m):       ",
          np.round(actual_delta_base, 5))

    print("Requested displacement (gripper, m):",
          np.round(requested_delta_ee, 5))
    print("Actual displacement (gripper, m):   ",
          np.round(actual_delta_ee, 5))

    print("Displacement error (mm):              ",
          np.round(displacement_error * 1000, 2))
    print(f"Final EE position error: "
          f"{final_position_error * 1000:.2f} mm")

    if final_position_error > POST_MOVE_WARN_ERR_M:
        print(
            "!! WARNING: Actual EE position differs from the target "
            f"by more than {POST_MOVE_WARN_ERR_M * 1000:.1f} mm."
        )
        print(
            "!! Check joint tracking, calibration, frame definitions, "
            "and possible mechanical obstruction."
        )
    else:
        print("Position verification: within warning threshold.")

    return final_position_error


def confirm(msg, dry_run):
    if dry_run:
        print(f"(dry run) skipping: {msg}")
        return False

    return input(f"{msg} [y/N]: ").strip().lower() == "y"


def solve_ik_guarded(kin, q_start, T_goal):
    """Solve IK and reject inaccurate, large-jump, or singular solutions."""
    q_ik = q_start.copy()
    ik_err = np.inf

    for it in range(IK_MAX_ITERS):
        q_ik = np.asarray(
            kin.inverse_kinematics(
                q_ik,
                T_goal,
                position_weight=1.0,
                orientation_weight=0.01,
            )
        )[:len(ARM_JOINTS)]

        ik_err = np.linalg.norm(
            kin.forward_kinematics(q_ik)[:3, 3]
            - T_goal[:3, 3]
        )

        if ik_err < IK_TOL_M:
            break

    print(f"IK iterations: {it + 1}")
    print("IK joints (deg):", np.round(q_ik, 2))
    print(f"IK solution position error: {ik_err * 1000:.1f} mm")

    if ik_err > IK_MAX_ERR_M:
        print(
            "!! IK did not converge. Goal may be unreachable "
            "or near a singularity."
        )
        return None

    jump = np.abs(q_ik - q_start)

    if np.any(jump > MAX_JOINT_JUMP_DEG):
        print(f"!! IK solution jumps too far (deg per joint): {np.round(jump, 1)}")
        return None

    print("Singularity check:")

    ok = check_singularity(kin, q_start, "start")
    ok &= check_singularity(kin, q_ik, "goal")
    ok &= check_path(kin, q_start, q_ik)

    if not ok:
        print(
            "!! Near a singularity. Not moving. Pick a goal away from "
            "full arm extension, the base's vertical axis, and a "
            "fully folded elbow."
        )
        return None

    return q_ik


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--port", required=True)
    p.add_argument("--id", required=True, help="Robot ID used during calibration")
    p.add_argument("--urdf", required=True, help="Absolute path to the URDF")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and check, never move",
    )
    p.add_argument(
        "--skip-fk",
        action="store_true",
        help="Skip the FK test move and go straight to IK",
    )

    goal = p.add_mutually_exclusive_group()
    goal.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Absolute EE target in base-frame coordinates (meters)",
    )
    goal.add_argument(
        "--offset",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help="Relative EE offset along base-frame axes (meters)",
    )

    args = p.parse_args()

    kin = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINTS,
    )

    # The joint values supplied to RobotKinematics are in degrees.
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.id,
            use_degrees=True,
        )
    )

    robot.connect()

    try:
        q0, grip = read_arm_joints(robot)

        print("Current joints (deg):", np.round(q0, 2))

        T0 = kin.forward_kinematics(q0)
        print_pose("Current EE", T0)
        check_singularity(kin, q0, "current pose")

        # ---------------- Test 1: FK ----------------
        q_fk = q0 + FK_JOINT_OFFSET_DEG

        print("\n[FK test] target joints (deg):", np.round(q_fk, 2))
        print_pose("Predicted EE", kin.forward_kinematics(q_fk))

        if not args.skip_fk and confirm("Move to FK target?", args.dry_run):
            move_joints(robot, q0, q_fk, grip)

            q1, _ = read_arm_joints(robot)

            print("Reached joints (deg):", np.round(q1, 2))
            print("Joint tracking error (deg):", np.round(q1 - q_fk, 2))

            T1 = kin.forward_kinematics(q1)
            print_pose("FK of reached joints", T1)
            print("Measure the real gripper tip to independently validate FK.")
        else:
            q1 = q0.copy()
            T1 = kin.forward_kinematics(q1)

        # ---------------- Test 2: IK ----------------
        # Preserve the current EE orientation and construct the target
        # by modifying the translation in BASE-FRAME coordinates.
        T_goal = T1.copy()

        if args.target is not None:
            T_goal[:3, 3] = np.asarray(args.target, dtype=float)
        elif args.offset is not None:
            T_goal[:3, 3] += np.asarray(args.offset, dtype=float)
        else:
            T_goal[:3, 3] += IK_EE_OFFSET_M

        step = np.linalg.norm(T_goal[:3, 3] - T1[:3, 3])

        print(f"\nRequested EE move distance: {step * 100:.2f} cm")
        print("[IK test] target EE position (m):",
              np.round(T_goal[:3, 3], 5))

        requested_delta_base, requested_delta_ee = (
            print_displacement_diagnostics(T1, T_goal)
        )

        q_ik = solve_ik_guarded(kin, q1, T_goal)

        if q_ik is None:
            print("IK move rejected; no IK motion commanded.")
            return

        if confirm("Move to IK solution?", args.dry_run):
            move_joints(robot, q1, q_ik, grip)

            # Read measured joint angles after motion.
            q2, _ = read_arm_joints(robot)
            T2 = kin.forward_kinematics(q2)

            print("Reached joints (deg):", np.round(q2, 2))
            print_pose("FK of reached joints", T2)

            verify_actual_motion(T1, T_goal, T2)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()

# """
# FK / IK sanity check for the SO-101 follower using LeRobot's RobotKinematics,
# with singularity guardrails on the IK move.

# Test 1 (FK): move to a joint-angle target, read the joints back, print the FK pose.
# Test 2 (IK): pick an end-effector point (current pose + offset), solve IK,
#              check the solution AND the path for singularities, move there,
#              read joints back, and report the position error.

# Run (lerobot conda env):
#   python fk_ik_test.py --port /dev/ttyACM0 --id my_follower \
#       --urdf /home/shilpitha/Documents/so101/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
#   python fk_ik_test.py ... --dry-run     # compute + check everything, never move
#   python fk_ik_test.py ... --skip-fk --target 0.20 0.0 0.10    # absolute EE goal, base frame (m)
#   python fk_ik_test.py ... --skip-fk --offset 0.0 0.0 0.03     # relative EE goal (m)
# """
# import argparse
# import time

# import numpy as np

# from lerobot.model.kinematics import RobotKinematics

# try:  # import path differs between LeRobot versions
#     from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
# except ImportError:
#     from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# # ---- Edit these test targets -------------------------------------------------
# # FK test: joint offsets (degrees) added to the CURRENT joints. Keep them small.
# FK_JOINT_OFFSET_DEG = np.array([10.0, -10.0, 10.0, 0.0, 0.0])
# # IK test: end-effector offset (meters, base frame) from the pose after the FK test.
# IK_EE_OFFSET_M = np.array([0.0, 0.0, 0.03])

# # ---- Motion safety -----------------------------------------------------------
# MAX_JOINT_JUMP_DEG = 30.0   # refuse any single move bigger than this per joint
# MOVE_TIME_S = 2.0           # duration of each interpolated move
# RATE_HZ = 30
# IK_MAX_ERR_M = 0.01#0.005        # IK solution must land within 5 mm of the goal
# IK_TOL_M = 0.001            # stop iterating once within 1 mm
# IK_MAX_ITERS = 200          # max IK solver steps

# # ---- Singularity guardrails (tune after looking at printed values) -----------
# # Computed on the 3x5 POSITION Jacobian (m/rad), since we mainly control position.
# SIGMA_MIN_THRESH = 0.02     # smallest singular value: m of tip motion per rad, worst direction
# COND_MAX = 25.0             # sigma_max / sigma_min: how lopsided the arm's mobility is
# PATH_CHECK_POINTS = 20      # waypoints along the joint-space path to check
# FD_EPS_DEG = 0.1            # finite-difference step for the Jacobian
# # -----------------------------------------------------------------------------


# def read_arm_joints(robot):
#     obs = robot.get_observation()
#     arm = np.array([obs[f"{j}.pos"] for j in ARM_JOINTS], dtype=float)
#     return arm, obs["gripper.pos"]


# def position_jacobian(kin, q_deg):
#     """Numerical 3x5 position Jacobian, units m/rad (central differences on FK)."""
#     J = np.zeros((3, len(q_deg)))
#     for i in range(len(q_deg)):
#         dq = np.zeros_like(q_deg)
#         dq[i] = FD_EPS_DEG
#         p_plus = kin.forward_kinematics(q_deg + dq)[:3, 3]
#         p_minus = kin.forward_kinematics(q_deg - dq)[:3, 3]
#         J[:, i] = (p_plus - p_minus) / (2 * np.deg2rad(FD_EPS_DEG))
#     return J


# def singularity_metrics(kin, q_deg):
#     """Returns (sigma_min, condition number, manipulability) of the position Jacobian."""
#     s = np.linalg.svd(position_jacobian(kin, q_deg), compute_uv=False)
#     sigma_min = s[-1]
#     cond = s[0] / max(sigma_min, 1e-12)
#     manip = float(np.prod(s))  # = sqrt(det(J J^T))
#     return sigma_min, cond, manip


# def check_singularity(kin, q_deg, label):
#     sigma_min, cond, manip = singularity_metrics(kin, q_deg)
#     ok = sigma_min >= SIGMA_MIN_THRESH and cond <= COND_MAX
#     tag = "OK " if ok else "BAD"
#     print(f"  [{tag}] {label}: sigma_min={sigma_min:.4f} m/rad  cond={cond:6.1f}  "
#           f"manipulability={manip:.2e}")
#     return ok


# def check_path(kin, q_start, q_goal):
#     """Check singularity metrics along the interpolated joint-space path."""
#     worst_sigma, worst_cond, worst_t = np.inf, 0.0, 0.0
#     for t in np.linspace(0.0, 1.0, PATH_CHECK_POINTS + 1):
#         sigma_min, cond, _ = singularity_metrics(kin, q_start + (q_goal - q_start) * t)
#         if sigma_min < worst_sigma:
#             worst_sigma, worst_cond, worst_t = sigma_min, cond, t
#     ok = worst_sigma >= SIGMA_MIN_THRESH and worst_cond <= COND_MAX
#     tag = "OK " if ok else "BAD"
#     print(f"  [{tag}] path worst point (t={worst_t:.2f}): sigma_min={worst_sigma:.4f} m/rad  "
#           f"cond={worst_cond:6.1f}")
#     return ok


# def move_joints(robot, start, goal, gripper):
#     """Linear interpolation in joint space so the arm doesn't snap."""
#     jump = np.abs(goal - start)
#     if np.any(jump > MAX_JOINT_JUMP_DEG):
#         raise RuntimeError(f"Move too large (deg per joint): {np.round(jump, 1)}")
#     steps = int(MOVE_TIME_S * RATE_HZ)
#     for i in range(1, steps + 1):
#         q = start + (goal - start) * i / steps
#         action = {f"{j}.pos": float(q[k]) for k, j in enumerate(ARM_JOINTS)}
#         action["gripper.pos"] = float(gripper)
#         robot.send_action(action)
#         time.sleep(1.0 / RATE_HZ)
#     time.sleep(0.5)  # let the servos settle


# def print_pose(label, T):
#     np.set_printoptions(precision=4, suppress=True)
#     print(f"{label} position (m): {T[:3, 3]}")
#     print(f"{label} rotation:\n{T[:3, :3]}")


# def confirm(msg, dry_run):
#     if dry_run:
#         print(f"(dry run) skipping: {msg}")
#         return False
#     return input(f"{msg} [y/N]: ").strip().lower() == "y"


# def solve_ik_guarded(kin, q_start, T_goal):
#     """Solve IK, then refuse the solution if it's inaccurate, a big jump, or singular."""
#     # LeRobot's inverse_kinematics() does ONE solver step per call (built for teleop
#     # loops at high rate), so iterate it until FK of the result reaches the goal.
#     q_ik = q_start.copy()
#     for it in range(IK_MAX_ITERS):
#         q_ik = np.asarray(kin.inverse_kinematics(q_ik, T_goal, position_weight=1.0,
#                                                  orientation_weight=0.01))[:len(ARM_JOINTS)]
#         ik_err = np.linalg.norm(kin.forward_kinematics(q_ik)[:3, 3] - T_goal[:3, 3])
#         if ik_err < IK_TOL_M:
#             break
#     print(f"IK iterations: {it + 1}")
#     print("IK joints (deg):", np.round(q_ik, 2))

#     # 1. Convergence: does FK of the solution actually hit the goal?
#     print(f"IK solution error (FK of solution vs goal): {ik_err * 1000:.1f} mm")
#     if ik_err > IK_MAX_ERR_M:
#         print("!! IK did not converge (goal may be out of reach or at a singularity).")
#         return None

#     # 2. Jump size: big joint changes for a small EE move are a classic singularity symptom.
#     jump = np.abs(q_ik - q_start)
#     if np.any(jump > MAX_JOINT_JUMP_DEG):
#         print(f"!! IK solution jumps too far (deg per joint): {np.round(jump, 1)}")
#         return None

#     # 3. Singularity at the goal and along the path there.
#     print("Singularity check:")
#     ok = check_singularity(kin, q_start, "start")
#     ok &= check_singularity(kin, q_ik, "goal ")
#     ok &= check_path(kin, q_start, q_ik)
#     if not ok:
#         print("!! Near a singularity. Not moving. Pick a goal away from full arm "
#               "extension, the base's vertical axis, and fully folded elbow.")
#         return None
#     return q_ik


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--port", required=True)
#     p.add_argument("--id", required=True, help="robot id used during calibration")
#     p.add_argument("--urdf", required=True, help="absolute path to so101_new_calib.urdf")
#     p.add_argument("--dry-run", action="store_true", help="compute and check, never move")
#     p.add_argument("--skip-fk", action="store_true", help="skip the FK move, go straight to IK")
#     goal = p.add_mutually_exclusive_group()
#     goal.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"),
#                       help="absolute EE target in the base frame (meters)")
#     goal.add_argument("--offset", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
#                       help="EE offset from the current pose, base-frame axes (meters)")
#     args = p.parse_args()

#     kin = RobotKinematics(urdf_path=args.urdf, target_frame_name="gripper_frame_link",
#                           joint_names=ARM_JOINTS)

#     # use_degrees=True is essential: kinematics expects degrees, not the -100..100 range
#     robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
#     robot.connect()

#     try:
#         q0, grip = read_arm_joints(robot)
#         print("Current joints (deg):", np.round(q0, 2))
#         print_pose("Current EE", kin.forward_kinematics(q0))
#         check_singularity(kin, q0, "current pose")

#         # ---------------- Test 1: FK ----------------
#         q_fk = q0 + FK_JOINT_OFFSET_DEG
#         print("\n[FK test] target joints (deg):", np.round(q_fk, 2))
#         print_pose("Predicted EE", kin.forward_kinematics(q_fk))
#         if not args.skip_fk and confirm("Move to FK target?", args.dry_run):
#             move_joints(robot, q0, q_fk, grip)
#             q1, _ = read_arm_joints(robot)
#             print("Reached joints (deg):", np.round(q1, 2))
#             print("Joint tracking error (deg):", np.round(q1 - q_fk, 2))
#             T1 = kin.forward_kinematics(q1)
#             print_pose("FK of reached joints", T1)
#             print(">> Measure the real gripper tip and compare with this position.")
#         else:
#             q1, T1 = q0, kin.forward_kinematics(q0)

#         # ---------------- Test 2: IK (guarded) ----------------
#         T_goal = T1.copy()  # keep current orientation
#         if args.target is not None:
#             T_goal[:3, 3] = np.array(args.target)          # absolute, base frame
#         elif args.offset is not None:
#             T_goal[:3, 3] += np.array(args.offset)         # relative, base-frame axes
#         else:
#             T_goal[:3, 3] += IK_EE_OFFSET_M                # default from top of file
#         step = np.linalg.norm(T_goal[:3, 3] - T1[:3, 3])
#         print(f"Requested EE move distance: {step * 100:.1f} cm")
#         print("\n[IK test] target EE position (m):", np.round(T_goal[:3, 3], 4))

#         q_ik = solve_ik_guarded(kin, q1, T_goal)
#         if q_ik is None:
#             return

#         if confirm("Move to IK solution?", args.dry_run):
#             move_joints(robot, q1, q_ik, grip)
#             q2, _ = read_arm_joints(robot)
#             T2 = kin.forward_kinematics(q2)
#             err = np.linalg.norm(T2[:3, 3] - T_goal[:3, 3])
#             print("Reached joints (deg):", np.round(q2, 2))
#             print_pose("FK of reached joints", T2)
#             print(f"End-to-end position error: {err * 1000:.1f} mm")

#     except KeyboardInterrupt:
#         print("\nInterrupted.")
#     finally:
#         robot.disconnect()
#         print("Disconnected.")


# if __name__ == "__main__":
#     main()