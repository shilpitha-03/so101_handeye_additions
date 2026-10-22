"""
Shared SO-101 kinematics with joint-offset correction.

Every script (FK test, hand-eye, IK test, pick-and-place) should import from here,
so the offsets live in ONE place.

Convention (all angles in degrees):
    q_urdf    = q_lerobot - JOINT_OFFSETS_DEG     (before FK / into the model)
    q_lerobot = q_urdf    + JOINT_OFFSETS_DEG     (after IK / back to the motors)
JOINT_OFFSETS_DEG = what LeRobot reads when the arm is in the URDF zero pose.
"""
import os

import numpy as np
from lerobot.model.kinematics import RobotKinematics

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Measured with a phone level at the URDF zero pose (2026-10-22). Edit here only.
JOINT_OFFSETS_DEG = np.array([0.48, 106.07, -89.05, 1.36, -0.13])

DEFAULT_URDF = os.path.expanduser(
    "~/Documents/so101/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
TCP_FRAME = "gripper_frame_link"   # between the fingertips; its z axis points out of the jaws


def to_urdf(q_lerobot_deg):
    return np.asarray(q_lerobot_deg, dtype=float) - JOINT_OFFSETS_DEG


def to_lerobot(q_urdf_deg):
    return np.asarray(q_urdf_deg, dtype=float) + JOINT_OFFSETS_DEG


def translation(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


class SO101Kinematics:
    """FK/IK that take and return LeRobot joint angles (degrees) and handle offsets.

    tip_offset_m: extra tool length along the TCP frame's z axis (e.g. a pencil).
    All poses are 4x4 transforms of the tool tip in the robot base frame (meters).
    """

    def __init__(self, urdf_path=DEFAULT_URDF, tcp_frame=TCP_FRAME, tip_offset_m=0.0):
        self.kin = RobotKinematics(urdf_path=os.path.expanduser(urdf_path),
                                   target_frame_name=tcp_frame, joint_names=ARM_JOINTS)
        self.T_tip = translation(z=tip_offset_m)

    def fk(self, q_lerobot_deg):
        """LeRobot joints (deg) -> T_base_tip (4x4)."""
        return self.kin.forward_kinematics(to_urdf(q_lerobot_deg)) @ self.T_tip

    def position(self, q_lerobot_deg):
        return self.fk(q_lerobot_deg)[:3, 3]

    def ik(self, q_seed_lerobot_deg, T_goal_tip, position_weight=1.0,
           orientation_weight=0.01, tol_m=0.001, max_iters=200):
        """T_base_tip goal -> (LeRobot joints in deg, position error in m).

        LeRobot's inverse_kinematics does one solver step per call, so iterate.
        A low orientation_weight suits the 5-DoF arm (position first).
        """
        T_goal_tcp = T_goal_tip @ np.linalg.inv(self.T_tip)
        q = to_urdf(q_seed_lerobot_deg)
        err = np.inf
        for _ in range(max_iters):
            q = np.asarray(self.kin.inverse_kinematics(
                q, T_goal_tcp, position_weight=position_weight,
                orientation_weight=orientation_weight))[:len(ARM_JOINTS)]
            err = np.linalg.norm(self.kin.forward_kinematics(q)[:3, 3] - T_goal_tcp[:3, 3])
            if err < tol_m:
                break
        return to_lerobot(q), err


def tilt_from_vertical_deg(T):
    """Angle between the tool's z axis and straight down (0 = pointing straight down)."""
    return float(np.degrees(np.arccos(np.clip(-T[2, 2], -1.0, 1.0))))
