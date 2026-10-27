"""SO-101 wrist pose for hand-eye calibration.

FK comes from your shared kinematics module (so101_kinematics.py), so the joint
offsets and URDF live in ONE place. This file only handles the robot connection.

Hand-eye frame: TCP frame of so101_kinematics (gripper_frame_link) with tip_offset_m=0.
It is rigidly attached to gripper_link and does not move with the jaw.
So the result outputs/T_gripper_camera.npy is the camera pose in gripper_frame_link.

Safety: connects with your robot id + calibration dir and calibrate=False, so it can
never start a recalibration of your arm.
"""
import os
import pathlib

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from so101_kinematics import ARM_JOINTS, SO101Kinematics, TCP_FRAME

# ----------------------------- EDIT THESE THREE -----------------------------
PORT = "/dev/ttyACM0"
ROBOT_ID = "so101_follower_vasco"   # = the .json filename of your existing follower calibration
CALIB_DIR = pathlib.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
# ----------------------------------------------------------------------------

# tip_offset_m MUST stay 0 here: the calibration is defined relative to the TCP frame itself.
KIN = SO101Kinematics(tip_offset_m=0.0)
EE_FRAME = TCP_FRAME

robot = None


def connect(torque_off=True):
    """Connect to the follower (once). torque_off=True lets you move it by hand."""
    global robot
    if robot is None:
        calib_file = CALIB_DIR / f"{ROBOT_ID}.json"
        if not calib_file.is_file():
            raise FileNotFoundError(f"No calibration file at {calib_file}. Fix ROBOT_ID / CALIB_DIR in so101_pose_provider.py")
        robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, calibration_dir=CALIB_DIR, use_degrees=True))
        robot.connect(calibrate=False)
    if torque_off:
        robot.bus.disable_torque()
    return robot


def get_joint_deg():
    """LeRobot joint angles (deg) for the 5 arm joints. Offsets are applied inside KIN.fk."""
    if robot is None:
        connect(torque_off=False)
    obs = robot.get_observation()
    return np.array([obs[f"{j}.pos"] for j in ARM_JOINTS], dtype=np.float64)


def fk_from_deg(q_lerobot_deg):
    """LeRobot joints (deg) -> T_base_tcp (4x4, meters)."""
    return np.asarray(KIN.fk(q_lerobot_deg), dtype=np.float64)


def get_T_base_gripper():
    return fk_from_deg(get_joint_deg())


# Hand-move mode by default (used by 01/04 scripts). The teleop script sets this env var first.
if os.environ.get("SO101_NO_AUTOCONNECT") != "1":
    connect(torque_off=True)


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    q = get_joint_deg()
    print("LeRobot joint deg (pan, lift, elbow, wrist_flex, wrist_roll):", np.round(q, 1))
    print(f"T_base_{EE_FRAME}:\n", fk_from_deg(q))