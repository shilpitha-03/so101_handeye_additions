"""SO-101 wrist pose from LeRobot joint readings + URDF forward kinematics (pinocchio).

Fixed vs. the original:
  * uses frame "gripper_link" (rigid with the wrist camera) instead of "gripper",
    which is the moving-jaw JOINT frame and shifts whenever the jaw opens/closes
  * gripper (jaw) value is not fed into FK (it was a 0-100 value treated as degrees)
  * passes your robot id + calibration dir, and connects with calibrate=False,
    so it can NEVER start a recalibration of your arm
"""
import pathlib

import numpy as np
import pinocchio as pin
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ----------------------------- EDIT THESE THREE -----------------------------
PORT = "/dev/ttyACM0"
ROBOT_ID = "my_awesome_follower_arm"   # = the .json filename of your existing follower calibration
CALIB_DIR = pathlib.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
# ----------------------------------------------------------------------------

URDF_PATH = str(pathlib.Path(__file__).resolve().parent.parent / "calibration" / "so101_updated.urdf")
EE_FRAME = "gripper_link"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

_calib_file = CALIB_DIR / f"{ROBOT_ID}.json"
if not _calib_file.is_file():
    raise FileNotFoundError(f"No calibration file at {_calib_file}. Fix ROBOT_ID / CALIB_DIR in so101_pose_provider.py")

robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, calibration_dir=CALIB_DIR, use_degrees=True))
robot.connect(calibrate=False)
robot.bus.disable_torque()

model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()
_FRAME_ID = model.getFrameId(EE_FRAME)
assert model.nq == 6, f"Unexpected URDF dof: {model.nq}"


def get_joint_deg():
    obs = robot.get_observation()
    return np.array([obs[f"{j}.pos"] for j in ARM_JOINTS], dtype=np.float64)


def fk_from_deg(q_deg):
    q = np.zeros(model.nq)
    q[:5] = np.deg2rad(q_deg)          # jaw stays 0: it doesn't move gripper_link
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[_FRAME_ID].homogeneous.copy()


def get_T_base_gripper():
    return fk_from_deg(get_joint_deg())


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    q = get_joint_deg()
    print("joint deg (pan, lift, elbow, wrist_flex, wrist_roll):", np.round(q, 1))
    print("T_base_gripper:\n", fk_from_deg(q))
