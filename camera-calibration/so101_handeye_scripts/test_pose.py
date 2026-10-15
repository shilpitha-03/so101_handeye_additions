from so101_pose_provider import get_T_base_gripper
import numpy as np

np.set_printoptions(suppress=True)

q = get_T_base_gripper()

print("Joint state:", q)