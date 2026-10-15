"""
Step 3: Validate the result.

Run:
    python 03_validate_handeye.py

This re-computes the board pose in the SO-101 base frame for every sample.
Because the board stayed fixed, the board position should not jump around.
"""

import argparse

import numpy as np

import config
from common import print_T, rotation_angle_deg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=config.SAMPLES_NPZ_PATH)
    parser.add_argument("--T", default=config.HAND_EYE_NPY_PATH)
    args = parser.parse_args()

    data = np.load(args.samples, allow_pickle=True)
    T_base_gripper_all = data["T_base_gripper"].astype(np.float64)
    T_camera_charuco_all = data["T_camera_charuco"].astype(np.float64)
    T_gripper_camera = np.load(args.T).astype(np.float64)

    print_T("T_gripper_camera", T_gripper_camera)

    T_base_charuco_all = []
    for T_bg, T_ct in zip(T_base_gripper_all, T_camera_charuco_all):
        T_base_charuco_all.append(T_bg @ T_gripper_camera @ T_ct)
    T_base_charuco_all = np.asarray(T_base_charuco_all)

    positions = T_base_charuco_all[:, :3, 3]
    mean_pos = positions.mean(axis=0)
    errors = np.linalg.norm(positions - mean_pos, axis=1)

    print("\nPredicted fixed board position in SO-101 base frame:")
    print("mean position [m]:", mean_pos)
    print("per-sample translation errors [mm]:")
    print(np.round(errors * 1000, 2))
    print(f"mean error: {errors.mean() * 1000:.2f} mm")
    print(f"max error:  {errors.max() * 1000:.2f} mm")

    R0 = T_base_charuco_all[0, :3, :3]
    rot_errors = np.array([rotation_angle_deg(R0.T @ T[:3, :3]) for T in T_base_charuco_all])
    print("\nper-sample orientation spread vs sample 0 [deg]:")
    print(np.round(rot_errors, 2))
    print(f"mean orientation spread: {rot_errors.mean():.2f} deg")
    print(f"max orientation spread:  {rot_errors.max():.2f} deg")

    print("\nRule of thumb:")
    print("  A low-cost arm may not give industrial accuracy, but huge errors usually mean wrong units, wrong FK frame, wrong transform direction, or the board moved.")


if __name__ == "__main__":
    main()
