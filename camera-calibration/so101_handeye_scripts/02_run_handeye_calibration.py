"""
Step 2: Run OpenCV hand-eye calibration.

Input:
    data/handeye_samples.npz

Output:
    outputs/T_gripper_camera.npy

Run:
    python 02_run_handeye_calibration.py
"""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

import config
from common import ensure_dir, make_T, print_T, rotation_angle_deg


METHODS = {
    "TSAI": cv.CALIB_HAND_EYE_TSAI,
    "PARK": cv.CALIB_HAND_EYE_PARK,
    "HORAUD": cv.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv.CALIB_HAND_EYE_DANIILIDIS,
}


def validate(T_base_gripper_all: np.ndarray, T_camera_charuco_all: np.ndarray, T_gripper_camera: np.ndarray):
    """
    Since the ChArUco board was fixed on the table, all predicted board poses in
    base frame should be nearly equal:

        T_base_charuco_i = T_base_gripper_i @ T_gripper_camera @ T_camera_charuco_i
    """
    T_base_charuco_all = []
    for T_bg, T_ct in zip(T_base_gripper_all, T_camera_charuco_all):
        T_base_charuco_all.append(T_bg @ T_gripper_camera @ T_ct)
    T_base_charuco_all = np.asarray(T_base_charuco_all)

    positions = T_base_charuco_all[:, :3, 3]
    mean_pos = positions.mean(axis=0)
    trans_errors_m = np.linalg.norm(positions - mean_pos, axis=1)

    # Rotation consistency relative to first predicted board orientation.
    R0 = T_base_charuco_all[0, :3, :3]
    rot_errors_deg = []
    for T in T_base_charuco_all:
        dR = R0.T @ T[:3, :3]
        rot_errors_deg.append(rotation_angle_deg(dR))
    rot_errors_deg = np.asarray(rot_errors_deg)

    return {
        "mean_board_position_m": mean_pos,
        "mean_translation_error_m": float(trans_errors_m.mean()),
        "max_translation_error_m": float(trans_errors_m.max()),
        "mean_rotation_error_deg": float(rot_errors_deg.mean()),
        "max_rotation_error_deg": float(rot_errors_deg.max()),
        "T_base_charuco_all": T_base_charuco_all,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=config.SAMPLES_NPZ_PATH)
    parser.add_argument("--out-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--method", default="ALL", choices=["ALL"] + list(METHODS.keys()))
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    data = np.load(args.samples, allow_pickle=True)
    T_base_gripper_all = data["T_base_gripper"].astype(np.float64)
    T_camera_charuco_all = data["T_camera_charuco"].astype(np.float64)

    n = len(T_base_gripper_all)
    print(f"Loaded {n} samples from {args.samples}")
    if n < 3:
        raise RuntimeError("Need at least 3 different poses. Recommended: 20 to 30.")
    if n < 10:
        print("WARNING: Fewer than 10 samples. Use 20 to 30 if possible.")

    # OpenCV expects:
    #   R_gripper2base, t_gripper2base = ^base T_gripper
    #   R_target2cam,   t_target2cam   = ^camera T_target
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for T_bg, T_ct in zip(T_base_gripper_all, T_camera_charuco_all):
        R_gripper2base.append(T_bg[:3, :3])
        t_gripper2base.append(T_bg[:3, 3].reshape(3, 1))

        R_target2cam.append(T_ct[:3, :3])
        t_target2cam.append(T_ct[:3, 3].reshape(3, 1))

    methods_to_run = METHODS if args.method == "ALL" else {args.method: METHODS[args.method]}

    results = []
    for name, method in methods_to_run.items():
        print(f"\n=== Method: {name} ===")
        try:
            R_cam2gripper, t_cam2gripper = cv.calibrateHandEye(
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
                method=method,
            )
            T_gripper_camera = make_T(R_cam2gripper, t_cam2gripper)
            if not np.all(np.isfinite(T_gripper_camera)):
                print("Result contains NaN/Inf, skipping")
                continue
        except Exception as e:
            print("Calibration failed:", e)
            continue

        metrics = validate(T_base_gripper_all, T_camera_charuco_all, T_gripper_camera)
        print_T(f"T_gripper_camera_{name}", T_gripper_camera)
        print(
            "Validation board-position error: "
            f"mean={metrics['mean_translation_error_m']*1000:.2f} mm, "
            f"max={metrics['max_translation_error_m']*1000:.2f} mm"
        )
        print(
            "Validation board-orientation spread: "
            f"mean={metrics['mean_rotation_error_deg']:.2f} deg, "
            f"max={metrics['max_rotation_error_deg']:.2f} deg"
        )

        out_path = Path(args.out_dir) / f"T_gripper_camera_{name}.npy"
        np.save(out_path, T_gripper_camera)
        results.append((name, T_gripper_camera, metrics, out_path))
        print("Saved", out_path)

    if not results:
        raise RuntimeError("No hand-eye method produced a valid result.")

    # Pick the method with lowest mean board translation spread.
    best = min(results, key=lambda x: x[2]["mean_translation_error_m"])
    best_name, best_T, best_metrics, _ = best
    final_path = Path(args.out_dir) / "T_gripper_camera.npy"
    np.save(final_path, best_T)

    print("\n=== Selected result ===")
    print(f"Best method by validation spread: {best_name}")
    print_T("T_gripper_camera", best_T)
    print(
        "Mean validation error: "
        f"{best_metrics['mean_translation_error_m']*1000:.2f} mm | "
        f"Max: {best_metrics['max_translation_error_m']*1000:.2f} mm"
    )
    print("Saved final hand-eye transform:", final_path)
    print("\nNext run:")
    print("    python 03_validate_handeye.py")


if __name__ == "__main__":
    main()
