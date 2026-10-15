"""
Step 4: Example usage after calibration.

This example detects the same ChArUco board and converts its pose into the SO-101
base frame:

    T_base_charuco = T_base_gripper @ T_gripper_camera @ T_camera_charuco

For an object/ArUco marker, replace T_camera_charuco with T_camera_object.
"""

import argparse

import cv2 as cv
import numpy as np

import config
from common import draw_charuco_debug, estimate_charuco_pose, load_intrinsics, open_camera, parse_camera_arg, print_T
from so101_pose_provider import get_T_base_gripper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=str(config.CAMERA_INDEX_OR_PATH))
    parser.add_argument("--T", default=config.HAND_EYE_NPY_PATH)
    args = parser.parse_args()

    K, D = load_intrinsics()
    T_gripper_camera = np.load(args.T).astype(np.float64)
    print_T("T_gripper_camera", T_gripper_camera)

    cap = open_camera(parse_camera_arg(args.camera))
    print("Press p to print current board pose in base frame. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = estimate_charuco_pose(frame, K, D)
        debug = draw_charuco_debug(frame, result, K, D)
        cv.putText(debug, "p=print T_base_charuco | q=quit", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.imshow("use hand-eye calibration", debug)

        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            if not result.ok:
                print("No valid ChArUco pose:", result.message)
                continue
            T_base_gripper = get_T_base_gripper()
            T_base_charuco = T_base_gripper @ T_gripper_camera @ result.T_camera_charuco
            print_T("T_base_gripper", T_base_gripper)
            print_T("T_camera_charuco", result.T_camera_charuco)
            print_T("T_base_charuco", T_base_charuco)
            print("\nFor an object, use exactly this pattern:")
            print("    T_base_object = T_base_gripper @ T_gripper_camera @ T_camera_object")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
