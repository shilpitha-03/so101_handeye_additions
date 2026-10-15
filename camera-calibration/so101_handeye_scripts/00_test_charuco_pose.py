"""
Step 0: Test that your existing camera_matrix.npy + dist_coeffs.npy can detect
and estimate the ChArUco board pose from the wrist camera.

Run:
    python 00_test_charuco_pose.py --camera 0

Keys:
    q / ESC : quit
    s       : save debug image to data/debug_charuco.png
"""

import argparse

import cv2 as cv

import config
from common import draw_charuco_debug, estimate_charuco_pose, load_intrinsics, open_camera, parse_camera_arg, print_T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=str(config.CAMERA_INDEX_OR_PATH), help="Camera index or video path")
    parser.add_argument("--image", default=None, help="Optional single image path instead of live camera")
    args = parser.parse_args()

    K, D = load_intrinsics()
    print("Loaded intrinsics:")
    print("camera_matrix=\n", K)
    print("dist_coeffs=", D.reshape(-1))

    if args.image:
        image = cv.imread(args.image)
        if image is None:
            raise RuntimeError(f"Could not read image: {args.image}")
        result = estimate_charuco_pose(image, K, D)
        debug = draw_charuco_debug(image, result, K, D)
        print(result.message)
        if result.ok:
            print_T("T_camera_charuco", result.T_camera_charuco)
        cv.imshow("charuco pose test", debug)
        cv.waitKey(0)
        return

    cap = open_camera(parse_camera_arg(args.camera))
    print("Move the wrist camera so the ChArUco board is visible.")
    print("Press q or ESC to quit. Press s to save debug image.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        result = estimate_charuco_pose(frame, K, D)
        debug = draw_charuco_debug(frame, result, K, D)
        cv.imshow("charuco pose test", debug)

        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            cv.imwrite("data/debug_charuco.png", debug)
            print("Saved data/debug_charuco.png")
            if result.ok:
                print_T("T_camera_charuco", result.T_camera_charuco)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
