"""
Capture hand-eye samples for SO-101 wrist camera.
GUI-ready for Linux/X11. Shows live wrist camera feed.
Automatically fetches wrist pose from so101_pose_provider.
"""

import argparse
import json
import time
from pathlib import Path
import cv2 as cv
import numpy as np
import config

from common import draw_charuco_debug, ensure_dir, estimate_charuco_pose, load_intrinsics, open_camera, parse_camera_arg, print_T
from so101_pose_provider import get_T_base_gripper

# -------------------- Wrist pose helper --------------------
def try_get_robot_pose() -> np.ndarray:
    """Return 4x4 wrist pose dynamically from robot"""
    return np.asarray(get_T_base_gripper(), dtype=np.float64).reshape(4,4)

# -------------------- Main capture --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="/dev/video2",
                        help="Camera index or device path, e.g., /dev/video2")
    parser.add_argument("--out", default=config.SAMPLES_NPZ_PATH,
                        help="Output NPZ path for samples")
    parser.add_argument("--image-dir", default=config.SAMPLES_IMAGE_DIR,
                        help="Directory to save captured images")
    args = parser.parse_args()

    ensure_dir(str(Path(args.out).parent))
    ensure_dir(args.image_dir)

    K, D = load_intrinsics()
    cap = open_camera(parse_camera_arg(args.camera))

    T_base_gripper_list = []
    T_camera_charuco_list = []
    image_paths = []
    timestamps = []

    print("\n=== Capture hand-eye samples ===")
    print("Press 's' to save a sample when ChArUco detection is OK.")
    print("Press 'q' or ESC to quit.")
    print("Move the wrist to different poses for multiple samples.")
    print("Recommended samples:", config.RECOMMENDED_SAMPLE_COUNT)

    last_result = None
    last_frame = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        # Detect ChArUco board
        result = estimate_charuco_pose(frame, K, D)
        last_result = result
        last_frame = frame.copy()

        # Draw axes for debug
        debug = draw_charuco_debug(frame, result, K, D)
        cv.putText(debug,f"samples: {len(T_base_gripper_list)} | s=save | q=quit",
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # GUI display
        cv.imshow("capture hand-eye samples", debug)
        key = cv.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        if key == ord("s"):
            if last_result is None or not last_result.ok:
                print("Cannot save: ChArUco pose invalid")
                continue

            # Automatically fetch current robot pose
            try:
                T_base_gripper = try_get_robot_pose()
            except Exception as e:
                print(f"Failed to get robot pose: {e}")
                continue

            T_camera_charuco = last_result.T_camera_charuco

            idx = len(T_base_gripper_list)
            image_path = str(Path(args.image_dir)/f"sample_{idx:03d}.png")
            cv.imwrite(image_path, last_frame)

            # Store sample
            T_base_gripper_list.append(T_base_gripper)
            T_camera_charuco_list.append(T_camera_charuco)
            image_paths.append(image_path)
            timestamps.append(time.time())

            print_T("T_base_gripper", T_base_gripper)
            print_T("T_camera_charuco", T_camera_charuco)
            print(f"Saved image: {image_path}")
            print(f"Total samples: {len(T_base_gripper_list)}")

    cap.release()
    cv.destroyAllWindows()

    if len(T_base_gripper_list) == 0:
        print("No samples collected.")
        return

    np.savez(args.out,
             T_base_gripper=np.asarray(T_base_gripper_list, dtype=np.float64),
             T_camera_charuco=np.asarray(T_camera_charuco_list, dtype=np.float64),
             image_paths=np.asarray(image_paths),
             timestamps=np.asarray(timestamps, dtype=np.float64),
             board_config=json.dumps({
                 "squares_x": config.SQUARES_X,
                 "squares_y": config.SQUARES_Y,
                 "square_length_m": config.SQUARE_LENGTH_M,
                 "marker_length_m": config.MARKER_LENGTH_M,
                 "aruco_dict_name": config.ARUCO_DICT_NAME
             }))

    print(f"\nSaved {len(T_base_gripper_list)} samples to {args.out}")
    if len(T_base_gripper_list) < 10:
        print("WARNING: fewer than 10 samples. Calibration may be weak.")
    print("Next step: run 02_run_handeye_calibration.py")

if __name__=="__main__":
    main()