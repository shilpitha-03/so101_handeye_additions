"""
Calibration 2 — Wrist Camera Intrinsics
- Live stream from USB camera
- Move the board to different angles/distances while wrist camera is fixed
- SPACE to capture a valid frame
- Saves camera_matrix.npy and dist_coeffs.npy to calibration_files/wrist/
- Note: during this step the robot arm should be stationary
         Only the BOARD moves — you hold it at different poses
"""

import cv2
import numpy as np
import pathlib
import platform
import time

from config import (
    WRIST_CAMERA_ID,
    WRIST_CAMERA_MATRIX_PATH,
    WRIST_DIST_COEFFS_PATH,
    SQUARES_X,
    SQUARES_Y,
    SQUARE_LENGTH,
    MARKER_LENGTH,
    ARUCO_DICT,
    CALIB_MIN_IMAGES,
    MIN_CHARUCO_CORNERS,
)


WINDOW_NAME = "Calib 2 - Wrist Camera Intrinsics"


def _build_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )

    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.CharucoDetector(
        board,
        charuco_params,
        detector_params,
    )

    return board, dictionary, detector


def _backend_candidates():
    """
    Prefer the native backend for the OS, then fall back to OpenCV default.
    """
    system = platform.system()

    candidates = []

    if system == "Linux":
        candidates.append(getattr(cv2, "CAP_V4L2", cv2.CAP_ANY))
    elif system == "Windows":
        candidates.append(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
    elif system == "Darwin":
        candidates.append(getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))

    candidates.append(cv2.CAP_ANY)

    # Remove duplicates while preserving order
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)

    return unique


def _get_backend_name(cap):
    try:
        return cap.getBackendName()
    except Exception:
        return "unknown"


def _configure_camera(cap):
    """
    Do not force camera resolution.
    Use the camera/backend default resolution, same as the original script.
    """
    pass


def _open_camera(camera_id):
    """
    Try to open the wrist camera using the best backend for this OS.
    """
    last_cap = None

    for backend in _backend_candidates():
        if backend == cv2.CAP_ANY:
            cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(camera_id, backend)

        if cap.isOpened():
            _configure_camera(cap)

            print("\nCamera opened successfully")
            print(f"  Camera ID : {camera_id}")
            print(f"  Backend   : {_get_backend_name(cap)}")
            print(f"  Width     : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            print(f"  Height    : {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            print(f"  FPS       : {cap.get(cv2.CAP_PROP_FPS)}")

            return cap

        last_cap = cap
        cap.release()

    if last_cap is not None:
        last_cap.release()

    raise RuntimeError(f"Cannot open wrist camera at index: {camera_id}")


def _make_blank_frame(message):
    """
    Shows a visible error frame instead of leaving the window black/frozen.
    """
    display = np.zeros((480, 640, 3), dtype=np.uint8)

    lines = [
        "No valid camera frame received.",
        message,
        "",
        "Check WRIST_CAMERA_ID in config.py",
        "Press Q to quit.",
    ]

    y = 80
    for line in lines:
        cv2.putText(
            display,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y += 40

    return display


def _normalize_frame(frame):
    """
    Ensure frame is BGR with 3 channels.
    """
    if frame is None:
        return None

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


def _draw_debug_info(display, frame, captured):
    mean_val = float(frame.mean())

    debug_text = (
        f"cam={WRIST_CAMERA_ID} | "
        f"shape={frame.shape} | "
        f"mean={mean_val:.1f} | "
        f"captured={captured}/{CALIB_MIN_IMAGES}"
    )

    cv2.putText(
        display,
        debug_text,
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if mean_val < 2.0:
        cv2.putText(
            display,
            "WARNING: frame is almost black - wrong camera ID or camera feed problem",
            (20, display.shape[0] - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def run():
    """
    Entry point — can be called from main.py or run independently.
    """
    print("\n========== CALIB 2 : WRIST CAMERA INTRINSICS ==========")
    print("Keep robot arm STATIONARY. Move the BOARD to different")
    print("angles, distances, and positions in front of wrist camera.")
    print(f"SPACE -> capture frame ({CALIB_MIN_IMAGES} needed) | Q -> quit\n")

    board, dictionary, detector = _build_detector()

    cap = _open_camera(WRIST_CAMERA_ID)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Camera warmup
    print("\nWarming up camera...")
    warmup_ok = 0

    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            warmup_ok += 1
        time.sleep(0.02)

    print(f"Warmup frames received: {warmup_ok}/30\n")

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None
    captured = 0
    failed_frames = 0

    try:
        while captured < CALIB_MIN_IMAGES:
            ret, frame = cap.read()

            if not ret or frame is None or frame.size == 0:
                failed_frames += 1

                display = _make_blank_frame(
                    f"Failed frames: {failed_frames}"
                )

                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    print("  Quit by user.")
                    break

                continue

            frame = _normalize_frame(frame)

            if frame is None:
                failed_frames += 1
                continue

            failed_frames = 0

            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])
                print(f"Image size used for calibration: {image_size}")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

            display = frame.copy()

            enough = (
                charuco_ids is not None
                and len(charuco_ids) >= MIN_CHARUCO_CORNERS
            )

            if enough:
                cv2.aruco.drawDetectedCornersCharuco(
                    display,
                    charuco_corners,
                    charuco_ids,
                )

                cv2.putText(
                    display,
                    f"SPACE capture [{captured}/{CALIB_MIN_IMAGES}] | "
                    f"{len(charuco_ids)} corners | Q quit",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                detected = 0 if charuco_ids is None else len(charuco_ids)

                cv2.putText(
                    display,
                    f"Only {detected} corners - need >= {MIN_CHARUCO_CORNERS}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            _draw_debug_info(display, frame, captured)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                print("  Quit by user.")
                break

            if key == ord(" "):
                if enough:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    captured += 1

                    print(
                        f"  [{captured}/{CALIB_MIN_IMAGES}] captured - "
                        f"{len(charuco_ids)} corners detected"
                    )
                else:
                    print("  Not enough corners - reposition board")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured < 4:
        raise RuntimeError(
            f"Only {captured} frames captured - need at least 4 to calibrate."
        )

    if captured < CALIB_MIN_IMAGES:
        print(
            f"  WARNING: only {captured}/{CALIB_MIN_IMAGES} frames - "
            f"calibration may be less accurate."
        )

    print(f"\n  Running calibrateCameraCharuco on {captured} frames...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,
        None,
    )

    print(f"  Reprojection error : {ret:.4f} px  target < 1.0, ideal < 0.5")

    if ret > 1.0:
        print("  WARNING: high reprojection error - recapture with better coverage")
    else:
        print("  Reprojection error acceptable")

    print(f"\n  Camera matrix:\n{camera_matrix}")
    print(f"\n  Distortion coefficients:\n{dist_coeffs}")

    out_dir = pathlib.Path(WRIST_CAMERA_MATRIX_PATH).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(WRIST_CAMERA_MATRIX_PATH, camera_matrix)
    np.save(WRIST_DIST_COEFFS_PATH, dist_coeffs)

    print(f"\n  Saved: camera_matrix.npy, dist_coeffs.npy -> {out_dir}")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "reprojection_error": ret,
    }


if __name__ == "__main__":
    run()