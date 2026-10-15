"""
Calibration 1 — Side-View Camera Intrinsics
- Live stream from Android IP camera
- SPACE to capture a valid frame
- Needs CALIB_MIN_IMAGES valid frames
- Uses CharucoBoard.matchImagePoints() + cv2.calibrateCamera()  [OpenCV 4.7+ correct API]
- Saves camera_matrix.npy and dist_coeffs.npy to calibration_files/side_view/
- Target reprojection error < 1.0 px, ideally < 0.5 px
"""

import cv2
import platform
import time
import numpy as np
import pathlib
from config import (
    SIDE_CAMERA_ID,
    SIDE_CAMERA_MATRIX_PATH, SIDE_DIST_COEFFS_PATH,
    SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH,
    ARUCO_DICT, CALIB_MIN_IMAGES, MIN_CHARUCO_CORNERS,
)


WINDOW_NAME = "Calib 1 — Side-View Intrinsics"


def _build_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH, MARKER_LENGTH,
        dictionary
    )
    # Do NOT call setLegacyPattern(True) — we use OpenCV 4.6+ axis convention.
    # Phase 2 diag([1,-1,-1]) correction is calibrated for new-API Z direction.
    charuco_params = cv2.aruco.CharucoParameters()
    # No intrinsics in charuco_params — not yet available during calibration.
    # detectBoard() will use homography fallback, which is expected and correct here.
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    return board, dictionary, detector


def _normalize_camera_id(camera_id):
    """Convert numeric camera IDs stored as strings (for example "0") to int."""
    if isinstance(camera_id, str) and camera_id.strip().isdigit():
        return int(camera_id.strip())
    return camera_id


def _is_url_camera(camera_id):
    if not isinstance(camera_id, str):
        return False
    camera_id_lower = camera_id.lower().strip()
    return camera_id_lower.startswith(("http://", "https://", "rtsp://", "udp://", "tcp://"))


def _backend_candidates(camera_id):
    system = platform.system()
    camera_is_url = _is_url_camera(camera_id)
    candidates = []

    if camera_is_url:
        # IP / Android camera streams should not be opened with V4L2 / DirectShow.
        # Try stream-capable backends first so we do not get an opened-but-black window.
        candidates.append(getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY))
        if system == "Linux":
            candidates.append(getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY))
        candidates.append(cv2.CAP_ANY)
    else:
        if system == "Linux":
            candidates.append(getattr(cv2, "CAP_V4L2", cv2.CAP_ANY))
        elif system == "Windows":
            candidates.append(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
            candidates.append(getattr(cv2, "CAP_MSMF", cv2.CAP_ANY))
        elif system == "Darwin":  # macOS
            candidates.append(getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))

        candidates.append(cv2.CAP_ANY)

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


def _valid_frame(frame):
    return frame is not None and frame.size > 0 and frame.shape[0] > 1 and frame.shape[1] > 1


def _prepare_frame(frame):
    """Make sure frame is BGR before ChArUco detection and display."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _read_first_valid_frame(cap, attempts=40):
    """
    Warm up the camera and return the first usable frame.
    This prevents accepting a backend that opens but only shows a black/empty window.
    """
    last_frame_was_black = False

    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and _valid_frame(frame):
            frame = _prepare_frame(frame)

            # Completely black frames usually mean the wrong backend/stream path was opened.
            # Very dark real frames still pass as long as they are not all zeros.
            if np.max(frame) == 0:
                last_frame_was_black = True
                time.sleep(0.03)
                continue

            return frame, last_frame_was_black

        time.sleep(0.03)

    return None, last_frame_was_black


def _open_camera(camera_id):
    source = _normalize_camera_id(camera_id)

    last_error = "camera did not open"
    for backend in _backend_candidates(source):
        if backend == cv2.CAP_ANY:
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, backend)

        if not cap.isOpened():
            cap.release()
            last_error = f"backend {backend} could not open camera"
            continue

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        first_frame, black_frames_seen = _read_first_valid_frame(cap)
        if first_frame is None:
            backend_name = _get_backend_name(cap)
            cap.release()
            if black_frames_seen:
                last_error = f"backend {backend_name} opened but returned only black frames"
            else:
                last_error = f"backend {backend_name} opened but returned no valid frames"
            continue

        print("\nCamera opened successfully")
        print(f"  Camera ID : {camera_id}")
        print(f"  Backend   : {_get_backend_name(cap)}")
        print(f"  Width     : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"  Height    : {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  FPS       : {cap.get(cv2.CAP_PROP_FPS)}")
        return cap, first_frame

    raise RuntimeError(f"Cannot open side camera: {camera_id} ({last_error})")


def run():
    """
    Entry point — callable from main.py or independently.
    No arguments needed.
    """
    print("\n========== CALIB 1 : SIDE-VIEW INTRINSICS ==========")
    print("Move the board to different angles, distances, and positions.")
    print("Cover all regions of the frame — corners, edges, and center.")
    print(f"SPACE → capture frame ({CALIB_MIN_IMAGES} needed) | Q → quit\n")

    board, dictionary, detector = _build_detector()

    cap, first_frame = _open_camera(SIDE_CAMERA_ID)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    all_obj_points = []
    all_img_points = []
    image_size = None
    captured = 0
    pending_frame = first_frame

    while captured < CALIB_MIN_IMAGES:
        if pending_frame is not None:
            frame = pending_frame
            pending_frame = None
            ret = True
        else:
            ret, frame = cap.read()

        if not ret or not _valid_frame(frame):
            print("  WARNING: failed to grab frame, retrying...")
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("  Quit by user.")
                break
            continue

        frame = _prepare_frame(frame)

        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        display = frame.copy()
        enough = charuco_ids is not None and len(charuco_ids) >= MIN_CHARUCO_CORNERS

        if enough:
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
            cv2.putText(display,
                        f"SPACE capture [{captured}/{CALIB_MIN_IMAGES}] | "
                        f"{len(charuco_ids)} corners | Q quit",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            detected = 0 if charuco_ids is None else len(charuco_ids)
            cv2.putText(display,
                        f"Only {detected} corners — need >= {MIN_CHARUCO_CORNERS}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("  Quit by user.")
            break

        if key == ord(' ') and enough:
            # matchImagePoints maps detected corners to 3D board object points
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_points is not None and obj_points.shape[0] >= MIN_CHARUCO_CORNERS:
                all_obj_points.append(obj_points)
                all_img_points.append(img_points)
                captured += 1
                print(f"  [{captured}/{CALIB_MIN_IMAGES}] captured — "
                      f"{len(charuco_ids)} corners detected")
            else:
                print("  matchImagePoints returned insufficient points — skip frame")
        elif key == ord(' ') and not enough:
            print("  Not enough corners — reposition board")

    cap.release()
    cv2.destroyAllWindows()

    if captured < 4:
        raise RuntimeError(
            f"Only {captured} frames captured — need at least 4 to calibrate.")

    if captured < CALIB_MIN_IMAGES:
        print(f"  WARNING: only {captured}/{CALIB_MIN_IMAGES} frames — "
              f"calibration may be less accurate.")

    # ── Run calibration ───────────────────────────────────────────────
    print(f"\n  Running cv2.calibrateCamera on {captured} frames...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, all_img_points, image_size, None, None
    )

    print(f"  Reprojection error : {ret:.4f} px  (target < 1.0, ideal < 0.5)")
    if ret > 1.0:
        print("  WARNING: high reprojection error — recapture with better coverage")
    else:
        print("  Reprojection error acceptable")

    print(f"\n  Camera matrix:\n{camera_matrix}")
    print(f"\n  Distortion coefficients:\n{dist_coeffs}")

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = pathlib.Path(SIDE_CAMERA_MATRIX_PATH).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(SIDE_CAMERA_MATRIX_PATH, camera_matrix)
    np.save(SIDE_DIST_COEFFS_PATH, dist_coeffs)
    print(f"\n  Saved: camera_matrix.npy, dist_coeffs.npy → {out_dir}")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "reprojection_error": ret,
    }