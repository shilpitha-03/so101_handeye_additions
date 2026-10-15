"""
Phase 1 — Capture & Collect
- Opens side-view camera stream
- Shows live ChArUco detection with axes overlay
- Spacebar to capture a valid frame (needs >= MIN_CHARUCO_CORNERS)
- Uses board.matchImagePoints() + cv2.solvePnP(IPPE)  [OpenCV 4.7+ correct API]
- Saves raw rvecs and tvecs (shape N×3) to session results folder
"""

import cv2
import platform
import numpy as np
from config import (
    SIDE_CAMERA_ID, SIDE_CAMERA_MATRIX_PATH, SIDE_DIST_COEFFS_PATH,
    SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH,
    ARUCO_DICT, PHASE1_N_FRAMES, MIN_CHARUCO_CORNERS,
)


def _build_detector(K, D):
    """Build CharucoDetector with intrinsics for accurate reprojection-based interpolation."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH, MARKER_LENGTH,
        dictionary,
    )
    # No setLegacyPattern — using OpenCV 4.6+ convention; Phase 2 corrects axis direction.
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.cameraMatrix = K
    charuco_params.distCoeffs = D
    # With intrinsics provided, detectBoard() uses reprojection-based interpolation (accurate)
    # instead of homography (inaccurate under distortion).
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    return board, detector


def _estimate_pose(board, charuco_corners, charuco_ids, K, D):
    """
    Pose estimation using the modern API: board.matchImagePoints() + cv2.solvePnP().
    Uses SOLVEPNP_IPPE — correct method for coplanar object points (flat board).
    Returns (ok, rvec, tvec) where rvec/tvec are shape (3,1).
    """
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts is None or obj_pts.shape[0] < 4:
        return False, None, None

    # SOLVEPNP_IPPE returns 2 solutions for coplanar points.
    # retval = number of solutions (should be 2), rvecs/tvecs are lists.
    retval, rvecs, tvecs, reprojErrs = cv2.solvePnPGeneric(
        obj_pts, img_pts, K, D,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if retval == 0:
        return False, None, None

    # Select the solution with the lowest reprojection error.
    best = int(np.argmin([e[0] for e in reprojErrs]))
    return True, rvecs[best], tvecs[best]  # both shape (3,1)

def _backend_candidates():
    system = platform.system()
    candidates = []

    if system == "Linux":
        candidates.append(getattr(cv2, "CAP_V4L2", cv2.CAP_ANY))
    elif system == "Windows":
        candidates.append(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
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


def _open_camera(camera_id):
    for backend in _backend_candidates():
        if backend == cv2.CAP_ANY:
            cap = cv2.VideoCapture(camera_id)
        else:
            cap = cv2.VideoCapture(camera_id, backend)

        if cap.isOpened():
            print("\nCamera opened successfully")
            print(f"  Camera ID : {camera_id}")
            print(f"  Backend   : {_get_backend_name(cap)}")
            print(f"  Width     : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            print(f"  Height    : {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            print(f"  FPS       : {cap.get(cv2.CAP_PROP_FPS)}")
            return cap

        cap.release()

    raise RuntimeError(f"Cannot open side camera: {camera_id}")


def run(session_dir):
    """
    Entry point for Phase 1.
    session_dir: pathlib.Path — where to save outputs.
    Returns: dict with keys 'rvecs', 'tvecs' — np arrays of shape (N, 3).
    """
    print("\n========== PHASE 1 : CAPTURE ==========")

    K = np.load(SIDE_CAMERA_MATRIX_PATH)
    D = np.load(SIDE_DIST_COEFFS_PATH)

    board, detector = _build_detector(K, D)

    cap = _open_camera(SIDE_CAMERA_ID)
    cv2.namedWindow("Phase 1 — Capture", cv2.WINDOW_NORMAL)

    print("Camera opened.")
    print(f"SPACE  → capture frame  ({PHASE1_N_FRAMES} needed)")
    print("Q      → quit early")
    print("Watch: BLUE axis (Z) must point UP toward camera\n")

    rvecs_collected = []
    tvecs_collected = []
    collected = 0

    # State from last successful live detection — reused on SPACE press.
    last_ok = False
    last_rvec = None
    last_tvec = None
    last_corners = None
    last_ids = None

    while collected < PHASE1_N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("  WARNING: failed to grab frame, retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        display = frame.copy()
        enough = charuco_ids is not None and len(charuco_ids) >= MIN_CHARUCO_CORNERS
        print(charuco_ids)

        if enough:
            ok, rvec, tvec = _estimate_pose(board, charuco_corners, charuco_ids, K, D)
            if ok:
                last_ok, last_rvec, last_tvec = True, rvec, tvec
                last_corners, last_ids = charuco_corners, charuco_ids
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
                cv2.drawFrameAxes(display, K, D, rvec, tvec, SQUARE_LENGTH * 2)
                cv2.putText(display,
                            f"SPACE to capture [{collected}/{PHASE1_N_FRAMES}]"
                            f" | Z (blue) UP | Q quit",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                last_ok = False
                cv2.putText(display, "Pose failed — adjust board",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        else:
            last_ok = False
            detected = 0 if charuco_ids is None else len(charuco_ids)
            cv2.putText(display,
                        f"Only {detected} corners — need >= {MIN_CHARUCO_CORNERS}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Phase 1 — Capture", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("  Quit by user.")
            break

        if key == ord(' '):
            if last_ok:
                # Reuse the already-computed pose — no second solve on same frame.
                rvecs_collected.append(last_rvec.flatten())   # store as (3,)
                tvecs_collected.append(last_tvec.flatten())
                collected += 1
                print(f"  [{collected}/{PHASE1_N_FRAMES}] captured"
                      f" — tvec: {last_tvec.flatten().round(4)}")
            else:
                print("  No valid pose — reposition board and try again")

    cap.release()
    cv2.destroyAllWindows()

    if collected == 0:
        raise RuntimeError("No frames collected. Phase 1 failed.")

    if collected < PHASE1_N_FRAMES:
        print(f"  WARNING: only {collected}/{PHASE1_N_FRAMES} frames collected.")

    rvecs_arr = np.array(rvecs_collected)   # shape (N, 3)
    tvecs_arr = np.array(tvecs_collected)   # shape (N, 3)

    # ── Sanity check: consistency across frames ───────────────────────
    tvec_std = tvecs_arr.std(axis=0)
    rvec_std = rvecs_arr.std(axis=0)
    print(f"\n  rvec std across frames: {rvec_std.round(5)}  (target < 0.01 rad)")
    print(f"  tvec std across frames: {tvec_std.round(5)}  (target < 0.005 m)")

    # Detect and report outlier frames (tvec > 2 std from median)
    tvec_median = np.median(tvecs_arr, axis=0)
    deviations = np.abs(tvecs_arr - tvec_median)
    outlier_mask = (deviations > 2 * tvec_std).any(axis=1)
    if outlier_mask.any():
        print(f"  WARNING: {outlier_mask.sum()} frame(s) are outliers by tvec."
              f" Consider re-running Phase 1 for more consistent results.")

    if tvec_std.max() > 0.005:
        print("  WARNING: high tvec std — board may have moved during capture"
              " or pose is noisy. Check board stability and re-run if needed.")

    # ── Save ──────────────────────────────────────────────────────────
    # Saved as (N, 3) float64. Phase 2 must reshape to (3,1) before Rodrigues.
    np.save(session_dir / "p1_rvecs_raw.npy", rvecs_arr)
    np.save(session_dir / "p1_tvecs_raw.npy", tvecs_arr)
    print(f"  Saved: p1_rvecs_raw.npy, p1_tvecs_raw.npy → {session_dir}")

    return {"rvecs": rvecs_arr, "tvecs": tvecs_arr}