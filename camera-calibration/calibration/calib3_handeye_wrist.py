"""
Calibration 3 — Wrist Camera Hand-Eye Calibration (Eye-in-Hand)
================================================================
Solves AX = XB  →  X = T_cam2gripper
  (fixed transform: camera optical frame relative to gripper_link frame)

What is collected per pose
--------------------------
  B — board pose in wrist camera frame    (ChArUco detection, this script)
  A — gripper_link pose in robot base     (SO-101 FK via FeetechMotorsBus)

Procedure
---------
  1. Cover the fixed ChArUco board on the table with white paper.
  2. Connect follower arm USB (FOLLOWER_PORT in config.py).
  3. Run this script (or call run() from main.py).
  4. Teleop the arm to a pose using the leader arm.
  5. Hold the HANDHELD calibration board visible in the wrist camera.
  6. Press SPACE — captures board pose from camera + FK from arm simultaneously.
  7. Move arm to a new pose (>30 deg rotation, ideally on a different axis).
  8. Repeat until HANDEYE_MIN_POSES captured, then press Q.

Pose diversity requirements (critical for good calibration)
-----------------------------------------------------------
  - Each new pose must differ from the previous by >30 deg of ROTATION.
  - Vary ALL joints across the session — do not just rotate the last wrist joint.
  - Include arm tilts left/right, forward/back, and varied elbow angles.
  - Avoid purely translational moves between poses.
  - Minimum: HANDEYE_MIN_POSES (15).  Recommended: 18-20.

OpenCV API
----------
  All pose estimation uses the modern OpenCV 4.7+ API:
    board.matchImagePoints() + cv2.solvePnPGeneric(SOLVEPNP_IPPE)
  The deprecated cv2.aruco.estimatePoseCharucoBoard() is NOT used.
  No axis correction is applied — raw solvePnP output (board-in-camera)
  is exactly R_target2cam / t_target2cam that cv2.calibrateHandEye expects.

Output
------
  calibration_files/wrist/T_cam2gripper.npy        — 4x4 float64
  calibration_files/wrist/handeye_R_board2cam.npy
  calibration_files/wrist/handeye_t_board2cam.npy
  calibration_files/wrist/handeye_R_gripper2base.npy
  calibration_files/wrist/handeye_t_gripper2base.npy

Log file
--------
  Standalone: logs/standalone_YYYYMMDD_HHMMSS.log
  From main.py: results/session_XXX/session.log
  Log contains: raw ticks, degrees, FK matrices, board poses,
                reprojection errors, solver inputs/outputs, sanity check numbers.
"""

import pathlib

import cv2
import numpy as np
import platform
import time

from logging_utils import get_logger
from calibration.so101_fk import (
    _NOMINAL_T_CAM_IN_GRIPPER_XYZ,
    _validate_rotation,
    compute_gripper2base,
    degrees_to_radians,
    load_calibration_params,
    raw_ticks_to_degrees,
)
from config import (
    ARUCO_DICT,
    FOLLOWER_CALIB_JSON,
    FOLLOWER_PORT,
    HANDEYE_MIN_POSES,
    MARKER_LENGTH,
    MIN_CHARUCO_CORNERS,
    SQUARE_LENGTH,
    SQUARES_X,
    SQUARES_Y,
    WRIST_CAMERA_ID,
    WRIST_CAMERA_MATRIX_PATH,
    WRIST_DIST_COEFFS_PATH,
    WRIST_HANDEYE_PATH,
)

# lerobot v0.5 imports — only needed when arm is physically connected.
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

logger = get_logger(__name__)

# Joint definitions for the 5 revolute arm joints.
# Motor 6 (gripper) deliberately excluded — does not affect gripper_link pose.
_ARM_JOINTS = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
}

# Order that compute_gripper2base() / FK chain expects.
_FK_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_SHARED_BUS = None


# ── FK: read arm + compute gripper pose ──────────────────────────────────────

def get_gripper2base(
    homing_offsets: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Read current joint angles from the SO-101 follower arm and compute
    gripper_link pose in robot base frame via forward kinematics.

    Strategy
    --------
    Opens FeetechMotorsBus, reads a single snapshot with normalize=False
    (zero hardware writes — no risk of disturbing motor registers or
    interfering with ongoing teleoperation), then closes immediately.

    Conversion: raw_tick → degrees = (raw - homing_offset) * 360.0 / 4096.0
    Then degrees → radians → FK.

    Returns
    -------
    (R [3x3], t [3,])  on success.
    (None, None)        on any connection or read error — caller skips the pose.
    """
    global _SHARED_BUS

    try:
        if _SHARED_BUS is None:
            _SHARED_BUS = FeetechMotorsBus(port=FOLLOWER_PORT, motors=_ARM_JOINTS)
            _SHARED_BUS.connect()
        raw: dict = _SHARED_BUS.sync_read("Present_Position", normalize=False)
        logger.debug("Raw ticks from arm: %s", raw)

    except Exception as exc:
        logger.error("Arm read failed on port %s: %s", FOLLOWER_PORT, exc)
        return None, None

    finally:
        pass

    # Convert ticks → degrees
    degrees = raw_ticks_to_degrees(raw, homing_offsets)

    missing = [j for j in _FK_ORDER if j not in degrees]
    if missing:
        logger.error("sync_read missing joints: %s", missing)
        return None, None

    joint_deg = [degrees[j] for j in _FK_ORDER]
    joint_rad = degrees_to_radians(joint_deg)

    logger.info(
        "  Joint deg: pan=%.1f  lift=%.1f  elbow=%.1f  wflex=%.1f  wroll=%.1f",
        *joint_deg,
    )
    logger.debug(
        "  Joint rad: %s", [round(r, 5) for r in joint_rad.tolist()],
    )

    R, t = compute_gripper2base(joint_rad)

    if not _validate_rotation(R, "FK"):
        logger.error("FK produced invalid rotation matrix — check arm connection.")
        return None, None

    logger.debug(
        "  FK result  t=[%.4f %.4f %.4f]  R=\n%s",
        t[0], t[1], t[2], np.round(R, 5),
    )
    return R, t


# ── ChArUco detector ─────────────────────────────────────────────────────────

def _build_detector(K: np.ndarray, D: np.ndarray):
    """
    Build CharucoDetector with camera intrinsics for reprojection-based
    corner interpolation (more accurate than homography under distortion).
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )
    # Do NOT call board.setLegacyPattern() — using OpenCV 4.6+ convention.
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.cameraMatrix = K
    charuco_params.distCoeffs   = D

    detector_params = cv2.aruco.DetectorParameters()
    # CORNER_REFINE_SUBPIX — CORNER_REFINE_CONTOUR has a regression in OpenCV 4.7+
    # (returns integer corners, no actual refinement).
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    logger.debug(
        "CharucoDetector built: board=%dx%d  sq=%.4fm  mk=%.4fm",
        SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH,
    )
    return board, detector


# ── Board pose estimation ─────────────────────────────────────────────────────

def _estimate_board_pose(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """
    Estimate ChArUco board pose using the modern OpenCV 4.7+ API.

    Uses board.matchImagePoints() + cv2.solvePnPGeneric(SOLVEPNP_IPPE).
    SOLVEPNP_IPPE is correct for coplanar object points (flat board) and
    returns two solutions — we select by lowest reprojection error.

    Returns raw solvePnP output (board-in-camera frame), which is exactly
    R_target2cam / t_target2cam for cv2.calibrateHandEye().
    NO axis correction is applied here.

    Returns
    -------
    (R_board2cam [3x3], t_board2cam [3,], reprojection_error)
    or (None, None, inf) on failure.
    """
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts is None or obj_pts.shape[0] < 4:
        logger.debug("matchImagePoints: insufficient points (%s)",
                     0 if obj_pts is None else obj_pts.shape[0])
        return None, None, float("inf")

    retval, rvecs, tvecs, reprojErrs = cv2.solvePnPGeneric(
        obj_pts, img_pts, K, D,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if retval == 0:
        logger.debug("solvePnPGeneric returned no solutions")
        return None, None, float("inf")

    best       = int(np.argmin([e[0] for e in reprojErrs]))
    reproj_err = float(reprojErrs[best][0])
    R_b2c, _   = cv2.Rodrigues(rvecs[best])
    t_b2c      = tvecs[best].flatten()

    logger.debug(
        "Board pose: t=[%.4f %.4f %.4f]  reproj_err=%.4f px  (solution %d/%d)",
        t_b2c[0], t_b2c[1], t_b2c[2], reproj_err, best, retval,
    )
    if reproj_err > 2.0:
        logger.warning(
            "High reprojection error %.4f px — board may be blurry or partially occluded.",
            reproj_err,
        )

    return R_b2c, t_b2c, reproj_err


def _backend_candidates():

    system = platform.system()
    print("&&" * 20)
    print(system)

    candidates = []

    if system == "Linux":
        candidates.append(getattr(cv2, "CAP_V4L2", cv2.CAP_ANY))

    elif system == "Windows":
        candidates.append(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))

    elif system == "Darwin":
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


def _configure_camera(cap):

    # Optional:
    # keep camera native defaults
    pass


def _open_camera(camera_id):

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

    raise RuntimeError(
        f"Cannot open wrist camera at index: {camera_id}"
    )


# ── Main calibration loop ─────────────────────────────────────────────────────

def run() -> dict:
    """
    Entry point — call from main.py or run independently.

    Returns
    -------
    dict with keys:
      "T_cam2gripper" : np.ndarray (4x4) on success.
      "captured"      : int — number of poses captured.
      "status"        : "ok" | "fk_error".
    """
    logger.info("=" * 60)
    logger.info("CALIB 3 — Wrist Camera Hand-Eye Calibration")
    logger.info("=" * 60)
    logger.info("Follower port  : %s", FOLLOWER_PORT)
    logger.info("Calib JSON     : %s", FOLLOWER_CALIB_JSON)
    logger.info("Min poses      : %d", HANDEYE_MIN_POSES)
    logger.info("")
    logger.info("Before starting:")
    logger.info("  1. Cover the FIXED ChArUco board on the table with white paper.")
    logger.info("  2. Prepare the HANDHELD calibration board.")
    logger.info("  3. Ensure follower arm USB is connected on %s.", FOLLOWER_PORT)
    logger.info("")
    logger.info("During capture:")
    logger.info("  - Teleop arm to a new pose using the leader arm.")
    logger.info("  - Hold handheld board visible in wrist camera view.")
    logger.info("  - SPACE = capture | Q = stop and run calibration")
    logger.info("")
    logger.info("CRITICAL: each pose must change by >30 deg rotation on varied axes.")
    logger.info("          Do not just rotate the wrist — vary shoulder and elbow too.")

    # ── Load calibration JSON ─────────────────────────────────────────────────
    try:
        calib_params = load_calibration_params(FOLLOWER_CALIB_JSON)
        logger.info("Homing offsets loaded from %s", FOLLOWER_CALIB_JSON)
        for j, v in calib_params.items():
            logger.debug("  %-16s homing_offset=%d", j, v)
    except FileNotFoundError as exc:
        logger.error("FATAL: %s", exc)
        logger.error("Cannot proceed without calibration JSON.")
        return {"status": "fk_error", "captured": 0}

    # ── Test arm connection before opening camera ─────────────────────────────
    logger.info("")
    logger.info("Testing arm connection...")
    R_test, t_test = get_gripper2base(calib_params)
    if R_test is None:
        logger.error("FATAL: cannot read arm on port %s.", FOLLOWER_PORT)
        logger.error("Check USB connection and permissions:")
        logger.error("  sudo chmod 666 %s  OR  sudo usermod -a -G dialout $USER", FOLLOWER_PORT)
        return {"status": "fk_error", "captured": 0}
    logger.info(
        "Arm connected OK — gripper_link at current pose: [%.4f %.4f %.4f] m",
        t_test[0], t_test[1], t_test[2],
    )

    # ── Load camera intrinsics ────────────────────────────────────────────────
    K = np.load(WRIST_CAMERA_MATRIX_PATH)
    D = np.load(WRIST_DIST_COEFFS_PATH)
    logger.debug("Wrist K:\n%s", np.round(K, 3))
    logger.debug("Wrist D: %s", np.round(D.flatten(), 6).tolist())

    board, detector = _build_detector(K, D)

    # ── Open wrist camera ─────────────────────────────────────────────────────
    cap = _open_camera(WRIST_CAMERA_ID)

    cv2.namedWindow(
        "Calib 3 — Wrist Hand-Eye",
        cv2.WINDOW_NORMAL,
    )

    # Camera warmup
    logger.info("Warming up wrist camera...")

    warmup_ok = 0

    for _ in range(30):

        ret, frame = cap.read()

        if ret and frame is not None:
            warmup_ok += 1

        time.sleep(0.02)

    logger.info("Warmup frames received: %d/30", warmup_ok)

    # ── Storage ───────────────────────────────────────────────────────────────
    R_board2cam_list    = []
    t_board2cam_list    = []
    R_gripper2base_list = []
    t_gripper2base_list = []
    reproj_errors       = []
    captured = 0

    # Cache last good detection so SPACE can reuse it without re-solving.
    _last: dict = {
        "ok": False, "R": None, "t": None,
        "reproj": float("inf"), "corners": None, "ids": None,
    }

    logger.info("Camera ready. Waiting for captures...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame — retrying.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        display = frame.copy()
        n_corners = 0 if charuco_ids is None else len(charuco_ids)
        enough    = n_corners >= MIN_CHARUCO_CORNERS

        if enough:
            R_b2c, t_b2c, reproj = _estimate_board_pose(
                charuco_corners, charuco_ids, board, K, D,
            )
            if R_b2c is not None:
                _last.update(
                    ok=True, R=R_b2c, t=t_b2c, reproj=reproj,
                    corners=charuco_corners, ids=charuco_ids,
                )
                rvec_disp, _ = cv2.Rodrigues(R_b2c)
                cv2.aruco.drawDetectedCornersCharuco(
                    display, charuco_corners, charuco_ids,
                )
                cv2.drawFrameAxes(
                    display, K, D, rvec_disp, t_b2c, SQUARE_LENGTH * 2,
                )
                cv2.putText(
                    display,
                    f"SPACE [{captured}/{HANDEYE_MIN_POSES}]"
                    f"  corners={n_corners}  reproj={reproj:.2f}px  Q=done",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                )
            else:
                _last["ok"] = False
                cv2.putText(
                    display, "Pose estimation failed — reposition board",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
                )
        else:
            _last["ok"] = False
            cv2.putText(
                display,
                f"Only {n_corners} corners — need >= {MIN_CHARUCO_CORNERS}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

        cv2.imshow("Calib 3 — Wrist Hand-Eye", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            logger.info("Q pressed — stopping capture with %d poses.", captured)
            break

        if key == ord(" "):
            if not _last["ok"]:
                logger.info("SPACE: no valid board pose — reposition board.")
                continue

            R_b2c   = _last["R"]
            t_b2c   = _last["t"]
            reproj  = _last["reproj"]

            # Read arm at this exact instant — same moment as the camera pose.
            R_g2b, t_g2b = get_gripper2base(calib_params)
            if R_g2b is None:
                logger.warning("SPACE: arm read failed — skipping this pose.")
                continue

            R_board2cam_list.append(R_b2c)
            t_board2cam_list.append(t_b2c.reshape(3, 1))
            R_gripper2base_list.append(R_g2b)
            t_gripper2base_list.append(t_g2b.reshape(3, 1))
            reproj_errors.append(reproj)

            captured += 1
            logger.info(
                "[%2d/%d] captured  |  board t_cam=[%.3f %.3f %.3f]"
                "  |  gripper t_base=[%.3f %.3f %.3f]"
                "  |  reproj=%.3f px",
                captured, HANDEYE_MIN_POSES,
                t_b2c[0], t_b2c[1], t_b2c[2],
                t_g2b[0], t_g2b[1], t_g2b[2],
                reproj,
            )
            logger.debug(
                "  R_board2cam:\n%s\n  R_gripper2base:\n%s",
                np.round(R_b2c, 5), np.round(R_g2b, 5),
            )

            if captured >= HANDEYE_MIN_POSES:
                logger.info(
                    "Reached %d poses — press Q to calibrate or keep capturing.",
                    HANDEYE_MIN_POSES,
                )

    cap.release()
    cv2.destroyAllWindows()

    # ── Minimum viable check ──────────────────────────────────────────────────
    if captured < 3:
        raise RuntimeError(
            f"Only {captured} poses captured — need at least 3 for calibrateHandEye."
        )

    if captured < HANDEYE_MIN_POSES:
        logger.warning(
            "Only %d/%d poses — result may be less accurate. "
            "Consider re-running with more diverse poses.",
            captured, HANDEYE_MIN_POSES,
        )

    # ── Log reprojection error summary ───────────────────────────────────────
    reproj_arr = np.array(reproj_errors)
    logger.info(
        "Reprojection errors across %d poses: "
        "mean=%.3f px  min=%.3f px  max=%.3f px",
        captured, reproj_arr.mean(), reproj_arr.min(), reproj_arr.max(),
    )
    if reproj_arr.max() > 2.0:
        logger.warning(
            "Max reprojection error %.3f px > 2.0 px — "
            "one or more poses may have had board blur or occlusion.",
            reproj_arr.max(),
        )

    # ── Save raw poses (always — even if solve were to fail) ─────────────────
    out_dir = pathlib.Path(WRIST_HANDEYE_PATH).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "handeye_R_board2cam.npy",    np.array(R_board2cam_list))
    np.save(out_dir / "handeye_t_board2cam.npy",    np.array(t_board2cam_list))
    np.save(out_dir / "handeye_R_gripper2base.npy", np.array(R_gripper2base_list))
    np.save(out_dir / "handeye_t_gripper2base.npy", np.array(t_gripper2base_list))
    logger.info("Raw poses saved → %s", out_dir)

    # ── Log full inputs for post-hoc debugging ────────────────────────────────
    logger.debug("calibrateHandEye inputs (%d poses):", captured)
    for i in range(captured):
        logger.debug(
            "  pose %2d  R_g2b det=%.6f  |  t_b2c=[%.4f %.4f %.4f]  t_g2b=[%.4f %.4f %.4f]",
            i,
            np.linalg.det(R_gripper2base_list[i]),
            t_board2cam_list[i].flatten()[0],
            t_board2cam_list[i].flatten()[1],
            t_board2cam_list[i].flatten()[2],
            t_gripper2base_list[i].flatten()[0],
            t_gripper2base_list[i].flatten()[1],
            t_gripper2base_list[i].flatten()[2],
        )

    # ── Solve AX = XB ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info(
        "Running cv2.calibrateHandEye (PARK method) on %d poses...", captured,
    )
    logger.debug(
        "PARK method chosen: robust for varied pose sets. "
        "Alternatives (TSAI, HORAUD, ANDREFF, DANIILIDIS) available but PARK "
        "recommended when pose diversity is good and N >= 15."
    )

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list,    # A — gripper in base, list of (3x3) float64
        t_gripper2base_list,    # A — gripper in base, list of (3x1) float64
        R_board2cam_list,       # B — board in camera, list of (3x3) float64
        t_board2cam_list,       # B — board in camera, list of (3x1) float64
        method=cv2.CALIB_HAND_EYE_PARK,
    )

    t_result = t_cam2gripper.flatten()

    # ── Build 4x4 result ──────────────────────────────────────────────────────
    T_cam2gripper         = np.eye(4, dtype=np.float64)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3,  3] = t_result

    # ── Log full result ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Result ──────────────────────────────────────────────────")
    logger.info(
        "T_cam2gripper:\n"
        "  [%8.5f %8.5f %8.5f %8.5f]\n"
        "  [%8.5f %8.5f %8.5f %8.5f]\n"
        "  [%8.5f %8.5f %8.5f %8.5f]\n"
        "  [%8.5f %8.5f %8.5f %8.5f]",
        *T_cam2gripper.flatten().tolist(),
    )
    logger.info("Translation (x, y, z) : [%.4f  %.4f  %.4f] m", *t_result)

    # ── Rotation validity ─────────────────────────────────────────────────────
    det      = np.linalg.det(R_cam2gripper)
    orth_err = np.max(np.abs(R_cam2gripper @ R_cam2gripper.T - np.eye(3)))
    logger.info("det(R_cam2gripper)    : %.8f  (must be 1.0)", det)
    logger.info("max |R·Rᵀ - I|        : %.2e  (must be ~0)", orth_err)

    r_ok = abs(det - 1.0) < 0.01 and orth_err < 1e-4
    if r_ok:
        logger.info("Rotation matrix       : VALID")
    else:
        logger.warning(
            "Rotation matrix INVALID — likely cause: insufficient pose diversity. "
            "Re-run with more varied arm configurations."
        )

    # ── Sanity check vs URDF nominal camera offset ────────────────────────────
    nominal = _NOMINAL_T_CAM_IN_GRIPPER_XYZ
    diff    = np.linalg.norm(t_result - nominal)
    logger.info("")
    logger.info("── Sanity check vs URDF camera mount ───────────────────────")
    logger.info("URDF nominal  t (xyz) : %s m", nominal.tolist())
    logger.info("Calibrated    t (xyz) : [%.4f  %.4f  %.4f] m", *t_result)
    logger.info("Euclidean difference  : %.1f cm", diff * 100)
    logger.debug(
        "Per-axis diff (x, y, z): [%.4f  %.4f  %.4f] m",
        *(t_result - nominal),
    )

    if diff < 0.03:
        logger.info("Sanity check : PASSED (within 3 cm of URDF design)")
    elif diff < 0.06:
        logger.warning(
            "Sanity check: %.1f cm from URDF design (3-6 cm range). "
            "Acceptable if pose diversity was good, but consider re-running.",
            diff * 100,
        )
    else:
        logger.warning(
            "Sanity check: %.1f cm from URDF design (>6 cm). "
            "Result likely inaccurate. Probable causes: insufficient pose diversity, "
            "FK calibration error, or camera physically shifted on mount.",
            diff * 100,
        )
        logger.warning(
            "Recommendation: re-run with 18-20 poses, ensuring each pose "
            "has >30 deg rotation change on a different axis from the previous."
        )

    # ── Save result ───────────────────────────────────────────────────────────
    np.save(WRIST_HANDEYE_PATH, T_cam2gripper)
    logger.info("")
    logger.info("Saved: T_cam2gripper.npy → %s", WRIST_HANDEYE_PATH)
    logger.info("=" * 60)
    
    global _SHARED_BUS

    if _SHARED_BUS is not None:

        try:
            logger.info("Disconnecting Feetech bus...")
            _SHARED_BUS.disconnect()

        except Exception as exc:
            logger.warning(
                "Error while disconnecting bus: %s",
                exc,
            )

        _SHARED_BUS = None

    return {
        "T_cam2gripper": T_cam2gripper,
        "captured": captured,
        "status": "ok",
    }