import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2 as cv
import numpy as np

import config


# ----------------------------
# Utils
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rpy_to_R(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def xyz_rpy_to_T(x, y, z, roll, pitch, yaw, degrees=True):
    return make_T(rpy_to_R(roll, pitch, yaw, degrees), np.array([x, y, z], dtype=np.float64))


def load_intrinsics():
    K = np.load(config.CAMERA_MATRIX_PATH).astype(np.float64)
    D = np.load(config.DIST_COEFFS_PATH).astype(np.float64)
    return K, D


# ----------------------------
# ArUco / Charuco setup
# ----------------------------

def get_aruco_dictionary():
    if not hasattr(cv, "aruco"):
        raise RuntimeError("OpenCV aruco missing. Install opencv-contrib-python.")

    if not hasattr(cv.aruco, config.ARUCO_DICT_NAME):
        raise RuntimeError("Invalid ARUCO_DICT_NAME")

    return cv.aruco.getPredefinedDictionary(getattr(cv.aruco, config.ARUCO_DICT_NAME))


def create_charuco_board():
    dictionary = get_aruco_dictionary()

    if hasattr(cv.aruco, "CharucoBoard_create") and not hasattr(cv.aruco, "CharucoBoard"):
        return cv.aruco.CharucoBoard_create(
            config.SQUARES_X,
            config.SQUARES_Y,
            config.SQUARE_LENGTH_M,
            config.MARKER_LENGTH_M,
            dictionary,
        )

    # fallback constructor style
    n_markers = (config.SQUARES_X * config.SQUARES_Y) // 2
    first_id = getattr(config, "FIRST_MARKER_ID", 0)
    board = cv.aruco.CharucoBoard(
        (config.SQUARES_X, config.SQUARES_Y),
        config.SQUARE_LENGTH_M,
        config.MARKER_LENGTH_M,
        dictionary,
        np.arange(first_id, first_id + n_markers, dtype=np.int32),
    )
    # Boards made by older generators (pre-OpenCV 4.6, some online tools) use a different
    # layout when SQUARES_Y is even. detect_board.py tells you which one yours is.
    board.setLegacyPattern(getattr(config, "LEGACY_PATTERN", False))
    return board


# ----------------------------
# Pose result
# ----------------------------

@dataclass
class CharucoPoseResult:
    ok: bool
    T_camera_charuco: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    message: str = ""
    n_corners: int = 0
    reproj_px: Optional[float] = None   # RMS reprojection error of the detected corners


# ----------------------------
# SAFE Charuco pose estimation
# ----------------------------

def estimate_charuco_pose(image_bgr: np.ndarray, K: np.ndarray, D: np.ndarray) -> CharucoPoseResult:
    board = create_charuco_board()
    dictionary = get_aruco_dictionary()

    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

    # ----------------------------
    # Step 1: Detect markers (SAFE)
    # ----------------------------
    try:
        params = cv.aruco.DetectorParameters_create()
    except Exception:
        params = cv.aruco.DetectorParameters()

    marker_corners, marker_ids, _ = cv.aruco.detectMarkers(
        gray, dictionary, parameters=params
    )

    if marker_ids is None or len(marker_ids) == 0:
        return CharucoPoseResult(False, message="No markers detected")

    # ----------------------------
    # Step 2: Charuco interpolation (SAFE)
    # ----------------------------
    try:
        retval, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board, K, D
        )
    except Exception as e:
        return CharucoPoseResult(False, message=f"Charuco interpolation failed: {e}")

    if charuco_ids is None or retval is None or retval < 4:
        return CharucoPoseResult(False, message="Not enough charuco corners")

    # ----------------------------
    # Step 3: Pose estimation (SAFE)
    # ----------------------------
    try:
        ok, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            board,
            K,
            D,
            None,
            None,
        )
    except Exception as e:
        return CharucoPoseResult(False, message=f"Pose estimation failed: {e}")

    if not ok:
        return CharucoPoseResult(False, message="Pose estimation returned false")

    R, _ = cv.Rodrigues(rvec)
    T = make_T(R, tvec)

    reproj = None
    try:
        obj = np.asarray(board.getChessboardCorners(), dtype=np.float64)[charuco_ids.ravel()]
        proj, _ = cv.projectPoints(obj, rvec, tvec, K, D)
        reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - charuco_corners.reshape(-1, 2)) ** 2, axis=1))))
    except Exception:
        pass

    return CharucoPoseResult(
        True,
        T_camera_charuco=T,
        rvec=rvec,
        tvec=tvec,
        message=f"OK: {len(charuco_ids)} corners",
        n_corners=int(len(charuco_ids)),
        reproj_px=reproj,
    )


# ----------------------------
# Debug drawing
# ----------------------------

def draw_charuco_debug(image_bgr, result, K, D):
    out = image_bgr.copy()

    if result.ok:
        cv.drawFrameAxes(out, K, D, result.rvec, result.tvec, 0.05)

    cv.putText(
        out,
        result.message,
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if result.ok else (0, 0, 255),
        2,
    )

    return out


# ----------------------------
# Camera
# ----------------------------

def open_camera(index_or_path):
    cap = cv.VideoCapture(index_or_path, cv.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {index_or_path}")

    return cap

def print_T(name: str, T: np.ndarray) -> None:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    print(f"\n{name}:")
    with np.printoptions(precision=6, suppress=True):
        print(T)


def parse_camera_arg(value: str):
    try:
        return int(value)
    except ValueError:
        return value

def rotation_angle_deg(R):
    """
    Compute rotation angle (in degrees) from rotation matrix difference.
    """
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    return np.degrees(angle)