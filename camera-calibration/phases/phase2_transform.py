"""
Phase 2 — Transform Chain
Reads raw rvecs/tvecs from Phase 1 (or session folder).
Steps:
  1. Chordal mean of rotation matrices + arithmetic mean of tvecs
  2. Build T_board_in_cam  (raw OpenCV 4.6+ convention, Z into board)
  3. Apply axis correction C = diag([1,-1,-1,1]) to full 4×4 matrix
     — flips Y and Z to restore Z-out-of-board (camera-toward) convention
  4. Invert → T_cam_in_board  (corrected board frame)
  5. Compose with T_board_in_robot → T_cam_in_robot  (robot base = world frame)
Saves all intermediate transforms for Phase 3.
"""

import numpy as np
import cv2
from config import (
    BOARD_X_FROM_ROBOT,
    BOARD_Y_FROM_ROBOT,
    BOARD_Z_FROM_ROBOT,
)


def _make_T(R, t):
    """Build 4×4 homogeneous transform from 3×3 R and 3-vec t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t.flatten()
    return T


def _chordal_mean_rotation(rvecs_arr):
    """
    Compute the chordal L2 mean of a set of rotation vectors.
    Correct for SO(3) — naive rvec arithmetic mean is wrong.
    rvecs_arr: shape (N, 3)
    Returns: R_mean (3×3), rvec_mean (3,)
    """
    R_sum = np.zeros((3, 3))
    for rvec in rvecs_arr:
        R_i, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        R_sum += R_i
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    # Guarantee det = +1 (proper rotation, not reflection)
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    rvec_mean, _ = cv2.Rodrigues(R_mean)
    return R_mean, rvec_mean.flatten()


def run(session_dir, phase1_data=None):
    """
    Entry point for Phase 2.
    session_dir  : pathlib.Path
    phase1_data  : dict from Phase 1 (optional — loads from disk if None)
    Returns      : dict with all intermediate and final transforms
    """
    print("\n========== PHASE 2 : TRANSFORM CHAIN ==========")

    # ── Load Phase 1 data ─────────────────────────────────────────────
    if phase1_data is None:
        print("  Loading Phase 1 data from disk...")
        rvecs_arr = np.load(session_dir / "p1_rvecs_raw.npy")
        tvecs_arr = np.load(session_dir / "p1_tvecs_raw.npy")
    else:
        rvecs_arr = phase1_data["rvecs"]   # shape (N, 3)
        tvecs_arr = phase1_data["tvecs"]   # shape (N, 3)

    print(f"  Frames loaded: {len(rvecs_arr)}")

    # ── Step 1: Average poses ─────────────────────────────────────────
    # Rotation: chordal mean (correct on SO(3) manifold)
    # Translation: arithmetic mean (correct — Euclidean)
    R_mean, rvec_mean = _chordal_mean_rotation(rvecs_arr)
    t_mean = tvecs_arr.mean(axis=0)   # shape (3,)

    print(f"  mean rvec : {rvec_mean.round(6)}")
    print(f"  mean tvec : {t_mean.round(6)}")

    # ── Step 2: Build T_board_in_cam (raw OpenCV 4.6+ convention) ─────
    # rvec/tvec from solvePnP: board origin in camera frame.
    # Z points INTO the board (OpenCV 4.6+ corner indexing change).
    T_board_in_cam = _make_T(R_mean, t_mean)
    print(f"\n  T_board_in_cam (raw, Z into board):\n{T_board_in_cam.round(6)}")

    # ── Step 3: Apply axis correction ────────────────────────────────
    # C applied on the RIGHT: changes the board coordinate frame convention.
    # Left-multiply would change the camera frame — that is wrong.
    # Right-multiply negates the Y and Z axes of the board frame,
    # correcting for OpenCV 4.6+ corner indexing which flipped those axes.
    C = np.diag([1.0, -1.0, -1.0, 1.0])
    T_board_in_cam_corrected = T_board_in_cam @ C
    print(f"\n  T_board_in_cam (corrected, Z out of board):\n"
          f"{T_board_in_cam_corrected.round(6)}")

    # ── Step 4: Invert → T_cam_in_board ──────────────────────────────
    # For 4×4 homogeneous: R_inv = R.T, t_inv = -R.T @ t
    # Using np.linalg.inv is fine here (R is orthonormal so it's stable).
    T_cam_in_board = np.linalg.inv(T_board_in_cam_corrected)
    cam_pos_in_board = T_cam_in_board[:3, 3]
    print(f"\n  T_cam_in_board:\n{T_cam_in_board.round(6)}")
    print(f"  Camera position in board frame: {cam_pos_in_board.round(4)}")

    # ── Step 5: Compose with T_board_in_robot ────────────────────────
    # Robot base = world origin.
    # Board is at (BOARD_X_FROM_ROBOT, BOARD_Y_FROM_ROBOT, BOARD_Z_FROM_ROBOT)
    # with identity rotation (board axes aligned to robot axes).
    T_board_in_robot = np.eye(4)
    T_board_in_robot[0, 3] = BOARD_X_FROM_ROBOT
    T_board_in_robot[1, 3] = BOARD_Y_FROM_ROBOT
    T_board_in_robot[2, 3] = BOARD_Z_FROM_ROBOT

    T_cam_in_robot = T_board_in_robot @ T_cam_in_board
    cam_pos_in_robot = T_cam_in_robot[:3, 3]
    print(f"\n  T_cam_in_robot (robot base = world frame):\n{T_cam_in_robot.round(6)}")
    print(f"  Camera position in robot frame: {cam_pos_in_robot.round(4)}")

    # ── Sanity checks ─────────────────────────────────────────────────
    x, y, z = cam_pos_in_robot
    print(f"\n  Sanity checks:")
    print(f"    X = {x:.4f} m  (expect > {BOARD_X_FROM_ROBOT:.3f} — camera is behind the board)")
    print(f"    Y = {y:.4f} m  (lateral offset from robot centerline)")
    print(f"    Z = {z:.4f} m  (expect POSITIVE — camera is above the table)")

    ok = True
    if z <= 0:
        print("    ✗ Z is NEGATIVE — board was not flat during capture, or axis correction failed")
        ok = False
    else:
        print("    ✓ Z positive — correct")

    if x <= BOARD_X_FROM_ROBOT:
        print(f"    ✗ X <= {BOARD_X_FROM_ROBOT:.3f} — camera appears in front of board, check measurement")
        ok = False
    else:
        print(f"    ✓ X > {BOARD_X_FROM_ROBOT:.3f} — correct")

    # Rotation matrix orthonormality check
    R_out = T_cam_in_robot[:3, :3]
    det = np.linalg.det(R_out)
    ortho_err = np.max(np.abs(R_out @ R_out.T - np.eye(3)))
    print(f"    det(R) = {det:.6f}  (expect 1.0)")
    print(f"    max |R·Rᵀ - I| = {ortho_err:.2e}  (expect < 1e-10)")
    if abs(det - 1.0) > 1e-6 or ortho_err > 1e-10:
        print("    ✗ Rotation matrix is not orthonormal — numerical issue")
        ok = False
    else:
        print("    ✓ Rotation matrix orthonormal")

    if not ok:
        print("\n  WARNING: one or more sanity checks failed — do not use Phase 3 output")
    else:
        print("\n  All sanity checks passed — ready for Phase 3")

    # ── Save ──────────────────────────────────────────────────────────
    np.save(session_dir / "p2_T_board_in_cam.npy",           T_board_in_cam)
    np.save(session_dir / "p2_T_board_in_cam_corrected.npy", T_board_in_cam_corrected)
    np.save(session_dir / "p2_T_cam_in_board.npy",           T_cam_in_board)
    np.save(session_dir / "p2_T_cam_in_robot.npy",           T_cam_in_robot)
    print(f"\n  Saved: p2_*.npy → {session_dir}")

    return {
        "T_board_in_cam":           T_board_in_cam,
        "T_board_in_cam_corrected": T_board_in_cam_corrected,
        "T_cam_in_board":           T_cam_in_board,
        "T_cam_in_robot":           T_cam_in_robot,
    }