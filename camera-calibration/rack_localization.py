"""
Rack Localization — measure rack corner position in robot base frame.

Method:
  Single ArUco marker (ID 4, DICT_5X5_100) taped on the front face of the
  rack's nearest bottom-left corner leg. Bottom edge of marker flush with
  leg foot.

  Corner 3 of the marker (bottom-left in OpenCV clockwise ordering) is
  physically coincident with the rack corner point we want. No manual offset
  measurement needed — the script computes corner 3's world position directly
  from the pose.

Transform chain:
  solvePnPGeneric(IPPE_SQUARE) → T_marker_in_cam
      ↓  compose with T_cam_in_robot  (from Phase 2)
  T_marker_in_robot
      ↓  extract corner 3 position  (t + R @ [-half, -half, 0])
  rack_corner_in_robot  (x, y, z)  — ready for Isaac Sim

Physical setup:
  - Cover the fixed ChArUco board with white paper (no ID conflict)
  - Marker black square side = 21.68 mm  (RACK_MARKER_SIZE in config)
  - Run this script once — result is permanent until rack moves

Usage:
  python rack_localization.py
  python rack_localization.py --session results/session_YYYYMMDD_HHMMSS
"""

import argparse
import pathlib
import datetime
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from config import (
    SIDE_CAMERA_ID,
    SIDE_CAMERA_MATRIX_PATH,
    SIDE_DIST_COEFFS_PATH,
    ARUCO_DICT,
    RESULTS_BASE_DIR,
)

# ── Rack marker config ────────────────────────────────────────────────────────
RACK_MARKER_ID   = 4
RACK_MARKER_SIZE = 0.02168   # meters — black square side, measured with calipers
N_FRAMES         = 30        # frames to average (more = lower noise)
# ─────────────────────────────────────────────────────────────────────────────


def _build_detector():
    """
    ArucoDetector with CORNER_REFINE_SUBPIX.
    CORNER_REFINE_CONTOUR has a regression in OpenCV 4.7+ (returns integer
    corners, no actual refinement). SUBPIX is stable for single markers.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementMaxIterations = 100
    params.cornerRefinementMinAccuracy   = 0.05
    return cv2.aruco.ArucoDetector(dictionary, params)


def _ippe_square_obj_points(marker_size):
    """
    Object points for SOLVEPNP_IPPE_SQUARE.
    Origin = marker center, Z=0 = marker plane, Z points out toward camera.
    Order (matching detectMarkers clockwise from top-left):
      corner 0: top-left     [-half,  half, 0]
      corner 1: top-right    [ half,  half, 0]
      corner 2: bottom-right [ half, -half, 0]
      corner 3: bottom-left  [-half, -half, 0]  ← rack corner point
    """
    h = marker_size / 2.0
    return np.array([
        [-h,  h, 0],
        [ h,  h, 0],
        [ h, -h, 0],
        [-h, -h, 0],
    ], dtype=np.float64)


def _estimate_pose(corners_2d, obj_pts, K, D):
    """
    Estimate marker pose using solvePnPGeneric with IPPE_SQUARE.
    Returns (rvec, tvec) of shape (3,1) each — best solution by reprojection error.
    Returns (None, None) if estimation fails.
    """
    img_pts = corners_2d.reshape(4, 1, 2).astype(np.float64)
    retval, rvecs, tvecs, reprojErrs = cv2.solvePnPGeneric(
        obj_pts, img_pts, K, D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if retval == 0:
        return None, None
    best = int(np.argmin([e[0] for e in reprojErrs]))
    return rvecs[best], tvecs[best]


def _corner3_world_pos(rvec, tvec, marker_size):
    """
    Compute the world position of marker corner 3 (bottom-left).
    In marker frame: corner 3 is at [-half, -half, 0].
    Transform: p_cam = R @ p_marker + t
    """
    R, _ = cv2.Rodrigues(rvec)
    h = marker_size / 2.0
    p_marker = np.array([-h, -h, 0.0])
    p_cam = R @ p_marker + tvec.flatten()
    return p_cam   # in camera frame


def _chordal_mean_rotation(rvecs_list):
    """
    Chordal mean of rotation matrices — correct averaging on SO(3).
    rvecs_list: list of (3,1) arrays.
    Returns R_mean (3×3), rvec_mean (3,).
    """
    R_sum = np.zeros((3, 3))
    for rv in rvecs_list:
        R_i, _ = cv2.Rodrigues(rv)
        R_sum += R_i
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    rvec_mean, _ = cv2.Rodrigues(R_mean)
    return R_mean, rvec_mean.flatten()


def run(session_dir=None):
    """
    Main entry point.
    session_dir: pathlib.Path to an existing session containing
                 p2_T_cam_in_robot.npy  (from Phase 2).
                 If None, uses the most recent session in RESULTS_BASE_DIR.
    """
    print("\n========== RACK LOCALIZATION ==========")
    print(f"Marker ID     : {RACK_MARKER_ID}")
    print(f"Marker size   : {RACK_MARKER_SIZE*1000:.2f} mm (black square side)")
    print(f"Frames target : {N_FRAMES}")
    print("\nSetup checklist:")
    print("  ✓ ChArUco board covered with white paper")
    print("  ✓ Marker on front face of rack bottom-left corner leg")
    print("  ✓ Marker bottom edge flush with leg foot\n")

    # ── Resolve session dir ───────────────────────────────────────────
    if session_dir is None:
        base = pathlib.Path(RESULTS_BASE_DIR)
        sessions = sorted(base.glob("session_*"))
        if not sessions:
            raise RuntimeError(f"No sessions found in {RESULTS_BASE_DIR}. "
                               "Run the main pipeline first.")
        session_dir = sessions[-1]
        print(f"  Using most recent session: {session_dir}")
    else:
        session_dir = pathlib.Path(session_dir)

    T_cam_in_robot_path = session_dir / "p2_T_cam_in_robot.npy"
    if not T_cam_in_robot_path.exists():
        raise RuntimeError(f"p2_T_cam_in_robot.npy not found in {session_dir}. "
                           "Run Phase 2 first.")

    # ── Load camera data ──────────────────────────────────────────────
    K = np.load(SIDE_CAMERA_MATRIX_PATH)
    D = np.load(SIDE_DIST_COEFFS_PATH)
    T_cam_in_robot = np.load(T_cam_in_robot_path)

    print(f"  Loaded T_cam_in_robot from: {session_dir}")
    print(f"  Camera position in robot frame: "
          f"{T_cam_in_robot[:3,3].round(4)}\n")

    obj_pts = _ippe_square_obj_points(RACK_MARKER_SIZE)
    detector = _build_detector()

    # ── Open camera ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(SIDE_CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {SIDE_CAMERA_ID}")

    print(f"Camera opened. Collecting {N_FRAMES} frames...")
    print("SPACE → capture frame | Q → quit early\n")

    rvecs_collected = []
    tvecs_collected = []
    c3_cam_collected = []   # corner 3 positions in camera frame
    collected = 0

    last_ok    = False
    last_rvec  = None
    last_tvec  = None
    last_corners = None

    while collected < N_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("  WARNING: failed to grab frame, retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_all, ids_all, _ = detector.detectMarkers(gray)

        display = frame.copy()
        last_ok = False

        # ── Find our marker ID ────────────────────────────────────────
        if ids_all is not None:
            ids_flat = ids_all.flatten()
            matches  = np.where(ids_flat == RACK_MARKER_ID)[0]

            if len(matches) > 0:
                idx     = matches[0]
                corners = corners_all[idx]   # shape (1, 4, 2)

                rvec, tvec = _estimate_pose(corners[0], obj_pts, K, D)

                if rvec is not None:
                    last_ok      = True
                    last_rvec    = rvec
                    last_tvec    = tvec
                    last_corners = corners

                    # Draw axes and corners on display
                    cv2.aruco.drawDetectedMarkers(display, [corners], np.array([[RACK_MARKER_ID]]))
                    cv2.drawFrameAxes(display, K, D, rvec, tvec, RACK_MARKER_SIZE)

                    # Compute corner 3 position in cam frame for live display
                    c3_cam = _corner3_world_pos(rvec, tvec, RACK_MARKER_SIZE)

                    # Show corner 3 projected back to image for visual confirmation
                    c3_img, _ = cv2.projectPoints(
                        np.array([[-RACK_MARKER_SIZE/2, -RACK_MARKER_SIZE/2, 0.0]]),
                        rvec, tvec, K, D
                    )
                    cx, cy = int(c3_img[0][0][0]), int(c3_img[0][0][1])
                    cv2.circle(display, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.putText(display, "C3=rack corner",
                                (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 255), 2)

                    cv2.putText(display,
                                f"SPACE [{collected}/{N_FRAMES}] | Q quit",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 255, 0), 2)
                    cv2.putText(display,
                                f"tvec: {tvec.flatten().round(4)}",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(display, "Pose failed — adjust marker",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 165, 255), 2)
            else:
                cv2.putText(display,
                            f"ID {RACK_MARKER_ID} not detected — check coverage",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 255), 2)
        else:
            cv2.putText(display, "No markers detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255), 2)

        cv2.imshow("Rack Localization", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("  Quit by user.")
            break

        if key == ord(' '):
            if last_ok:
                c3_cam = _corner3_world_pos(last_rvec, last_tvec, RACK_MARKER_SIZE)
                rvecs_collected.append(last_rvec.copy())
                tvecs_collected.append(last_tvec.flatten().copy())
                c3_cam_collected.append(c3_cam.copy())
                collected += 1
                print(f"  [{collected}/{N_FRAMES}] captured"
                      f" — tvec: {last_tvec.flatten().round(4)}"
                      f" | c3_cam: {c3_cam.round(4)}")
            else:
                print("  No valid pose — reposition and try again")

    cap.release()
    cv2.destroyAllWindows()

    if collected == 0:
        raise RuntimeError("No frames collected. Rack localization failed.")
    if collected < N_FRAMES:
        print(f"\n  WARNING: only {collected}/{N_FRAMES} frames collected.")

    # ── Average poses ─────────────────────────────────────────────────
    # Rotation: chordal mean (correct on SO(3))
    # Translation: arithmetic mean (Euclidean)
    R_mean, _ = _chordal_mean_rotation(rvecs_collected)
    t_mean    = np.array(tvecs_collected).mean(axis=0)   # marker center in cam

    # Average corner 3 positions in camera frame
    c3_cam_arr  = np.array(c3_cam_collected)             # (N, 3)
    c3_cam_mean = c3_cam_arr.mean(axis=0)
    c3_cam_std  = c3_cam_arr.std(axis=0)

    print(f"\n  Frames used           : {collected}")
    print(f"  mean tvec (cam frame) : {t_mean.round(6)}")
    print(f"  c3 std across frames  : {c3_cam_std.round(5)}"
          f"  (target < 0.002 m each axis)")

    if c3_cam_std.max() > 0.002:
        print("  WARNING: high std on corner 3 — marker may have moved"
              " or pose is noisy. Consider re-running.")

    # ── Transform corner 3 from camera frame → robot base frame ───────
    # T_cam_in_robot maps points in camera frame to robot base frame.
    # p_robot = R_cam2robot @ p_cam + t_cam2robot
    R_cam2robot = T_cam_in_robot[:3, :3]
    t_cam2robot = T_cam_in_robot[:3,  3]

    rack_corner_robot = R_cam2robot @ c3_cam_mean + t_cam2robot

    x, y, z = rack_corner_robot

    # ── Sanity checks ─────────────────────────────────────────────────
    print(f"\n  Rack corner in robot base frame:")
    print(f"    X = {x:.6f} m  (forward from robot base)")
    print(f"    Y = {y:.6f} m  (left of robot base — expect positive, rack is on left)")
    print(f"    Z = {z:.6f} m  (height — expect ~{0.01145:.5f} m, leg height)")

    ok = True
    if x <= 0:
        print("  ✗ X negative — rack is behind robot base, check setup")
        ok = False
    else:
        print("  ✓ X positive")

    if y <= 0:
        print("  ✗ Y negative — rack should be on robot's left (+Y side)")
        ok = False
    else:
        print("  ✓ Y positive — rack is on left side")

    if abs(z - 0.01145) > 0.015:
        print(f"  ✗ Z = {z:.4f} m — expected ~0.01145 m (leg height)."
              f" Difference: {abs(z-0.01145)*1000:.1f} mm."
              f" Check marker bottom edge alignment.")
    else:
        print(f"  ✓ Z ≈ leg height ({abs(z-0.01145)*1000:.1f} mm deviation)")

    if not ok:
        print("\n  WARNING: sanity check failed — verify physical setup and re-run.")
    else:
        print("\n  All sanity checks passed.")

    # ── Final output ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RACK CORNER POSITION IN ROBOT BASE FRAME")
    print("=" * 55)
    print(f"  X : {x: .6f} m   (forward)")
    print(f"  Y : {y: .6f} m   (left)")
    print(f"  Z : {z: .6f} m   (height)")
    print("=" * 55)

    # ── Save ──────────────────────────────────────────────────────────
    out_path = session_dir / "rack_corner_robot_frame.npy"
    np.save(out_path, rack_corner_robot)
    print(f"\n  Saved: rack_corner_robot_frame.npy → {session_dir}")

    # Also save a human-readable text file
    txt_path = session_dir / "rack_corner_robot_frame.txt"
    with open(txt_path, "w") as f:
        f.write(f"Rack corner position in robot base frame\n")
        f.write(f"Measured: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Frames averaged: {collected}\n")
        f.write(f"Marker ID: {RACK_MARKER_ID}\n")
        f.write(f"Marker size: {RACK_MARKER_SIZE*1000:.2f} mm\n\n")
        f.write(f"X = {x:.6f} m  (forward from robot base)\n")
        f.write(f"Y = {y:.6f} m  (left of robot base)\n")
        f.write(f"Z = {z:.6f} m  (height above table)\n")
    print(f"  Saved: rack_corner_robot_frame.txt → {session_dir}")

    return {"rack_corner_robot": rack_corner_robot}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rack corner localization")
    parser.add_argument("--session", type=str, default=None,
                        help="Path to session dir containing p2_T_cam_in_robot.npy. "
                             "Defaults to most recent session.")
    args = parser.parse_args()
    run(session_dir=args.session)