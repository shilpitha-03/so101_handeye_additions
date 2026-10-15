"""
Phase 3 — Axis Convention Conversion: OpenCV → Isaac Sim 5.x (Xform UI path)

We place the camera by entering values on an XFORM prim in the Isaac Sim UI,
with the Camera prim as a child at local (0,0,0). The Xform UI shows world-frame
values (+X forward, +Z up) — the same as our robot base frame. So:

Translation: T_cam_in_robot[:3,3] goes in directly. No axis remapping.

Rotation chain:
  R_cam_in_robot  (from Phase 2 — camera pose in robot/world frame)
      ↓  post-multiply 180° around X
      — OpenCV camera: +X right, +Y down, +Z forward (into scene)
      — USD camera:    +X right, +Y up,   -Z forward (into scene)
      — 180° X maps OpenCV → USD so the camera child "looks" correctly
  R_opengl  →  quaternion (w, x, y, z) scalar-first
      ↓  enter directly on Xform orient in UI

NOTE: The formula [rW, -rZ, rX, rY] from Isaac Sim docs is for the Camera()
Python API path (position=[-dZ, dX, dY]), NOT for placing an Xform in the UI.
Do not apply that remapping here.
"""

import numpy as np
from scipy.spatial.transform import Rotation


# 180° rotation around X: converts OpenCV camera axes → USD/OpenGL camera axes
_R_180x = Rotation.from_euler('x', 180, degrees=True).as_matrix()


def run(session_dir, phase2_data=None):
    """
    Entry point for Phase 3.
    session_dir : pathlib.Path
    phase2_data : dict from Phase 2 (optional — loads from disk if None)
    Returns     : dict with isaac_position and isaac_quaternion
    """
    print("\n========== PHASE 3 : ISAAC SIM CONVERSION ==========")

    if phase2_data is None:
        print("  Loading Phase 2 data from disk...")
        T_cam_in_robot = np.load(session_dir / "p2_T_cam_in_robot.npy")
    else:
        T_cam_in_robot = phase2_data["T_cam_in_robot"]

    t = T_cam_in_robot[:3, 3]    # camera position in world/robot frame
    R = T_cam_in_robot[:3, :3]   # camera orientation in world/robot frame

    print(f"  Input translation (robot frame): {t.round(6)}")
    print(f"  Input rotation matrix:\n{R.round(6)}")

    # ── Translation ───────────────────────────────────────────────────
    # Robot base frame == Isaac Sim world frame (+X forward, +Z up).
    # No remapping needed — enter directly into Xform Translate in the UI.
    isaac_position = t.copy()

    # ── Rotation ──────────────────────────────────────────────────────
    # R is the orientation of the camera expressed in world frame.
    # Columns of R are the camera's local X, Y, Z axes in world coordinates.
    # OpenCV camera: local +Z points forward (into scene), local +Y points down.
    # USD camera:    local -Z points forward (into scene), local +Y points up.
    # Post-multiplying by R_180x rotates the camera's local frame by 180° around
    # its own X axis — flipping Y and Z — giving the USD camera convention.
    R_usd = R @ _R_180x

    # Sanity: verify det = +1
    det = np.linalg.det(R_usd)
    if abs(det - 1.0) > 1e-6:
        print(f"  WARNING: det(R_usd) = {det:.6f} — rotation matrix issue")

    # Convert to quaternion, scalar-first (w, x, y, z) for Isaac Sim UI / USD
    q_xyzw = Rotation.from_matrix(R_usd).as_quat()  # scipy: scalar-last
    qx, qy, qz, qw = q_xyzw
    isaac_quaternion = np.array([qw, qx, qy, qz])   # scalar-first, no axis remap

    norm = np.linalg.norm(isaac_quaternion)

    # ── Sanity checks ─────────────────────────────────────────────────
    print(f"\n  Sanity checks:")
    print(f"    Z = {isaac_position[2]:.4f} m  (expect positive — camera above table)")
    if isaac_position[2] <= 0:
        print("    ✗ Z negative — check Phase 2 output")
    else:
        print("    ✓ Z positive")

    print(f"    Quaternion norm = {norm:.8f}  (must be 1.0)")
    if abs(norm - 1.0) > 1e-4:
        print("    ✗ Quaternion not unit length — check rotation chain")
    else:
        print("    ✓ Unit quaternion")

    # ── Final output ──────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  ENTER THESE ON THE XFORM PRIM IN ISAAC SIM UI")
    print("  (stage units must be meters — check Edit > Preferences)")
    print("=" * 58)
    print(f"\n  Translate X : {isaac_position[0]: .6f}  m   (forward from robot base)")
    print(f"  Translate Y : {isaac_position[1]: .6f}  m   (left of robot base)")
    print(f"  Translate Z : {isaac_position[2]: .6f}  m   (height above table)")
    print(f"\n  Orient  W   : {isaac_quaternion[0]: .6f}")
    print(f"  Orient  X   : {isaac_quaternion[1]: .6f}")
    print(f"  Orient  Y   : {isaac_quaternion[2]: .6f}")
    print(f"  Orient  Z   : {isaac_quaternion[3]: .6f}")
    print("\n  Validation steps in Isaac Sim:")
    print("  1. Edit > Preferences > Stage > Meters Per Unit = 1.0")
    print("  2. Create Xform prim 'side_camera_rig' — enter values above")
    print("  3. Add Camera prim as child — leave at local (0, 0, 0)")
    print("  4. Place a box at (0.135, 0, 0) to mark the ChArUco board origin")
    print("  5. Switch viewport to side_camera_rig/Camera — board box should")
    print("     appear at roughly the same position as in the real image")
    print("=" * 58)

    # Save
    np.save(session_dir / "p3_isaac_position.npy",   isaac_position)
    np.save(session_dir / "p3_isaac_quaternion.npy", isaac_quaternion)
    print(f"\n  Saved: p3_*.npy → {session_dir}")

    return {
        "isaac_position":   isaac_position,
        "isaac_quaternion": isaac_quaternion,
    }