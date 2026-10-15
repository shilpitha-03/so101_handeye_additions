"""
SO-101 Forward Kinematics — calibration/so101_fk.py
====================================================
Standalone module.  No lerobot dependency.  Pure numpy.

Computes gripper_link pose in robot base frame given 5 joint angles.
FK reference frame: gripper_link  (the output of the wrist_roll joint).
This is the correct reference for hand-eye calibration because the wrist
camera is rigidly attached to gripper_link via fixed joints in the URDF.

Joint order (matches URDF and lerobot motor IDs):
    1  shoulder_pan
    2  shoulder_lift
    3  elbow_flex
    4  wrist_flex
    5  wrist_roll

gripper (motor 6) is NOT included — it does not affect gripper_link pose.

Input
-----
joint_angles_rad : array-like, shape (5,)
    Angles in radians, order above.
    Origin: lerobot normalized degrees (from none.json calibration) converted
    via np.radians().

Output
------
R_gripper2base : np.ndarray, shape (3, 3)  — rotation matrix
t_gripper2base : np.ndarray, shape (3,)    — translation, metres

Usage
-----
    from calibration.so101_fk import compute_gripper2base
    R, t = compute_gripper2base([q1, q2, q3, q4, q5])  # radians

Standalone validation (run from project root):
    python calibration/so101_fk.py
"""

import json
import pathlib

import numpy as np

from logging_utils import get_logger

logger = get_logger(__name__)

# ── URDF joint chain ──────────────────────────────────────────────────────────
# Each entry: (joint_name, xyz_in_parent, rpy_in_parent)
# Source: so101_updated.urdf, joint origin attributes, in chain order.
# All values copied verbatim from URDF — do not edit without re-deriving from URDF.
# Every joint rotates around its LOCAL Z axis (axis xyz="0 0 1" in URDF).
_JOINT_CHAIN = [
    # joint_name        xyz (m)                                   rpy (rad)
    ("shoulder_pan",  [ 0.0388353, -8.97657e-09,  0.0624],   [ 3.14159,  4.18253e-17, -3.14159]),
    ("shoulder_lift", [-0.0303992, -0.0182778,   -0.0542],   [-1.5708,  -1.5708,       0.0    ]),
    ("elbow_flex",    [-0.11257,   -0.028,         1.73763e-16],[0.0,    0.0,           1.5708 ]),
    ("wrist_flex",    [-0.1349,     0.0052,         3.62355e-17],[0.0,   0.0,          -1.5708 ]),
    ("wrist_roll",    [ 5.55112e-17,-0.0611,        0.0181],  [ 1.5708,  0.0486795,    3.14159]),
]

# URDF-derived nominal camera offset from gripper_link (for post-calib validation).
# Chain: gripper_link → vinmooog_mount → vinmooog_webcam (both fixed joints).
# Exposed for use in calib3 sanity check only — never used in computation.
_NOMINAL_T_CAM_IN_GRIPPER_XYZ = np.array([0.0, 0.07, -0.0172])  # metres


# ── Low-level math ────────────────────────────────────────────────────────────

def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    """Intrinsic RPY → rotation matrix  (R = Rz @ Ry @ Rx)."""
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(y), np.sin(y)
    Rx = np.array([[1,  0,   0 ], [0,  cx, -sx], [0,  sx,  cx]], dtype=np.float64)
    Ry = np.array([[cy, 0,  sy ], [0,  1,   0 ], [-sy, 0,  cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0 ], [sz, cz,  0 ], [0,   0,   1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _make_T(xyz, rpy) -> np.ndarray:
    """Build 4×4 homogeneous transform from xyz + rpy."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_to_R(*rpy)
    T[:3,  3] = xyz
    return T


def _rotz(q: float) -> np.ndarray:
    """4×4 pure rotation around Z by angle q (radians)."""
    c, s = np.cos(q), np.sin(q)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=np.float64)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_gripper2base(joint_angles_rad) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward kinematics: joint angles → gripper_link pose in base_link frame.

    Parameters
    ----------
    joint_angles_rad : array-like of float, length 5
        [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
        in radians.  Gripper (joint 6) excluded — does not affect gripper_link.

    Returns
    -------
    R_gripper2base : np.ndarray, shape (3, 3)
    t_gripper2base : np.ndarray, shape (3,)

    Raises
    ------
    ValueError  if joint_angles_rad does not have exactly 5 elements.
    """
    q = np.asarray(joint_angles_rad, dtype=np.float64).flatten()
    if q.shape[0] != 5:
        raise ValueError(
            f"Expected 5 joint angles (shoulder_pan … wrist_roll), got {q.shape[0]}."
        )

    logger.debug("FK input (rad): %s", np.round(q, 6).tolist())

    T = np.eye(4, dtype=np.float64)
    for i, (name, xyz, rpy) in enumerate(_JOINT_CHAIN):
        T_fixed = _make_T(xyz, rpy)
        T_rot   = _rotz(q[i])
        T       = T @ T_fixed @ T_rot
        logger.debug(
            "  after %-14s  pos=[%7.4f %7.4f %7.4f]",
            name, T[0, 3], T[1, 3], T[2, 3],
        )

    R = T[:3, :3].copy()
    t = T[:3,  3].copy()

    det      = np.linalg.det(R)
    orth_err = np.max(np.abs(R @ R.T - np.eye(3)))
    logger.debug(
        "FK result  pos=[%.4f %.4f %.4f]  det(R)=%.8f  orth_err=%.2e",
        t[0], t[1], t[2], det, orth_err,
    )

    return R, t


def degrees_to_radians(joint_degrees) -> np.ndarray:
    """
    Convert lerobot normalized degree values to radians.
    joint_degrees : array-like, length 5 (shoulder_pan … wrist_roll).
    """
    return np.radians(np.asarray(joint_degrees, dtype=np.float64).flatten())


def raw_ticks_to_degrees(raw_ticks, calib_params):

    result = {}

    for name, raw in raw_ticks.items():

        if name not in calib_params:

            continue

        mid = (calib_params[name]["range_min"] + calib_params[name]["range_max"]) / 2.0

        result[name] = (raw - mid) * 360.0 / 4096.0

    logger.debug("raw_ticks → degrees: %s", {k: f"{v:.3f}" for k, v in result.items()})

    return result
 


def load_calibration_params(calib_json_path):
    with open(calib_json_path) as f:
        data = json.load(f)
    return {
        joint: {
            "range_min": data[joint]["range_min"],
            "range_max": data[joint]["range_max"],
        }
        for joint in data
    }


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_rotation(R: np.ndarray, label: str = "") -> bool:
    """
    Check R is a proper rotation matrix.
    Logs WARNING if checks fail, DEBUG with numbers either way.
    Returns True if valid.
    """
    det      = np.linalg.det(R)
    orth_err = np.max(np.abs(R @ R.T - np.eye(3)))
    logger.debug(
        "Rotation check [%s]: det=%.8f  max|R·Rᵀ-I|=%.2e", label, det, orth_err,
    )
    ok = True
    if abs(det - 1.0) > 1e-6:
        logger.warning("Rotation check [%s]: det(R)=%.8f — expected 1.0", label, det)
        ok = False
    if orth_err > 1e-6:
        logger.warning(
            "Rotation check [%s]: max|R·Rᵀ-I|=%.2e — expected ~0", label, orth_err,
        )
        ok = False
    return ok


# ── Standalone validation entry point ─────────────────────────────────────────

def _run_validation(calib_json_path: str | None = None) -> None:
    """
    Run from project root:  python calibration/so101_fk.py

    Tests:
      1. Zero config    — gripper_link pos expected ≈ [0.293, 0.000, 0.234] m.
      2. Rotation validity across 7 representative configs.
      3. Orthonormality stress test across 50 random configs.
      4. Loads calibration JSON (if present) and prints homing offsets.
    """
    logger.info("=" * 60)
    logger.info("SO-101 FK — standalone validation")
    logger.info("=" * 60)

    # ── Test 1: zero config ───────────────────────────────────────
    logger.info("")
    logger.info("[1] Zero config (all joints = 0 rad)")
    R0, t0 = compute_gripper2base([0, 0, 0, 0, 0])
    logger.info("    gripper_link pos : %s m", t0.round(4).tolist())
    logger.info("    expected approx  : [0.293, 0.0, 0.234] m")
    _validate_rotation(R0, "zero-config")

    # ── Test 2: representative configs ───────────────────────────
    logger.info("")
    logger.info("[2] Rotation validity at representative configs")
    test_configs = [
        ("all zeros",          [0, 0, 0, 0, 0]),
        ("pan  +45 deg",       [np.radians( 45), 0, 0, 0, 0]),
        ("pan  -45 deg",       [np.radians(-45), 0, 0, 0, 0]),
        ("lift +60 deg",       [0, np.radians(60), 0, 0, 0]),
        ("elbow+45 lift+30",   [0, np.radians(30), np.radians(45), 0, 0]),
        ("wrist_roll +90 deg", [0, 0, 0, 0, np.radians(90)]),
        ("realistic pick",     [np.radians(10), np.radians(50),
                                np.radians(-40), np.radians(20), np.radians(30)]),
    ]
    all_ok = True
    for label, angles in test_configs:
        R, t = compute_gripper2base(angles)
        ok   = _validate_rotation(R, label)
        all_ok = all_ok and ok
        status = "OK  " if ok else "FAIL"
        logger.info("    [%s] %-28s  pos=%s", status, label, t.round(3).tolist())

    # ── Test 3: random stress test ────────────────────────────────
    logger.info("")
    logger.info("[3] Orthonormality stress test — 50 random configs")
    rng = np.random.default_rng(42)
    worst_det  = 0.0
    worst_orth = 0.0
    for _ in range(50):
        angles = rng.uniform(-np.pi / 2, np.pi / 2, 5)
        R, _   = compute_gripper2base(angles)
        worst_det  = max(worst_det,  abs(np.linalg.det(R) - 1.0))
        worst_orth = max(worst_orth, np.max(np.abs(R @ R.T - np.eye(3))))
    logger.info("    Worst det error  : %.2e  (threshold 1e-10)", worst_det)
    logger.info("    Worst orth error : %.2e  (threshold 1e-10)", worst_orth)
    stress_ok = worst_det < 1e-10 and worst_orth < 1e-10
    logger.info("    Stress test      : %s", "PASSED" if stress_ok else "FAILED")
    all_ok = all_ok and stress_ok

    # ── Test 4: URDF nominal camera offset ───────────────────────
    logger.info("")
    logger.info("[4] URDF-nominal camera offset from gripper_link")
    logger.info("    T_cam_in_gripper xyz: %s m", _NOMINAL_T_CAM_IN_GRIPPER_XYZ.tolist())
    logger.info("    Hand-eye result should match within ~3 cm.")

    # ── Test 5: calibration JSON ──────────────────────────────────
    default_calib = (
        pathlib.Path.home()
        / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json"
    )
    path = pathlib.Path(calib_json_path) if calib_json_path else default_calib
    logger.info("")
    if path.exists():
        logger.info("[5] Calibration JSON: %s", path)
        offsets = load_homing_offsets(path)
        for joint, val in offsets.items():
            logger.info("    %-16s homing_offset = %d", joint, val)
    else:
        logger.info("[5] Calibration JSON not found at %s — skipping.", path)
        logger.info("    Normal if robot not connected.")

    logger.info("")
    logger.info("=" * 60)
    logger.info("FK validation: %s", "PASSED" if all_ok else "FAILED — see warnings above")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    calib_path = sys.argv[1] if len(sys.argv) > 1 else None
    _run_validation(calib_path)