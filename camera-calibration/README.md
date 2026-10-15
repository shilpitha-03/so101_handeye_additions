# Real-to-Sim Camera Calibration Pipeline

Real-to-sim camera calibration for an SO-101 arm: calibrates side-view and wrist cameras, computes camera poses in the robot base frame, and converts them for Isaac Sim.

**Project:** SO-101 tabletop pick-and-place — VLA data generation
**Sim:** Isaac Sim 5.x | **Robot:** SO-101 (lerobot USD) | **Cameras:** Side-view (Android IP) + Wrist (USB)
**OpenCV:** `opencv-contrib-python >= 4.13` | **Python:** `>= 3.12`

**Related blog posts:**
- [ChArUco Camera Calibration — the calibration step that makes or breaks your sim-to-real pipeline](https://www.blog.fireloop.ai/the-calibration-step-that-makes-or-breaks-your-sim-to-real-pipeline/)
- [Hand-Eye Calibration — connecting robot vision to robot motion](https://www.blog.fireloop.ai/hand-eye-calibration-connecting-robot-vision-to-robot-motion/)
- Extrinsic Parameters — coming soon

---

## Overview

Full pipeline to calibrate real-world cameras, place them accurately in Isaac Sim, and measure the 3D position of fixed objects (rack) relative to the robot base.

World origin = robot base `(0, 0, 0)`. Everything is measured relative to it.

**Pipeline flow:**

```
Step 1   — Calib 1      : side-view camera intrinsics
Step 1.1 — Extract      : derive Isaac Sim sensor parameters from intrinsics
Step 1.2 — Convert      : verify parameters, print Isaac Sim GUI cheatsheet
Step 2   — Calib 2      : wrist camera intrinsics
Step 3   — Calib 3      : wrist camera hand-eye calibration  ← FK placeholder, partial
           ↓
Step 4   — Phase 1      : side-view pose capture (ChArUco board)
Step 5   — Phase 2      : transform chain → camera pose in robot base frame
Step 6   — Phase 3      : axis conversion → Isaac Sim position + quaternion
           ↓
Enter extrinsic values (Phase 3) + intrinsic values (Step 1.2) in Isaac Sim UI
Run isaacsim_script.py inside Isaac Sim Script Editor to apply distortion coefficients
           ↓
Step 7   — Rack localization → rack corner in robot base frame (two methods)
```

**What needs re-running after a change:**

If nothing has changed (camera, setup, robot position), only Steps 4–7 need re-running.
If the camera is replaced or moved, re-run from Step 1 accordingly.
Steps 1.1 and 1.2 only need re-running if the camera or calibration changes — the sensor parameters are fixed once the camera is fixed.

---

## Folder Structure

```
project/
├── pyproject.toml / uv.lock                  # dependencies
├── main.py                                   # orchestrator for Steps 1–6
├── config.py                                 # all constants — edit here only
├── rack_localization.py                      # Step 7 — standalone script
├── calibration/
│   ├── __init__.py
│   ├── calib1_intrinsics_sideview.py         # side-view intrinsics
│   ├── calib2_intrinsics_wrist.py            # wrist intrinsics
│   └── calib3_handeye_wrist.py               # wrist hand-eye (FK pending)
├── extraction/                               # camera sensor setup for Isaac Sim
│   ├── extract_calibration.py                # Step 1.1 — derive sensor params, save JSON
│   ├── convert_verify.py                     # Step 1.2 — verify params, print GUI cheatsheet
│   └── isaacsim_script.py                    # run inside Isaac Sim Script Editor only
├── phases/
│   ├── __init__.py
│   ├── phase1_capture.py                     # pose capture from side-view
│   ├── phase2_transform.py                   # transform chain
│   └── phase3_conversion.py                  # Isaac Sim axis conversion
├── so101_handeye_scripts/                    # standalone SO-101 hand-eye scripts (own README)
├── calibration_files/                        # example calibration outputs — replace with your own
│   ├── side_view/
│   │   ├── camera_matrix.npy
│   │   ├── dist_coeffs.npy
│   │   └── camera_params.json                # output of extract_calibration.py
│   └── wrist/
│       ├── camera_matrix.npy
│       ├── dist_coeffs.npy
│       ├── T_cam2gripper.npy                 # hand-eye result
│       ├── handeye_R_board2cam.npy
│       ├── handeye_t_board2cam.npy
│       ├── handeye_R_gripper2base.npy        # saved when FK available
│       └── handeye_t_gripper2base.npy
└── results/                                  # generated at runtime, git-ignored
    └── session_YYYYMMDD_HHMMSS/              # one folder per pipeline run
        ├── p1_rvecs_raw.npy                  # shape (N, 3)
        ├── p1_tvecs_raw.npy                  # shape (N, 3)
        ├── p2_T_board_in_cam.npy
        ├── p2_T_board_in_cam_corrected.npy
        ├── p2_T_cam_in_board.npy
        ├── p2_T_cam_in_robot.npy
        ├── p3_isaac_position.npy
        ├── p3_isaac_quaternion.npy
        ├── rack_corner_marker.npy            # Method A result
        ├── rack_corner_click.npy             # Method B result
        ├── rack_corner_robot_frame.npy       # consensus
        └── rack_corner_robot_frame.txt       # human-readable summary
```

---

## Physical Setup Requirements

Verify before every run:

| Item | Requirement |
|------|-------------|
| Robot base | Bolted down, never moved |
| Fixed ChArUco board | Flat on table surface, fixed, never moved |
| Board ID 0 corner | 13.5 cm along robot +X, Y offset = 0, Z offset = 0 |
| Side-view camera | Locked in final position, never moved |
| Wrist camera | Attached to gripper |
| Calibration board | Separate print — same specs as fixed board |
| Rack | Fixed on table surface, never moved |
| Rack marker | ID 4, DICT_5X5_100, upright on front face of rack corner leg |

**Coordinate convention (robot base = world origin):**

```
+X → forward (direction arm naturally reaches)
+Y → left of robot
+Z → up
```

---

## Dependencies

Requires Python 3.12+. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

```bash
# Recommended: uv
uv sync
source .venv/bin/activate

# Or with pip
pip install "lerobot[feetech]>=0.5.1" "opencv-contrib-python>=4.13.0.92" "scipy>=1.17.1"
```

---

## Configuration — `config.py`

All parameters live in `config.py`. Edit only this file — never hardcode values elsewhere.

| Key | Description |
|-----|-------------|
| `SIDE_CAMERA_ID` | Android IP stream URL — **set this to your own camera stream** (placeholder `"#"` in the repo) |
| `FOLLOWER_PORT` | Serial port of the SO-101 follower arm (e.g. `/dev/ttyACM0`) — set to your own |
| `WRIST_CAMERA_ID` | USB camera index (default: 0) |
| `SQUARES_X / Y` | Board dimensions: 8 × 11 (columns × rows) |
| `SQUARE_LENGTH` | 30.01 mm |
| `MARKER_LENGTH` | 22.01 mm |
| `ARUCO_DICT` | `DICT_5X5_100` |
| `CALIB_MIN_IMAGES` | Minimum captures for intrinsics (default: 20) |
| `MIN_CHARUCO_CORNERS` | Minimum ChArUco corners per frame (default: 6) — single constant used everywhere |
| `PHASE1_N_FRAMES` | Frames to collect in Phase 1 (default: 15) |
| `HANDEYE_MIN_POSES` | Minimum poses for hand-eye (default: 15) |
| `BOARD_X_FROM_ROBOT` | 0.135 m — tape measured, do not change unless re-measured |
| `BOARD_Y_FROM_ROBOT` | 0.0 m — assumed on robot centerline |
| `BOARD_Z_FROM_ROBOT` | 0.0 m — assumed table surface = robot Z=0 |

---

## Running the Pipeline

**Full pipeline (Steps 1–6):**

```bash
python main.py
```

**Skip steps already done** — comment out blocks in `main.py`. The safe default is all calibration steps commented out, with Phases 1–3 active:

```python
# ── Step 1: Side-view intrinsics ──────────────────────────────────────
# calib1()                   # skip if already calibrated

# ── Step 1.1 / 1.2: Sensor extraction ────────────────────────────────
# calib1_extraction()        # skip if camera_params.json already exists
# convert_verify()           # skip if GUI cheatsheet already printed

# ── Step 2: Wrist intrinsics ──────────────────────────────────────────
# calib2()                   # skip if already calibrated

# ── Step 3: Wrist hand-eye ───────────────────────────────────────────
# calib3()                   # skip until FK is ready

# ── Steps 4–6: Pose pipeline ─────────────────────────────────────────
p1_data = phase1(session_dir)
p2_data = phase2(session_dir, phase1_data=p1_data)
p3_data = phase3(session_dir, phase2_data=p2_data)
```

**Run extraction scripts independently (outside Isaac Sim):**

```bash
# Step 1.1 — extract sensor parameters from calibration files
python extraction/extract_calibration.py
# Reads:  calibration_files/side_view/camera_matrix.npy + dist_coeffs.npy
# Writes: calibration_files/side_view/camera_params.json

# Step 1.2 — verify parameters, print Isaac Sim GUI cheatsheet
python extraction/convert_verify.py
# Reads:  calibration_files/side_view/camera_params.json
# Prints: every value to type into the Isaac Sim Property panel
```

**Apply distortion coefficients inside Isaac Sim (Step 1.3):**

`isaacsim_script.py` must be run inside Isaac Sim's Script Editor — it cannot be called from a regular Python environment.

```
1. Open Isaac Sim and load your scene
2. Create the camera prim via the GUI using the cheatsheet from convert_verify.py
3. Open Window > Script Editor
4. Paste the contents of extraction/isaacsim_script.py
5. Update CAMERA_PRIM_PATH and PARAMS_JSON_PATH at the top of the script
6. Click Run (Ctrl+Enter)
7. Save the scene (Ctrl+S)
```

**Run a single phase on an existing session:**

```python
import pathlib
from phases.phase2_transform import run as phase2

session_dir = pathlib.Path("results/session_20250511_143022")
phase2(session_dir)   # loads p1_*.npy from disk automatically
```

**Rack localization (Step 7):**

```bash
python rack_localization.py                           # both methods, most recent session
python rack_localization.py --method marker           # marker method only
python rack_localization.py --method click            # pixel click method only
python rack_localization.py --session results/session_YYYYMMDD_HHMMSS
```

---

## OpenCV Version Notes — Critical

All scripts are written for **OpenCV 4.6+ new API**. Several breaking changes from older versions affect this pipeline:

**ChArUco board convention change (4.6+):** Corner indexing changed from bottom-left to top-left start. This causes the board Z axis to point **into** the board plane (toward the table) instead of out of it. Phase 2 corrects for this with a right-multiplication by `diag([1,-1,-1,1])` on the full 4×4 `T_board_in_cam` matrix. Do NOT call `board.setLegacyPattern(True)` — this would conflict with the Phase 2 correction.

**Deprecated calibration API:** `cv2.aruco.calibrateCameraCharuco()` and `cv2.aruco.estimatePoseCharucoBoard()` do not exist in OpenCV 4.7+. All scripts use the correct modern API:
- Calibration: `board.matchImagePoints()` + `cv2.calibrateCamera()`
- Pose estimation: `board.matchImagePoints()` + `cv2.solvePnPGeneric(SOLVEPNP_IPPE)`
- Single marker: `cv2.solvePnPGeneric(SOLVEPNP_IPPE_SQUARE)`

**Corner refinement:** `CORNER_REFINE_CONTOUR` has a regression in OpenCV 4.7+ (returns integer corners, no actual refinement). All scripts use `CORNER_REFINE_SUBPIX`.

---

## Calibration Details

### Calib 1 — Side-View Intrinsics

> Background: [ChArUco Camera Calibration](https://www.blog.fireloop.ai/the-calibration-step-that-makes-or-breaks-your-sim-to-real-pipeline/)

- Camera: Android phone IP stream
- Move the **calibration board** (separate print) to varied angles, distances, positions
- Robot arm can be in any position — irrelevant for this step
- Target: reprojection error < 1.0 px, ideally < 0.5 px
- Output: `calibration_files/side_view/camera_matrix.npy`, `dist_coeffs.npy`

**Tips for good coverage:**
- Fill all regions of the frame — corners, edges, center
- Include tilted, rotated, close, and far poses
- Avoid capturing from the same angle twice
- Minimum 20 frames, more is better

### Calib 2 — Wrist Camera Intrinsics

- Camera: USB camera (index 1)
- Keep robot arm stationary — move only the calibration board
- Same coverage tips as Calib 1
- Target: reprojection error < 1.0 px
- Output: `calibration_files/wrist/camera_matrix.npy`, `dist_coeffs.npy`

### Calib 3 — Wrist Hand-Eye Calibration

> Background: [Hand-Eye Calibration](https://www.blog.fireloop.ai/hand-eye-calibration-connecting-robot-vision-to-robot-motion/)

Solves AX = XB → finds fixed transform `T_cam2gripper` (camera relative to gripper tool frame).

**Status: partially implemented — FK placeholder pending SO-101 URDF integration.**

- Camera: wrist USB camera (fixed on gripper)
- Move robot to varied poses using leader arm → hold calibration board visible → SPACE to capture
- Repeat 15–20 times with varied arm orientations
- Each pose must include > 30° rotation change **across varied axes** — poses rotating around only one joint will fail even if angular change is > 30°
- Method used: `CALIB_HAND_EYE_PARK` — robust for varied pose sets

**What to implement when FK is ready:**
- Fill in `get_gripper2base()` in `calib3_handeye_wrist.py`
- It must return `(R_gripper2base [3×3], t_gripper2base [3,])` from current joint angles
- Re-run `calib3` independently — board poses already saved from previous runs

**Output:** `calibration_files/wrist/T_cam2gripper.npy`

---

## Phase Details

### Phase 1 — Pose Capture (Side-View)

- Opens side-view camera stream with calibrated intrinsics loaded
- `CharucoDetector` uses `K` and `D` for reprojection-based corner interpolation (accurate), not homography
- Shows live ChArUco detection with coordinate axes overlay
- SPACE to capture | Q to quit
- Needs ≥ `MIN_CHARUCO_CORNERS` (6) corners per frame, collects `PHASE1_N_FRAMES` (15) frames
- Pose estimated with `SOLVEPNP_IPPE` — correct method for coplanar boards. Returns 2 solutions; best selected by reprojection error
- Saved: `p1_rvecs_raw.npy`, `p1_tvecs_raw.npy` — shape `(N, 3)` float64

**What to watch during capture:**
- With the new OpenCV 4.6+ API (no `setLegacyPattern`), the Z axis (blue) will point **into the board** (downward toward the table) — this is correct and expected
- Phase 2 corrects this axis direction — do not be alarmed by the Z direction in Phase 1
- The board must be flat and not moving between captures
- Check that `tvec` values are consistent across frames (Phase 1 prints std — target < 0.005 m)

**Note on saved data:** Arrays are saved as `(N, 3)`. Phase 2 reshapes to `(3, 1)` before passing to `cv2.Rodrigues`.

### Phase 2 — Transform Chain

Reads Phase 1 output. Produces camera pose in robot base frame.

**Rotation averaging:** Uses chordal mean (correct on SO(3) manifold), not naive arithmetic mean of Rodrigues vectors. Naive averaging of rotation vectors is mathematically wrong.

**Transform chain:**

```
T_board_in_cam  (raw OpenCV 4.6+ output — Z into board)
    → right-multiply by diag([1,-1,-1,1])   ← corrects board frame Y and Z axes
      MUST apply to full 4×4 matrix — both rotation block AND translation transform
    → invert → T_cam_in_board  (corrected board frame)
    → compose with T_board_in_robot (0.135, 0, 0) identity rotation
    → T_cam_in_robot  (robot base frame = Isaac Sim world frame)
```

**Why right-multiply (not left-multiply):**
- Right-multiplying `T_board_in_cam @ C` changes the **input (board) frame** convention — which is what we want to fix
- Left-multiplying `C @ T_board_in_cam` would change the **output (camera) frame** — wrong

**Critical property of `T_cam_in_robot`:**
- Translation column `[:3, 3]` = camera position in robot frame — validated correct
- Rotation block `[:3, :3]` = camera orientation in robot frame
- When transforming a raw `solvePnP` tvec from any other marker to robot frame, apply `T_cam_in_robot` directly: `p_robot = R_c2r @ tvec + t_c2r`. This was validated: board origin tvec gives `[0.135, 0, 0]` correctly.

**Sanity checks (printed automatically):**
- Z must be positive — camera is above table
- X must be > 0.135 m — camera is behind the board
- `det(R) = 1.0` and `max |R·Rᵀ - I| < 1e-10` — rotation matrix orthonormality

### Phase 3 — Isaac Sim Axis Conversion

Converts `T_cam_in_robot` to Isaac Sim 5.x placement values for the **Xform prim UI path**.

**Important distinction — two different paths exist:**

| Path | When used | Translation | Quaternion |
|------|-----------|-------------|------------|
| Xform prim UI (our path) | Placing camera via UI Transform properties | Direct from `T_cam_in_robot[:3,3]` | `[qw, qx, qy, qz]` — no axis remap |
| Camera Python API | `Camera(position=..., orientation=...)` | `[-dZ, dX, dY]` | `[rW, -rZ, rX, rY]` |

The formula `[rW, -rZ, rX, rY]` from Isaac Sim docs is for the Python API path only. It does NOT apply when entering values in the UI.

**Convention chain:**

```
OpenCV camera  : +X right, +Y down,  +Z forward (into scene)
      ↓ post-multiply R by 180° rotation around X axis (R @ R_180x)
        — rotates the camera's LOCAL frame, not the world frame
USD camera     : +X right, +Y up,    -Z forward
      ↓ extract quaternion scalar-first (w, x, y, z) — no axis remap for Xform UI
Isaac Sim Xform: enter directly
```

**Translation:** enter `T_cam_in_robot[:3, 3]` directly. Robot base frame = Isaac Sim world frame (+X forward, +Z up). No remapping needed.

**Quaternion norm** is checked automatically — must be 1.0.

---

## Camera Sensor Setup — `extraction/`

These three scripts set up the **camera intrinsics** (sensor properties) in Isaac Sim to match the physical camera. This is separate from the **extrinsics** (position and orientation) which Phase 3 handles. Both must be done for a correct real-to-sim match.

### Script 1 — `extract_calibration.py` (Step 1.1)

Reads `camera_matrix.npy` and `dist_coeffs.npy` from Calib 1, derives all Isaac Sim sensor parameters, and saves them to `calibration_files/side_view/camera_params.json`.

**Parameters you must configure inside the script:**

| Parameter | Description |
|-----------|-------------|
| `CALIB_PATH` | Folder containing `.npy` files — `calibration_files/side_view/` |
| `CALIB_WIDTH / HEIGHT` | Resolution used during calibration — check your captured images |
| `PHYSICAL_FOCAL_LENGTH_MM` | Real focal length of the lens in mm — read from lens or camera datasheet |
| `F_STOP` | f-number printed on the lens. Set to `0.0` to disable depth-of-field blur (recommended for robotics/perception) |
| `FOCUS_DISTANCE_M` | Approximate working distance camera-to-scene in metres — only matters when `F_STOP > 0` |
| `OUTPUT_JSON` | Output path — default `calibration_files/side_view/camera_params.json` |

**Why pixel size is not asked directly:** Instead of requiring the physical pixel size (hard to find for phone cameras), the script derives it automatically from the known physical focal length and the calibrated `fx` value: `effective_pixel_size_mm = physical_focal_length_mm / fx`. This is more reliable for phone cameras that downscale from a larger sensor.

Run this script on your regular Python environment — not inside Isaac Sim:

```bash
python extraction/extract_calibration.py
```

### Script 2 — `convert_verify.py` (Step 1.2)

Reads `camera_params.json` from Script 1, runs sanity checks, and prints a numbered cheatsheet of every value to enter in the Isaac Sim Property panel.

**Checks performed:**
- Focal length round-trip — recomputes `fx`/`fy` from cm values and confirms they match original calibration
- FOV sanity — compares computed FOV against known hardware spec
- `cx`/`cy` offset — warns if principal point is far from image centre (Isaac Sim cannot model this perfectly)
- Distortion magnitude — flags unusually large `k1`/`k2`/`k3` values that may indicate poor calibration

Run on your regular Python environment:

```bash
python extraction/convert_verify.py
```

### Script 3 — `isaacsim_script.py` (Step 1.3)

Applies OpenCV pinhole distortion coefficients (`k1`, `k2`, `p1`, `p2`, `k3`) to an existing camera prim inside Isaac Sim 5.x via the Script Editor. Also corrects `cx`, `cy`, `fx`, `fy`, and `imageSize` inside the distortion schema — the Python API does not set these correctly by default.

**This script must run inside Isaac Sim's Script Editor.** It cannot be called from a regular Python environment or from `main.py`.

What it does:
- Wraps the existing camera prim with the Camera API
- Calls `set_opencv_pinhole_properties()` to apply the `OmniLensDistortion OpenCvPinholeAPI` schema
- Directly patches `cx`, `cy`, `fx`, `fy`, `imageSize` on the USD prim
- Prints a full verification of every distortion attribute for confirmation

**Important notes:**
- Run this script AFTER creating the camera prim via the GUI and entering all values from Script 2's cheatsheet
- The Fisheye Lens panel in the GUI shows deprecated fields — ignore them. The `OmniLensDistortion` schema takes priority in Isaac Sim 5.x
- Save the scene (Ctrl+S) after running to persist changes
- Tested on Isaac Sim 5.0 and 5.1

---

## Entering Values in Isaac Sim

**Extrinsics (camera position and orientation — from Phase 3):**

1. Check stage units: **Edit → Preferences → Stage → Meters Per Unit = 1.0** (off by 100× if set to cm)
2. Create **Xform prim** named `side_camera_rig` — enter Phase 3 Translate and Orient values
3. Add **Camera prim** as child of that Xform — leave child at local `(0, 0, 0)`
4. Always enter values on the **Xform**, not directly on the Camera prim (camera prim UI shows USD axes, not world axes)
5. Place a box at `(0.135, 0, 0)` to represent the ChArUco board origin — use for visual validation
6. Switch viewport to `side_camera_rig/Camera` — the board box should appear at roughly the same position as in the real image

**Intrinsics (sensor properties — from extraction scripts):**

7. With the camera prim selected in the Stage panel, open the Property panel
8. Enter every value from `convert_verify.py`'s printed cheatsheet into the exact GUI fields listed
9. Run `isaacsim_script.py` in the Script Editor to apply distortion coefficients
10. Save the scene (Ctrl+S)

---

## Rack Localization — `rack_localization.py`

Measures the rack corner position in robot base frame. Standalone script — runs independently after Phase 2.

### Physical Setup

- Cover the fixed ChArUco board completely with white paper (prevents ID conflicts)
- Print ArUco marker ID 4 from `DICT_5X5_100` — measure black square side with calipers
- Mount marker on the **front face** of the rack's nearest bottom-left corner leg (the face pointing toward the camera)
- Orientation: marker must be **upright** — Y axis pointing up (robot +Z direction). Pattern should read normally when viewed face-on from the camera, not sideways
- Bottom edge of marker (including white quiet zone) flush with the bottom of the rack leg foot
- With correct placement, marker corner 3 (bottom-left in OpenCV clockwise ordering) is physically at the rack corner point

### Two Methods

**Method A — ArUco Marker (automated, 30-frame average):**

Detects the marker and uses the `tvec` (marker center in camera frame) averaged over 30 frames. Key design decisions:

- Only `tvec` is used — not corner offsets via rvec. IPPE_SQUARE pose estimation has an ambiguity problem for small markers viewed obliquely: both solutions have similar reprojection error and neither `tvec[2] > 0` filtering nor reprojection error selection reliably picks the correct rotation. The `tvec` is stable across frames (std < 2mm) even when rvec flips between solutions.
- Marker center → rack corner X: subtract half-marker-size in robot X direction (validated: marker +X axis maps exactly to robot +X)
- Rack corner Y = marker center Y (no offset — marker X axis has zero Y component in robot frame)
- Rack corner Z = `RACK_LEG_HEIGHT` from physical measurement — vision Z is unreliable for a small marker at oblique angle

**Method B — Pixel Click (interactive, single frame):**

Freeze a live frame, click the rack corner pixel, cast a ray from the camera through that pixel, intersect with the horizontal plane `Z = RACK_LEG_HEIGHT` in robot frame.

- `cv2.undistortPoints` handles full lens distortion before ray construction
- Ray direction transformed to robot frame using rotation block of `T_cam_in_robot`
- Accuracy: ~1–2mm per pixel at ~1m distance — depends on click precision
- Z comes from physical measurement by construction (it is the plane constraint)

**Cross-validation:** both methods output X, Y, Z. Agreement within 5mm = high confidence. Results are averaged into a consensus value.

### Rack Marker Config

| Key | Value | Notes |
|-----|-------|-------|
| `RACK_MARKER_ID` | 4 | From `DICT_5X5_100` |
| `RACK_MARKER_SIZE` | 0.02168 m | Black square side — measure with calipers |
| `RACK_LEG_HEIGHT` | 0.01145 m | Rack corner height above table — physically measured |

---

## Known Issues and Decisions

| Issue | Cause | Resolution |
|-------|-------|------------|
| Z axis points into board in Phase 1 | OpenCV 4.6+ corner indexing change | Expected and correct — Phase 2 corrects it. Display shows corrected axes for visual confirmation but raw data saved |
| `calibrateCameraCharuco` missing | Removed in OpenCV 4.7+ | Use `board.matchImagePoints()` + `cv2.calibrateCamera()` |
| `estimatePoseCharucoBoard` missing | Removed in OpenCV 4.7+ | Use `board.matchImagePoints()` + `cv2.solvePnPGeneric()` |
| `setLegacyPattern(True)` conflict | Restores pre-4.6 axes, conflicts with Phase 2 correction | Never call `setLegacyPattern` — use new API throughout |
| Naive rvec averaging is wrong | SO(3) is a manifold, not a vector space | Chordal mean via SVD used in Phase 2 and Phase 1 stats |
| diag([1,-1,-1]) must right-multiply | Left-multiply changes wrong frame | `T_board_in_cam @ C` — right-multiply changes board frame |
| diag must apply to full 4×4 | tvec also transforms under frame change | `C = diag([1,-1,-1,1])` applied to full matrix, not just rotation block |
| Isaac Sim quaternion remap | Only for Python API path, not Xform UI | Xform UI: plain `[qw, qx, qy, qz]`, no axis remap |
| Off by 100× in sim | Stage units in cm (0.01) | Edit → Preferences → Stage → Meters Per Unit = 1.0 |
| IPPE ambiguity for single marker | Small marker at oblique angle — both solutions similar reprojection error | Use tvec only (stable), not rvec-derived corner offsets |
| CORNER_REFINE_CONTOUR broken | Regression in OpenCV 4.7+ — returns integer corners | Use CORNER_REFINE_SUBPIX |
| Camera prim UI shows wrong axes | USD camera convention (+Y up, -Z forward) differs from world axes | Always set values on parent Xform, not on Camera prim directly |

---

## What Triggers a Full Re-Run

| Change | Steps to re-run |
|--------|-----------------|
| Side-view camera replaced or moved | Calib 1 → Extract (1.1, 1.2, 1.3) → Phase 1 → Phase 2 → Phase 3 |
| Wrist camera replaced | Calib 2 → Calib 3 |
| Robot re-bolted (position changed) | Re-measure board X distance → update `BOARD_X_FROM_ROBOT` → Phase 1 → Phase 2 → Phase 3 |
| Board moved or reprinted at different size | Re-measure → Phase 1 → Phase 2 → Phase 3 |
| Rack moved | Re-run `rack_localization.py` |
| Everything from scratch | Full pipeline Step 1 → Step 7 including extraction |

---

## Pending / Future Work

| Item | Status | Location |
|------|--------|----------|
| SO-101 forward kinematics | Pending URDF integration | `calib3_handeye_wrist.py` → `get_gripper2base()` |
| Isaac Sim validation | Run after camera placement — render vs real comparison | Manual step |
| Vial slot positions | After rack — derived from rack corner + slot grid dimensions | New script TBD |
| Wrist camera placement | After hand-eye — same Phase 1–3 pipeline adapted for wrist | New session |

---

## Notes

- The files in `calibration_files/` and `my_follower.json` are calibration outputs for one specific hardware setup. They are included as examples — regenerate them for your own cameras and arm.
- Back up `calibration_files/` after every successful calibration run (`camera_matrix.npy`, `dist_coeffs.npy`, and `camera_params.json` for the side-view camera).
- `results/session_*` folders are not committed. `rack_localization.py` needs a session produced by Phases 1–2 (it reads `p2_T_cam_in_robot.npy`), so run `python main.py` first.

## License

MIT — see [LICENSE](LICENSE). The SO-101 URDF in `calibration/` comes from the SO-101 open-source hardware project and remains under its own license.
