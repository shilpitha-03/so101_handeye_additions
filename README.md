# SO-101 Wrist Camera (Eye-in-Hand) Calibration

Find the pose of a camera mounted on the SO-101 follower's gripper, relative to the gripper, using a ChArUco board and `cv2.calibrateHandEye`.

**Output:** `outputs/T_gripper_camera.npy`, a 4×4 transform of the camera in the `gripper_link` frame.

This folder is `so101_handeye_scripts/` from [fireloop-ai/camera-calibration](https://github.com/fireloop-ai/camera-calibration), with bug fixes and extra tools (listed below).

---

## Contents

1. [What changed vs. upstream](#what-changed-vs-upstream)
2. [Files](#files)
3. [Requirements](#requirements)
4. [Step 0: Set your paths](#step-0-set-your-paths-every-new-terminal)
5. [Step 1: Install](#step-1-install-once)
6. [Step 2: Find ports, calibration files, and camera](#step-2-find-ports-calibration-files-and-camera)
7. [Step 3: Configure the arms](#step-3-configure-the-arms)
8. [Step 4: Identify your ChArUco board](#step-4-identify-your-charuco-board)
9. [Step 5: Measure the printed board](#step-5-measure-the-printed-board)
10. [Step 6: Camera intrinsics](#step-6-camera-intrinsics)
11. [Step 7: Test board pose detection](#step-7-test-board-pose-detection)
12. [Step 8: Test the follower connection](#step-8-test-the-follower-connection)
13. [Step 9: Collect hand-eye samples](#step-9-collect-hand-eye-samples)
14. [Step 10: Solve](#step-10-solve)
15. [Step 11: Validate](#step-11-validate)
16. [Step 12: Live check](#step-12-live-check)
17. [Using the result](#using-the-result)
18. [When to recalibrate](#when-to-recalibrate)
19. [Troubleshooting](#troubleshooting)

---

## What changed vs. upstream

| Issue in upstream | Effect | Fix |
|---|---|---|
| FK used the `"gripper"` frame, which is the **moving jaw's** joint frame | Calibration silently depended on how open the jaw was | FK now uses `gripper_link`, rigid with the camera |
| Jaw value (0–100) fed into FK as degrees | Corrupted FK | Jaw excluded from FK |
| Follower connected with no `id` | LeRobot found no calibration file and would launch an **interactive recalibration of your arm** | Requires `ROBOT_ID` and `CALIB_DIR`, connects with `calibrate=False`, and errors out instead of recalibrating |
| No support for boards with marker IDs not starting at 0, or the legacy ChArUco layout | Many printed boards weren't detected | `FIRST_MARKER_ID` and `LEGACY_PATTERN` in `config.py` |
| No intrinsics script in this folder | Hand-eye would use someone else's lens parameters | `calib_intrinsics.py` |
| Hand-moving the arm with torque off | Arm sags between image and joint read, adding error | `01_capture_teleop.py` drives the follower from the leader, with torque on and a stillness check |

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Board definition, file paths. **Edit this.** |
| `common.py` | ChArUco detection, pose estimation, camera helpers |
| `so101_pose_provider.py` | Follower connection and FK to `gripper_link`. **Edit the 3 lines at the top.** |
| `detect_board.py` | Works out your printed board's layout settings |
| `calib_intrinsics.py` | Camera intrinsic calibration (writes `camera_matrix.npy`, `dist_coeffs.npy`) |
| `check_undistort.py` | Visual check of the intrinsics |
| `00_test_charuco_pose.py` | Live board-pose test |
| `01_capture_teleop.py` | **Recommended:** sample capture while teleoperating with the leader arm. **Edit the 3 lines at the top.** |
| `01_capture_handeye_samples.py` | Alternative: capture by moving the follower by hand (torque off) |
| `02_run_handeye_calibration.py` | Runs all 5 OpenCV hand-eye methods and saves the best |
| `03_validate_handeye.py` | Consistency check of the result |
| `04_use_calibration_example.py` | Live board pose in the base frame (hand-move mode) |
| `../calibration/so101_updated.urdf` | SO-101 URDF (`so101_new_calib` convention, matching LeRobot calibration) |

All scripts use paths relative to this folder, so **always run them from inside `so101_handeye_scripts/`.** Step 0 handles this.

---

## Requirements

- Linux (cameras are opened with V4L2)
- SO-101 follower arm with a camera rigidly mounted on the **fixed part of the gripper** (not the moving jaw), already calibrated with LeRobot
- SO-101 leader arm (optional, recommended), already calibrated with LeRobot
- A printed ChArUco board, flat and rigid
- conda (Miniforge or Miniconda), Python 3.12

---

## Step 0: Set your paths (every new terminal)

Replace the path with wherever you cloned the repo:

```bash
conda activate so101-handeye
export S=/path/to/camera-calibration/so101_handeye_scripts
cd "$S"
```

Every command below assumes you are in `$S`.

---

## Step 1: Install (once)

```bash
conda create -y -n so101-handeye python=3.12
conda activate so101-handeye
cd "$S"

pip install "lerobot[feetech]==0.6.1" pin pillow
# LeRobot installs opencv-python-headless, which conflicts with the contrib build and has no GUI windows:
pip uninstall -y opencv-python-headless opencv-python opencv-contrib-python
pip install --no-deps "opencv-contrib-python>=4.9,<4.14"
```

Verify:

```bash
python -c "import cv2; print(cv2.__version__, hasattr(cv2.aruco,'interpolateCornersCharuco')); cv2.namedWindow('t'); cv2.destroyAllWindows()"
```

This should print a version and `True` without crashing.

`QFontDatabase: Cannot find font directory` warnings are harmless. To silence them:

```bash
sudo apt install -y fonts-dejavu-core
ln -s /usr/share/fonts/truetype/dejavu "$(python -c 'import cv2,os;print(os.path.dirname(cv2.__file__))')/qt/fonts"
```

---

## Step 2: Find ports, calibration files, and camera

**Arm ports:**

```bash
lerobot-find-port                 # run once per arm; unplug/replug when asked
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1    # use your ports
```

**LeRobot calibration files:**

```bash
ls ~/.cache/huggingface/lerobot/calibration/robots/*/          # follower
ls ~/.cache/huggingface/lerobot/calibration/teleoperators/*/   # leader
```

Note the folder (e.g. `so_follower`, or `so101_follower` from older LeRobot versions) and the filename without `.json`. That filename is the arm's id.

**Wrist camera:**

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
```

Each camera lists several device files. Use the **first `/dev/videoN`** under the wrist camera's name:

- The first `/dev/videoN` is the video stream. Use this one.
- The second `/dev/videoN` is the metadata stream, which produces no images.
- `/dev/mediaN` is a device topology description, not used here.

If you're unsure which camera is the wrist camera, unplug it, rerun the command, and see which entry disappears. Alternatively, `lerobot-find-cameras opencv` saves a test image per camera.

The examples below use `/dev/video4`. **Replace it with yours everywhere.**

**Autofocus:** if the camera has it, turn it off for the whole procedure:

```bash
v4l2-ctl -d /dev/video4 -l | grep -i focus
v4l2-ctl -d /dev/video4 -c focus_automatic_continuous=0     # only if that control exists
```

---

## Step 3: Configure the arms

**Follower:** edit the top of `so101_pose_provider.py`:

```python
PORT = "/dev/ttyACM0"
ROBOT_ID = "<your_follower_id>"
CALIB_DIR = pathlib.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
```

**Leader** (only if using teleop capture): edit the top of `01_capture_teleop.py`:

```python
LEADER_PORT = "/dev/ttyACM1"
LEADER_ID = "<your_leader_id>"
LEADER_CALIB_DIR = pathlib.Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader"
```

Use `so101_follower` / `so101_leader` in the paths if that's what Step 2 showed.

> Close `lerobot-teleoperate` and any other program using the arms or camera before running these scripts. Each device can only be used by one program at a time.

---

## Step 4: Identify your ChArUco board

A board's label usually doesn't say which count is columns vs. rows, whether it uses the legacy layout, or which marker ID it starts at. Find out from a photo:

```bash
python detect_board.py --camera /dev/video4 --squares 7,6 --dict DICT_4X4_50
```

- `--squares`: the two square counts printed on your board, in either order.
- `--dict`: your board's dictionary. Any `DICT_4X4_*` works for 4×4 boards with fewer than 50 markers.

Show the **whole board**, flat and well lit, and press SPACE. Example output:

```
Detected 21 markers, IDs: [1, 2, ..., 21]
...
USE: SQUARES_X = 7, SQUARES_Y = 6, LEGACY_PATTERN = True, FIRST_MARKER_ID = 1
```

Copy those four values plus your dictionary into `config.py`:

```python
SQUARES_X = 7
SQUARES_Y = 6
ARUCO_DICT_NAME = "DICT_4X4_50"
LEGACY_PATTERN = True
FIRST_MARKER_ID = 1
```

If it prints `Nothing matched`, show the whole board, check `--squares` and `--dict`, and retry.

The board's rotation on the table doesn't matter later. Marker IDs fix its orientation.

---

## Step 5: Measure the printed board

Printers often scale slightly. Measure across 5 squares with calipers or a ruler and divide by 5. In `config.py`, set (in meters):

```python
SQUARE_LENGTH_M = 0.025      # your measured square size
MARKER_LENGTH_M = 0.018      # measured square × (nominal marker / nominal square)
```

Example: a nominal 25 mm / 18 mm board that measures 24.8 mm gives `SQUARE_LENGTH_M = 0.0248` and `MARKER_LENGTH_M = 0.0248 * 18/25 = 0.017856`.

Mount the board on something flat and rigid.

---

## Step 6: Camera intrinsics

This measures focal length, image center, and lens distortion. Everything later depends on it.

```bash
python calib_intrinsics.py --camera /dev/video4
```

Keep the arm still and move the **board** in front of the camera by hand. Click the camera window, then:

- Press `s` to save a view. This only works when the overlay text is green. Hold still before each save.
- Collect **20–25 varied views**:
  - board in the center, **each image corner**, and along each edge (the corners matter most)
  - near and far
  - tilted 20–45° left, right, up and down
  - rotated in-plane 0°, 45° and 90°
- Press `c` to calibrate and save. `q` quits without saving.

Example output:

```
RMS reprojection error: 0.219 px
K = [[571.5, 0, 295.0], [0, 570.1, 288.2], [0, 0, 1]]
image size = (640, 480)
Saved camera_matrix.npy dist_coeffs.npy
```

How to judge it:

| RMS | Meaning |
|---|---|
| < 0.5 px | Good |
| 0.5–1.0 px | Acceptable |
| > 1.0 px | Redo it: motion blur, a bent board, or too few tilted and corner views |

The two focal lengths (`K[0,0]` and `K[1,1]`) should agree within a few percent.

**Visual check:** wide-angle lenses need this, because a low RMS doesn't prove the edges are modeled well.

```bash
python check_undistort.py --camera /dev/video4
```

Hold straight edges near the image **edges and corners**. They should look straight in the right-hand (undistorted) view. If they still bow, redo Step 6 with more corner and edge views. Curved black borders at the outer edge are normal.

---

## Step 7: Test board pose detection

```bash
python 00_test_charuco_pose.py --camera /dev/video4
```

- You should see green `OK: N corners` and axes drawn on a board corner that stay stable as you move the board.
- Press `s` to print the pose. At about 30 cm from the camera, the translation z should read about `0.30`. If it's far off, recheck Step 5.
- Press `q` to quit.

---

## Step 8: Test the follower connection

**Support the follower by hand first.** This turns its torque off, and it will sag.

```bash
python so101_pose_provider.py
```

It prints 5 joint angles in degrees (pan, lift, elbow, wrist_flex, wrist_roll) and the 4×4 `T_base_gripper`. Move **one** joint, rerun, and confirm only that number changes. Do this for all 5 joints.

---

## Step 9: Collect hand-eye samples

**Board setup:**

- Tape the board flat on the table, all four edges, about 20–30 cm in front of the follower's base.
- **It must not move until Step 12 is done.** Don't bump the table or the robot base.

### Option A: Teleop (recommended)

Put the **leader roughly in the follower's current pose** first. When the script connects, the follower moves to match the leader, as in normal teleop.

```bash
python 01_capture_teleop.py --camera /dev/video4
```

For each sample:

1. Move the follower with the leader so the camera sees most of the board, with green `OK: N corners`, ideally 20 or more.
2. **Hold the leader still**, click the camera window, and press `s`.
3. Keep holding still until the terminal prints `Saved sample N`, about 1 s. Keys pressed during that second are ignored.
4. If it prints `Arm still moving`, hold steadier and press `s` again.

To finish, bring the follower to its **rest pose** with the leader, then press `q`. The follower's torque turns off when the script exits. Samples are saved on `q`, and also on Ctrl+C.

### Option B: By hand (no leader)

```bash
python 01_capture_handeye_samples.py --camera /dev/video4
```

Hold the arm still for about 1 s before each `s`, keeping your hand out of the camera's view. Finish with `q`. **Ctrl+C loses all samples** in this script.

### Which poses to collect (both options)

Collect about 25 samples. **Rotation variety is what makes the solve work.**

- Wrist roll spread across roughly −60° to +60°
- Views from the left, right, front, and above, tilted 20–40°. Don't take every view looking straight down.
- Distance varied from about 15 to 40 cm
- No near-duplicate poses
- Rule of thumb: about 5 positions around the board × 5 different wrist roll and tilt combinations each

Back up the samples:

```bash
cp data/handeye_samples.npz data/handeye_samples_backup_$(date +%Y%m%d_%H%M).npz
```

---

## Step 10: Solve

```bash
python 02_run_handeye_calibration.py
```

This runs TSAI, PARK, HORAUD, ANDREFF and DANIILIDIS, prints each result with its board-position spread, and saves the most consistent one to `outputs/T_gripper_camera.npy`.

PARK, HORAUD and TSAI should agree within a few millimeters. ANDREFF is often worse, and that's normal. If **all** methods are bad, the problem is in the data (see Troubleshooting).

---

## Step 11: Validate

```bash
python 03_validate_handeye.py
```

The board was fixed, so its computed position in the base frame should be the same for every sample.

| Mean position error | Meaning |
|---|---|
| < ~5 mm | Good for an SO-101 |
| 5–15 mm | Usable for most tasks |
| > 20 mm | Something is wrong. Don't use the result |

The orientation spread should be a few degrees at most.

Physical sanity check of the camera's offset from the wrist:

```bash
python -c "import numpy as np; T=np.load('outputs/T_gripper_camera.npy'); np.set_printoptions(suppress=True,precision=4); print(T); print('offset from gripper_link: %.1f mm' % (np.linalg.norm(T[:3,3])*1000))"
```

The offset should match where the camera physically sits, typically a few centimeters.

---

## Step 12: Live check

Leave the board where it was.

**With teleop.** Use a separate `--out` so this run can never overwrite your samples:

```bash
python 01_capture_teleop.py --camera /dev/video4 --out data/live_check.npz
```

Move to 4–5 very different poses that see the board. At each one, hold still and press **`p`** (not `s`). Finish at the rest pose and press `q`.

**By hand:**

```bash
python 04_use_calibration_example.py --camera /dev/video4
```

Press `p` at each pose.

**Pass:** the printed `T_base_charuco` translation (last column, in meters) stays the same within a few millimeters across all poses.

Back up the result:

```bash
cp outputs/T_gripper_camera.npy ~/T_gripper_camera_$(date +%Y%m%d).npy
```

---

## Using the result

```python
import numpy as np
import so101_pose_provider as P        # connects to the follower (torque off) on import

T_gripper_camera = np.load("outputs/T_gripper_camera.npy")
T_base_gripper = P.get_T_base_gripper()             # read at the moment the image is taken
T_base_object = T_base_gripper @ T_gripper_camera @ T_camera_object   # T_camera_object from your detector
```

- **Units:** meters.
- **Frames:** `base_link` → `gripper_link` → camera (OpenCV convention: x right, y down, z forward).

---

## When to recalibrate

| Change | Redo |
|---|---|
| Camera mount bumped, loosened or moved | Steps 9–12 |
| Arm recalibrated in LeRobot (joint zeros change) | Steps 8–12 |
| Camera resolution, focus or lens changed | Steps 6–12 |
| Different board | Steps 4–5 (and 6 if you want), then 9–12 |

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `FileNotFoundError: No calibration file` | Wrong `ROBOT_ID`/`CALIB_DIR` (or leader equivalents). Recheck Step 2 |
| `Permission denied` on `/dev/ttyACM*` | `sudo chmod 666 /dev/ttyACM*`, or add your user to the `dialout` group |
| Port or camera busy, or timeouts | Another program is using it. Close `lerobot-teleoperate`, other scripts, and camera apps |
| `No markers detected` | Wrong `/dev/videoN` (use the first one), bad lighting, blur, or wrong dictionary |
| Board detected but `Charuco interpolation failed` / 0 corners | Wrong `SQUARES_X`/`SQUARES_Y`/`LEGACY_PATTERN`/`FIRST_MARKER_ID`. Rerun Step 4 |
| Distances wrong by a constant % | `SQUARE_LENGTH_M`/`MARKER_LENGTH_M` don't match the print. Redo Step 5 |
| Intrinsics RMS > 1 px | Blur, a non-flat board, or not enough corner and tilted views |
| Straight lines bow after undistortion | Not enough views near the image edges and corners. Redo Step 6 |
| Hand-eye error > 20 mm | Board moved, arm moving during capture, too little rotation variety, or wrong board size. Recollect Step 9 |
| A few samples much worse than others | Movement during those captures. Recollect |
| Error still large after a clean recollection | Joint zeros may not match the URDF. The LeRobot calibration must be done in the standard middle-of-range pose. Recalibrate the arm with LeRobot, then redo Steps 8–12 |
| Follower jumps when teleop capture starts | Expected. It moves to the leader's pose. Match the leader to the follower before starting |
| Arm drops when a script exits | Expected. Torque turns off. Rest the arm first |
| `QFontDatabase` warnings | Harmless. See Step 1 to silence them |
