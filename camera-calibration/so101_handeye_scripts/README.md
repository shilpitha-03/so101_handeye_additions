# SO-101 Wrist Camera Hand-Eye Calibration Scripts

You already have:

```text
camera_matrix.npy
dist_coeffs.npy
```

Those are camera intrinsics. These scripts produce:

```text
outputs/T_gripper_camera.npy
```

That is the eye-in-hand calibration transform for a camera mounted on the SO-101 wrist.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

Use `opencv-contrib-python`, not only `opencv-python`, because ChArUco/ArUco lives in the contrib package.

## 2. Put your intrinsic files in this folder

Copy these into the same directory as the scripts:

```text
camera_matrix.npy
dist_coeffs.npy
```

## 3. Edit `config.py`

Set your real ChArUco board values:

```python
SQUARES_X = ...
SQUARES_Y = ...
SQUARE_LENGTH_M = ...
MARKER_LENGTH_M = ...
ARUCO_DICT_NAME = ...
CAMERA_INDEX_OR_PATH = ...
```

Use meters for square and marker length.

## 4. Test ChArUco pose detection

```bash
python 00_test_charuco_pose.py --camera 0
```

You should see the board axes drawn on the image. If this fails, fix `config.py` before continuing.

## 5. Edit `so101_pose_provider.py`

The function `get_T_base_gripper()` must return the current SO-101 wrist/gripper pose as a 4x4 matrix:

```python
T_base_gripper = get_T_base_gripper()
```

It must be in meters and must transform points from the gripper/wrist frame to the robot base frame:

```python
point_base = T_base_gripper @ point_gripper
```

If you only have joint angles, convert them to the wrist/gripper pose with FK using your SO-101 URDF.

## 6. Capture hand-eye samples

Fix the ChArUco board on the table. Do not move it.
Move the robot wrist camera around the board and press `s` at each pose.

```bash
python 01_capture_handeye_samples.py --camera 0
```

Collect around 20 to 30 samples. Use different wrist rotations, not only different positions.

## 7. Run calibration

```bash
python 02_run_handeye_calibration.py
```

This saves:

```text
outputs/T_gripper_camera.npy
```

## 8. Validate

```bash
python 03_validate_handeye.py
```

The predicted board pose in the robot base frame should be similar across all samples.

## 9. Use it

For any detected object pose in camera frame:

```python
T_base_object = T_base_gripper @ T_gripper_camera @ T_camera_object
```

For a 3D point:

```python
point_base = T_base_gripper @ T_gripper_camera @ point_camera
```
