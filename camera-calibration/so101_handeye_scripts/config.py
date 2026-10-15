"""
Edit this file first.

All lengths must be in meters if your SO-101 FK pose is in meters.
Use the exact same ChArUco board settings that you used for camera calibration.
"""

# ---------------- Camera intrinsics from your previous calibration ----------------
CAMERA_MATRIX_PATH = "camera_matrix.npy"
DIST_COEFFS_PATH = "dist_coeffs.npy"

# ---------------- Camera selection ----------------
# For OpenCV USB camera: usually 0, 1, 2...
# For video/file stream you can pass --camera on the command line.
CAMERA_INDEX_OR_PATH = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ---------------- ChArUco board settings ----------------
# IMPORTANT: change these to your real board values.
# squares_x = number of chessboard squares in X direction
# squares_y = number of chessboard squares in Y direction
SQUARES_X = 7
SQUARES_Y = 6

# Real physical sizes in meters.
# Example: 30 mm = 0.030
SQUARE_LENGTH_M = 0.025   # replace with your measured value
MARKER_LENGTH_M = 0.018   # = measured square * 18/25

# Dictionary must match the board you printed.
# Common options:
#   "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_100", "DICT_5X5_250",
#   "DICT_5X5_1000", "DICT_6X6_250", "DICT_APRILTAG_36h11"
ARUCO_DICT_NAME = "DICT_4X4_50"

# Set from detect_board.py output.
LEGACY_PATTERN = True

# Your board's markers are numbered 1..21 (not 0..20).
FIRST_MARKER_ID = 1

# Minimum ChArUco corners required before accepting a pose.
# Use 8 or higher for better quality. If your board is small/partly visible, lower to 6.
MIN_CHARUCO_CORNERS = 6

# ---------------- Output paths ----------------
SAMPLES_NPZ_PATH = "data/handeye_samples.npz"
SAMPLES_IMAGE_DIR = "data/images"
OUTPUT_DIR = "outputs"
HAND_EYE_NPY_PATH = "outputs/T_gripper_camera.npy"

# ---------------- Collection advice ----------------
# Recommended for real calibration: 20 to 30 samples.
RECOMMENDED_SAMPLE_COUNT = 20
 