"""Visual check of the intrinsics: shows original | undistorted side by side.

Straight edges (table edge, door frame, the board) held near the image EDGES and CORNERS
should look straight in the right-hand (undistorted) view.

    python check_undistort.py --camera /dev/video4      (q / ESC to quit)
"""
import argparse

import cv2 as cv
import numpy as np

import config
from common import open_camera, parse_camera_arg

ap = argparse.ArgumentParser()
ap.add_argument("--camera", required=True)
a = ap.parse_args()

K = np.load(config.CAMERA_MATRIX_PATH)
D = np.load(config.DIST_COEFFS_PATH)
cap = open_camera(parse_camera_arg(a.camera))
while True:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read frame")
    cv.imshow("original | undistorted   (q = quit)", np.hstack([frame, cv.undistort(frame, K, D)]))
    if cv.waitKey(1) & 0xFF in (ord("q"), 27):
        break
cap.release()
cv.destroyAllWindows()
