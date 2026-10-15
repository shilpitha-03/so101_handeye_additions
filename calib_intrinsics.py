"""Calibrate wrist-camera intrinsics with the same ChArUco board and the same camera settings
used by the hand-eye scripts. Writes camera_matrix.npy and dist_coeffs.npy here.

Keys: s = save frame (only when enough corners), c = calibrate & quit, q/ESC = quit without saving.
"""
import argparse
import cv2 as cv
import numpy as np
import config
from common import create_charuco_board, get_aruco_dictionary, open_camera, parse_camera_arg

ap = argparse.ArgumentParser()
ap.add_argument("--camera", required=True)
a = ap.parse_args()

board, dictionary = create_charuco_board(), get_aruco_dictionary()
try:
    params = cv.aruco.DetectorParameters_create()
except Exception:
    params = cv.aruco.DetectorParameters()

def detect(gray):
    mc, mi, _ = cv.aruco.detectMarkers(gray, dictionary, parameters=params)
    if mi is None or len(mi) == 0:
        return None, None
    n, cc, ci = cv.aruco.interpolateCornersCharuco(mc, mi, gray, board)
    if cc is None or n < 12:
        return None, None
    return cc, ci

cap = open_camera(parse_camera_arg(a.camera))
all_c, all_i, img_size = [], [], None
while True:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read frame")
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]
    cc, ci = detect(gray)
    view = frame.copy()
    if cc is not None:
        cv.aruco.drawDetectedCornersCharuco(view, cc, ci)
    cv.putText(view, f"{img_size[0]}x{img_size[1]}  saved: {len(all_c)}  s=save c=calibrate q=quit",
               (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if cc is not None else (0, 0, 255), 2)
    cv.imshow("intrinsics", view)
    k = cv.waitKey(1) & 0xFF
    if k == ord("s") and cc is not None:
        all_c.append(cc); all_i.append(ci); print("saved", len(all_c))
    elif k == ord("c"):
        break
    elif k in (ord("q"), 27):
        raise SystemExit("Quit without calibrating")
cap.release(); cv.destroyAllWindows()

if len(all_c) < 15:
    raise SystemExit(f"Only {len(all_c)} frames. Get at least 15-25 varied views.")
rms, K, D, _, _ = cv.aruco.calibrateCameraCharuco(all_c, all_i, board, img_size, None, None)
print(f"RMS reprojection error: {rms:.3f} px  (aim for < 0.5; > 1.0 means redo)")
print("K =\n", K, "\nD =", D.ravel(), "\nimage size =", img_size)
np.save(config.CAMERA_MATRIX_PATH, K); np.save(config.DIST_COEFFS_PATH, D)
print("Saved", config.CAMERA_MATRIX_PATH, config.DIST_COEFFS_PATH)
