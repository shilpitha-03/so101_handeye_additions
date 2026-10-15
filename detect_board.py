"""Figure out which ChArUco settings match a board you already printed.

Point the wrist camera at the WHOLE board, flat and well lit, then press SPACE.
(Or pass --image photo.png.) Prints detected marker IDs and scores every layout.
"""
import argparse
import cv2 as cv
import numpy as np
from common import open_camera, parse_camera_arg

ap = argparse.ArgumentParser()
ap.add_argument("--camera", default=None)
ap.add_argument("--image", default=None)
ap.add_argument("--squares", default="6,7", help="the two square counts printed on the board")
ap.add_argument("--dict", default="DICT_4X4_50")
a = ap.parse_args()

if a.image:
    frame = cv.imread(a.image)
else:
    cap = open_camera(parse_camera_arg(a.camera))
    while True:
        ok, frame = cap.read()
        cv.imshow("SPACE = grab, q = quit", frame)
        k = cv.waitKey(1) & 0xFF
        if k == 32: break
        if k in (ord("q"), 27): raise SystemExit
    cap.release(); cv.destroyAllWindows()
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

d = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, a.dict))
mc, mi, _ = cv.aruco.detectMarkers(gray, d, parameters=cv.aruco.DetectorParameters())
if mi is None:
    raise SystemExit(f"No {a.dict} markers found. Wrong dictionary (try DICT_5X5_50 / DICT_6X6_50) or bad image.")
ids = sorted(mi.ravel().tolist())
print(f"Detected {len(ids)} markers, IDs: {ids}")
first_id = ids[0]
print(f"Lowest marker ID seen: {first_id}  (assumed to be the board's first marker)")

a1, a2 = map(int, a.squares.split(","))
results = []
for sx, sy in [(a1, a2), (a2, a1)]:
    for legacy in (False, True):
        n_markers = (sx * sy) // 2
        b = cv.aruco.CharucoBoard((sx, sy), 0.025, 0.018, d,
                                  np.arange(first_id, first_id + n_markers, dtype=np.int32))
        if legacy: b.setLegacyPattern(True)
        if max(ids) >= first_id + n_markers:
            results.append((0, sx, sy, legacy)); continue
        n, cc, ci = cv.aruco.interpolateCornersCharuco(mc, mi, gray, b)
        results.append((int(n or 0), sx, sy, legacy))
results.sort(reverse=True)
max_corners = (a1 - 1) * (a2 - 1)
print(f"\nChArUco corners found (max possible {max_corners}):")
for n, sx, sy, legacy in results:
    print(f"  SQUARES_X={sx} SQUARES_Y={sy} legacy={legacy!s:5}  ->  {n}")
best = results[0]
tie = [r for r in results if r[0] == best[0]]
if best[0] == 0:
    print("\nNothing matched. Show the whole board, or the squares count / dictionary is different.")
elif len({(r[1], r[2]) for r in tie}) > 1:
    print("\nAmbiguous: show the WHOLE board, less tilted, and rerun.")
elif len(tie) > 1:
    print(f"\nUSE: SQUARES_X = {best[1]}, SQUARES_Y = {best[2]}, LEGACY_PATTERN = False  (legacy doesn't matter for this board), FIRST_MARKER_ID = {first_id}")
else:
    print(f"\nUSE: SQUARES_X = {best[1]}, SQUARES_Y = {best[2]}, LEGACY_PATTERN = {best[3]}, FIRST_MARKER_ID = {first_id}")
