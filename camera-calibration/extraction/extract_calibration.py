"""
==============================================================================
SCRIPT 1 of 3 — Extract & Inspect Calibration Values
==============================================================================
PURPOSE:
    Load camera_matrix.npy and dist_coeffs.npy produced by ChArUco / OpenCV
    calibration, derive all values needed for Isaac Sim, and save them to a
    JSON file that the other two scripts consume.

USAGE:
    Run this on your regular Python environment (NOT inside Isaac Sim).

        python 01_extract_calibration.py

REQUIREMENTS:
    pip install numpy opencv-contrib-python

WHAT YOU NEED TO FILL IN (Section marked ── CONFIGURE ──):
    1. CALIB_PATH        — folder containing your .npy files
    2. CALIB_WIDTH/HEIGHT— the resolution you calibrated at (check your images)
    3. PHYSICAL_FOCAL_LENGTH_MM
                         — the real focal length of your lens in millimetres.
                           Read it off the lens or the camera datasheet.
                           For Poco M3 main camera: 4.7 mm  (Camera FV-5 data)
    4. F_STOP            — the f-number printed on the lens (e.g. 1.8 for Poco M3).
                           Set to 0.0 to disable depth-of-field blur in the sim
                           (recommended for robotics / perception work).
    5. FOCUS_DISTANCE_M  — rough working distance from camera to scene in metres.
                           Only matters when F_STOP > 0.
    6. OUTPUT_JSON       — where to save the derived parameters

WHY PIXEL SIZE IS NOT ASKED DIRECTLY:
    Instead of asking for pixel size (which is hard to find for phone cameras),
    this script derives the effective pixel size automatically from the known
    physical focal length and the calibrated fx value:
        effective_pixel_size_mm = physical_focal_length_mm / fx
    This is more reliable for phone cameras that downscale from a larger sensor.
==============================================================================
"""

import json
import math
import sys

import cv2
import numpy as np

# ── CONFIGURE ─────────────────────────────────────────────────────────────────

CALIB_PATH   = "./calibration_files/side_view"                   # folder with camera_matrix.npy / dist_coeffs.npy
CALIB_WIDTH  = 1280                  # image width  used during calibration (pixels)
CALIB_HEIGHT = 720                   # image height used during calibration (pixels)

# Physical lens/sensor specs — from datasheet or spec sheet
PHYSICAL_FOCAL_LENGTH_MM = 4.7      # real focal length in mm  (Poco M3 = 4.7 mm)
F_STOP                   = 0.0      # f-number (set 0.0 to disable DoF blur)
FOCUS_DISTANCE_M         = 1.0      # distance to focus plane in metres

# Clipping planes (metres) — near/far planes of the rendered scene
CLIP_NEAR = 0.05                    # 5 cm  — nothing closer will be rendered
CLIP_FAR  = 100000.0                # 100 km — effectively infinite

OUTPUT_JSON = "./calibration_files/side_view/camera_params.json"  # output file read by scripts 2 and 3

# ── END CONFIGURE ─────────────────────────────────────────────────────────────


def run():
    # ------------------------------------------------------------------
    # 1. Load calibration files
    # ------------------------------------------------------------------
    try:
        K = np.load(f"{CALIB_PATH}/camera_matrix.npy")
        D = np.load(f"{CALIB_PATH}/dist_coeffs.npy").flatten()
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}\nCheck CALIB_PATH and file names.")

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # ------------------------------------------------------------------
    # 2. Derive effective pixel size from physical focal length
    # ------------------------------------------------------------------
    # fx (pixels) = physical_focal_length_mm / pixel_size_mm
    # => pixel_size_mm = physical_focal_length_mm / fx
    pixel_size_mm = PHYSICAL_FOCAL_LENGTH_MM / fx

    # ------------------------------------------------------------------
    # 3. Compute aperture and focal length in mm
    # ------------------------------------------------------------------
    h_aperture_mm   = pixel_size_mm * CALIB_WIDTH
    v_aperture_mm   = pixel_size_mm * CALIB_HEIGHT
    focal_length_mm = (fx * pixel_size_mm + fy * pixel_size_mm) / 2.0

    # ------------------------------------------------------------------
    # 4. Convert mm → cm  (Isaac Sim GUI and API use 1/10 of stage unit = cm)
    # ------------------------------------------------------------------
    focal_length_cm  = focal_length_mm  / 10.0
    h_aperture_cm    = h_aperture_mm    / 10.0
    v_aperture_cm    = v_aperture_mm    / 10.0

    # ------------------------------------------------------------------
    # 5. Field of view (for reference / sanity check)
    # ------------------------------------------------------------------
    fov_x_deg = math.degrees(2 * math.atan(CALIB_WIDTH  / (2 * fx)))
    fov_y_deg = math.degrees(2 * math.atan(CALIB_HEIGHT / (2 * fy)))

    # ------------------------------------------------------------------
    # 6. Distortion coefficients — pad to 12 elements
    #    OpenCV standard order: [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
    # ------------------------------------------------------------------
    D_padded = list(D) + [0.0] * (12 - len(D))
    dist_labels = ["k1","k2","p1","p2","k3","k4","k5","k6","s1","s2","s3","s4"]

    # ------------------------------------------------------------------
    # 7. Sanity checks
    # ------------------------------------------------------------------
    cx_offset = cx - CALIB_WIDTH  / 2
    cy_offset = cy - CALIB_HEIGHT / 2
    aspect    = fy / fx

    print("=" * 60)
    print("  CALIBRATION EXTRACTION RESULTS")
    print("=" * 60)

    print(f"\n── Resolution ──")
    print(f"  Width  : {CALIB_WIDTH} px")
    print(f"  Height : {CALIB_HEIGHT} px")

    print(f"\n── Camera Matrix (pixel units) ──")
    print(f"  fx : {fx:.6f} px")
    print(f"  fy : {fy:.6f} px")
    print(f"  cx : {cx:.6f} px  (offset from centre: {cx_offset:+.2f} px)")
    print(f"  cy : {cy:.6f} px  (offset from centre: {cy_offset:+.2f} px)")
    print(f"  fy/fx : {aspect:.6f}  (1.0 = square pixels, ideal)")

    print(f"\n── Distortion Coefficients ──")
    for lbl, val in zip(dist_labels, D_padded):
        marker = "  (zero-padded)" if val == 0.0 and lbl not in ["k1","k2","p1","p2","k3"] else ""
        print(f"  {lbl:3s} : {val:+.10f}{marker}")

    print(f"\n── Field of View ──")
    print(f"  FOV X : {fov_x_deg:.4f} deg")
    print(f"  FOV Y : {fov_y_deg:.4f} deg")

    print(f"\n── Physical Conversion ──")
    print(f"  Effective pixel size : {pixel_size_mm*1000:.4f} µm  (derived from focal length)")
    print(f"  Physical focal length: {PHYSICAL_FOCAL_LENGTH_MM} mm  (from datasheet)")
    print(f"  Focal length (avg)   : {focal_length_mm:.4f} mm  (computed)")
    if abs(focal_length_mm - PHYSICAL_FOCAL_LENGTH_MM) > 0.5:
        print(f"  ⚠  Computed focal length differs from datasheet by "
              f"{abs(focal_length_mm - PHYSICAL_FOCAL_LENGTH_MM):.3f} mm.")
        print(f"     Check PHYSICAL_FOCAL_LENGTH_MM or CALIB_WIDTH/HEIGHT.")
    else:
        print(f"  ✓  Focal length matches datasheet within 0.5 mm.")

    print(f"\n── Isaac Sim GUI Values (enter these in Property Panel) ──")
    print(f"  Focal Length       : {focal_length_cm:.6f}  cm")
    print(f"  Horizontal Aperture: {h_aperture_cm:.6f}  cm")
    print(f"  Vertical Aperture  : {v_aperture_cm:.6f}  cm")
    print(f"  f-Stop             : {F_STOP}  (0.0 = no depth-of-field blur)")
    print(f"  Focus Distance     : {FOCUS_DISTANCE_M}  m")
    print(f"  Clipping Range     : near={CLIP_NEAR} m,  far={CLIP_FAR} m")
    print(f"  Resolution         : {CALIB_WIDTH} x {CALIB_HEIGHT}  (set on viewport/render product)")

    # ------------------------------------------------------------------
    # 8. Save JSON for downstream scripts
    # ------------------------------------------------------------------
    params = {
        "_readme": (
            "Generated by 01_extract_calibration.py. "
            "Used by 02_convert_and_verify.py and 03_isaac_sim_apply_distortion.py."
        ),
        "calibration": {
            "width":  CALIB_WIDTH,
            "height": CALIB_HEIGHT,
            "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "dist_coeffs": D_padded,
            "dist_labels": dist_labels,
        },
        "physical": {
            "physical_focal_length_mm":  PHYSICAL_FOCAL_LENGTH_MM,
            "effective_pixel_size_mm":   pixel_size_mm,
            "f_stop":                    F_STOP,
            "focus_distance_m":          FOCUS_DISTANCE_M,
            "clip_near_m":               CLIP_NEAR,
            "clip_far_m":                CLIP_FAR,
        },
        "isaac_sim_gui": {
            "_note": (
                "These values go into the camera prim Property Panel in Isaac Sim. "
                "Focal Length, Horizontal Aperture, and Vertical Aperture are in cm "
                "(1/10 of stage unit, assuming stage unit = 1 metre). "
                "Focus Distance and Clipping Range are in metres."
            ),
            "focal_length_cm":       focal_length_cm,
            "horizontal_aperture_cm": h_aperture_cm,
            "vertical_aperture_cm":   v_aperture_cm,
            "f_stop":                F_STOP,
            "focus_distance_m":      FOCUS_DISTANCE_M,
            "clip_near_m":           CLIP_NEAR,
            "clip_far_m":            CLIP_FAR,
        },
        "derived": {
            "fov_x_deg": fov_x_deg,
            "fov_y_deg": fov_y_deg,
            "focal_length_mm":       focal_length_mm,
            "horizontal_aperture_mm": h_aperture_mm,
            "vertical_aperture_mm":   v_aperture_mm,
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\n✓  Parameters saved to: {OUTPUT_JSON}")
    print("   Next step: run 02_convert_and_verify.py to review and confirm,")
    print("              then run 03_isaac_sim_apply_distortion.py inside Isaac Sim.")
    print("=" * 60)

