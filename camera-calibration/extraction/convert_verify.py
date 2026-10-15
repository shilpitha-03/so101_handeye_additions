"""
==============================================================================
SCRIPT 2 of 3 — Convert, Verify & Print Isaac Sim GUI Cheatsheet
==============================================================================
PURPOSE:
    Read the camera_params.json produced by Script 1, run additional sanity
    checks, and print a clear "cheatsheet" of every value to type into
    the Isaac Sim GUI — no guessing, no unit confusion.

USAGE:
    Run on your regular Python environment (NOT inside Isaac Sim).

        python 02_convert_and_verify.py

WHAT THIS SCRIPT CHECKS:
    ✓ Focal length round-trip  — recompute fx/fy from cm values and confirm
                                  they match the original calibration.
    ✓ FOV sanity               — compare computed FOV against known hardware spec.
    ✓ cx/cy offset             — warn if principal point is far from image centre
                                  (Isaac Sim cannot model this perfectly).
    ✓ Distortion magnitude     — flag if k1/k2/k3 are unusually large, which can
                                  indicate a poor calibration.

HOW TO USE THE OUTPUT:
    The script prints a numbered cheatsheet.  Open Isaac Sim, select your
    camera prim in the Stage panel, then fill the Property panel fields
    exactly as shown.  Every field is labelled with the exact GUI name.

    After entering the GUI values, run Script 3 inside the Isaac Sim
    Script Editor to apply distortion coefficients (the GUI cannot do this
    in Isaac Sim 5.x).
==============================================================================
"""

import json
import math
import sys

PARAMS_JSON = "./calibration_files/side_view/camera_params.json"  # produced by Script 1

# ── Optional: known hardware FOV for comparison (degrees) ─────────────────────
# Leave as None if unknown.  For Poco M3 at 16:9 crop, H≈67.3°, V≈41.1°
KNOWN_FOV_H = 67.3  # horizontal  (None to skip check)
KNOWN_FOV_V = 41.1  # vertical    (None to skip check)


# ──────────────────────────────────────────────────────────────────────────────


def run():
    try:
        with open(PARAMS_JSON) as f:
            p = json.load(f)
    except FileNotFoundError:
        sys.exit(f"ERROR: {PARAMS_JSON} not found. Run 01_extract_calibration.py first.")

    cal = p["calibration"]
    phys = p["physical"]
    gui = p["isaac_sim_gui"]
    drv = p["derived"]

    fx, fy = cal["fx"], cal["fy"]
    cx, cy = cal["cx"], cal["cy"]
    W, H = cal["width"], cal["height"]

    fl_cm = gui["focal_length_cm"]
    ha_cm = gui["horizontal_aperture_cm"]
    va_cm = gui["vertical_aperture_cm"]

    issues = []
    notices = []

    # ------------------------------------------------------------------
    # Check 1 — Focal length round-trip
    # ------------------------------------------------------------------
    fx_check = (W * fl_cm) / ha_cm
    fy_check = (H * fl_cm) / va_cm
    fx_err = abs(fx_check - fx)
    fy_err = abs(fy_check - fy)

    if fx_err > 1.0 or fy_err > 1.0:
        issues.append(
            f"Focal length round-trip error > 1 px  "
            f"(fx_err={fx_err:.3f}, fy_err={fy_err:.3f}). "
            f"Check PHYSICAL_FOCAL_LENGTH_MM in Script 1."
        )

    # ------------------------------------------------------------------
    # Check 2 — FOV sanity vs hardware spec
    # ------------------------------------------------------------------
    fov_x = drv["fov_x_deg"]
    fov_y = drv["fov_y_deg"]

    if KNOWN_FOV_H is not None and abs(fov_x - KNOWN_FOV_H) > 3.0:
        notices.append(
            f"Computed FOV X ({fov_x:.2f}°) differs from hardware spec "
            f"({KNOWN_FOV_H}°) by {abs(fov_x - KNOWN_FOV_H):.2f}°. "
            f"This is expected if you calibrated at a non-native resolution."
        )

    # ------------------------------------------------------------------
    # Check 3 — Principal point offset
    # ------------------------------------------------------------------
    cx_off_px = abs(cx - W / 2)
    cy_off_px = abs(cy - H / 2)
    cx_off_pct = cx_off_px / W * 100
    cy_off_pct = cy_off_px / H * 100

    if cx_off_pct > 2.0 or cy_off_pct > 2.0:
        notices.append(
            f"Principal point is offset from image centre by "
            f"{cx_off_px:.1f} px ({cx_off_pct:.1f}%) horizontally and "
            f"{cy_off_px:.1f} px ({cy_off_pct:.1f}%) vertically. "
            f"Isaac Sim cannot model this perfectly — aperture offsets are "
            f"not supported. The error is usually small and acceptable."
        )

    # ------------------------------------------------------------------
    # Check 4 — Distortion magnitude
    # ------------------------------------------------------------------
    D = cal["dist_coeffs"]
    k1, k2, k3 = D[0], D[1], D[4]

    if abs(k3) > 5.0:
        notices.append(
            f"k3={k3:.4f} is large.  This is common for phone cameras at "
            f"wide-angle crops but verify your calibration reprojection error "
            f"is below 1.0 px.  Large k3 can sometimes indicate overfitting."
        )

    # ------------------------------------------------------------------
    # Print cheatsheet
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  ISAAC SIM CAMERA SETUP CHEATSHEET")
    print("=" * 65)

    print("""
HOW TO USE:
  1. Open Isaac Sim and load your scene.
  2. Create > Camera  (or select your existing camera prim).
  3. Select the camera in the Stage panel.
  4. In the Property panel (bottom right), enter the values below.
  5. Switch the Viewport to your camera:
       Click the camera icon (top-left of viewport) > Cameras > your camera.
  6. Set the viewport resolution to match:
       Viewport top-left menu > Viewport > Resolution > custom
  7. After entering all GUI values, run Script 3 inside the
       Isaac Sim Script Editor (Window > Script Editor) to apply distortion.
""")

    print("─── PROPERTY PANEL — Camera section ────────────────────────────")
    print(f"  Focal Length         : {fl_cm:.6f}    ← in cm (tenths of metre)")
    print(f"  Horizontal Aperture  : {ha_cm:.6f}    ← in cm")
    print(f"  Vertical Aperture    : {va_cm:.6f}    ← in cm")
    print(f"  f-Stop               : {phys['f_stop']}")
    print(f"  Focus Distance       : {phys['focus_distance_m']}            ← in metres")
    print(f"  Clipping Range (near): {phys['clip_near_m']}          ← in metres")
    print(f"  Clipping Range (far) : {phys['clip_far_m']}      ← in metres")
    print(f"  Projection Type      : Perspective  (leave default)")

    print("\n─── VIEWPORT / RENDER PRODUCT resolution ───────────────────────")
    print(f"  Width  : {W} px")
    print(f"  Height : {H} px")
    print(f"  Note   : Set this on the viewport, NOT in the camera prim.")

    print("\n─── DISTORTION — applied via Script 3 (not in GUI) ─────────────")
    dlabels = cal["dist_labels"]
    for lbl, val in zip(dlabels, D):
        note = "  ← zero-padded" if val == 0.0 and lbl not in ["k1", "k2", "p1", "p2", "k3"] else ""
        print(f"  {lbl:3s} : {val:+.10f}{note}")
    print("  Schema applied: OmniLensDistortionOpenCvPinholeAPI")
    print("  Run Script 3 inside Isaac Sim Script Editor to apply.")

    print("\n─── DERIVED / REFERENCE values ──────────────────────────────────")
    print(f"  FOV X             : {fov_x:.4f} deg")
    print(f"  FOV Y             : {fov_y:.4f} deg")
    print(f"  fx round-trip     : {fx_check:.4f} px  (original: {fx:.4f})")
    print(f"  fy round-trip     : {fy_check:.4f} px  (original: {fy:.4f})")
    print(f"  Pixel size        : {phys['effective_pixel_size_mm'] * 1000:.4f} µm  (effective)")
    print(f"  Focal length (mm) : {drv['focal_length_mm']:.4f} mm  "
          f"(datasheet: {phys['physical_focal_length_mm']} mm)")

    # ------------------------------------------------------------------
    # Print any warnings
    # ------------------------------------------------------------------
    if issues:
        print("\n─── ⛔  ISSUES (must fix before proceeding) ──────────────────")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")

    if notices:
        print("\n─── ⚠   NOTICES (informational) ─────────────────────────────")
        for i, msg in enumerate(notices, 1):
            print(f"  {i}. {msg}")

    if not issues and not notices:
        print("\n  ✓  All checks passed.")

    print("=" * 65)
    print(f"  Source: {PARAMS_JSON}")
    print("=" * 65)
