"""
==============================================================================
SCRIPT 3 of 3 — Apply Distortion Coefficients in Isaac Sim Script Editor
==============================================================================
PURPOSE:
    Apply OpenCV pinhole distortion coefficients (from camera_matrix.npy /
    dist_coeffs.npy) to an existing camera prim inside Isaac Sim 5.x.
    Also corrects the cx, cy, fx, fy, and imageSize inside the distortion
    schema, which the Python API does not set correctly by default.

HOW TO RUN:
    1. Open Isaac Sim.
    2. Load your scene and create your camera prim via the GUI, entering
       the values from Script 2's cheatsheet in the Property panel.
    3. Open Window > Script Editor.
    4. Paste this entire script into the editor.
    5. Update CAMERA_PRIM_PATH and PARAMS_JSON_PATH below.
    6. Click Run (or press Ctrl+Enter).

WHAT THIS SCRIPT DOES:
    Step A — Wraps the existing camera prim with the Camera API and calls
             set_opencv_pinhole_properties() to apply the OmniLensDistortion
             OpenCvPinholeAPI schema with your k1,k2,p1,p2,k3 coefficients.
    Step B — Directly patches cx, cy, fx, fy, imageSize on the USD prim
             because the API call leaves them at incorrect default values.
    Step C — Prints a full verification of every distortion attribute so
             you can confirm everything is correct before saving the scene.

NOTES:
    • This script does NOT recreate the camera — it only patches the
      existing prim.  Run it after you have set all other values in the GUI.
    • The Fisheye Lens panel in the GUI still shows old deprecated fields.
      Ignore them — the OmniLensDistortion schema takes priority in Isaac Sim 5.x.
    • Save the scene (Ctrl+S) after running this script to persist changes.

ISAAC SIM VERSION: 5.x  (tested on 5.0 and 5.1)
==============================================================================
"""

# ── CONFIGURE ─────────────────────────────────────────────────────────────────

# Path to your camera prim in the stage.
# Check it in the Stage panel — click the camera and look at the path bar.
CAMERA_PRIM_PATH = "/Studio/side_camera_1/side_camera_1"

# Path to the JSON file produced by Script 1.
# If you cannot access the filesystem from Isaac Sim, set PARAMS_JSON_PATH = None
# and fill in the values manually in the MANUAL FALLBACK section below.
PARAMS_JSON_PATH = "/path/to/camera_params.json"

# ── END CONFIGURE ─────────────────────────────────────────────────────────────


import json
import os

import omni.usd
from isaacsim.sensors.camera import Camera
from pxr import Gf


# ── Load parameters ────────────────────────────────────────────────────────────

def load_params(json_path):
    """Load from JSON if available, otherwise use manual fallback values."""
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            p = json.load(f)
        cal = p["calibration"]
        gui = p["isaac_sim_gui"]
        print(f"[Script 3] Loaded parameters from: {json_path}")
        return {
            "fx":     cal["fx"],
            "fy":     cal["fy"],
            "cx":     cal["cx"],
            "cy":     cal["cy"],
            "width":  cal["width"],
            "height": cal["height"],
            "D":      cal["dist_coeffs"],   # already 12-element list
        }
    else:
        print("[Script 3] JSON file not found — using MANUAL FALLBACK values.")
        print("           Edit the values below in the MANUAL FALLBACK section.")

        # ── MANUAL FALLBACK — paste your values here if JSON is unavailable ──
        # Copy these from Script 2's output or from your own npy files.
        return {
            "fx":     960.953424,
            "fy":     959.878935,
            "cx":     644.948097,
            "cy":     355.666277,
            "width":  1280,
            "height": 720,
            # Full 12-element list: [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
            "D": [
                 0.24834103,   # k1
                -1.86899562,   # k2
                 0.00095079,   # p1
                 0.00070837,   # p2
                 5.91545234,   # k3
                 0.0, 0.0, 0.0,  # k4, k5, k6
                 0.0, 0.0, 0.0, 0.0,  # s1, s2, s3, s4
            ],
        }
        # ──────────────────────────────────────────────────────────────────────


def main():
    params = load_params(PARAMS_JSON_PATH)
    fx, fy = params["fx"], params["fy"]
    cx, cy = params["cx"], params["cy"]
    W, H   = params["width"], params["height"]
    D      = params["D"]

    print(f"\n[Script 3] Target prim : {CAMERA_PRIM_PATH}")

    # ── Step A — Apply OpenCV pinhole distortion schema ────────────────────────
    print("[Script 3] Step A: Applying OmniLensDistortionOpenCvPinholeAPI schema...")

    camera = Camera(prim_path=CAMERA_PRIM_PATH)
    camera.initialize()
    camera.set_opencv_pinhole_properties(pinhole=D)

    print("           set_opencv_pinhole_properties() called successfully.")

    # ── Step B — Patch cx, cy, fx, fy, imageSize ──────────────────────────────
    # The Camera API sets these to defaults from the render product (often wrong).
    # We overwrite them directly on the USD prim with the calibration values.
    print("[Script 3] Step B: Patching cx, cy, fx, fy, imageSize on USD prim...")

    stage = omni.usd.get_context().get_stage()
    prim  = stage.GetPrimAtPath(CAMERA_PRIM_PATH)

    if not prim.IsValid():
        print(f"[Script 3] ERROR: Prim not found at '{CAMERA_PRIM_PATH}'.")
        print("           Check CAMERA_PRIM_PATH matches your Stage panel path.")
        return

    def patch(attr_name, value):
        attr = prim.GetAttribute(attr_name)
        if attr:
            attr.Set(value)
        else:
            print(f"           WARNING: attribute '{attr_name}' not found — skipped.")

    patch("omni:lensdistortion:opencvPinhole:cx",        cx)
    patch("omni:lensdistortion:opencvPinhole:cy",        cy)
    patch("omni:lensdistortion:opencvPinhole:fx",        fx)
    patch("omni:lensdistortion:opencvPinhole:fy",        fy)
    patch("omni:lensdistortion:opencvPinhole:imageSize", Gf.Vec2i(W, H))

    print("           cx, cy, fx, fy, imageSize patched.")

    # ── Step C — Verification ─────────────────────────────────────────────────
    print("[Script 3] Step C: Verifying final state of distortion schema...\n")

    applied = prim.GetAppliedSchemas()
    print(f"  Applied schemas : {applied}")

    if "OmniLensDistortionOpenCvPinholeAPI" not in applied:
        print("  ⛔  OmniLensDistortionOpenCvPinholeAPI was NOT applied!")
        print("     This usually means the prim path is wrong or the camera")
        print("     was not initialized correctly.  Check CAMERA_PRIM_PATH.")
        return

    print("\n  omni:lensdistortion attributes:")
    all_ok = True
    dist_labels = ["k1","k2","p1","p2","k3","k4","k5","k6","s1","s2","s3","s4"]

    for attr in prim.GetAttributes():
        name = attr.GetName()
        if "lensdistortion" not in name.lower():
            continue
        val = attr.Get()
        print(f"    {name} = {val}")

        # spot-check key values
        if name.endswith(":cx") and abs(float(val) - cx) > 1.0:
            print(f"      ⚠  cx mismatch! expected {cx:.4f}")
            all_ok = False
        if name.endswith(":cy") and abs(float(val) - cy) > 1.0:
            print(f"      ⚠  cy mismatch! expected {cy:.4f}")
            all_ok = False
        if name.endswith(":fx") and abs(float(val) - fx) > 1.0:
            print(f"      ⚠  fx mismatch! expected {fx:.4f}")
            all_ok = False
        if name.endswith(":fy") and abs(float(val) - fy) > 1.0:
            print(f"      ⚠  fy mismatch! expected {fy:.4f}")
            all_ok = False

    if all_ok:
        print("\n  ✓  All distortion attributes look correct.")
    else:
        print("\n  ⚠  Some values did not match — review warnings above.")

    print("\n[Script 3] Done.")
    print("           Save the scene now (Ctrl+S) to persist the distortion schema.")
    print("           Next step: validation — compare sim renders against real images.")


main()