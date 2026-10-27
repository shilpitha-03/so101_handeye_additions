"""Capture hand-eye samples while TELEOPERATING the follower with the leader arm,
with a live quality/coverage panel next to the camera view.

The follower stays under torque (no sagging). Hold the leader still and press s.
On each 's' it waits for the follower to settle, checks it really stopped, then
records a fresh camera frame + follower joint pose together.

Panel (right side):
  CURRENT VIEW  GOOD / OK / BAD for the pose you're holding now, and why
  COVERAGE      ROT and TRANS observability of the saved set, consistency (mm)
                of a live solve, outlier warning for the last sample, and which
                joint to vary next
  READY         when coverage + consistency are good enough to solve

Keys (click the camera window first):
    s : save sample (refused if the view is BAD)    S : save anyway
    u : undo last sample                            p : board pose in base (after step 10)
    q : save all samples & quit   (Ctrl+C also saves)
"""
import argparse
import json
import os
import pathlib
import textwrap
import time
from pathlib import Path

os.environ["SO101_NO_AUTOCONNECT"] = "1"   # we connect the follower ourselves, torque ON

import cv2 as cv
import numpy as np
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

import config
import handeye_quality as HQ
import so101_pose_provider as P
from common import draw_charuco_debug, ensure_dir, estimate_charuco_pose, load_intrinsics, open_camera, parse_camera_arg, print_T

# ----------------------------- EDIT THESE THREE -----------------------------
LEADER_PORT = "/dev/ttyACM1"
LEADER_ID = "so101_leader_vasco"    # = the .json filename of your existing leader calibration
LEADER_CALIB_DIR = pathlib.Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader"
# ----------------------------------------------------------------------------

SETTLE_S = 0.8          # keep teleoperating this long after pressing s, then capture
STILL_DEG = 0.5         # follower must move less than this (any joint) over 0.2 s
PANEL_W = 430
PANEL_MIN_H = 600
COLORS = {HQ.GOOD: (60, 200, 60), HQ.OK: (0, 200, 255), HQ.BAD: (60, 60, 230), None: (200, 200, 200)}


# ------------------------------------------------------------------ drawing helpers
def _text(img, txt, y, color=(230, 230, 230), scale=0.5, thick=1, x=12):
    cv.putText(img, txt, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)
    return y + int(28 * scale + 8)


def _wrapped(img, txt, y, color=(230, 230, 230), width=48, scale=0.48):
    for line in textwrap.wrap(txt, width):
        y = _text(img, line, y, color, scale)
    return y


def _bar(img, y, label, value, good, ok, unit, fmt, grade):
    y = _text(img, f"{label}: {fmt.format(value)} {unit}   (target {fmt.format(good)})", y, COLORS[grade])
    x0, w, h = 12, PANEL_W - 24, 10
    cv.rectangle(img, (x0, y - 8), (x0 + w, y - 8 + h), (80, 80, 80), 1)
    frac = min(value / (good * 1.5), 1.0) if good > 0 else 0
    cv.rectangle(img, (x0, y - 8), (x0 + int(w * frac), y - 8 + h), COLORS[grade], -1)
    xo = x0 + int(w * ok / (good * 1.5)); xg = x0 + int(w / 1.5)
    cv.line(img, (xo, y - 11), (xo, y + 4), (0, 200, 255), 1)
    cv.line(img, (xg, y - 11), (xg, y + 4), (60, 200, 60), 1)
    return y + 18


def draw_panel(h, verdict, why, cand_lines, st, advice, flash):
    pan = np.full((h, PANEL_W, 3), 30, np.uint8)
    y = _text(pan, "CURRENT VIEW", 24, (180, 180, 180), 0.5)
    y = _text(pan, verdict, y + 6, COLORS[verdict], 0.9, 2)
    y = _wrapped(pan, why, y, COLORS[verdict])
    for txt, g in cand_lines:
        y = _text(pan, txt, y, COLORS[g], 0.47)
    cv.line(pan, (10, y), (PANEL_W - 10, y), (90, 90, 90), 1)
    y += 22

    y = _text(pan, f"COVERAGE   samples {st['n']}/{HQ.TARGET_SAMPLES}", y, (180, 180, 180), 0.5)
    if st["n"] >= 2:
        y = _bar(pan, y, "ROT", st["rot"], HQ.ROT_GOOD, HQ.ROT_OK, "deg", "{:.1f}", st["rot_grade"])
        y = _bar(pan, y, "TRANS", st["trans"], HQ.TRANS_GOOD, HQ.TRANS_OK, "", "{:.2f}", st["trans_grade"])
    else:
        y = _text(pan, "ROT / TRANS: need 2+ samples", y, COLORS[None], 0.47)
    if st["spread"] is not None:
        y = _text(pan, f"Consistency: {st['spread']:.1f} mm   (target < {HQ.SPREAD_GOOD:.0f})", y,
                  COLORS[st["spread_grade"]], 0.5)
        if st["last_outlier"]:
            y = _wrapped(pan, f"Last sample off by {st['last_resid']:.0f} mm: likely bad. Press u to undo.",
                         y, COLORS[HQ.BAD])
    elif not st["solvable"]:
        y = _wrapped(pan, "Consistency: can't solve yet, all motion is about one axis", y, COLORS[HQ.BAD])
    else:
        y = _text(pan, f"Consistency: after {HQ.MIN_SAMPLES_SOLVE} samples", y, COLORS[None], 0.47)
    if advice:
        y = _wrapped(pan, advice, y + 4, (255, 200, 120))
    if st["ready"]:
        msg = "READY TO SOLVE" if st["n"] >= HQ.TARGET_SAMPLES else f"READY (a few more helps: {st['n']}/{HQ.TARGET_SAMPLES})"
        y = _text(pan, msg, y + 6, COLORS[HQ.GOOD], 0.65, 2)
    elif (st["n"] < HQ.MIN_SAMPLES_READY and st["rot_grade"] == HQ.GOOD and st["trans_grade"] == HQ.GOOD):
        y = _text(pan, f"Coverage good, collect {HQ.MIN_SAMPLES_READY - st['n']} more (min {HQ.MIN_SAMPLES_READY})",
                  y + 4, COLORS[HQ.OK], 0.47)

    if flash:
        lines = textwrap.wrap(flash[0], 48)[:3]
        yf = h - 30 - 17 * len(lines)
        cv.rectangle(pan, (0, yf - 16), (PANEL_W, h - 26), (30, 30, 30), -1)
        for line in lines:
            yf = _text(pan, line, yf, flash[1], 0.48)
    _text(pan, "s save  S force  u undo  p pose  q quit", h - 12, (150, 150, 150), 0.45)
    return pan


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--out", default=config.SAMPLES_NPZ_PATH)
    ap.add_argument("--image-dir", default=config.SAMPLES_IMAGE_DIR)
    a = ap.parse_args()
    ensure_dir(str(Path(a.out).parent)); ensure_dir(a.image_dir)

    if not (LEADER_CALIB_DIR / f"{LEADER_ID}.json").is_file():
        raise FileNotFoundError(f"No leader calibration at {LEADER_CALIB_DIR / (LEADER_ID + '.json')}. Edit the top of this file.")

    K, D = load_intrinsics()
    T_gc = np.load(config.HAND_EYE_NPY_PATH) if Path(config.HAND_EYE_NPY_PATH).is_file() else None

    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID, calibration_dir=LEADER_CALIB_DIR, use_degrees=True))
    leader.connect(calibrate=False)
    follower = P.connect(torque_off=False)
    cap = open_camera(parse_camera_arg(a.camera))

    def grab():
        follower.send_action(leader.get_action())
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame")
        return frame

    cov = HQ.Coverage()
    paths, stamps = [], []
    q_prev, t_prev, speed = None, None, 0.0
    advice, frame_i, flash, flash_until = None, 0, None, 0.0

    def set_flash(msg, grade):
        nonlocal flash, flash_until
        flash, flash_until = (msg, COLORS[grade]), time.time() + 3.0
        print(msg)

    print("\nTeleop running. Watch the panel; hold the leader still and press s on GOOD views. q = save & quit.")
    try:
        while True:
            frame = grab()
            q = P.get_joint_deg()
            now = time.time()
            if q_prev is not None:
                inst = np.max(np.abs(q - q_prev)) / max(now - t_prev, 1e-3)
                speed = 0.6 * speed + 0.4 * inst
            q_prev, t_prev = q, now
            G = P.fk_from_deg(q)

            res = estimate_charuco_pose(frame, K, D)
            verdict, cand_lines, why = cov.evaluate(G, res, q, speed, P.fk_from_deg)
            if frame_i % 10 == 0:
                advice = cov.advice(q, P.fk_from_deg)
            frame_i += 1

            view = draw_charuco_debug(frame, res, K, D)
            cv.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), COLORS[verdict], 6)
            if time.time() > flash_until:
                flash = None
            ph = max(view.shape[0], PANEL_MIN_H)
            panel = draw_panel(ph, verdict, why, cand_lines, cov.status(), advice, flash)
            if view.shape[0] < ph:
                view = cv.copyMakeBorder(view, 0, ph - view.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(30, 30, 30))
            cv.imshow("teleop hand-eye capture", np.hstack([view, panel]))
            key = cv.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                if len(cov):
                    cov.pop(); paths.pop(); stamps.pop()
                    set_flash(f"Removed last sample ({len(cov)} left).", HQ.OK)
                continue
            if key not in (ord("s"), ord("S"), ord("p")):
                continue

            # settle: keep teleop + camera running so the buffer is fresh
            t_end = time.time() + SETTLE_S
            while time.time() < t_end:
                grab(); cv.waitKey(1)
            q1 = P.get_joint_deg(); grab(); time.sleep(0.2); grab()
            q2 = P.get_joint_deg()
            if np.max(np.abs(q2 - q1)) > STILL_DEG:
                set_flash(f"Arm still moving ({np.max(np.abs(q2 - q1)):.2f} deg). Hold still and press again.", HQ.BAD)
                continue
            frame = grab()
            q_cap = P.get_joint_deg()
            T_bg = P.fk_from_deg(q_cap)
            res = estimate_charuco_pose(frame, K, D)
            if not res.ok:
                set_flash(f"Board not detected in the capture frame: {res.message}", HQ.BAD)
                continue

            if key == ord("p"):
                if T_gc is None:
                    set_flash("No outputs/T_gripper_camera.npy yet (run step 10 first).", HQ.OK)
                else:
                    print_T("T_base_charuco", T_bg @ T_gc @ res.T_camera_charuco)
                continue

            verdict, _, why = cov.evaluate(T_bg, res, q_cap, 0.0, P.fk_from_deg)
            if verdict == HQ.BAD and key == ord("s"):
                set_flash(f"Not saved: {why}. Press S to save anyway.", HQ.BAD)
                continue

            idx = len(cov)
            path = str(Path(a.image_dir) / f"sample_{idx:03d}.png")
            cv.imwrite(path, frame)
            cov.add(T_bg, res.T_camera_charuco, q_cap)
            paths.append(path); stamps.append(time.time())
            st = cov.status()
            msg = f"Saved sample {idx + 1} ({verdict}: {why})"
            if st["spread"] is not None:
                msg += f" | consistency {st['spread']:.1f} mm, this sample {st['last_resid']:.1f} mm"
            set_flash(msg, verdict)
            advice = cov.advice(q_cap, P.fk_from_deg)
    finally:
        cap.release(); cv.destroyAllWindows()
        if len(cov):
            np.savez(a.out, T_base_gripper=np.asarray(cov.G), T_camera_charuco=np.asarray(cov.C),
                     joints_lerobot_deg=np.asarray(cov.Q),
                     image_paths=np.asarray(paths), timestamps=np.asarray(stamps),
                     board_config=json.dumps({"squares_x": config.SQUARES_X, "squares_y": config.SQUARES_Y,
                                              "square_length_m": config.SQUARE_LENGTH_M,
                                              "marker_length_m": config.MARKER_LENGTH_M,
                                              "aruco_dict_name": config.ARUCO_DICT_NAME}))
            st = cov.status()
            print(f"\nSaved {len(cov)} samples to {a.out}")
            print(f"Coverage: ROT {st['rot']:.1f} deg, TRANS {st['trans']:.2f}"
                  + (f", consistency {st['spread']:.1f} mm" if st["spread"] is not None else ""))
        print("Disconnecting (follower torque turns OFF now).")
        follower.disconnect(); leader.disconnect()


if __name__ == "__main__":
    main()
