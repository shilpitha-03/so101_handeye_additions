#!/usr/bin/env python
"""
Pick the red cube and place it on the blue tray, using the wrist camera.

No ChArUco board at run time. Depth comes from the hand-eye chain instead:
    T_base_cam = FK(q at the frame) @ T_gripper_camera
Each detected pixel becomes a ray from the camera, intersected with a horizontal
plane in the base frame at  table_z + feature height  (cube top / tray top).

Uses the SAME intrinsics (camera_matrix.npy, dist_coeffs.npy) and camera opening
(common.open_camera) as the hand-eye calibration, and FK/IK from so101_kinematics.
Color segmentation is reused from the V2 detector file (detect_colors / select_plate_target).

Phases
  1. OBSERVE  teleop with the leader to a pose that sees the cube and the tray.
              Live overlay shows both objects in base-frame mm.
              c / p = teach cube / tray color, SPACE = lock targets, q = quit.
  2. PLAN     IK for every waypoint, checked for reach, tilt and table clearance.
  3. EXECUTE  automatic after one Enter (or --step to confirm every move).
              STOP any time: any key in the RUNNING window, Enter in the terminal, or Ctrl+C.
              On stop the arm HOLDS where it is (torque on, cube stays gripped); then
              r = resume, h = return to rest, x = exit still holding.
              A blocked arm (collision) or an empty grasp also stops and holds.
              Torque only turns off when you choose to, at the rest pose.

Gripper: only ONE jaw moves. The TCP is offset so the FIXED jaw comes down
--jaw-clearance-mm outside the near cube face and the open MOVING jaw beyond the far
face; closing pushes the cube onto the fixed jaw. Which side the moving jaw opens to is
derived from the URDF (override with --jaw-axis). The cube's resulting off-centre position
in the gripper is accounted for when placing it on the tray.

Example:
  python pick_place_automatic.py --camera /dev/video4 \
      --port /dev/ttyACM0 --id so101_follower_vasco \
      --leader-port /dev/ttyACM1 --leader-id so101_leader_vasco \
      --table-z-mm -7 --cube-size-mm 25 --tray-top-mm 16
Add --dry-run to plan and print everything without moving after the observe phase.
"""
import argparse
import importlib.util
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config                                                        # noqa: E402
from common import open_camera, parse_camera_arg                     # noqa: E402
from so101_kinematics import ARM_JOINTS, SO101Kinematics, tilt_from_vertical_deg  # noqa: E402

ALL_JOINTS = ARM_JOINTS + ["gripper"]
RATE_HZ = 30
DETECT_FRAMES = 12           # frames averaged when locking targets
MAX_SPREAD_MM = 4.0          # locked target must be this stable across frames
MAX_IK_ERR_MM = 3.0
TILT_STEPS_DEG = (0, 5, 10, 15, 20, 25, 30, 35, 40)
MIN_TABLE_CLEAR_MM = 5.0     # TCP never below table + this
REACH_MM = (90.0, 380.0)     # horizontal distance from base z axis
STILL_DEG_S = 3.0
TRACK_ERR_DEG = 12.0        # measured vs commanded joint gap that counts as blocked
TRACK_TICKS = 5             # ...sustained this many ticks (~0.17 s)


# ============================================================ setup helpers
def load_detector(path):
    spec = importlib.util.spec_from_file_location("v2_detector", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("detect_colors", "select_plate_target", "load_profiles", "write_json", "NAMES"):
        if not hasattr(mod, name):
            raise ImportError(f"{path} has no {name}; is this the V2 detector file?")
    return mod


def R_grasp(yaw, tilt_rad, azimuth):
    """Grasp orientation for a 5-DoF arm.

    TCP z points down, tilted by tilt_rad away from the robot (in the vertical plane through
    the pan axis, the only tilt a 5-DoF arm can reach). TCP x (jaw axis) is the horizontal
    direction at `yaw`, made perpendicular to z."""
    z = np.array([np.cos(azimuth) * np.sin(tilt_rad), np.sin(azimuth) * np.sin(tilt_rad), -np.cos(tilt_rad)])
    e = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = e - (e @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def pose(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


# ============================================================ perception
class Perception:
    def __init__(self, det, colors_path, K, D, X, kin0, args):
        self.det, self.colors_path = det, Path(colors_path)
        self.profiles = det.load_profiles(self.colors_path)
        self.K, self.D, self.X, self.kin0 = K, D, X, kin0
        self.table_z = args.table_z_mm / 1000.0
        self.h = {"cube": args.cube_top_mm / 1000.0, "plate": args.tray_top_mm / 1000.0}
        self.plate_mode = args.tray_target

    def ray_to_plane(self, uv, T_base_cam, height):
        xy = cv.undistortPoints(np.array([[uv]], np.float64), self.K, self.D)[0, 0]
        ray = T_base_cam[:3, :3] @ np.array([xy[0], xy[1], 1.0])
        o = T_base_cam[:3, 3]
        z_plane = self.table_z + height
        if ray[2] > -1e-3:
            return None                                  # ray not pointing down to the table
        s = (z_plane - o[2]) / ray[2]
        return o + s * ray if s > 0 else None

    def detect(self, frame, q):
        """Returns dict name -> {status, uv, base (m), contour, yaw (cube only)}."""
        T_base_cam = self.kin0.fk(q) @ self.X
        dets, _ = self.det.detect_colors(frame, self.profiles)
        dets["plate"], _ = self.det.select_plate_target(frame, dets["plate"], self.plate_mode)
        out = {}
        for name in ("cube", "plate"):
            d = dets[name]
            o = {"status": d["status"], "uv": d.get("uv"), "contour": d.get("contour"), "base": None}
            if d["status"] == "OK":
                p = None
                if d.get("pixel_source") == "color_centroid" and d.get("contour") is not None:
                    p = self.metric_centroid(d["contour"], T_base_cam, self.h[name])
                if p is None:
                    p = self.ray_to_plane(d["uv"], T_base_cam, self.h[name])
                if p is None:
                    o["status"] = "ray misses table plane"
                else:
                    o["base"] = p
                    if name == "cube" and d.get("contour") is not None:
                        o["yaw"] = self.cube_yaw(d["contour"], T_base_cam)
            out[name] = o
        return out

    def metric_centroid(self, contour, T_base_cam, height):
        """Area centroid of the region AFTER projecting its outline onto the feature plane.
        The pixel centroid is biased by perspective (the near side looks bigger)."""
        pts = [self.ray_to_plane(c, T_base_cam, height) for c in contour.reshape(-1, 2).astype(float)]
        if any(p is None for p in pts) or len(pts) < 3:
            return None
        xy = np.array(pts)[:, :2]
        x, y = xy[:, 0], xy[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        A = cross.sum() / 2
        if abs(A) < 1e-8:
            return None
        cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * A)
        cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * A)
        return np.array([cx, cy, pts[0][2]])

    def cube_yaw(self, contour, T_base_cam):
        """Cube edge direction in base XY (rad, mod 90 deg).

        The contour is first projected onto the cube-top plane (removes perspective),
        then a min-area rectangle is fitted there in millimetres."""
        pts = [self.ray_to_plane(c, T_base_cam, self.h["cube"]) for c in contour.reshape(-1, 2).astype(float)]
        pts = [p[:2] for p in pts if p is not None]
        if len(pts) < 4:
            return None
        box = cv.boxPoints(cv.minAreaRect((np.array(pts) * 1000).astype(np.float32)))
        e = box[1] - box[0]
        return float(np.mod(np.arctan2(e[1], e[0]), np.pi / 2))

    def teach(self, frame, name):
        title = f"Select a small uniform patch of the {'cube' if name == 'cube' else 'tray'} top, ENTER accepts"
        roi = cv.selectROI(title, frame, showCrosshair=True, fromCenter=False)
        cv.destroyWindow(title)
        if roi[2] and roi[3]:
            x, y, w, h = map(int, roi)
            patch = cv.cvtColor(frame[y:y + h, x:x + w], cv.COLOR_BGR2LAB)
            self.profiles[name]["lab"] = np.median(patch.reshape(-1, 3), axis=0).tolist()
            self.colors_path.parent.mkdir(parents=True, exist_ok=True)
            self.det.write_json(self.colors_path, self.profiles)
            print(f"Taught {name} color, saved to {self.colors_path}")


def draw(frame, res, msg, still, grip=None):
    view = frame.copy()
    colors = {"cube": (60, 60, 255), "plate": (255, 160, 40)}
    for i, name in enumerate(("cube", "plate")):
        r = res[name]
        label = "CUBE" if name == "cube" else "TRAY"
        if r["contour"] is not None:
            cv.drawContours(view, [r["contour"]], -1, colors[name], 2)
        if r["base"] is not None:
            u, v = np.rint(r["uv"]).astype(int)
            cv.drawMarker(view, (u, v), colors[name], cv.MARKER_CROSS, 18, 2)
            b = r["base"] * 1000
            txt = f"{label}: base x {b[0]:.0f} y {b[1]:.0f} z {b[2]:.0f} mm"
            if name == "cube" and r.get("yaw") is not None:
                txt += f"  yaw {np.degrees(r['yaw']):.0f}"
        else:
            txt = f"{label}: {r['status']}"
        cv.putText(view, txt, (10, 24 + 22 * i), cv.FONT_HERSHEY_SIMPLEX, 0.55, colors[name], 2, cv.LINE_AA)
    cv.putText(view, ("Arm still" if still else "Arm MOVING") + ("" if grip is None else f"   gripper {grip:.0f}"),
               (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.55, (60, 200, 60) if still else (60, 60, 230), 2, cv.LINE_AA)
    cv.putText(view, msg, (10, view.shape[0] - 14), cv.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv.LINE_AA)
    return view


# ============================================================ robot helpers
class Stopped(Exception):
    """Raised inside a move after the arm has been told to hold where it is."""
    def __init__(self, reason, waypoint_idx):
        super().__init__(reason)
        self.reason, self.waypoint_idx = reason, waypoint_idx


class StopControl:
    """Stop requests from: any key in the status window, Enter in the terminal, or Ctrl+C.

    All terminal input goes through one reader thread, so prompts and the Enter-to-stop
    watcher never compete for stdin."""
    WINDOW = "pick & place: RUNNING"

    def __init__(self):
        self.lines = queue.Queue()
        self.flag, self.reason = False, ""
        threading.Thread(target=self._reader, daemon=True).start()
        signal.signal(signal.SIGINT, self._sigint)

    def _reader(self):
        while True:
            try:
                self.lines.put(input())
            except EOFError:
                return

    def _sigint(self, *_):
        self.flag, self.reason = True, "Ctrl+C"

    def clear(self):
        self.flag, self.reason = False, ""
        while not self.lines.empty():
            self.lines.get_nowait()

    def requested(self):
        if self.flag:
            return True
        if (cv.waitKey(1) & 0xFF) != 255:
            self.flag, self.reason = True, "key pressed in the status window"
        elif not self.lines.empty():
            self.lines.get_nowait()
            self.flag, self.reason = True, "Enter pressed in the terminal"
        return self.flag

    def prompt(self, msg, valid=None):
        """Wait for a terminal answer. With `valid`, keep asking until the answer is one of them
        (Enter = ''). Ctrl+C while waiting answers 'x' (exit and keep holding)."""
        while True:
            print(msg, end="", flush=True)
            while True:
                try:
                    ans = self.lines.get(timeout=0.2).strip().lower()
                    break
                except queue.Empty:
                    if self.flag and self.reason == "Ctrl+C":
                        print("x  (Ctrl+C)")
                        return "x"
            if valid is None or ans in valid:
                return ans
            print(f"   please type one of: {', '.join(repr(v) if v else 'Enter' for v in valid)}")

    def status(self, lines, color=(60, 200, 60)):
        img = np.full((170, 720, 3), 25, np.uint8)
        for i, (txt, c) in enumerate(lines):
            cv.putText(img, txt, (14, 34 + 34 * i), cv.FONT_HERSHEY_SIMPLEX, 0.7, c or color, 2, cv.LINE_AA)
        cv.imshow(self.WINDOW, img)
        cv.waitKey(1)


class Arm:
    def __init__(self, args):
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        # Torque is NEVER switched off implicitly: not on disconnect, not on a stop, not on a crash.
        # It only goes off through release(), which is only offered once the arm is at its rest pose.
        self.robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True,
                                                       disable_torque_on_disconnect=False))
        self.robot.connect(calibrate=False)        # never starts a recalibration
        self.last_cmd = None
        self.leader = None
        if args.leader_port:
            try:
                from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
            except ImportError:
                from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
            self.leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id, use_degrees=True))
            self.leader.connect(calibrate=False)

    def q6(self):
        obs = self.robot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in ALL_JOINTS], dtype=float)

    def send(self, q6):
        q6 = np.asarray(q6, float)
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(ALL_JOINTS, q6)})
        self.last_cmd = q6

    def hold(self):
        """Freeze: command the arm joints to where they are NOW (so the servos stop instead of
        finishing the last goal) and keep the last gripper command (so a held cube stays gripped)."""
        goal = self.q6()
        if self.last_cmd is not None:
            goal[5] = self.last_cmd[5]
        self.send(goal)

    def teleop_step(self):
        action = self.leader.get_action()
        self.robot.send_action(action)
        self.last_cmd = np.array([action[f"{j}.pos"] for j in ALL_JOINTS], dtype=float)

    def move(self, q6_path, speed_deg_s, stop=None):
        """Follow 6-joint waypoints with linear joint interpolation.

        Every tick: check for a stop request, and compare the measured arm joints with the
        previous command. A sustained large gap means the arm is blocked (collision) ->
        hold and raise Stopped."""
        cur = self.q6()
        if self.last_cmd is not None:
            cur[5] = self.last_cmd[5]
        bad = 0
        for k, target in enumerate(q6_path):
            target = np.asarray(target, float)
            dist = max(np.max(np.abs(target[:5] - cur[:5])), np.abs(target[5] - cur[5]) * 0.5)
            n = max(int(dist / speed_deg_s * RATE_HZ), 3)
            for i in range(1, n + 1):
                t0 = time.perf_counter()
                if stop is not None and stop.requested():
                    self.hold()
                    raise Stopped(stop.reason, k)
                cmd = cur + (target - cur) * i / n
                if self.last_cmd is not None:
                    lag = np.max(np.abs(self.q6()[:5] - self.last_cmd[:5]))
                    bad = bad + 1 if lag > TRACK_ERR_DEG else 0
                    if bad >= TRACK_TICKS:
                        self.hold()
                        raise Stopped(f"arm is not following (lag {lag:.0f} deg): possible collision", k)
                self.send(cmd)
                time.sleep(max(0.0, 1.0 / RATE_HZ - (time.perf_counter() - t0)))
            cur = target
        t_end = time.time() + 0.4                        # let the servos settle (still stoppable)
        while time.time() < t_end:
            if stop is not None and stop.requested():
                self.hold()
                raise Stopped(stop.reason, len(q6_path))
            time.sleep(0.02)

    def release(self):
        self.robot.bus.disable_torque()

    def disconnect(self):
        self.robot.disconnect()                          # torque stays as it is
        if self.leader is not None:
            self.leader.disconnect()


# ============================================================ planning
def jaw_axis_from_urdf(kin0, open_rad=0.4):
    """Direction (TCP frame) from the FIXED jaw toward the MOVING jaw.

    The SO-101 has one fixed jaw (part of gripper_link) and one moving jaw. We glue the
    TCP point to the moving jaw, open the jaw in the URDF, and see where that point goes."""
    r = kin0.kin.robot
    def frame(n):
        return np.asarray(r.get_T_world_frame(n))
    r.set_joint("gripper", 0.0); r.update_kinematics()
    T_tcp = frame(kin0.kin.target_frame_name)
    p_on_jaw = np.linalg.inv(frame("moving_jaw_so101_v1_link")) @ np.r_[T_tcp[:3, 3], 1.0]
    r.set_joint("gripper", open_rad); r.update_kinematics()
    moved = (frame("moving_jaw_so101_v1_link") @ p_on_jaw)[:3] - T_tcp[:3, 3]
    r.set_joint("gripper", 0.0); r.update_kinematics()
    d = T_tcp[:3, :3].T @ moved
    d[2] = 0.0                                   # keep the across-the-jaws component
    return d / np.linalg.norm(d)


AXES = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0]}


class Planner:
    """Grasp planning for the SO-101's asymmetric gripper.

    Only one finger moves. So the TCP is NOT put over the cube centre: it is offset so the
    FIXED jaw comes down just outside one cube face (jaw clearance), the open MOVING jaw
    comes down beyond the opposite face, and closing pushes the cube against the fixed jaw.
    After the grasp the cube sits against the fixed jaw, which is accounted for at the tray."""

    def __init__(self, kin0, args):
        self.kin = kin0
        self.table_z = args.table_z_mm / 1000.0
        self.args = args
        # pan axis location in the base frame (the arm can only tilt in planes through it)
        try:
            self.pan_xy = np.asarray(kin0.kin.robot.get_T_world_frame("shoulder_link"))[:2, 3]
        except Exception:
            self.pan_xy = np.zeros(2)
        if args.jaw_axis == "auto":
            try:
                self.jaw_tcp = jaw_axis_from_urdf(kin0)
                src = "from URDF"
            except Exception as e:
                raise RuntimeError(f"Could not derive the moving-jaw side from the URDF ({e}); pass --jaw-axis") from e
        else:
            self.jaw_tcp = np.array(AXES[args.jaw_axis], float)
            src = "from --jaw-axis"
        print(f"Moving jaw opens toward TCP {np.round(self.jaw_tcp, 2)} ({src}); "
              f"fixed-jaw clearance {args.jaw_clearance_mm} mm, fixed-jaw face offset {args.fixed_jaw_offset_mm} mm")

    def jaw_h(self, R):
        """Unit horizontal direction (base frame) from fixed jaw to moving jaw for orientation R."""
        v = R @ self.jaw_tcp
        v[2] = 0.0
        return v / np.linalg.norm(v)

    def check(self, q, T_goal, max_tilt):
        T = self.kin.fk(q)
        err = np.linalg.norm(T[:3, 3] - T_goal[:3, 3]) * 1000
        tilt = tilt_from_vertical_deg(T)
        z_clear = (T[2, 3] - self.table_z) * 1000
        r = np.linalg.norm(T[:2, 3]) * 1000
        problems = []
        if err > MAX_IK_ERR_MM:
            problems.append(f"IK error {err:.1f} mm")
        if tilt > max_tilt:
            problems.append(f"tilt {tilt:.1f} deg > {max_tilt:.0f}")
        if z_clear < MIN_TABLE_CLEAR_MM:
            problems.append(f"only {z_clear:.1f} mm above table")
        if not REACH_MM[0] <= r <= REACH_MM[1]:
            problems.append(f"reach {r:.0f} mm outside {REACH_MM}")
        return problems

    def ik_at(self, q_seed, p, R, max_tilt):
        T_goal = pose(R, p)
        q, _ = self.kin.ik(q_seed, T_goal, orientation_weight=0.1, tol_m=1e-4, max_iters=300)
        return q, self.check(q, T_goal, max_tilt)

    def segment(self, q_seed, center_xy, points_fn, yaws):
        """Least tilt (then least joint motion over yaw options) at which every point that
        points_fn(R) returns is reachable with the same orientation R.
        Returns qs, R, points, tilt, problems."""
        az = np.arctan2(center_xy[1] - self.pan_xy[1], center_xy[0] - self.pan_xy[0])
        last = ["no tilt option tried"]
        for tilt in TILT_STEPS_DEG:
            if tilt > self.args.max_tilt_deg:
                break
            best = None
            for yaw in yaws:
                R = R_grasp(yaw, np.radians(tilt), az)
                pts = points_fn(R)
                qs, q, ok = [], q_seed, True
                for p in pts:
                    q, pr = self.ik_at(q, p, R, tilt + 5)
                    if pr:
                        last, ok = pr, False
                        break
                    qs.append(q)
                if ok:
                    cost = np.sum(np.abs(qs[0] - q_seed))
                    if best is None or cost < best[0]:
                        best = (cost, qs, R, pts)
            if best is not None:
                return best[1], best[2], best[3], tilt, []
        return None, None, None, None, last

    def line(self, q_start, p0, p1, R, max_tilt, steps, q_end=None):
        """Straight-line Cartesian path at a fixed orientation, IK seeded step to step.
        If a step's IK falls into a bad local solution, retry seeded from a blend toward the
        known end solution; reject the path if consecutive waypoints jump more than 25 deg."""
        qs, q = [], np.asarray(q_start, float)
        for i in range(1, steps + 1):
            p = p0 + (p1 - p0) * i / steps
            seeds = [q] + ([q + (q_end - q) * (1.0 / (steps - i + 1)), q_end] if q_end is not None else [])
            for seed in seeds:
                qn, pr = self.ik_at(seed, p, R, max_tilt)
                if not pr and np.max(np.abs(qn - q)) < 25:
                    break
            else:
                return None, pr or [f"joint jump {np.max(np.abs(qn - q)):.0f} deg between waypoints"]
            q = qn
            qs.append(q)
        return qs, []

    def plan(self, q_obs, cube, tray):
        a = self.args
        g_open, g_closed = a.gripper_open, a.gripper_closed
        half = a.cube_size_mm / 2000.0
        clear = a.jaw_clearance_mm / 1000.0
        d_fixed = a.fixed_jaw_offset_mm / 1000.0
        up = np.array([0, 0, 1.0])
        steps = []

        if a.open_gap_mm is not None and a.open_gap_mm < a.cube_size_mm + a.jaw_clearance_mm + 5:
            return steps, [f"open gap {a.open_gap_mm} mm < cube {a.cube_size_mm} + clearance "
                           f"{a.jaw_clearance_mm} + 5 mm margin: open the gripper wider (--gripper-open)"]

        def add(desc, q_path, grip):
            steps.append((desc, [np.r_[q, grip] for q in q_path]))

        # ---- pick: TCP offset so the fixed jaw lands `clear` outside the near cube face
        z_grasp = max(self.table_z + a.cube_top_mm / 1000.0 - half + a.grasp_z_offset_mm / 1000.0,
                      self.table_z + MIN_TABLE_CLEAR_MM / 1000.0)
        c = np.array([cube["base"][0], cube["base"][1], z_grasp])

        def pick_points(R):
            g = c - self.jaw_h(R) * (half + clear - d_fixed)
            return [g, g + up * a.approach_mm / 1000.0, g + up * a.lift_mm / 1000.0]

        cube_yaw = cube.get("yaw") or 0.0
        qs, R, pts, tilt, pr = self.segment(q_obs[:5], c[:2], pick_points,
                                            [cube_yaw + k * np.pi / 2 for k in range(4)])
        if pr:
            return steps, [f"pick: {x}" for x in pr]
        grasp, pre, lift = pts
        add(f"Move above cube, open gripper  (TCP {np.round(pre*1000)} mm, tilt {tilt} deg; "
            f"TCP offset {(half + clear - d_fixed)*1000:.1f} mm from cube centre to the fixed-jaw side)", [qs[1]], g_open)
        down, pr = self.line(qs[1], pre, grasp, R, tilt + 5, 6, q_end=qs[0])
        if pr:
            return steps, [f"descend: {x}" for x in pr]
        add(f"Descend beside cube  (TCP {np.round(grasp*1000)} mm)", down, g_open)
        add("Close gripper (moving jaw pushes the cube onto the fixed jaw)", [down[-1]], g_closed)
        upq, pr = self.line(down[-1], grasp, lift, R, tilt + 5, 6, q_end=qs[2])
        if pr:
            return steps, [f"lift: {x}" for x in pr]
        add(f"Lift cube  (TCP {np.round(lift*1000)} mm)", upq, g_closed)

        # ---- place: after closing, the cube centre sits at TCP + jaw_h * (half - d_fixed)
        t = np.array([tray["base"][0], tray["base"][1],
                      self.table_z + a.tray_top_mm / 1000.0 + half + a.place_clear_mm / 1000.0])

        def place_points(Rp):
            p = t - self.jaw_h(Rp) * (half - d_fixed)
            pp = p + up * a.approach_mm / 1000.0
            pp[2] = max(pp[2], lift[2])
            back = p - self.jaw_h(Rp) * a.release_backoff_mm / 1000.0     # slide fixed jaw off the cube
            return [p, pp, back, back + up * a.approach_mm / 1000.0]

        yaw_now = np.arctan2(R[1, 0], R[0, 0])
        qs_p, Rp, pts, tilt_p, pr = self.segment(upq[-1], t[:2], place_points,
                                                 [yaw_now + k * np.pi / 2 for k in range(4)])
        if pr:
            return steps, [f"place: {x}" for x in pr]
        place, pre_place, back, back_up = pts
        add(f"Move above tray  (TCP {np.round(pre_place*1000)} mm, tilt {tilt_p} deg)", [qs_p[1]], g_closed)
        dn, pr = self.line(qs_p[1], pre_place, place, Rp, tilt_p + 5, 6, q_end=qs_p[0])
        if pr:
            return steps, [f"lower onto tray: {x}" for x in pr]
        add(f"Lower onto tray  (TCP {np.round(place*1000)} mm)", dn, g_closed)
        add("Open gripper (release)", [dn[-1]], g_open)
        off, pr = self.line(dn[-1], place, back, Rp, tilt_p + 5, 3, q_end=qs_p[2])
        if pr:
            return steps, [f"back off: {x}" for x in pr]
        add(f"Back fixed jaw off the cube ({a.release_backoff_mm:.0f} mm)", off, g_open)
        rq, pr = self.line(off[-1], back, back_up, Rp, tilt_p + 5, 6, q_end=qs_p[3])
        if pr:
            return steps, [f"retreat: {x}" for x in pr]
        add("Retreat up", rq, g_open)
        return steps, []


# ============================================================ main
def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", required=True)
    p.add_argument("--port", required=True); p.add_argument("--id", required=True)
    p.add_argument("--leader-port", required=True); p.add_argument("--leader-id", required=True)
    p.add_argument("--detector", default=str(HERE / "v2_object_detector.py"), help="path to the V2 detector file")
    p.add_argument("--colors", default=None, help="color profiles json (default: the detector's DEFAULT_COLORS)")
    p.add_argument("--hand-eye", default=config.HAND_EYE_NPY_PATH)
    p.add_argument("--table-z-mm", type=float, required=True, help="table surface height in the robot base frame")
    p.add_argument("--cube-size-mm", type=float, required=True)
    p.add_argument("--cube-top-mm", type=float, default=None, help="cube top above table (default: cube size)")
    p.add_argument("--tray-top-mm", type=float, required=True, help="tray top surface above table")
    p.add_argument("--tray-target", choices=("centroid", "auto", "white-mark"), default="centroid")
    p.add_argument("--gripper-open", type=float, default=40.0)
    p.add_argument("--gripper-closed", type=float, default=2.0)
    p.add_argument("--approach-mm", type=float, default=60.0)
    p.add_argument("--lift-mm", type=float, default=80.0)
    p.add_argument("--place-clear-mm", type=float, default=10.0, help="release height above the tray top")
    p.add_argument("--grasp-z-offset-mm", type=float, default=0.0, help="+ raises the grasp point")
    p.add_argument("--speed-deg-s", type=float, default=25.0)
    p.add_argument("--jaw-axis", choices=("auto", "+x", "-x", "+y", "-y"), default="auto",
                   help="TCP axis from the fixed jaw toward the moving jaw (auto: from the URDF)")
    p.add_argument("--jaw-clearance-mm", type=float, default=6.0,
                   help="gap between the fixed jaw and the cube face while descending")
    p.add_argument("--fixed-jaw-offset-mm", type=float, default=0.0,
                   help="distance from the TCP to the fixed jaw's gripping face (0 if the TCP sits on it)")
    p.add_argument("--open-gap-mm", type=float, default=None,
                   help="measured jaw gap at --gripper-open; the plan is rejected if the cube won't fit")
    p.add_argument("--release-backoff-mm", type=float, default=5.0,
                   help="after release, slide the fixed jaw this far off the cube before lifting")
    p.add_argument("--max-tilt-deg", type=float, default=35.0,
                   help="max jaw tilt from vertical; far targets need some tilt on a 5-DoF arm")
    p.add_argument("--dry-run", action="store_true", help="plan and print, no autonomous motion")
    p.add_argument("--step", action="store_true", help="confirm every move with Enter (default: automatic)")
    a = p.parse_args()
    if a.cube_top_mm is None:
        a.cube_top_mm = a.cube_size_mm
    return a


def main():
    a = parse()
    det = load_detector(a.detector)
    colors = a.colors or getattr(det, "DEFAULT_COLORS", HERE / "v2_detection/colors.json")
    K, D = np.load(config.CAMERA_MATRIX_PATH), np.load(config.DIST_COEFFS_PATH)
    X = np.load(a.hand_eye)
    kin0 = SO101Kinematics(tip_offset_m=0.0)        # hand-eye is defined on the bare TCP
    perc = Perception(det, colors, K, D, X, kin0, a)
    planner = Planner(kin0, a)

    print(f"Intrinsics: {Path(config.CAMERA_MATRIX_PATH).resolve()}")
    print(f"Hand-eye:   {Path(a.hand_eye).resolve()}")
    print(f"Colors:     {Path(colors).resolve()}")
    print(f"Table z {a.table_z_mm} mm | cube {a.cube_size_mm} mm (top {a.cube_top_mm}) | tray top {a.tray_top_mm} mm")

    stop = StopControl()
    arm = Arm(a)
    cap = open_camera(parse_camera_arg(a.camera))   # same opening as the calibration
    q_rest = arm.q6()
    at_rest = True
    exit_holding = False          # user already chose "exit and keep holding"

    def go_rest(q_via=None):
        nonlocal at_rest
        stop.clear()
        stop.status([("Returning to rest pose", None), ("Any key here / Enter in terminal = STOP", (200, 200, 200))])
        path = ([np.r_[q_via[:5], a.gripper_open]] if q_via is not None else []) + [q_rest]
        arm.move(path, a.speed_deg_s, stop)
        at_rest = True

    try:
        # sync follower to leader smoothly
        goal = np.array([arm.leader.get_action()[f"{j}.pos"] for j in ALL_JOINTS])
        at_rest = False
        arm.move([goal], a.speed_deg_s, stop)

        # ---------------- OBSERVE
        print("\nOBSERVE: teleop so the camera sees the cube and the tray. "
              "c/p teach colors, SPACE lock targets, q quit.")
        q_prev, t_prev, speed, locked = None, None, 0.0, None
        while locked is None:
            if stop.flag:                                 # Ctrl+C: follower holds its last goal
                return
            arm.teleop_step()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            if frame.shape[:2] != (480, 640):
                raise RuntimeError(f"camera gives {frame.shape[1]}x{frame.shape[0]}, intrinsics are for 640x480")
            q = arm.q6()
            now = time.time()
            if q_prev is not None:
                speed = 0.6 * speed + 0.4 * np.max(np.abs(q[:5] - q_prev[:5])) / max(now - t_prev, 1e-3)
            q_prev, t_prev = q, now
            res = perc.detect(frame, q[:5])
            cv.imshow("pick & place: observe", draw(frame, res, "c cube color | p tray color | SPACE lock | q quit",
                                                    speed < STILL_DEG_S, q[5]))
            k = cv.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                arm.hold()
                return
            if k in (ord("c"), ord("p")):
                perc.teach(frame, "cube" if k == ord("c") else "plate")
            if k == ord(" "):
                pts = {"cube": [], "plate": []}
                yaws = []
                for _ in range(DETECT_FRAMES):
                    arm.teleop_step()
                    ok, frame = cap.read()
                    q = arm.q6()
                    r = perc.detect(frame, q[:5])
                    for n in pts:
                        if r[n]["base"] is not None:
                            pts[n].append(r[n]["base"])
                    if r["cube"].get("yaw") is not None:
                        yaws.append(r["cube"]["yaw"])
                    cv.waitKey(1)
                bad = [n for n in pts if len(pts[n]) < DETECT_FRAMES * 0.7]
                spread = {n: np.max(np.linalg.norm(np.array(v) - np.median(v, 0), axis=1)) * 1000
                          for n, v in pts.items() if v}
                if bad:
                    print(f"Not locked: {', '.join('cube' if b == 'cube' else 'tray' for b in bad)} not detected reliably.")
                elif any(s > MAX_SPREAD_MM for s in spread.values()):
                    print(f"Not locked: detections jitter ({spread}). Hold still / improve lighting.")
                else:
                    cube = {"base": np.median(pts["cube"], 0)}
                    if yaws:  # circular mean on the 90-degree symmetric angle
                        cube["yaw"] = float(np.angle(np.mean(np.exp(4j * np.array(yaws)))) / 4) % (np.pi / 2)
                    tray = {"base": np.median(pts["plate"], 0)}
                    locked = (cube, tray, arm.q6())
        arm.hold()
        cv.destroyAllWindows()
        cube, tray, q_obs = locked
        c, t = cube["base"] * 1000, tray["base"] * 1000
        print(f"\nLOCKED  cube  base {np.round(c, 1)} mm, yaw {np.degrees(cube.get('yaw', 0)):.0f} deg")
        print(f"        tray  base {np.round(t, 1)} mm   (cube -> tray {np.linalg.norm((t - c)[:2]):.0f} mm)")

        # ---------------- PLAN
        steps, problems = planner.plan(q_obs, cube, tray)
        print("\nPLAN")
        for i, (desc, path) in enumerate(steps, 1):
            print(f" {i}. {desc}   [{len(path)} waypoint(s), final joints {np.round(path[-1][:5], 1)}, grip {path[-1][5]:.0f}]")
        if problems:
            print("\nPlan rejected, nothing will move:")
            for pr in problems:
                print("  -", pr)
            return
        if a.dry_run:
            print("\n--dry-run: no autonomous motion.")
            return

        # ---------------- EXECUTE
        mode = "step-by-step" if a.step else "AUTOMATIC"
        print(f"\nEXECUTE ({mode}). Put the leader down; the follower now moves on its own.")
        print("STOP at any time: any key in the RUNNING window, Enter in this terminal, or Ctrl+C.")
        print("On stop the arm holds where it is (torque stays on, the cube stays gripped).")
        stop.status([("Ready. Start from the terminal.", (0, 200, 255))])
        if stop.prompt("\nEnter = start, q = cancel: ", ["", "q"]) in ("q", "x"):
            return
        stop.clear()

        i, start_idx = 0, 0
        while i < len(steps):
            desc, path = steps[i]
            if a.step and start_idx == 0:
                c = stop.prompt(f"\n[{i+1}/{len(steps)}] {desc}\n   Enter = go, s = skip, q = stop here: ", ["", "s", "q"])
                if c in ("q", "x"):
                    arm.hold()
                    break
                if c == "s":
                    i += 1
                    continue
                stop.clear()
            print(f"[{i+1}/{len(steps)}] {desc}")
            stop.status([(f"RUNNING {i+1}/{len(steps)}", None), (desc[:52], (230, 230, 230)),
                         ("Any key here / Enter in terminal = STOP", (0, 200, 255))])
            try:
                arm.move(path[start_idx:], a.speed_deg_s, stop)
                start_idx = 0
            except Stopped as e:
                at_rest = False
                stop.status([("STOPPED - holding position", (60, 60, 230)), (e.reason[:52], (230, 230, 230)),
                             ("Choose in the terminal", (0, 200, 255))])
                print(f"\n!! STOPPED ({e.reason}). Holding position, torque on.")
                stop.flag = False
                c = stop.prompt("   r = resume, h = return to rest pose, x = exit and keep holding: ", ["r", "h", "x"])
                if c == "r":
                    start_idx += e.waypoint_idx
                    stop.clear()
                    continue
                if c == "h":
                    go_rest(q_obs)
                else:
                    exit_holding = True
                break
            at_rest = False
            if desc.startswith("Close gripper"):
                time.sleep(0.3)
                g = arm.q6()[5]
                if g < a.gripper_closed + 3:
                    arm.hold()
                    print(f"\n!! Gripper closed fully ({g:.1f}): the cube was probably missed. Holding.")
                    c = stop.prompt("   r = continue anyway, h = return to rest pose, x = exit and keep holding: ", ["r", "h", "x"])
                    if c == "h":
                        go_rest(q_obs)
                        break
                    if c != "r":
                        exit_holding = True
                        break
                    stop.clear()
            i += 1
        else:
            print("\nPick and place complete. Returning to rest pose.")
            go_rest(q_obs)
    except Stopped as e:                                  # stop during sync / return to rest
        print(f"\n!! STOPPED ({e.reason}). Holding position, torque on.")
    finally:
        cap.release()
        try:
            if not at_rest and not exit_holding:
                stop.flag = False
                c = stop.prompt("\nArm is NOT at rest and is holding under torque.\n"
                                "   h = move to rest pose, k = exit and keep holding: ", ["h", "k", "x"])
                if c == "h":
                    try:
                        go_rest()
                    except Stopped as e:
                        print(f"!! STOPPED ({e.reason}). Holding.")
                    if not at_rest:
                        exit_holding = True
            if at_rest:
                c = stop.prompt("\nArm at rest. Enter = release torque, k = keep holding: ", ["", "k", "x"])
                if c == "":
                    arm.release()
                    print("Torque released.")
                else:
                    print("Keeping torque on.")
            else:
                print("Exiting with the arm HOLDING its position (torque on). "
                      "Support the arm before cutting power, or rerun to move it.")
        finally:
            cv.destroyAllWindows()
            arm.disconnect()


if __name__ == "__main__":
    main()