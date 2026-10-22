#!/usr/bin/env python3
"""Standalone V2 cube/plate detector. Importing never opens cameras or a robot.

Geometry and segmentation carried over from validate_cube_plate_intrinsics.py.
Public API: V2ObjectDetector.process(frame), teach_color(), render(), save_colors().
All public 3D outputs are metres in the OpenCV camera frame: X right, Y down,
Z forward. A visible ChArUco board and known feature heights supply metric depth.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2 as cv
import numpy as np

BASE = Path(__file__).resolve().parent
V2 = Path('/home/asma/soarm101_team_calibration/independent_calibrations/v2_20261019_182328_439303')
DEFAULT_COLORS = BASE / 'v2_detection/colors.json'
NAMES = ('cube', 'plate')
COLORS = {'cube': (70, 90, 255), 'plate': (255, 180, 40)}
WINDOW = 'V2 - live camera points'
CONTROLS = 'V2 color tolerances'


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n')


def make_board(square_cm=2.5):
    b = cv.aruco.CharucoBoard((7, 6), square_cm/100, square_cm/100 * 18/25,
                             cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50),
                             np.arange(1, 22, dtype=np.int32))
    b.setLegacyPattern(True)
    return b


def load_profiles(path):
    profiles = {name: dict(lab=None, l_tol=24, ab_tol=14,
                          min_area=30 if name == 'cube' else 100) for name in NAMES}
    if path.exists():
        saved = json.loads(path.read_text())
        for name in NAMES:
            item = saved.get(name, {})
            if item.get('lab') is not None:
                lab = np.asarray(item['lab'], dtype=float)
                if lab.shape != (3,) or not np.isfinite(lab).all() or np.any((lab < 0) | (lab > 255)):
                    raise ValueError(f'Invalid LAB profile: {name}')
                profiles[name]['lab'] = lab.tolist()
            for key, limit in (('l_tol', 100), ('ab_tol', 100), ('min_area', 3000)):
                if key in item:
                    profiles[name][key] = max(0, min(limit, int(item[key])))
    return profiles


def detect_colors(frame, profiles, roi=None):
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB).astype(np.float32)
    detections, masks = {}, {}
    for name, profile in profiles.items():
        mask = np.zeros(frame.shape[:2], np.uint8)
        detections[name] = dict(status='Teach color with ' + ('C' if name == 'cube' else 'P'))
        if profile['lab'] is not None:
            diff = np.abs(lab - np.asarray(profile['lab'], np.float32))
            mask = ((diff[..., 0] <= profile['l_tol']) & (diff[..., 1] <= profile['ab_tol'])
                    & (diff[..., 2] <= profile['ab_tol'])).astype(np.uint8) * 255
            allowed = np.zeros_like(mask)
            if roi is None:
                allowed[:] = 255
            else:
                x, y, w, h = roi
                allowed[y:y+h, x:x+w] = 255
            mask &= allowed
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
            mask &= allowed
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = sorted((c for c in contours if cv.contourArea(c) >= max(1, profile['min_area'])),
                              key=cv.contourArea, reverse=True)
            detections[name] = dict(status='No matching region')
            if contours:
                c = contours[0]
                M = cv.moments(c)
                uv = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
                status = 'OK'
                if len(contours) > 1 and cv.contourArea(contours[1]) >= .6 * cv.contourArea(c):
                    status = 'Ambiguous regions: narrow ROI/color'
                x, y, w, h = cv.boundingRect(c)
                rx, ry, rw, rh = roi or (0, 0, frame.shape[1], frame.shape[0])
                if x <= rx or y <= ry or x+w >= rx+rw or y+h >= ry+rh:
                    status = 'Region clipped by image/ROI'
                detections[name] = dict(status=status, uv=uv, contour=c, area_px2=cv.contourArea(c),
                                         pixel_source='color_centroid')
        masks[name] = mask
    overlap = np.count_nonzero(masks['cube'] & masks['plate'])
    smaller = min(np.count_nonzero(masks[n]) for n in NAMES)
    if smaller and overlap / smaller > .25:
        for n in NAMES:
            detections[n]['status'] = 'Colors overlap: teach distinct patches'
    return detections, masks


def detect_plate_mark(frame, plate):
    """Find one small neutral bright mark strictly inside a valid colored plate.

    Does not fall back to a region center or a previous frame. The mark's outer
    contour includes small dark ink holes; its centroid is an approximate mark
    center, not recognition of an arbitrary point drawn on the sticker.
    """
    mask = np.zeros(frame.shape[:2], np.uint8)
    result = dict(plate, target_mode='white-mark', pixel_source='white_mark_centroid')
    center = result.pop('uv', None)
    if center is not None:
        result['region_center_uv'] = center
    if plate['status'] != 'OK':
        return result, mask
    inside = np.zeros_like(mask)
    cv.drawContours(inside, [plate['contour']], -1, 255, -1)
    distance = cv.distanceTransform(inside, cv.DIST_L2, 3)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Relative brightness rejects gray-blue texture; saturation separates white
    # from the colored plate. No learned world/pixel offset is used.
    value_min = max(110., float(np.median(hsv[..., 2][inside > 0])) + 35.)
    mask[(inside > 0) & (hsv[..., 1] <= 75) & (hsv[..., 2] >= value_min)] = 255
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    candidates = []
    plate_area = cv.contourArea(plate['contour'])
    for contour in contours:
        area = cv.contourArea(contour)
        if not 8 <= area <= min(600., .10 * plate_area):
            continue
        points = contour.reshape(-1, 2)
        if np.min(distance[points[:, 1], points[:, 0]]) < 2:
            continue
        _, (width, height), _ = cv.minAreaRect(contour)
        if min(width, height) < 2 or max(width, height) / min(width, height) > 3:
            continue
        moments = cv.moments(contour)
        candidates.append(dict(uv=np.array([moments['m10']/moments['m00'],
                                           moments['m01']/moments['m00']]),
                               contour=contour, area_px2=area))
    result['mark_candidate_count'] = len(candidates)
    result['status'] = 'White mark missing: check visibility'
    if len(candidates) > 1:
        result['status'] = 'Multiple white marks: target ambiguous'
    elif candidates:
        mark = candidates[0]
        result.update(status='OK', uv=mark['uv'], mark_contour=mark['contour'],
                      mark_area_px2=mark['area_px2'])
    return result, mask


def select_plate_target(frame, plate, mode):
    """Select the current-frame target; never reuse a past position.

    Auto permits a center fallback only when a valid plate has zero plausible
    marks. Multiple marks or invalid plate segmentation still withhold a target.
    """
    if mode == 'centroid':
        return dict(plate, target_mode=mode), None
    if mode not in ('auto', 'white-mark'):
        raise ValueError('Unknown plate target mode: '+mode)
    target, mask = detect_plate_mark(frame, plate)
    target['target_mode'] = mode
    if mode == 'auto' and plate['status'] == 'OK' and target.get('mark_candidate_count') == 0:
        target = dict(plate, target_mode=mode, fallback_reason='No white mark detected',
                      mark_candidate_count=0, mark_detection_status=target['status'])
    return target, mask


def plate_target_label(detection):
    if detection['status'] != 'OK':
        return 'PLATE TARGET'
    return {'white_mark_centroid': 'WHITE DOT', 'color_centroid': 'PLATE CENTER',
            'manual_marked_point': 'MANUAL POINT'}.get(detection.get('pixel_source'), 'PLATE TARGET')


def solve_board(board, corners, ids, model, max_rmse):
    result = dict(status='Need >=12 ChArUco corners', rmse_px=None)
    if ids is None or len(ids) < 12:
        return result
    obj, pix = board.matchImagePoints(corners, ids)
    try:
        ok, rv, tv = cv.solvePnP(obj, pix, model['K'], model['D'], flags=cv.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(rv).all() or not np.isfinite(tv).all():
            return dict(status='Board pose failed', rmse_px=None)
        R, _ = cv.Rodrigues(rv)
        origin = -R.T @ tv.reshape(3)
        projected, _ = cv.projectPoints(obj, rv, tv, model['K'], model['D'])
        rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-pix.reshape(-1, 2))**2, axis=1))))
        status = 'OK'
        if np.any((obj.reshape(-1, 3) @ R.T + tv.reshape(3))[:, 2] <= 0):
            status = 'Board behind camera'
        elif origin[2] >= 0:
            status = 'Unexpected board normal; check board/setup'
        elif rmse > max_rmse:
            status = 'Board reprojection error too high'
        return dict(status=status, rmse_px=rmse, R=R, t=tv.reshape(3), rvec=rv,
                    camera_origin_board_m=origin)
    except cv.error as e:
        return dict(status='Board solve error: ' + str(e).splitlines()[0], rmse_px=None)


def project_feature(uv, height_cm, model, pose):
    """Ray/plane intersection; raw board Z=-height, not camera depth."""
    if pose['status'] != 'OK':
        return dict(status=pose['status'])
    if height_cm is None:
        return dict(status='Set feature height in detector configuration')
    xy = cv.undistortPoints(np.array([[uv]], np.float64), model['K'], model['D'])[0, 0]
    ray = pose['R'].T @ np.array([*xy, 1.])
    origin = pose['camera_origin_board_m']
    if abs(ray[2]) < 1e-6:
        return dict(status='Viewing ray parallel to height plane')
    distance = (-height_cm / 100 - origin[2]) / ray[2]
    if not np.isfinite(distance) or distance <= 0:
        return dict(status='Feature plane behind camera')
    board_point = origin + distance * ray
    camera_point = pose['R'] @ board_point + pose['t']
    return dict(status='OK', board_cm=board_point * 100, camera_cm=camera_point * 100,
                range_cm=float(np.linalg.norm(camera_point) * 100),
                depth_cm=float(camera_point[2] * 100), height_cm=height_cm,
                origin_range_cm=float(np.linalg.norm(board_point) * 100))


def text(image, message, x, y, color=(230, 230, 230), scale=.47):
    cv.putText(image, message, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)


class V2ObjectDetector:
    """One instance per video stream; process() never reuses an earlier pose/point.

    Call process with raw, unmirrored 640x480 BGR uint8 frames from the calibrated
    camera. Do not pre-undistort the image. No GUI or hardware is
    opened by this class. Heights refer to the tracked TOP surface above board.
    """

    def __init__(self, intrinsics_dir=V2, colors_path=DEFAULT_COLORS,
                 cube_height_cm=0.8, plate_height_cm=1.6,
                 plate_target='auto', square_size_cm=2.5, max_board_rmse=1.5):
        if plate_target not in ('auto', 'centroid', 'white-mark'):
            raise ValueError('plate_target must be auto, centroid, or white-mark')
        for value in (square_size_cm, max_board_rmse):
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Square size and maximum board RMSE must be positive')
        folder = Path(intrinsics_dir)
        kp, dp = (folder / 'camera_matrix_standard_5coef.npy',
                  folder / 'dist_coeffs_standard_5coef.npy')
        K, D = np.load(kp, allow_pickle=False), np.load(dp, allow_pickle=False).reshape(-1)
        if (K.shape != (3, 3) or D.size != 5 or not np.isfinite(K).all()
                or not np.isfinite(D).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0, 0, 1])):
            raise ValueError('Invalid STANDARD V2 five-coefficient calibration')
        self.model = dict(K=K.astype(float), D=D.astype(float))
        self.calibration = dict(matrix_path=str(kp.resolve()), distortion_path=str(dp.resolve()),
            matrix_sha256=hashlib.sha256(kp.read_bytes()).hexdigest(),
            distortion_sha256=hashlib.sha256(dp.read_bytes()).hexdigest(), image_size=[640, 480])
        self.colors_path = Path(colors_path)
        self.profiles = load_profiles(self.colors_path)
        self.board = make_board(square_size_cm)
        self.charuco = cv.aruco.CharucoDetector(self.board)
        self.heights = {}
        self.set_heights(cube_height_cm, plate_height_cm)
        self.plate_target = plate_target
        self.max_board_rmse = max_board_rmse
        self.roi = None
        self._sequence = 0

    @staticmethod
    def _check_frame(frame):
        if not isinstance(frame, np.ndarray) or frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
            raise ValueError('Expected raw 640x480 BGR uint8 frame; do not resize or undistort it')

    def set_heights(self, cube_height_cm, plate_height_cm):
        values = dict(cube=cube_height_cm, plate=plate_height_cm)
        if any(h is not None and not np.isfinite(h) for h in values.values()):
            raise ValueError('Feature heights must be finite, or None to withhold that 3D point')
        self.heights = values

    @staticmethod
    def _check_roi(roi):
        if len(roi) != 4 or any(not np.isfinite(v) or int(v) != v for v in roi):
            raise ValueError('ROI must contain four integers: x, y, width, height')
        x, y, w, h = map(int, roi)
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > 640 or y+h > 480:
            raise ValueError('ROI must be nonempty and inside 640x480')
        return x, y, w, h

    def set_workspace(self, roi=None):
        """Restrict color segmentation only; the full frame still supplies board pose."""
        self.roi = None if roi is None else self._check_roi(roi)

    def teach_color(self, frame, name, roi):
        """Learn median LAB from a small uniform patch (exclude sides/white dot)."""
        self._check_frame(frame)
        if name not in NAMES:
            raise ValueError('Object name must be cube or plate')
        x, y, w, h = self._check_roi(roi)
        patch = cv.cvtColor(frame[y:y+h, x:x+w], cv.COLOR_BGR2LAB)
        self.profiles[name]['lab'] = np.median(patch.reshape(-1, 3), axis=0).tolist()

    def save_colors(self):
        self.colors_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.colors_path, self.profiles)

    def process(self, frame, *, capture_time_s=None):
        """Return a JSON-serializable snapshot; every invalid camera_m is None.

        capture_time_s is optional caller-supplied acquisition time, in its own
        clock domain. processed_time_s is wall-clock processing time, NOT a
        hardware exposure timestamp. Associate FK with the same captured frame.
        """
        self._check_frame(frame)
        if capture_time_s is not None and not np.isfinite(capture_time_s):
            raise ValueError('capture_time_s must be finite')
        self._sequence += 1
        corners, ids, _, _ = self.charuco.detectBoard(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        pose = solve_board(self.board, corners, ids, self.model, self.max_board_rmse)
        detections, _ = detect_colors(frame, self.profiles, self.roi)
        detections['plate'], _ = select_plate_target(frame, detections['plate'], self.plate_target)
        board = dict(valid=pose['status'] == 'OK', status=pose['status'],
            corners=0 if ids is None else len(ids), rmse_px=pose['rmse_px'],
            center_camera_m=None, center_uv=None, T_camera_board=None)
        if board['valid']:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = pose['R'], pose['t']
            center = np.array([7, 6, 0]) * self.board.getSquareLength() / 2
            uv, _ = cv.projectPoints(center[None], pose['rvec'], pose['t'],
                                     self.model['K'], self.model['D'])
            board.update(T_camera_board=T.tolist(), center_uv=uv.reshape(2).tolist(),
                         center_camera_m=(pose['R'] @ center + pose['t']).tolist())
        objects = {}
        for name in NAMES:
            detection = detections[name]
            point = (project_feature(detection['uv'], self.heights[name], self.model, pose)
                     if detection['status'] == 'OK' else dict(status=detection['status']))
            valid = point['status'] == 'OK'
            objects[name] = dict(valid=valid, status=point['status'],
                detection_status=detection['status'], source=detection.get('pixel_source'),
                pixel_uv=detection['uv'].tolist() if detection['status'] == 'OK' else None,
                camera_m=(point['camera_cm']/100).tolist() if valid else None,
                board_m=(point['board_cm']/100).tolist() if valid else None,
                range_m=point['range_cm']/100 if valid else None,
                depth_m=point['depth_cm']/100 if valid else None,
                feature_height_cm=self.heights[name], fallback_reason=detection.get('fallback_reason'),
                contour=detection['contour'].reshape(-1, 2).tolist() if 'contour' in detection else None)
        both = all(objects[n]['valid'] for n in NAMES)
        delta = (np.array(objects['plate']['board_m']) - objects['cube']['board_m']) if both else None
        return dict(schema_version=1, frame_id=self._sequence, capture_time_s=capture_time_s,
            processed_time_s=time.time(), coordinate_frame='opencv_camera', units='metres',
            calibration=dict(self.calibration), plate_target_mode=self.plate_target,
            workspace_roi=self.roi, board=board, objects=objects,
            cube_to_plate_m=float(np.linalg.norm(delta)) if both else None,
            cube_to_plate_xy_m=float(np.linalg.norm(delta[:2])) if both else None)

    def render(self, frame, result, *, frozen=False):
        """Draw a result on its matching frame; display-only values are centimetres."""
        self._check_frame(frame)
        canvas = np.full((810, 640, 3), 22, np.uint8)
        canvas[35:515] = frame
        view = canvas[35:515]
        text(canvas, 'STANDARD V2 | ' + ('FROZEN - review only' if frozen else 'LIVE'),
             12, 24, (80, 230, 230), .6)
        board = result['board']
        if board['valid'] and board['T_camera_board'] is not None:
            T_camera_board = np.asarray(board['T_camera_board'], dtype=float)
            rvec, _ = cv.Rodrigues(T_camera_board[:3, :3])
            # Same board-origin frame and 4 cm axes as the comparison preview.
            cv.drawFrameAxes(view, self.model['K'], self.model['D'],
                             rvec, T_camera_board[:3, 3], .04)
        for name, obj in result['objects'].items():
            color = COLORS[name]
            if obj['contour'] is not None:
                cv.drawContours(view, [np.array(obj['contour'], np.int32).reshape(-1, 1, 2)], -1, color, 2)
            if obj['pixel_uv'] is not None:
                uv = tuple(np.rint(obj['pixel_uv']).astype(int))
                cv.drawMarker(view, uv, color, cv.MARKER_CROSS, 18, 2)
                label = name if name == 'cube' else ('white dot' if obj['source'] == 'white_mark_centroid' else 'plate center')
                text(view, label, uv[0]+8, uv[1]-8, color)
        if board['center_uv'] is not None:
            uv = tuple(np.rint(board['center_uv']).astype(int))
            cv.drawMarker(view, uv, (255, 80, 255), cv.MARKER_CROSS, 16, 2)
            text(view, 'board center', uv[0]+8, uv[1]+18, (255, 80, 255))
        roi = result['workspace_roi']
        if roi is not None:
            x, y, w, h = roi
            cv.rectangle(view, (x, y), (x+w-1, y+h-1), (160, 160, 160), 1)
        rmse = '--' if board['rmse_px'] is None else f"{board['rmse_px']:.3f}px"
        text(canvas, f"Board: {board['status']} | {board['corners']}/30 | RMSE {rmse}", 10, 536, scale=.42)
        for i, name in enumerate(NAMES):
            obj, y = result['objects'][name], 560+i*66
            text(canvas, f"{name.upper()}: {obj['source'] or obj['status']}", 10, y, COLORS[name])
            if obj['valid']:
                x, yy, z = np.array(obj['camera_m'])*100
                text(canvas, f'Camera X {x:+.2f}   Y {yy:+.2f}   Z {z:+.2f} cm', 10, y+20)
                text(canvas, f"Range {obj['range_m']*100:.2f} cm | feature height {obj['feature_height_cm']:g} cm", 10, y+39)
            else:
                text(canvas, '3D unavailable: '+obj['status'], 10, y+22, (100, 170, 255), .44)
        distance = result['cube_to_plate_m']
        message = 'Cube -> plate: unavailable'
        if distance is not None:
            message = f"Cube -> plate: 3D {distance*100:.2f} cm | board XY {result['cube_to_plate_xy_m']*100:.2f} cm"
            a, b = [tuple(np.rint(result['objects'][n]['pixel_uv']).astype(int)) for n in NAMES]
            cv.line(view, a, b, (0, 230, 230), 1)
        text(canvas, message, 10, 699, (0, 230, 230))
        text(canvas, 'Camera axes: X right, Y down, Z forward | API uses metres', 10, 722, scale=.43)
        text(canvas, 'C cube color | P plate color | D target: '+self.plate_target, 10, 745, scale=.43)
        text(canvas, 'SPACE freeze/live | W workspace | X clear ROI | S save | Q quit', 10, 767, scale=.43)
        text(canvas, 'Known heights + visible board required. No robot commands.', 10, 791, scale=.43)
        return canvas


def camera_point_to_base(camera_m, T_base_gripper, T_gripper_camera):
    """Pure math for AFTER eye-in-hand calibration. All translations in metres.

    T_gripper_camera maps camera -> gripper, using the SAME gripper frame as FK.
    T_base_gripper must correspond to the frame acquisition. Does not move robot.
    """
    point = np.asarray(camera_m, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Expected a valid camera point with three finite metre coordinates')
    transforms = []
    for value in (T_base_gripper, T_gripper_camera):
        T = np.asarray(value, dtype=float)
        if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1])
                or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)):
            raise ValueError('Expected a rigid 4x4 transform with translation in metres')
        transforms.append(T)
    return (transforms[0] @ transforms[1] @ np.r_[point, 1])[:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', default='/dev/video2')
    parser.add_argument('--intrinsics-dir', type=Path, default=V2)
    parser.add_argument('--colors', type=Path, default=DEFAULT_COLORS)
    parser.add_argument('--cube-height-cm', type=float, default=.8)
    parser.add_argument('--plate-height-cm', type=float, default=1.6)
    parser.add_argument('--plate-target', choices=('auto', 'centroid', 'white-mark'), default='auto')
    parser.add_argument('--square-size-cm', type=float, default=2.5)
    parser.add_argument('--max-board-rmse', type=float, default=1.5)
    parser.add_argument('--output-dir', type=Path, default=BASE/'v2_detection/captures')
    parser.add_argument('--image', type=Path, help='Process a saved raw 640x480 image')
    parser.add_argument('--headless', action='store_true', help='With --image: save PNG/JSON and exit')
    args = parser.parse_args()
    if args.headless and args.image is None:
        parser.error('--headless requires --image')
    detector = V2ObjectDetector(args.intrinsics_dir, args.colors, args.cube_height_cm,
        args.plate_height_cm, args.plate_target, args.square_size_cm, args.max_board_rmse)
    print('V2 intrinsics:', detector.calibration['matrix_path'])
    print('Color profiles:', detector.colors_path)
    print('Camera coordinates: X right, Y down, Z forward. API: metres; display: cm.')
    cap = None
    frame = cv.imread(str(args.image)) if args.image else None
    frozen = args.image is not None

    def save(result, panel):
        folder = args.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder/'points.json', result)
        write_json(folder/'colors.json', detector.profiles)
        for name, img in (('raw.png', frame), ('preview.png', panel)):
            if not cv.imwrite(str(folder/name), img):
                raise OSError('Could not save '+str(folder/name))
        print('Saved:', folder)

    try:
        if args.image:
            if frame is None:
                raise ValueError('Could not read '+str(args.image))
            detector._check_frame(frame)
        else:
            source = int(args.camera) if args.camera.isdigit() else args.camera
            cap = cv.VideoCapture(source, cv.CAP_V4L2)
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv.CAP_PROP_FPS, 30)
            if not cap.isOpened():
                raise RuntimeError('Cannot open camera '+args.camera)
        if args.headless:
            result = detector.process(frame)
            save(result, detector.render(frame, result, frozen=True))
            return
        cv.namedWindow(WINDOW, cv.WINDOW_AUTOSIZE)
        cv.namedWindow(CONTROLS, cv.WINDOW_NORMAL)
        for name in NAMES:
            for key, maximum in (('l_tol', 100), ('ab_tol', 100), ('min_area', 3000)):
                cv.createTrackbar(name+' '+key, CONTROLS, detector.profiles[name][key], maximum, lambda _: None)
        while True:
            if not frozen:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera frame read failed; no further points produced')
            for name in NAMES:
                for key in ('l_tol', 'ab_tol', 'min_area'):
                    detector.profiles[name][key] = cv.getTrackbarPos(name+' '+key, CONTROLS)
            result = detector.process(frame)
            result['preview_frozen'] = frozen
            panel = detector.render(frame, result, frozen=frozen)
            cv.imshow(WINDOW, panel)
            key = cv.waitKey(20) & 0xff
            if key in (27, ord('q')) or cv.getWindowProperty(WINDOW, cv.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' ') and cap is not None:
                frozen = not frozen
            elif key in (ord('c'), ord('p'), ord('w')):
                title = 'Select small color patch, ENTER accepts, ESC cancels' if key != ord('w') else 'Select workspace, ENTER accepts'
                roi = cv.selectROI(title, frame, showCrosshair=True, fromCenter=False)
                cv.destroyWindow(title)
                if roi[2] and roi[3]:
                    if key == ord('w'):
                        detector.set_workspace(roi)
                    else:
                        detector.teach_color(frame, 'cube' if key == ord('c') else 'plate', roi)
                        detector.save_colors()
            elif key == ord('x'):
                detector.set_workspace()
            elif key == ord('d'):
                modes = ('auto', 'centroid', 'white-mark')
                detector.plate_target = modes[(modes.index(detector.plate_target)+1) % len(modes)]
            elif key == ord('s'):
                save(result, panel)
        detector.save_colors()
    finally:
        if cap is not None:
            cap.release()
        if not args.headless:
            cv.destroyAllWindows()


if __name__ == '__main__':
    main()
