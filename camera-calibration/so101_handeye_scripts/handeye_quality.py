"""Live quality + coverage metrics for eye-in-hand calibration samples (AX = XB).

Per candidate (what the camera/arm see right now):
  * detection: corner count, reprojection error, viewing angle, distance
  * motion: is the arm still
  * novelty: how different it is from samples already saved
  * gain: how much it would improve the weakest coverage metric

Coverage of the saved set, from all pairwise relative gripper motions A_ij = inv(G_i) G_j:
  * ROT  = sqrt(2nd eigenvalue) of mean(r r^T), r = rotation vector of A_ij  [deg]
           The rotation of X needs relative motions about at least 2 non-parallel axes;
           the 2nd eigenvalue is the spread about the second-best axis.
  * TRANS = sqrt(min eigenvalue) of mean((R_ij - I)^T (R_ij - I))
           The translation of X is unobservable along any direction all motions rotate about.
  * CONSISTENCY (>= MIN_SAMPLES_SOLVE): solve with PARK, board position in base frame
           should be identical for every sample -> spread in mm, per-sample residuals.
"""
import cv2 as cv
import numpy as np

# ---- thresholds (tuned for SO-101 scale) -------------------------------------------
MIN_CORNERS_GOOD, MIN_CORNERS_OK = 15, 8
REPROJ_GOOD, REPROJ_OK = 0.8, 1.5                 # px
VIEW_GOOD = (15.0, 50.0)                          # deg between optical axis and board normal
VIEW_OK = (5.0, 65.0)
DIST_OK = (0.10, 0.50)                            # m
MOVING_DEG_S = 3.0                                # joint speed that counts as moving
SIMILAR_ROT_DEG, SIMILAR_POS_M = 8.0, 0.03        # closer than both = too similar
# Calibrated in simulation (25 samples, 0.3 deg joint noise, real ChArUco detection):
#   ROT 5 / TRANS 0.10 -> ~2.3 mm camera-position error; ROT 10 / TRANS 0.20 -> ~1 mm
ROT_GOOD, ROT_OK = 9.0, 6.0                       # deg
TRANS_GOOD, TRANS_OK = 0.18, 0.12
TARGET_SAMPLES, MIN_SAMPLES_READY, MIN_SAMPLES_SOLVE = 25, 15, 5
SPREAD_GOOD, SPREAD_OK = 5.0, 15.0                # mm
OUTLIER_MM = 6.0                                  # last-sample residual that is flagged (also > 3x median)

GOOD, OK, BAD = "GOOD", "OK", "BAD"
ADVICE_JOINTS = {0: "shoulder_pan (view the board from the left/right)",
                 3: "wrist_flex (tilt camera up/down)",
                 4: "wrist_roll"}


def _rotvec(R):
    return cv.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel()


def _pair_terms(Gi, Gj):
    A = np.linalg.inv(Gi) @ Gj
    r = _rotvec(A[:3, :3])
    M = A[:3, :3] - np.eye(3)
    return np.outer(r, r), M.T @ M


def _grade(value, good, ok, higher_is_better=True):
    if higher_is_better:
        return GOOD if value >= good else OK if value >= ok else BAD
    return GOOD if value <= good else OK if value <= ok else BAD


class Coverage:
    def __init__(self):
        self.G, self.C, self.Q = [], [], []        # T_base_gripper, T_camera_board, joints
        self._Srot = np.zeros((3, 3))
        self._Str = np.zeros((3, 3))
        self._npairs = 0
        self.solution = None                       # dict from _solve()

    # ---------------------------------------------------------------- bookkeeping
    def _recompute(self):
        self._Srot[:] = 0; self._Str[:] = 0; self._npairs = 0
        for i in range(len(self.G)):
            for j in range(i + 1, len(self.G)):
                a, b = _pair_terms(self.G[i], self.G[j])
                self._Srot += a; self._Str += b; self._npairs += 1

    def add(self, G, C, q):
        for Gi in self.G:
            a, b = _pair_terms(Gi, G)
            self._Srot += a; self._Str += b; self._npairs += 1
        self.G.append(np.asarray(G)); self.C.append(np.asarray(C)); self.Q.append(np.asarray(q))
        self.solution = self._solve()

    def pop(self):
        if self.G:
            self.G.pop(); self.C.pop(); self.Q.pop()
            self._recompute()
            self.solution = self._solve()

    def __len__(self):
        return len(self.G)

    # ---------------------------------------------------------------- coverage
    @staticmethod
    def _metrics(Srot, Str, n):
        if n == 0:
            return 0.0, 0.0, None, None
        wr, vr = np.linalg.eigh(Srot / n)          # ascending
        wt, vt = np.linalg.eigh(Str / n)
        rot = float(np.degrees(np.sqrt(max(wr[1], 0.0))))
        trans = float(np.sqrt(max(wt[0], 0.0)))
        return rot, trans, vr[:, 2], vt[:, 0]      # dominant rot axis, weakest trans dir

    def metrics(self):
        return self._metrics(self._Srot, self._Str, self._npairs)

    def metrics_with(self, G):
        Srot, Str, n = self._Srot.copy(), self._Str.copy(), self._npairs
        for Gi in self.G:
            a, b = _pair_terms(Gi, G)
            Srot += a; Str += b; n += 1
        return self._metrics(Srot, Str, n)

    # ---------------------------------------------------------------- consistency
    def _solve(self):
        if len(self.G) < MIN_SAMPLES_SOLVE:
            return None
        try:
            R, t = cv.calibrateHandEye([g[:3, :3] for g in self.G], [g[:3, 3] for g in self.G],
                                       [c[:3, :3] for c in self.C], [c[:3, 3] for c in self.C],
                                       method=cv.CALIB_HAND_EYE_PARK)
        except cv.error:
            return None
        if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
            return None                            # degenerate set (e.g. only wrist roll so far)
        X = np.eye(4); X[:3, :3] = R; X[:3, 3] = t.ravel()
        P = np.array([(g @ X @ c)[:3, 3] for g, c in zip(self.G, self.C)])
        resid = np.linalg.norm(P - np.median(P, axis=0), axis=1) * 1000
        return {"X": X, "spread_mm": float(np.mean(resid)), "resid_mm": resid}

    # ---------------------------------------------------------------- candidate
    def evaluate(self, G, res, q, speed_deg_s, fk):
        """Return (verdict, lines, why) for the current view. lines = [(text, grade)]."""
        lines, reasons_bad, reasons_ok = [], [], []

        # detection quality
        if res is None or not res.ok:
            lines.append(("Board: NOT DETECTED", BAD))
            return BAD, lines, "board not detected"
        n = res.n_corners
        g = _grade(n, MIN_CORNERS_GOOD, MIN_CORNERS_OK)
        lines.append((f"Corners: {n}", g))
        (reasons_bad if g == BAD else reasons_ok if g == OK else []).append("few corners")
        if res.reproj_px is not None:
            g = _grade(res.reproj_px, REPROJ_GOOD, REPROJ_OK, higher_is_better=False)
            lines.append((f"Reprojection: {res.reproj_px:.2f} px", g))
            (reasons_bad if g == BAD else reasons_ok if g == OK else []).append("blurry/noisy detection")
        Tcb = res.T_camera_charuco
        view = float(np.degrees(np.arccos(np.clip(abs(Tcb[2, 2]), 0, 1))))
        gv = GOOD if VIEW_GOOD[0] <= view <= VIEW_GOOD[1] else OK if VIEW_OK[0] <= view <= VIEW_OK[1] else BAD
        hint = " (tilt more)" if view < VIEW_GOOD[0] else " (too oblique)" if view > VIEW_GOOD[1] else ""
        lines.append((f"View angle: {view:.0f} deg{hint}", gv))
        if gv == BAD:
            reasons_bad.append("bad view angle")
        dist = float(np.linalg.norm(Tcb[:3, 3]))
        gd = GOOD if DIST_OK[0] <= dist <= DIST_OK[1] else BAD
        lines.append((f"Distance: {dist*100:.0f} cm", gd))
        if gd == BAD:
            reasons_bad.append("too close/far")

        # motion
        moving = speed_deg_s > MOVING_DEG_S
        lines.append(("Arm: MOVING" if moving else "Arm: still", BAD if moving else GOOD))

        # novelty vs saved samples
        if self.G:
            drot = min(np.degrees(np.linalg.norm(_rotvec(Gi[:3, :3].T @ G[:3, :3]))) for Gi in self.G)
            dpos = min(np.linalg.norm(Gi[:3, 3] - G[:3, 3]) for Gi in self.G)
            similar = drot < SIMILAR_ROT_DEG and dpos < SIMILAR_POS_M
            lines.append((f"Nearest saved: {drot:.0f} deg / {dpos*100:.1f} cm", BAD if similar else GOOD))
            if similar:
                reasons_bad.append("too similar to a saved pose")

        # gain in coverage
        gain_txt, gain_good = "", False
        if len(self.G) >= 2:
            rot0, tr0, _, _ = self.metrics()
            rot1, tr1, _, _ = self.metrics_with(G)
            d_rot, d_tr = rot1 - rot0, tr1 - tr0
            weakest_is_rot = (rot0 / ROT_GOOD) < (tr0 / TRANS_GOOD)
            # "improves" = raises the weakest metric by > 1 % of its current value
            gain_good = (d_rot > 0.01 * max(rot0, 1.0)) if weakest_is_rot else (d_tr > 0.01 * max(tr0, 0.02))
            gain_txt = f"Adds: ROT {d_rot:+.1f} deg, TRANS {d_tr:+.3f}"
            lines.append((gain_txt, GOOD if gain_good else OK))

        if reasons_bad:
            return BAD, lines, ", ".join(reasons_bad)
        if len(self.G) >= 2 and not gain_good:
            return OK, lines, "valid, but adds little new coverage"
        if reasons_ok:
            return OK, lines, ", ".join(reasons_ok)
        return GOOD, lines, "improves coverage" if len(self.G) >= 2 else "valid sample"

    # ---------------------------------------------------------------- advice
    def advice(self, q, fk):
        """Which joint to vary next, from the joint axes at pose q expressed in the gripper frame."""
        if len(self.G) < 2:
            return "Save a few varied poses to start"
        rot, trans, rot_axis, weak_t = self.metrics()
        need_rot = rot < ROT_GOOD
        need_tr = trans < TRANS_GOOD
        if not (need_rot or need_tr):
            return None
        G0 = fk(q)
        scores = {}
        for k in ADVICE_JOINTS:
            dq = np.zeros(5); dq[k] = 2.0
            axis = _rotvec(G0[:3, :3].T @ fk(q + dq)[:3, :3])
            nrm = np.linalg.norm(axis)
            if nrm < 1e-9:
                continue
            axis /= nrm
            # rotation: want an axis different from the one already dominating
            # translation: want an axis perpendicular to the weak direction
            ref = rot_axis if (need_rot and (rot / ROT_GOOD) <= (trans / TRANS_GOOD)) else weak_t
            scores[k] = 1.0 - float(axis @ ref) ** 2
        if not scores:
            return None
        best = max(scores, key=scores.get)
        what = "rotation variety" if need_rot else "translation coverage"
        return f"Need more {what}: vary {ADVICE_JOINTS[best]}"

    def status(self):
        rot, trans, _, _ = self.metrics()
        sol = self.solution
        n = len(self.G)
        ready = (n >= MIN_SAMPLES_READY and rot >= ROT_GOOD and trans >= TRANS_GOOD
                 and sol is not None and sol["spread_mm"] <= SPREAD_GOOD)
        return {
            "n": n,
            "rot": rot, "rot_grade": _grade(rot, ROT_GOOD, ROT_OK) if n >= 2 else BAD,
            "trans": trans, "trans_grade": _grade(trans, TRANS_GOOD, TRANS_OK) if n >= 2 else BAD,
            "spread": None if sol is None else sol["spread_mm"],
            "spread_grade": None if sol is None else _grade(sol["spread_mm"], SPREAD_GOOD, SPREAD_OK, False),
            "last_resid": None if sol is None else float(sol["resid_mm"][-1]),
            "last_outlier": False if sol is None else bool(
                sol["resid_mm"][-1] > max(OUTLIER_MM, 3 * np.median(sol["resid_mm"]))),
            "solvable": n < MIN_SAMPLES_SOLVE or sol is not None,
            "worst": None if sol is None else (int(np.argmax(sol["resid_mm"])), float(np.max(sol["resid_mm"]))),
            "ready": ready,
        }
