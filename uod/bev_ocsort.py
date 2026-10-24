"""Real-time BEV OC-SORT Tracker + Fixed-Lag Circular Buffer Smoother.

Operates directly in the metric rectified Bird's-Eye View (BEV) plane:
* Pedestrians (cls == 0): tracked via 2D bottom-center projected to BEV + 2-phase
  adaptive square footprint (12-frame burn-in -> smooth physical prior).
* Vehicles (cls != 0): pre-filtered by kp_ok == True, tracked via 4 ground corners
  projected to BEV + incremental Procrustes rigid rectangle + trajectory heading.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from homography_rt import apply_homography
from .classes import is_vehicle, UNIFIED_VEHICLE, UNIFIED_PEDESTRIAN

# --------------------------------------------------------------------------- #
# Vehicle family classification for association gating
# --------------------------------------------------------------------------- #

def _vehicle_family(unified_cls: int, source_cls: int) -> int:
    """Return vehicle family based on unified class and source class.
    
    Args:
        unified_cls: 0=pedestrian, 2=vehicle
        source_cls: original COCO class
        
    Returns:
        0 = pedestrian
        1 = bicycle/motorcycle
        2 = car/bus/truck
    """
    if unified_cls == UNIFIED_PEDESTRIAN:
        return 0
    if is_vehicle(source_cls):
        # For unified vehicles, check original COCO class to distinguish families
        if source_cls in (1, 3):  # bicycle, motorcycle
            return 1
        return 2  # car, bus, truck
    return unified_cls

# --------------------------------------------------------------------------- #
# Precomputed O(1) Savitzky-Golay Filter (Position + Analytical Velocity)
# --------------------------------------------------------------------------- #

class FastSavGol:
    """Zero-allocation Savitzky-Golay evaluator for position and 1st derivative."""

    def __init__(self, max_win: int = 15, poly: int = 2, dt: float = 1.0 / 30.0):
        self.max_win = max(3, int(max_win))
        self.poly = int(poly)
        self.dt = float(dt)
        self._cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for n in range(self.poly + 2, self.max_win + 1):
            for idx in range(n):
                self._cache[(n, idx)] = self._build_weights(n, idx)

    def _build_weights(self, n: int, eval_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        t = (np.arange(n, dtype=np.float64) - float(eval_idx)) * self.dt
        V = np.vander(t, N=self.poly + 1, increasing=True)  # [1, t, t^2]
        pinv = np.linalg.pinv(V)
        w_pos = pinv[0].copy()  # c0 = p(0)
        w_vel = pinv[1].copy()  # c1 = dp/dt(0)
        return w_pos, w_vel

    def eval(self, series: np.ndarray, eval_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate smoothed position and velocity at ``eval_idx`` of ``series`` (N, D)."""
        n = len(series)
        if n == 0:
            return np.zeros(2), np.zeros(2)
        if n == 1:
            return series[0].copy(), np.zeros_like(series[0])
        eval_idx = int(np.clip(eval_idx, 0, n - 1))
        if n < self.poly + 2:
            vel = (series[-1] - series[0]) / max((n - 1) * self.dt, 1e-6)
            return series[eval_idx].copy(), vel
        key = (n, eval_idx)
        if key not in self._cache:
            self._cache[key] = self._build_weights(n, eval_idx)
        w_pos, w_vel = self._cache[key]
        return w_pos @ series, w_vel @ series


# --------------------------------------------------------------------------- #
# Data Structures
# --------------------------------------------------------------------------- #

@dataclass
class BEVObservation:
    """Filtered detection already projected into normalized BEV (bev_unit gauge)."""
    cls: int
    source_cls: int
    conf: float
    box_xyxy: np.ndarray        # (4,) in image pixels
    kpts: np.ndarray            # (8, 2) in image pixels
    ground: np.ndarray          # (4, 2) in image pixels
    kp_ok: bool
    bev_center: np.ndarray      # (2,) in bev_unit coordinates
    bev_quad_raw: np.ndarray    # (4, 2) in bev_unit coordinates (centered or posed)
    obs_scale: float            # observed scale in bev_unit
    ground_indices: Optional[List[int]] = None   # which 4 of the 8 kpts are ground


# --------------------------------------------------------------------------- #
# OC-SORT Track in BEV Space
# --------------------------------------------------------------------------- #

class BEVKalmanTrack:
    """Single object track in BEV metric space with OC-SORT (ORU + OCM)."""

    _next_id = 1

    def __init__(self, obs: BEVObservation, frame_idx: int, dt: float,
                 lock_after_veh: int = 20, lock_after_ped: int = 12):
        self.id = BEVKalmanTrack._next_id
        BEVKalmanTrack._next_id += 1

        self.cls = int(obs.cls)
        self.class_votes: Dict[int, int] = {self.cls: 1}
        self.source_cls = int(obs.source_cls)
        self.vehicle_family = _vehicle_family(self.cls, self.source_cls)
        self.veh_scale = (
            max(float(obs.obs_scale), 1e-6) if self.cls != 0 else 0.0
        )
        self.dt = float(dt)

        # Decoupled 2D Constant-Velocity Kalman state: pos=(X, Y), vel=(Vx, Vy)
        # Covariance P is identical for X and Y -> stored as a single 2x2 matrix!
        self.pos = obs.bev_center.copy()
        self.vel = np.zeros(2, dtype=np.float64)
        self.P = np.array([[0.05, 0.0],
                           [0.0, 1.0]], dtype=np.float64)

        # Process & measurement noise in bev_unit
        self.q_pos = 0.002 if self.cls == 0 else 0.005
        self.q_vel = 0.08 if self.cls == 0 else 0.20
        self.r_meas = 0.015 if self.cls == 0 else 0.010

        # OC-SORT observation memory for ORU and OCM
        self.last_obs_pos = obs.bev_center.copy()
        self.last_obs_frame = int(frame_idx)
        self.obs_history: deque = deque(maxlen=10)  # (frame_idx, bev_center)
        self.obs_history.append((frame_idx, obs.bev_center.copy()))

        # Lifecycle counters
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.exited = False

        # Last 2D/3D visual payload
        self.box_xyxy = obs.box_xyxy.copy()
        self.kpts = obs.kpts.copy()
        self.ground = obs.ground.copy()
        self.kp_ok = bool(obs.kp_ok)
        self.ground_indices = (list(obs.ground_indices)
                               if obs.ground_indices is not None else [0, 1, 2, 3])

        # Pedestrian 2D box & 3D height state
        w2d = max(obs.box_xyxy[2] - obs.box_xyxy[0], 1.0)
        h2d = max(obs.box_xyxy[3] - obs.box_xyxy[1], 1.0)
        self.ped_wh_ema = (float(w2d), float(h2d))
        self.ped_height = float(h2d * 0.85)

        # Footprint shape state (Pedestrian 2-phase vs Vehicle Procrustes)
        self.lock_after_ped = int(lock_after_ped)
        self.lock_after_veh = int(lock_after_veh)
        self.phys_scale_ped = 0.19
        self.ped_scale = self.phys_scale_ped
        self.ped_W = 1.0
        self.ped_locked = False
        self.ped_center_ema = obs.bev_center.copy()

        # Vehicle rigid shape S (4, 2) centered in bev_unit
        Q0 = obs.bev_quad_raw - obs.bev_quad_raw.mean(axis=0)
        self.veh_S = self._rect_shape(Q0)
        self.veh_W = 1.0
        self.veh_locked = False
        self.veh_rms: deque = deque(maxlen=30)
        self.veh_heading = self._estimate_kabsch_angle(Q0, self.veh_S)
        
        # Store BEV ground footprint for largest-footprint selection
        self._bev_ground = obs.bev_quad_raw.copy() if obs.bev_quad_raw.shape == (4, 2) else None

    # ------------------------------------------------------------------ #
    def _kf_predict_step(self):
        dt = self.dt
        self.pos = self.pos + self.vel * dt
        # P = F P F^T + Q
        p00, p01, p10, p11 = self.P[0, 0], self.P[0, 1], self.P[1, 0], self.P[1, 1]
        self.P[0, 0] = p00 + dt * (p10 + p01) + (dt * dt) * p11 + self.q_pos
        self.P[0, 1] = p01 + dt * p11
        self.P[1, 0] = p10 + dt * p11
        self.P[1, 1] = p11 + self.q_vel

    def _kf_update_step(self, z: np.ndarray):
        y = z - self.pos
        s = self.P[0, 0] + self.r_meas
        k0 = self.P[0, 0] / s
        k1 = self.P[1, 0] / s
        self.pos = self.pos + k0 * y
        self.vel = self.vel + k1 * y
        p00, p01 = self.P[0, 0], self.P[0, 1]
        self.P[0, 0] -= k0 * p00
        self.P[0, 1] -= k0 * p01
        self.P[1, 0] -= k1 * p00
        self.P[1, 1] -= k1 * p01

    def predict(self) -> np.ndarray:
        """Advance Kalman state by one frame; damp velocity if coasting."""
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.vel *= 0.85  # prevent runaway drift during occlusion
        self._kf_predict_step()
        return self.pos

    def ocm_direction(self, dt_frames: int = 3) -> Optional[np.ndarray]:
        """OC-SORT Observation-Centric Momentum unit vector from recent observations."""
        if len(self.obs_history) < 2:
            return None
        f_cur, p_cur = self.obs_history[-1]
        best = self.obs_history[0]
        for f_old, p_old in reversed(self.obs_history):
            if f_cur - f_old >= dt_frames:
                best = (f_old, p_old)
                break
        d = p_cur - best[1]
        n = float(np.hypot(d[0], d[1]))
        if n < 0.04:  # stationary object has no momentum direction
            return None
        return d / n

    def update(self, obs: BEVObservation, frame_idx: int):
        """Update track with a matched BEV observation, running OC-SORT ORU if needed."""
        # 1. OC-SORT ORU (Observation-Centric Re-Update) across occlusion gaps
        gap = int(frame_idx - self.last_obs_frame)
        if self.time_since_update > 1 and gap > 1:
            # Rewind to last observed state and replay virtual linear observations
            p_start = self.last_obs_pos.copy()
            p_end = obs.bev_center
            self.pos = p_start.copy()
            self.vel = (p_end - p_start) / (gap * self.dt)
            for g in range(1, gap):
                self._kf_predict_step()
                z_virt = p_start + (p_end - p_start) * (g / float(gap))
                self._kf_update_step(z_virt)
            self._kf_predict_step()

        # 2. Standard Kalman update with current measurement
        self._kf_update_step(obs.bev_center)

        self.last_obs_pos = obs.bev_center.copy()
        self.last_obs_frame = int(frame_idx)
        self.obs_history.append((frame_idx, obs.bev_center.copy()))
        self.time_since_update = 0
        self.hits += 1

        # cls is LOCKED at track creation (unified class: 0=pedestrian, 2=vehicle)
        # No class voting - prevents ID switches when detector alternates subclasses

        # 4. Update 2D/3D visual geometry & footprint shape
        if self.cls == 0:
            self._update_pedestrian(obs)
        else:
            self._update_vehicle(obs)

    # ------------------------------------------------------------------ #
    def _update_pedestrian(self, obs: BEVObservation):
        # 2D Box EMA anchored at bottom-center + synthetic rigid 3D cuboid
        x1, y1, x2, y2 = obs.box_xyxy
        cx = (x1 + x2) * 0.5
        cy_bottom = float(y2)
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)
        pw, ph = self.ped_wh_ema
        w = 0.25 * w + 0.75 * pw
        h = 0.25 * h + 0.75 * ph
        self.ped_wh_ema = (float(w), float(h))

        self.box_xyxy = np.array([cx - w * 0.5, cy_bottom - h,
                                  cx + w * 0.5, cy_bottom], dtype=np.float64)
        height = self.ped_height
        depth = max(w * 0.6, h * 0.4)
        kp = np.zeros((8, 2), dtype=np.float64)
        kp[0] = [cx - w * 0.5, cy_bottom]
        kp[1] = [cx + w * 0.5, cy_bottom]
        kp[2] = [cx + w * 0.5, cy_bottom - depth]
        kp[3] = [cx - w * 0.5, cy_bottom - depth]
        kp[4] = [cx - w * 0.5, cy_bottom - height]
        kp[5] = [cx + w * 0.5, cy_bottom - height]
        kp[6] = [cx + w * 0.5, cy_bottom - height - depth]
        kp[7] = [cx - w * 0.5, cy_bottom - height - depth]
        self.kpts = kp
        self.ground = kp[:4].copy()
        self.kp_ok = True

        # Causal EMA on BEV center
        alpha_c = 0.15 if self.ped_locked else 0.4
        self.ped_center_ema = alpha_c * obs.bev_center + (1.0 - alpha_c) * self.ped_center_ema

        # Two-phase scale estimation (Burn-in 12 obs -> Smooth physical prior)
        obs_scale = obs.obs_scale
        if not self.ped_locked:
            w_cnt = self.ped_W
            self.ped_scale = self.ped_scale + (obs_scale - self.ped_scale) / (w_cnt + 1.0)
            self.ped_W = w_cnt + 1.0
            if self.ped_W >= self.lock_after_ped:
                self.ped_locked = True
        else:
            target_scale = 0.5 * obs_scale + 0.5 * self.phys_scale_ped
            scale_alpha = 0.30
            self.ped_scale = scale_alpha * target_scale + (1.0 - scale_alpha) * self.ped_scale

    def _update_vehicle(self, obs: BEVObservation):
        # Visual payload: ALWAYS the model's own predicted cuboid for the
        # detection associated with this track. No artificial re-posing,
        # rotation or reconstruction of stored geometry -- that is what made
        # the camera-view cuboids look distorted. The "largest footprint"
        # selection happens upstream at detection level (dedup keeps the
        # detection with the largest BEV footprint per object per frame), so
        # the cuboid drawn on the camera panel is exactly the model's output.

        self.box_xyxy = obs.box_xyxy.copy()
        self.kpts = obs.kpts.copy()
        self.ground = obs.ground.copy()
        if obs.bev_quad_raw is not None and obs.bev_quad_raw.shape == (4, 2):
            self._bev_ground = obs.bev_quad_raw.copy()

        obs_scale = max(float(obs.obs_scale), 1e-6)
        old_scale = max(float(self.veh_scale), 1e-6)
        ratio = max(obs_scale / old_scale, old_scale / obs_scale)

        if ratio <= 1.8:
            self.veh_scale = 0.9 * old_scale + 0.1 * obs_scale
        
        self.kp_ok = True

        Q = obs.bev_quad_raw - obs.bev_quad_raw.mean(axis=0)
        S_use = self._rect_shape(self.veh_S)
        angle = self._estimate_kabsch_angle(Q, S_use)
        self.veh_heading = angle

        # Ignora l'aggiornamento della forma se il box 2D tocca i bordi del frame (oggetto troncato)
        W_img, H_img = 1920, 1080  # O passa le dimensioni reali del frame
        x1, y1, x2, y2 = obs.box_xyxy
        margin = 25  # pixel dal bordo
        is_truncated = (x1 <= margin or y1 <= margin or x2 >= W_img - margin or y2 >= H_img - margin)

        if not self.veh_locked and not is_truncated:
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            posed = S_use @ R.T
            _v1 = Q[2] - Q[0]
            _v2 = Q[3] - Q[1]
            qa = 0.5 * abs(_v1[0] * _v2[1] - _v1[1] * _v2[0])
            res = Q - posed
            rms = float(np.sqrt((res * res).mean()))
            med = float(np.median(self.veh_rms)) if len(self.veh_rms) >= 5 else None
            if qa > 0.1 and (med is None or rms <= 4.0 * med):
                Qc = Q @ R
                w = self.veh_W
                self.veh_S = self.veh_S + (Qc - self.veh_S) / (w + 1.0)
                self.veh_S -= self.veh_S.mean(axis=0)
                self.veh_W = w + 1.0
                self.veh_rms.append(rms)
                if self.veh_W >= self.lock_after_veh:
                    self.veh_locked = True
                    self.veh_S = self._rect_shape(self.veh_S)

    # ------------------------------------------------------------------ #
    def get_bev_footprint(self, bev_unit: float, center_override: Optional[np.ndarray] = None,
                          heading_override: Optional[float] = None) -> np.ndarray:
        """Return the posed (4, 2) footprint in H's gauge (multiplied by bev_unit)."""
        if self.cls == 0:
            c = self.ped_center_ema if center_override is None else center_override
            h = 0.5 * self.ped_scale
            S0 = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)
            return (S0 + c) * bev_unit
        else:
            c = self.pos if center_override is None else center_override
            th = self.veh_heading if heading_override is None else heading_override
            cs, sn = np.cos(th), np.sin(th)
            R = np.array([[cs, -sn], [sn, cs]])
            S_use = self._rect_shape(self.veh_S)
            return (S_use @ R.T + c) * bev_unit

    @staticmethod
    def _rect_shape(S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=np.float64)
        C = S - S.mean(axis=0)
        e = np.roll(C, -1, axis=0) - C
        u = 0.5 * (e[0] - e[2])
        v = 0.5 * (e[1] - e[3])
        q = v if (v @ v) >= (u @ u) else u
        nq = float(np.hypot(q[0], q[1]))
        if nq < 1e-9:
            return S
        q = q / nq
        p = np.array([-q[1], q[0]])
        along, perp = C @ q, C @ p
        half_l = float(np.mean(np.abs(along)))
        half_w = float(np.mean(np.abs(perp)))
        sa = np.where(along >= 0, 1.0, -1.0)
        sp = np.where(perp >= 0, 1.0, -1.0)
        return (sp[:, None] * half_w) * p[None, :] + (sa[:, None] * half_l) * q[None, :]

    @staticmethod
    def _estimate_kabsch_angle(Q: np.ndarray, S: np.ndarray) -> float:
        C = Q.T @ S
        U, _, Vt = np.linalg.svd(C)
        dsign = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
        R = U @ np.array([[1.0, 0.0], [0.0, dsign]]) @ Vt
        return float(np.arctan2(R[1, 0], R[0, 0]))


# --------------------------------------------------------------------------- #
# BEV OC-SORT Multi-Object Tracker
# --------------------------------------------------------------------------- #

class BEVOCSortTracker:
    """OC-SORT operating in normalized BEV coordinates with 2-stage matching."""

    def __init__(self, fps: float = 30.0, max_age: int = 25, min_hits: int = 3,
                 high_conf: float = 0.45, ocm_weight: float = 0.25,
                 edge_frac: float = 0.006):
        self.fps = float(fps) if fps > 0 else 30.0
        self.dt = 1.0 / self.fps
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.high_conf = float(high_conf)
        self.ocm_weight = float(ocm_weight)
        self.edge_frac = float(edge_frac)
        self.tracks: List[BEVKalmanTrack] = []

        # Maximum physically plausible displacement per second in bev_unit (~2.85m)
        # Pedestrians: ~1.8 bev_unit/s (~5.1 m/s = 18.5 km/h)
        # Vehicles:    ~9.0 bev_unit/s (~25.6 m/s = 92 km/h)
        self.max_speed_ped = 1.8
        self.max_speed_veh = 9.0

    def _max_gate_dist(self, trk: BEVKalmanTrack) -> float:
        vmax = self.max_speed_ped if trk.cls == 0 else self.max_speed_veh
        base_rad = 0.45 if trk.cls == 0 else 1.25
        # Allow gate to expand slightly with coasting gap, capped physically
        return base_rad + min(trk.time_since_update, 8) * (vmax * self.dt)

    def _cost_matrix(self, trks: List[BEVKalmanTrack],
                     obs_list: List[BEVObservation],
                     use_last_obs: bool = False) -> np.ndarray:
        M, N = len(trks), len(obs_list)
        if M == 0 or N == 0:
            return np.empty((M, N), dtype=np.float64)

        cost = np.full((M, N), 1e6, dtype=np.float64)
        obs_pos = np.array([o.bev_center for o in obs_list], dtype=np.float64)
        obs_cls = np.array([o.cls for o in obs_list], dtype=int)
        obs_source_cls = np.array([o.source_cls for o in obs_list], dtype=int)
        obs_scales = np.maximum(
            np.array([o.obs_scale for o in obs_list], dtype=np.float64),
            1e-6,
        )

        for i, t in enumerate(trks):
            ref_pos = t.last_obs_pos if use_last_obs else t.pos
            dists = np.linalg.norm(obs_pos - ref_pos[None, :], axis=1)
            gate = self._max_gate_dist(t)

            # Class compatibility: pedestrians (0) never match vehicles (!= 0)
            if t.cls == 0:
                # Pedestrians only associate with other pedestrians.
                valid = (obs_cls == 0) & (dists <= gate)
                c_row = dists / max(gate, 1e-6)
            else:
                # All vehicle subclasses (car, truck, bus, bicycle, motorcycle)
                # associate with each other since they share unified class.
                # No family-based gating - any vehicle can match any vehicle track.
                track_scale = max(float(t.veh_scale), 1e-6)

                # obs_scale is the sqrt of footprint area in BEV.
                # Relaxed for vehicles: same physical object can appear as
                # different subclasses with different footprint sizes.
                track_scale = max(float(t.veh_scale), 1e-6)
                scale_ratio = np.maximum(
                    obs_scales / track_scale,
                    track_scale / obs_scales,
                )
                # Soft scale penalty: penalize different sizes but don't hard-reject,
                # so tracks don't get stranded when the detector alternates subclasses.
                scale_thresh = 3.0 if t.cls == 2 else 1.8
                scale_ok = scale_ratio <= scale_thresh
                scale_penalty = np.where(scale_ok, 0.0, 0.5 * np.minimum(np.log(scale_ratio), 1.0))

                # Spatial gate remains hard (must be within reasonable distance)
                valid = (
                    (obs_cls != 0)
                    & (dists <= gate)
                )
                c_row = dists / max(gate, 1e-6)

                # Add scale penalty to cost
                c_row += scale_penalty

                # Prefer footprint with similar scale
                c_row += 0.15 * np.minimum(np.abs(np.log(obs_scales / track_scale)), 1.0)

            # Add OC-SORT OCM (Observation-Centric Momentum) angular penalty
            if not use_last_obs:
                v_dir = t.ocm_direction()
                if v_dir is not None:
                    diff = obs_pos - t.last_obs_pos[None, :]
                    dn = np.linalg.norm(diff, axis=1)
                    dir_valid = dn > 0.04
                    cos_sim = np.ones(N, dtype=np.float64)
                    cos_sim[dir_valid] = (diff[dir_valid] @ v_dir) / dn[dir_valid]
                    # Penalty in [0, ocm_weight]
                    angle_pen = 0.5 * (1.0 - cos_sim) * self.ocm_weight
                    c_row += angle_pen

            cost[i, valid] = c_row[valid]

        return cost

    def _associate(self, trks: List[BEVKalmanTrack],
                   obs_list: List[BEVObservation],
                   use_last_obs: bool = False) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not trks or not obs_list:
            return [], list(range(len(trks))), list(range(len(obs_list)))

        cost = self._cost_matrix(trks, obs_list, use_last_obs=use_last_obs)
        row_ind, col_ind = linear_sum_assignment(cost)

        matches, un_t, un_o = [], [], []
        matched_t, matched_o = set(), set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= 1.25:
                matches.append((r, c))
                matched_t.add(r)
                matched_o.add(c)

        for i in range(len(trks)):
            if i not in matched_t:
                un_t.append(i)
        for j in range(len(obs_list)):
            if j not in matched_o:
                un_o.append(j)
        return matches, un_t, un_o

    def _check_border_exit(self, trk: BEVKalmanTrack, img_wh: Tuple[int, int]) -> bool:
        """Real-time version of refine_v2._exit_state: close tracks leaving the frame."""
        if trk.time_since_update == 0 or len(trk.obs_history) < 2:
            return False
        W, H = img_wh
        m = max(4.0, self.edge_frac * float(np.hypot(W, H)))
        b = trk.box_xyxy
        at_edge = (b[0] <= m) or (b[2] >= W - m) or (b[1] <= m) or (b[3] >= H - m)
        return bool(at_edge and trk.time_since_update >= 3)

    def update(self, observations: List[BEVObservation], frame_idx: int,
               img_wh: Tuple[int, int]) -> List[BEVKalmanTrack]:
        # 1. Predict all active tracks in BEV
        for t in self.tracks:
            t.predict()

        # Split observations into high and low confidence (ByteTrack 2-stage)
        hi_obs = [o for o in observations if o.conf >= self.high_conf]
        lo_obs = [o for o in observations if o.conf < self.high_conf]

        # 2. Stage 1: Associate Kalman predictions + OCM with high-conf observations
        m1, un_t1, un_o1 = self._associate(self.tracks, hi_obs, use_last_obs=False)
        for ti, oi in m1:
            self.tracks[ti].update(hi_obs[oi], frame_idx)

        # 3. Stage 2: Associate remaining active tracks with low-conf observations
        rem_trks = [self.tracks[i] for i in un_t1 if self.tracks[i].time_since_update <= 4]
        rem_idx = [i for i in un_t1 if self.tracks[i].time_since_update <= 4]
        m2, un_t2_local, _ = self._associate(rem_trks, lo_obs, use_last_obs=False)
        matched_rem = set()
        for loc_ti, oi in m2:
            real_ti = rem_idx[loc_ti]
            self.tracks[real_ti].update(lo_obs[oi], frame_idx)
            matched_rem.add(real_ti)

        # 4. Stage 3: OC-SORT Last-Observation Recovery for still-unmatched tracks
        un_t2 = [i for i in un_t1 if i not in matched_rem]
        left_trks = [self.tracks[i] for i in un_t2]
        left_obs = [hi_obs[j] for j in un_o1]
        m3, _, un_o3_local = self._associate(left_trks, left_obs, use_last_obs=True)
        for loc_ti, loc_oi in m3:
            real_ti = un_t2[loc_ti]
            real_oi = un_o1[loc_oi]
            self.tracks[real_ti].update(hi_obs[real_oi], frame_idx)

        # 5. Spawn new tracks from unmatched high-conf observations
        unmatched_hi = [un_o1[j] for j in un_o3_local]
        for oi in unmatched_hi:
            self.tracks.append(BEVKalmanTrack(hi_obs[oi], frame_idx, self.dt))

        # 6. Prune dead or border-exited tracks
        kept = []
        for t in self.tracks:
            if t.time_since_update > self.max_age:
                continue
            if self._check_border_exit(t, img_wh):
                continue
            # Drop tentative tracks that missed their second observation immediately
            if t.hits < self.min_hits and t.time_since_update > 4:
                continue
            kept.append(t)
        self.tracks = kept
        return self.tracks


# --------------------------------------------------------------------------- #
# Option B: Fixed-Lag Circular Buffer Smoother (Window W=11, Lag L=5)
# --------------------------------------------------------------------------- #

class FixedLagBuffer:
    """Real-time Fixed-Lag Circular Buffer (Option B).

    Holds the last ``window`` (e.g. 11) RenderStates in memory and emits frame
    ``t - lag`` (e.g. t - 5). Using the 5-frame look-ahead, it performs in
    real-time what previously required offline whole-clip passes:
    1. Flicker pruning & coast-tail trimming (tracks that never confirm or never
       recover in the next 5 frames are suppressed before rendering).
    2. Linear gap interpolation of 2D boxes, 3D keypoints, and BEV positions
       across short occlusion gaps (<= window frames).
    3. Zero-phase Savitzky-Golay smoothing of BEV trajectories and exact
       analytical velocity extraction (Vx, Vy in bev_unit/s).
    4. Stationary vehicle freeze (_still_flatten) & trajectory heading alignment.
    """

    def __init__(self, window: int = 11, poly: int = 2, fps: float = 30.0,
                 min_hits: int = 3, hold_still: bool = True, vel_heading: bool = True):
        self.window = max(3, int(window) | 1)  # ensure odd window (e.g. 11)
        self.lag = self.window // 2            # e.g. 5 frames
        self.fps = float(fps) if fps > 0 else 30.0
        self.dt = 1.0 / self.fps
        self.min_hits = int(min_hits)
        self.hold_still = bool(hold_still)
        self.vel_heading = bool(vel_heading)

        self.savgol = FastSavGol(max_win=self.window, poly=poly, dt=self.dt)
        self.buffer: deque = deque(maxlen=self.window)

        # Stationary vehicle anchor memory: tid -> frozen_bev_pos
        self._parked_anchor: Dict[int, Tuple[np.ndarray, float]] = {}

    def push(self, state, tracker_map: Dict[int, BEVKalmanTrack]):
        """Push frame ``t`` into the circular buffer; return smoothed frame ``t - lag`` (or None during warm-up)."""
        self.buffer.append(state)
        if len(self.buffer) < self.window:
            return None
        return self._emit_at_index(self.lag, tracker_map)

    def flush_remaining(self, tracker_map: Dict[int, BEVKalmanTrack]) -> List:
        """At end-of-video, flush the remaining ``lag`` frames still in the buffer."""
        out = []
        if not self.buffer:
            return out
        n = len(self.buffer)
        start_idx = (self.lag + 1) if n == self.window else 0
        for idx in range(start_idx, n):
            out.append(self._emit_at_index(idx, tracker_map))
        self.buffer.clear()
        return out

    # ------------------------------------------------------------------ #
    def _emit_at_index(self, target_idx: int, tracker_map: Dict[int, BEVKalmanTrack]):
        buf_list = list(self.buffer)
        target_state = buf_list[target_idx]
        unit = float(target_state.bev_unit or 1.0)
        if unit <= 1e-9:
            return target_state

        # 1. Index all track views across the circular window: tid -> [(buf_i, view)]
        track_series: Dict[int, List[Tuple[int, object]]] = {}
        for bi, st in enumerate(buf_list):
            for v in st.tracks:
                track_series.setdefault(v.id, []).append((bi, v))

        # 2. Interpolate short coasting gaps inside the window if bounded by real observations
        self._interpolate_window_gaps(buf_list, track_series, target_idx)

        # 3. Filter and zero-phase smooth each track present in target_state
        valid_views = []
        for v in target_state.tracks:
            tid = v.id
            series = track_series.get(tid, [])
            trk_obj = tracker_map.get(tid)

            # Check if track ever confirms or is just an unrecovered coasting tail
            max_hits = max((getattr(vw, "hits", 1) for _, vw in series), default=1)
            obs_after = any(bi >= target_idx and vw.time_since_update == 0 for bi, vw in series)

            if max_hits < self.min_hits:
                continue  # Flicker suppression: died before reaching min_hits
            if v.time_since_update > 0 and not obs_after:
                continue  # Coast-tail trimming: never re-observed in look-ahead

            # Extract BEV center series across the window
            pos_arr = np.array([vw.bev_pos for _, vw in series], dtype=np.float64)
            loc_idx = next((i for i, (bi, _) in enumerate(series) if bi == target_idx), len(series) - 1)

            # Zero-phase Savitzky-Golay position + analytical velocity
            sm_pos, sm_vel = self.savgol.eval(pos_arr, loc_idx)
            speed_lps = float(np.hypot(sm_vel[0], sm_vel[1]))  # in bev_unit / second

            heading_override = None
            if v.cls != 0:
                # Vehicle stationary hold & trajectory heading (from refine_v2)
                disp = float(np.linalg.norm(pos_arr[-1] - pos_arr[0])) if len(pos_arr) >= 2 else 0.0
                is_still = (speed_lps < 0.35) and (disp < 0.18)

                if self.hold_still and is_still:
                    if tid not in self._parked_anchor:
                        med_p = np.median(pos_arr, axis=0)
                        h_ang = trk_obj.veh_heading if trk_obj is not None else 0.0
                        self._parked_anchor[tid] = (med_p, h_ang)
                    anchor_p, anchor_h = self._parked_anchor[tid]
                    if float(np.linalg.norm(sm_pos - anchor_p)) < 0.25:
                        sm_pos = anchor_p.copy()
                        sm_vel = np.zeros(2, dtype=np.float64)
                        heading_override = anchor_h
                    else:
                        self._parked_anchor[tid] = (sm_pos.copy(), trk_obj.veh_heading if trk_obj else 0.0)
                else:
                    self._parked_anchor.pop(tid, None)
                    # Trajectory heading when moving straight enough
                    if self.vel_heading and speed_lps >= 0.8 and disp >= 0.35:
                        path_len = float(np.sum(np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)))
                        if disp >= 0.92 * max(path_len, 1e-6) and trk_obj is not None:
                            vel_ang = float(np.arctan2(sm_vel[1], sm_vel[0]))
                            # Align vehicle length axis with velocity vector (modulo pi)
                            S_rect = trk_obj._rect_shape(trk_obj.veh_S)
                            e = np.roll(S_rect, -1, axis=0) - S_rect
                            u_ax, v_ax = 0.5 * (e[0] - e[2]), 0.5 * (e[1] - e[3])
                            q_ax = v_ax if (v_ax @ v_ax) >= (u_ax @ u_ax) else u_ax
                            axis_ang = float(np.arctan2(q_ax[1], q_ax[0]))
                            target_th = vel_ang - axis_ang
                            diff_th = np.arctan2(np.sin(2.0 * (target_th - trk_obj.veh_heading)),
                                                 np.cos(2.0 * (target_th - trk_obj.veh_heading))) * 0.5
                            if abs(np.degrees(diff_th)) <= 45.0:
                                heading_override = trk_obj.veh_heading + diff_th

            v.bev_pos = sm_pos
            v.bev_vel = sm_vel
            if trk_obj is not None:
                v.cls = trk_obj.cls
                v.bev_quad = trk_obj.get_bev_footprint(unit, center_override=sm_pos,
                                                       heading_override=heading_override)
            valid_views.append(v)

        target_state.tracks = valid_views
        return target_state

    @staticmethod
    def _interpolate_window_gaps(buf_list: List, track_series: Dict[int, List[Tuple[int, object]]],
                                 target_idx: int):
        """If target_idx is a coasting frame bracketed by real observations in the window, interpolate."""
        for tid, series in track_series.items():
            # Find if target_idx is present and currently coasting
            target_entry = next(((bi, vw) for bi, vw in series if bi == target_idx), None)
            if target_entry is None or target_entry[1].time_since_update == 0:
                continue

            before = [(bi, vw) for bi, vw in series if bi < target_idx and vw.time_since_update == 0]
            after = [(bi, vw) for bi, vw in series if bi > target_idx and vw.time_since_update == 0]
            if not before or not after:
                continue

            b_idx, va = before[-1]
            a_idx, vb = after[0]
            w = float(target_idx - b_idx) / float(max(a_idx - b_idx, 1))
            vt = target_entry[1]

            vt.box_xyxy = va.box_xyxy * (1.0 - w) + vb.box_xyxy * w
            vt.bev_pos = va.bev_pos * (1.0 - w) + vb.bev_pos * w
            if va.kpts.shape == (8, 2) and vb.kpts.shape == (8, 2):
                vt.kpts = va.kpts * (1.0 - w) + vb.kpts * w
            if va.ground.shape == (4, 2) and vb.ground.shape == (4, 2):
                vt.ground = va.ground * (1.0 - w) + vb.ground * w
            vt.time_since_update = 0  # healed by look-ahead interpolation