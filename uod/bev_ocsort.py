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
# Funzione per controllare se un BB2D di un veicolo tocca il bordo del frame
# --------------------------------------------------------------------------- #
def _is_touching_border(box_xyxy: np.ndarray, img_wh: tuple, margin: float = 15.0) -> bool:
    """Restituisce True se il box 2D tocca uno dei bordi del frame."""
    if box_xyxy is None or len(box_xyxy) < 4 or img_wh is None:
        return False
    W, H = float(img_wh[0]), float(img_wh[1])
    margin = max(float(margin), 0.01 * float(np.hypot(W, H)))
    x1, y1, x2, y2 = float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])
    return (x1 <= margin) or (y1 <= margin) or (x2 >= W - margin) or (y2 >= H - margin)


def _is_exiting_frame(box_xyxy: np.ndarray, vel_2d: np.ndarray,
                      img_wh: Optional[Tuple[int, int]]) -> bool:
    """True only when a box is at an image edge and moving out through it."""
    if box_xyxy is None or vel_2d is None or img_wh is None:
        return False
    W, H = float(img_wh[0]), float(img_wh[1])
    if W <= 0.0 or H <= 0.0:
        return False
    margin = max(2.0, 0.003 * float(np.hypot(W, H)))
    x1, y1, x2, y2 = map(float, box_xyxy[:4])
    vx, vy = map(float, vel_2d[:2])
    return bool(
        (x1 <= margin and vx < -0.5)
        or (x2 >= W - margin and vx > 0.5)
        or (y1 <= margin and vy < -0.5)
        or (y2 >= H - margin and vy > 0.5)
    )

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
    box_bottom_bev: Optional[np.ndarray] = None  # (2,) bottom-center 2D projected in bev_unit
    shape_ok: bool = True                        # safe to learn/show the rigid BEV footprint
    border_truncated: bool = False               # box touches the image boundary


# --------------------------------------------------------------------------- #
# OC-SORT Track in BEV Space
# --------------------------------------------------------------------------- #

class BEVKalmanTrack:
    """Single object track in BEV metric space with OC-SORT (ORU + OCM)."""
    _next_id = 1

    def __init__(self, obs: BEVObservation, frame_idx: int, dt: float,
                 lock_after_veh: int = 20, lock_after_ped: int = 12, img_wh: Optional[Tuple[int, int]] = None):
        self.id = BEVKalmanTrack._next_id
        BEVKalmanTrack._next_id += 1
        self.cls = int(obs.cls)
        self.class_votes: Dict[int, int] = {self.cls: 1}
        self.source_cls = int(obs.source_cls)
        self.source_cls_votes: Dict[int, int] = {self.source_cls: 1} if self.source_cls >= 0 else {}
        self.vehicle_family = _vehicle_family(self.cls, self.source_cls)
        self.veh_scale = (
            max(float(obs.obs_scale), 1e-6) if self.cls != 0 else 0.0
        )
        self.dt = float(dt)

        # Decoupled 2D Constant-Velocity Kalman state: pos=(X, Y), vel=(Vx, Vy)
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
        self.obs_history: deque = deque(maxlen=12)  # (frame_idx, bev_center)
        self.obs_history.append((int(frame_idx), obs.bev_center.copy()))
        self.confirmed_dir: Optional[np.ndarray] = None

        # 2D image-space tracking memory (for border-exit & kp_ok=False maintenance)
        self.last_obs_box = obs.box_xyxy.copy()
        self.box_history: deque = deque(maxlen=12)  # (frame_idx, box_xyxy)
        self.box_history.append((int(frame_idx), obs.box_xyxy.copy()))
        self.vel_2d = np.zeros(2, dtype=np.float64)
        self.last_box_bottom_bev = (
            obs.box_bottom_bev.copy()
            if obs.box_bottom_bev is not None
            else obs.bev_center.copy()
        )

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
        self.shape_valid = bool(obs.shape_ok)
        self.bev_visible = bool(self.cls == 0 or self.shape_valid)
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
        if (self.shape_valid and obs.bev_quad_raw is not None
                and obs.bev_quad_raw.shape == (4, 2)):
            Q0 = obs.bev_quad_raw - obs.bev_quad_raw.mean(axis=0)
            self.veh_S = self._rect_shape(Q0)
            self.veh_heading = self._estimate_kabsch_angle(Q0, self.veh_S)
            self._bev_ground = obs.bev_quad_raw.copy()
        else:
            self.veh_S = np.array([[-0.2, -0.4], [0.2, -0.4], [0.2, 0.4], [-0.2, 0.4]], dtype=np.float64)
            self.veh_heading = 0.0
            self._bev_ground = None
        self.veh_W = 1.0
        self.veh_locked = False
        self.veh_rms: deque = deque(maxlen=30)

        # Stato di troncamento al bordo per consentire l'espansione del box 3D in ingresso
        self.img_wh = img_wh
        self.touching_border = _is_touching_border(obs.box_xyxy, img_wh)
        self.border_truncated = bool(self.touching_border or obs.border_truncated)
        self.full_view_hits = 0

        # Per i pedoni costruisci subito alla nascita il cuboide 3D sintetico non ruotato
        if self.cls == 0:
            self._update_pedestrian(obs)

    # ------------------------------------------------------------------ #
    def _kf_predict_step(self):
        dt = self.dt
        self.pos = self.pos + self.vel * dt
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
        if self.cls != 0:
            shift = self.vel_2d * float(self.time_since_update)
            predicted_box = self.box_xyxy + np.array(
                [shift[0], shift[1], shift[0], shift[1]], dtype=np.float64)
            self.bev_visible = bool(
                self.shape_valid
                and not _is_exiting_frame(predicted_box, self.vel_2d, self.img_wh)
            )
        return self.pos

    def ocm_direction(self, dt_frames: int = 3) -> Optional[np.ndarray]:
        """Direzione di moto BEV consolidata (immune al jitter del singolo frame pre-incrocio)."""
        if self.confirmed_dir is not None:
            return self.confirmed_dir
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
        if n < 0.05:
            return None
        return d / n

    def update(self, obs: BEVObservation, frame_idx: int):
        """Update track with a matched BEV observation, running OC-SORT ORU if needed."""
        cur_bb = obs.box_bottom_bev if obs.box_bottom_bev is not None else obs.bev_center
        delta_bev = cur_bb - self.last_box_bottom_bev

        self.touching_border = _is_touching_border(obs.box_xyxy, self.img_wh)
        if self.touching_border:
            self.border_truncated = True
            self.full_view_hits = 0

        # Rileva se un veicolo/motorino ha subito una rotazione/distorsione anomala del BB 3D
        # rispetto al movimento fluido del suo box 2D:
        use_3d_kpts = bool(obs.shape_ok and obs.kp_ok and obs.kpts.shape[0] >= 8
                           and obs.bev_quad_raw.shape == (4, 2))
        self._expanding_from_border = False
        if self.cls != 0 and use_3d_kpts: # se l'auto si sta espandendo, sblocchiamo l'anti-jitter
            s_trk = max(float(self.veh_scale), 1e-6)
            s_obs = max(float(obs.obs_scale), 1e-6)
            if not self.shape_valid:
                self._expanding_from_border = True
            elif self.border_truncated and s_obs > s_trk * 1.08 and (s_obs / s_trk) <= 3.5:
                self._expanding_from_border = True

        if self.cls != 0 and use_3d_kpts and self.hits >= 2 and not self._expanding_from_border:
            jump_3d_vs_2d = float(np.linalg.norm((obs.bev_center - self.last_obs_pos) - delta_bev))
            s_trk = max(float(self.veh_scale), 1e-6)
            s_obs = max(float(obs.obs_scale), 1e-6)
            s_ratio = max(s_obs / s_trk, s_trk / s_obs)
            # Se il centro 3D è saltato rispetto al box 2D (> 0.22 bev_unit) o la scala 3D
            # è esplosa/collassata (> 1.40), usa il box 2D per la posizione e congela il 3D!
            if jump_3d_vs_2d > 0.22 or s_ratio > 1.40:
                use_3d_kpts = False

        if self.cls != 0 and not use_3d_kpts:
            eff_center = self.last_obs_pos + delta_bev
        elif self.cls == 0 and self.hits >= 2:
            # Limita lo spostamento a 0.14 per ogni frame trascorso.
            step_vec = obs.bev_center - self.last_obs_pos
            step_norm = float(np.linalg.norm(step_vec))
            gap = max(1, int(self.time_since_update))
            max_step = 0.14 * gap

            if step_norm > max_step:
                eff_center = self.last_obs_pos + step_vec * (max_step / step_norm)
            else:
                eff_center = obs.bev_center.copy()
        else:
            eff_center = obs.bev_center.copy()

        self.last_box_bottom_bev = cur_bb.copy()

        # 1. Aggiorna velocità 2D immagine e direzione BEV consolidata (confirmed_dir)
        cur_box = np.asarray(obs.box_xyxy, dtype=np.float64)
        if len(self.box_history) >= 1:
            f_ref, b_ref = self.box_history[max(0, len(self.box_history) - 4)]
            dt_f = max(1, int(frame_idx) - int(f_ref))
            c_cur = np.array([(cur_box[0] + cur_box[2]) * 0.5, (cur_box[1] + cur_box[3]) * 0.5])
            c_ref = np.array([(b_ref[0] + b_ref[2]) * 0.5, (b_ref[1] + b_ref[3]) * 0.5])
            self.vel_2d = 0.6 * self.vel_2d + 0.4 * ((c_cur - c_ref) / float(dt_f))
        self.box_history.append((int(frame_idx), cur_box.copy()))
        self.last_obs_box = cur_box.copy()

        min_disp = 0.12 if self.cls == 0 else 0.22
        for f_old, p_old in reversed(self.obs_history):
            if int(frame_idx) - int(f_old) < 3:
                continue
            delta = eff_center - p_old
            norm = float(np.linalg.norm(delta))
            if norm >= min_disp:
                unit_d = delta / norm
                if self.confirmed_dir is None:
                    self.confirmed_dir = unit_d
                else:
                    blended = 0.80 * self.confirmed_dir + 0.20 * unit_d
                    bn = float(np.linalg.norm(blended))
                    if bn > 1e-6:
                        self.confirmed_dir = blended / bn
                break

        # 2. OC-SORT ORU (Observation-Centric Re-Update) across occlusion gaps
        gap = int(frame_idx - self.last_obs_frame)
        if self.time_since_update > 1 and gap > 1:
            p_start = self.last_obs_pos.copy()
            p_end = eff_center
            self.pos = p_start.copy()
            self.vel = (p_end - p_start) / (gap * self.dt)
            for g in range(1, gap):
                self._kf_predict_step()
                z_virt = p_start + (p_end - p_start) * (g / float(gap))
                self._kf_update_step(z_virt)
            self._kf_predict_step()

        # 3. Standard Kalman update with eff_center
        self._kf_update_step(eff_center)
        self.last_obs_pos = eff_center.copy()
        self.last_obs_frame = int(frame_idx)
        self.obs_history.append((int(frame_idx), eff_center.copy()))
        self.time_since_update = 0
        self.hits += 1

        # Aggiorna il voto sulla classe COCO originale
        if self.cls != 0 and int(obs.source_cls) >= 0:
            sc = int(obs.source_cls)
            self.source_cls_votes[sc] = self.source_cls_votes.get(sc, 0) + 1
            dom_sc = max(self.source_cls_votes.items(), key=lambda kv: kv[1])[0]
            self.source_cls = dom_sc
            self.vehicle_family = _vehicle_family(self.cls, dom_sc)

        # 4. Update 2D/3D visual geometry & footprint shape
        if self.cls == 0:
            self._update_pedestrian(obs, center_bev=eff_center)
        else:
            if use_3d_kpts:
                self._update_vehicle(obs)
            else:
                # La quality gate vale solo per la BEV: sulla camera mostra i
                # keypoint correnti del detector anche quando la shape è scartata.
                old_b = self.box_xyxy
                new_b = obs.box_xyxy
                dx = ((new_b[0] + new_b[2]) - (old_b[0] + old_b[2])) * 0.5
                dy = ((new_b[1] + new_b[3]) - (old_b[1] + old_b[3])) * 0.5
                shift_2d = np.array([dx, dy], dtype=np.float64)
                if obs.kpts.shape == (8, 2):
                    self.kpts = obs.kpts.copy()
                    self.kp_ok = bool(obs.kp_ok)
                else:
                    if self.kpts.shape == (8, 2):
                        self.kpts = self.kpts + shift_2d
                if obs.ground.shape == (4, 2):
                    self.ground = obs.ground.copy()
                elif not obs.shape_ok:
                    self.ground = np.zeros((0, 2), dtype=np.float64)
                elif self.ground.shape == (4, 2):
                    self.ground = self.ground + shift_2d
                self.box_xyxy = new_b.copy()

        if self.cls != 0:
            self.bev_visible = bool(
                self.shape_valid
                and not _is_exiting_frame(self.box_xyxy, self.vel_2d, self.img_wh)
            )
        else:
            self.bev_visible = True

    # ------------------------------------------------------------------ #
    def _update_pedestrian(self, obs: BEVObservation, center_bev: Optional[np.ndarray] = None):
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
        center_bev = obs.bev_center if center_bev is None else center_bev
        self.ped_center_ema = alpha_c * center_bev + (1.0 - alpha_c) * self.ped_center_ema

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
        """Aggiorna la forma rigida veicolo S e l'heading cercando il miglior shift ciclico
        dei 4 vertici (impedisce flip di 90°/180° quando i keypoint del motorino ruotano)."""
        if obs.bev_quad_raw.shape != (4, 2):
            self.box_xyxy = obs.box_xyxy.copy()
            return

        Q = obs.bev_quad_raw - obs.bev_quad_raw.mean(axis=0)
        S_rect = self._rect_shape(self.veh_S)

        # Cerca la migliore permutazione ciclica k in {0, 1, 2, 3} per evitare flip di 90°/180°
        best_k = 0
        best_ang = self.veh_heading
        best_res = float("inf")
        for k in range(4):
            Qk = np.roll(Q, -k, axis=0)
            ang = self._estimate_kabsch_angle(Qk, S_rect)
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]], dtype=np.float64)
            posed_k = S_rect @ R.T
            res_k = float(np.sqrt(((Qk - posed_k) ** 2).mean()))
            if res_k < best_res:
                best_res = res_k
                best_k = k
                best_ang = ang

        Q_aligned = np.roll(Q, -best_k, axis=0)

        # Differenza angolare modulo pi/2 (simmetria rettangolare) rispetto all'heading precedente
        d_ang = float(np.arctan2(np.sin(2.0 * (best_ang - self.veh_heading)),
                                 np.cos(2.0 * (best_ang - self.veh_heading))) * 0.5)

        expanding = getattr(self, "_expanding_from_border", False)

        # Se la rotazione in un singolo frame è estrema (> 35°) e abbiamo già storia,
        # è un glitch dei keypoint 3D (es. sul motorino): non aggiornare kpts/ground/heading!
        # (Ignora questo blocco se il veicolo sta emergendo dal bordo, perché il cambio di aspect ratio
        #  da "solo cofano" ad "auto intera" può ruotare l'asse maggiore di 90°).
        if not expanding and not self.border_truncated and self.hits >= 3 and abs(d_ang) > np.radians(35.0):
            old_b = self.box_xyxy
            new_b = obs.box_xyxy
            shift_2d = np.array([
                ((new_b[0] + new_b[2]) - (old_b[0] + old_b[2])) * 0.5,
                ((new_b[1] + new_b[3]) - (old_b[1] + old_b[3])) * 0.5,
            ], dtype=np.float64)
            if self.kpts.shape == (8, 2):
                self.kpts = self.kpts + shift_2d
            if self.ground.shape == (4, 2):
                self.ground = self.ground + shift_2d
            self.box_xyxy = new_b.copy()
            return

        # Aggiorna la geometria visiva
        self.box_xyxy = obs.box_xyxy.copy()
        self.kpts = obs.kpts.copy()
        self.ground = obs.ground.copy()
        self.kp_ok = True
        self.shape_valid = True
        self._bev_ground = obs.bev_quad_raw.copy()

        obs_scale = float(obs.obs_scale)
        old_scale = float(self.veh_scale)

        if expanding:
            # Il veicolo sta emergendo dal bordo: sostituisci la forma monca con quella più completa!
            self.veh_S = self._rect_shape(Q)
            self.veh_heading = self._estimate_kabsch_angle(Q, self.veh_S)
            self.veh_W = 1.0
            self.veh_locked = False
            self.veh_rms.clear()
            self.veh_scale = obs_scale
        else:
            self.veh_heading = float(self.veh_heading + 0.35 * d_ang)
            if obs_scale > 1e-6:
                if old_scale > 1e-6:
                    ratio = max(obs_scale / old_scale, old_scale / obs_scale)
                    if ratio <= 1.38:
                        alpha_s = 0.35 if self.border_truncated else 0.08
                        self.veh_scale = (1.0 - alpha_s) * old_scale + alpha_s * obs_scale
                else:
                    self.veh_scale = obs_scale

            if not self.veh_locked:
                _v1 = Q_aligned[2] - Q_aligned[0]
                _v2 = Q_aligned[3] - Q_aligned[1]
                qa = 0.5 * abs(_v1[0] * _v2[1] - _v1[1] * _v2[0])
                med = float(np.median(self.veh_rms)) if len(self.veh_rms) >= 5 else None
                if qa > 0.03 and (med is None or best_res <= 3.5 * med):
                    c, s = np.cos(best_ang), np.sin(best_ang)
                    R = np.array([[c, -s], [s, c]], dtype=np.float64)
                    Qc = Q_aligned @ R
                    w = self.veh_W
                    self.veh_S = self.veh_S + (Qc - self.veh_S) / (w + 1.0)
                    self.veh_S -= self.veh_S.mean(axis=0)
                    self.veh_W = w + 1.0
                    self.veh_rms.append(best_res)
                    if self.veh_W >= self.lock_after_veh:
                        self.veh_locked = True

        # Conta i frame completi lontano dal bordo: dopo 5 frame validi disattiva border_truncated
        if not self.touching_border:
            self.full_view_hits += 1
            if self.full_view_hits >= 5:
                self.border_truncated = False
        else:
            self.full_view_hits = 0

    # ------------------------------------------------------------------ #
    def get_bev_footprint(self, bev_unit: float, center_override: Optional[np.ndarray] = None,
                          heading_override: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the posed (4, 2) footprint in H's gauge (multiplied by bev_unit)."""
        if self.cls == 0:
            c = self.ped_center_ema if center_override is None else center_override
            h = 0.5 * self.ped_scale
            S0 = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)
            return (S0 + c) * bev_unit
        else:
            if not self.shape_valid:
                return None
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
    """OC-SORT operating in normalized BEV coordinates with multi-stage matching."""

    def __init__(self, fps: float = 30.0, max_age: int = 25, min_hits: int = 3,
                 high_conf: float = 0.45, ocm_weight: float = 0.25,
                 edge_frac: float = 0.018):
        self.fps = float(fps) if fps > 0 else 30.0
        self.dt = 1.0 / self.fps
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.high_conf = float(high_conf)
        self.ocm_weight = float(ocm_weight)
        self.edge_frac = float(edge_frac)
        self.tracks: List[BEVKalmanTrack] = []

        self.max_speed_ped = 1.8
        self.max_speed_veh = 3.8

    @staticmethod
    def _box_iou_2d(a: np.ndarray, b: np.ndarray) -> float:
        ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
        iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
        inter = ix * iy
        if inter <= 0.0:
            return 0.0
        ua = max(float(a[2] - a[0]) * float(a[3] - a[1]), 1.0)
        ub = max(float(b[2] - b[0]) * float(b[3] - b[1]), 1.0)
        return float(inter / (ua + ub - inter))

    @staticmethod
    def _is_near_border(box: np.ndarray, img_wh: Tuple[int, int], margin_frac: float = 0.020) -> bool:
        W, H = float(img_wh[0]), float(img_wh[1])
        if W <= 0 or H <= 0:
            return False
        m = max(18.0, margin_frac * float(np.hypot(W, H)))
        return bool(box[0] <= m or box[1] <= m or box[2] >= (W - m) or box[3] >= (H - m))

    def _max_gate_dist(self, trk: BEVKalmanTrack) -> float:
        vmax = self.max_speed_ped if trk.cls == 0 else self.max_speed_veh
        base_rad = 0.50 if trk.cls == 0 else 0.95
        max_cap = 0.95 if trk.cls == 0 else 1.70
        return min(max_cap, base_rad + min(max(trk.time_since_update - 1, 0), 6) * (vmax * self.dt))

    def _check_border_exit(self, trk: BEVKalmanTrack, img_wh: Tuple[int, int]) -> bool:
        if trk.time_since_update == 0:
            return False
        W, H = float(img_wh[0]), float(img_wh[1])
        if W <= 0 or H <= 0:
            return False
        diag = float(np.hypot(W, H))
        m = max(18.0, self.edge_frac * diag)
        b = np.asarray(trk.box_xyxy, dtype=np.float64)
        vx, vy = float(trk.vel_2d[0]), float(trk.vel_2d[1])
        tsu = max(1, int(trk.time_since_update))

        pred_b = b + np.array([vx * tsu, vy * tsu, vx * tsu, vy * tsu], dtype=np.float64)
        cx_p = (pred_b[0] + pred_b[2]) * 0.5
        cy_p = (pred_b[1] + pred_b[3]) * 0.5

        if cx_p <= 2.0 or cx_p >= (W - 2.0) or cy_p <= 2.0 or cy_p >= (H - 2.0):
            return True

        if len(trk.obs_history) >= 2:
            ex_l = (b[0] <= m or pred_b[0] <= m) and (vx < -0.5)
            ex_r = (b[2] >= W - m or pred_b[2] >= W - m) and (vx > 0.5)
            ex_t = (b[1] <= m or pred_b[1] <= m) and (vy < -0.5)
            ex_b = (b[3] >= H - m or pred_b[3] >= H - m) and (vy > 0.5)
            if ex_l or ex_r or ex_t or ex_b:
                return True

        m_tight = max(10.0, 0.008 * diag)
        at_tight = (b[0] <= m_tight) or (b[2] >= W - m_tight) or (b[1] <= m_tight) or (b[3] >= H - m_tight)
        return bool(at_tight and trk.time_since_update >= 2)

    def _cost_matrices(
        self,
        trks: List[BEVKalmanTrack],
        obs_list: List[BEVObservation],
        img_wh: Tuple[int, int],
        use_last_obs: bool = False,
        allow_kp_fallback: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        M, N = len(trks), len(obs_list)
        if M == 0 or N == 0:
            empty = np.empty((M, N), dtype=np.float64)
            return empty, empty

        INF = 1e6
        total_cost = np.full((M, N), INF, dtype=np.float64)
        gate_cost = np.full((M, N), INF, dtype=np.float64)

        for i, t in enumerate(trks):
            tsu = max(1, int(t.time_since_update))
            if tsu > 1 and self._check_border_exit(t, img_wh):
                continue

            base_rad = 0.50 if t.cls == 0 else 0.95
            gate = self._max_gate_dist(t)
            last_box = getattr(t, "last_obs_box", t.box_xyxy)
            vx2d, vy2d = float(t.vel_2d[0]), float(t.vel_2d[1])
            pred_box_2d = last_box + np.array([vx2d * tsu, vy2d * tsu, vx2d * tsu, vy2d * tsu])
            trk_at_border = self._is_near_border(last_box, img_wh, margin_frac=0.030)
            v_dir = t.ocm_direction()

            for j, o in enumerate(obs_list):
                if int(t.cls) != int(o.cls):
                    continue

                iou_last = self._box_iou_2d(last_box, o.box_xyxy)
                iou_pred = self._box_iou_2d(pred_box_2d, o.box_xyxy)
                iou_2d = max(iou_last, iou_pred)

                # Una traccia nata al bordo senza cuboide si riaggancia con il
                # box 2D anche quando la nuova osservazione ha già una shape 3D.
                if t.cls != 0 and not getattr(t, "shape_valid", True):
                    if iou_2d < 0.20:
                        continue
                    c2d = 0.45 * (1.0 - iou_2d) + 0.05 * (tsu - 1)
                    total_cost[i, j] = c2d
                    gate_cost[i, j] = c2d
                    continue

                # Gestione osservazioni veicolo con kp_ok=False (mantenimento traccia 2D anche se hits < min_hits!)
                if t.cls != 0 and (not o.kp_ok or not o.shape_ok):
                    if not allow_kp_fallback or tsu > 4:
                        continue
                    if iou_2d < 0.20:
                        continue
                    c2d = 0.45 * (1.0 - iou_2d) + 0.05 * (tsu - 1)
                    total_cost[i, j] = c2d
                    gate_cost[i, j] = c2d
                    continue

                # VETO "TRACCIA INTERNA IN COASTING -> INGRESSO DAL BORDO" (blocca motorino 84 -> auto rosa)
                obs_at_border = self._is_near_border(o.box_xyxy, img_wh, margin_frac=0.022)
                if obs_at_border and not trk_at_border and tsu > 1 and iou_2d < 0.10:
                    continue
                if obs_at_border and tsu > 1 and iou_2d < 0.05:
                    continue

                # Gate di distanza 2D immagine
                tc_x, tc_y = (last_box[0] + last_box[2]) * 0.5, (last_box[1] + last_box[3]) * 0.5
                oc_x, oc_y = (o.box_xyxy[0] + o.box_xyxy[2]) * 0.5, (o.box_xyxy[1] + o.box_xyxy[3]) * 0.5
                dist_2d = float(np.hypot(oc_x - tc_x, oc_y - tc_y))
                diag_t = float(np.hypot(last_box[2] - last_box[0], last_box[3] - last_box[1]))
                diag_o = float(np.hypot(o.box_xyxy[2] - o.box_xyxy[0], o.box_xyxy[3] - o.box_xyxy[1]))
                ref_diag = max(0.5 * (diag_t + diag_o), 16.0)
                max_2d_mult = (1.45 + 0.45 * min(tsu, 5)) if t.cls == 0 else (1.60 + 0.50 * min(tsu, 5))
                if dist_2d > max_2d_mult * ref_diag and iou_2d < 0.15:
                    continue

                # Posizione BEV ancorata al box 2D: immune ai salti di cP causati da rotazioni del BB 3D!
                ref_pos = t.last_obs_pos if use_last_obs else t.pos
                dist_bev_3d = float(np.linalg.norm(o.bev_center - ref_pos))
                if t.cls != 0 and o.box_bottom_bev is not None and iou_2d >= 0.25:
                    pos_from_2d = t.last_obs_pos + (o.box_bottom_bev - t.last_box_bottom_bev)
                    dist_bev_2d = float(np.linalg.norm(pos_from_2d - ref_pos))
                    dist_bev = min(dist_bev_3d, dist_bev_2d)
                    eff_obs_center = pos_from_2d if dist_bev_2d < dist_bev_3d else o.bev_center
                else:
                    dist_bev = dist_bev_3d
                    eff_obs_center = o.bev_center

                if dist_bev > gate:
                    continue
                c_base = (dist_bev / base_rad) + 0.06 * (tsu - 1) - 0.15 * iou_2d

                # Gate Famiglia Veicolo e Scala (se iou_2d >= 0.30 allo step corrente, ignora gli spike di scala del BB 3D!)
                if t.cls != 0:
                    has_2d_overlap = (tsu <= 2 and iou_2d >= 0.30)
                    fam_t = t.vehicle_family
                    fam_o = _vehicle_family(o.cls, o.source_cls)
                    if not has_2d_overlap:
                        if fam_t in (1, 2) and fam_o in (1, 2) and fam_t != fam_o:
                            continue
                    track_scale = max(float(t.veh_scale), 1e-6)
                    obs_scale = max(float(o.obs_scale), 1e-6)
                    grow_ratio = obs_scale / track_scale
                    shrink_ratio = track_scale / obs_scale
                    has_shape = bool(getattr(t, "shape_valid", True))
                    is_border_growth = (not has_shape or
                                        (getattr(t, "border_truncated", False)
                                         and grow_ratio > 1.0))
                    # gli spike di scala non vengono bloccati se l'auto sta entrando dal bordo
                    if not has_2d_overlap and has_shape:
                        max_grow = 3.5 if is_border_growth else (1.38 if tsu > 1 else 1.60)
                        max_shrink = 1.38 if tsu > 1 else 1.60
                        if grow_ratio > max_grow or shrink_ratio > max_shrink:
                            continue
                    if not is_border_growth:
                        c_base += 0.25 * min(abs(float(np.log(obs_scale / track_scale))), 1.0)

                # OCM + VETO CONTROCORRENTE (immune al jitter 2D/3D quando iou_2d mostra continuità)
                ocm_full = 0.0
                ocm_capped = 0.0
                if v_dir is not None:
                    d_last = eff_obs_center - t.last_obs_pos
                    dn_last = float(np.linalg.norm(d_last))
                    along_last = float(np.dot(d_last, v_dir))

                    cos_anc = (along_last / dn_last) if dn_last > 1e-6 else 1.0
                    if len(t.obs_history) >= 3:
                        _, p_anc = t.obs_history[max(0, len(t.obs_history) - 4)]
                        d_anc = eff_obs_center - p_anc
                        n_anc = float(np.linalg.norm(d_anc))
                        if n_anc >= 0.10:
                            cos_anc = float(np.dot(d_anc, v_dir)) / n_anc

                    # Hard veto SOLO se non c'è evidente sovrapposizione 2D con la propria box predetta
                    if (tsu > 1 or use_last_obs) and iou_2d < 0.25:
                        back_tol = -0.12 if t.cls == 0 else -0.18
                        if along_last < back_tol and cos_anc < -0.15:
                            continue
                        if dn_last > 0.12 and (along_last / dn_last) < -0.20:
                            continue

                    jitter_rad = 0.10 if t.cls == 0 else 0.18
                    if dn_last < jitter_rad and iou_2d >= 0.20:
                        eff_cos = max(cos_anc, 0.0)
                    else:
                        cos_last = (along_last / dn_last) if dn_last >= 0.06 else cos_anc
                        eff_cos = min(cos_last, cos_anc)

                    angle_norm = 0.5 * (1.0 - float(np.clip(eff_cos, -1.0, 1.0)))
                    ocm_full = angle_norm * self.ocm_weight
                    ocm_capped = angle_norm * min(self.ocm_weight, 0.25)

                total_cost[i, j] = max(0.0, c_base + ocm_full)
                gate_cost[i, j] = max(0.0, c_base + ocm_capped)

        return total_cost, gate_cost

    def _associate(self, trks: List[BEVKalmanTrack],
                   obs_list: List[BEVObservation],
                   img_wh: Tuple[int, int],
                   use_last_obs: bool = False,
                   allow_kp_fallback: bool = False,
                   max_cost: float = 1.25) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not trks or not obs_list:
            return [], list(range(len(trks))), list(range(len(obs_list)))
        total_cost, gate_cost = self._cost_matrices(
            trks, obs_list, img_wh,
            use_last_obs=use_last_obs,
            allow_kp_fallback=allow_kp_fallback,
        )
        row_ind, col_ind = linear_sum_assignment(total_cost)
        matches, un_t, un_o = [], [], []
        matched_t, matched_o = set(), set()
        for r, c in zip(row_ind, col_ind):
            if total_cost[r, c] < 1e5 and gate_cost[r, c] <= max_cost:
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

    def update(self, observations: List[BEVObservation], frame_idx: int,
               img_wh: Tuple[int, int]) -> List[BEVKalmanTrack]:
        # 1. Predict all active tracks in BEV
        for t in self.tracks:
            t.predict()

        hi_obs = [o for o in observations
                  if o.conf >= self.high_conf and o.kp_ok and o.shape_ok]
        lo_obs = [o for o in observations
                  if not (o.conf >= self.high_conf and o.kp_ok and o.shape_ok)]

        # 2. Stage 1: Associazione principale su hi_obs
        m1, un_t1, un_o1 = self._associate(self.tracks, hi_obs, img_wh, use_last_obs=False)
        for ti, oi in m1:
            self.tracks[ti].update(hi_obs[oi], frame_idx)

        # 3. Stage 2: Recupero tracce (tsu <= 4) con osservazioni low-conf o veicoli kp_ok=False
        rem_idx = [i for i in un_t1 if self.tracks[i].time_since_update <= 4]
        rem_trks = [self.tracks[i] for i in rem_idx]
        m2, _, _ = self._associate(
            rem_trks, lo_obs, img_wh,
            use_last_obs=False, allow_kp_fallback=True,
        )
        matched_rem = set()
        for loc_ti, oi in m2:
            real_ti = rem_idx[loc_ti]
            self.tracks[real_ti].update(lo_obs[oi], frame_idx)
            matched_rem.add(real_ti)

        # 4. Stage 3: OC-SORT Last-Observation Recovery (tsu <= 4)
        un_t2 = [i for i in un_t1 if i not in matched_rem and self.tracks[i].time_since_update <= 4]
        left_trks = [self.tracks[i] for i in un_t2]
        left_obs = [hi_obs[j] for j in un_o1]
        m3, _, un_o3_local = self._associate(
            left_trks, left_obs, img_wh,
            use_last_obs=True, allow_kp_fallback=False, max_cost=1.10,
        )
        for loc_ti, loc_oi in m3:
            real_ti = un_t2[loc_ti]
            real_oi = un_o1[loc_oi]
            self.tracks[real_ti].update(hi_obs[real_oi], frame_idx)

        # 5. Crea nuove tracce SOLO se l'osservazione non appartiene a una traccia esistente
        #    (controlla fino a tsu <= 3 e riaggancia in-place se una traccia isolata aveva
        #    rifiutato il match per uno spike temporaneo del BB 3D!)
        unmatched_hi = [un_o1[j] for j in un_o3_local]
        for oi in unmatched_hi:
            obs = hi_obs[oi]
            if not obs.kp_ok:
                continue

            dup = False
            best_recover_t = None
            best_recover_iou = 0.25 if obs.cls == 0 else 0.30

            for t in self.tracks:
                if int(t.cls) != int(obs.cls):
                    continue
                iou_curr = self._box_iou_2d(t.box_xyxy, obs.box_xyxy)
                dist_b = float(np.linalg.norm(t.pos - obs.bev_center))

                if t.time_since_update == 0:
                    # Già aggiornata in questo frame: verifica se obs è un semplice duplicato residuo
                    if iou_curr >= (0.30 if obs.cls == 0 else 0.45) or (obs.cls == 0 and dist_b <= 0.24):
                        dup = True
                        break
                elif t.time_since_update <= 3 and not self._check_border_exit(t, img_wh):
                    # Traccia rimasta senza match da 1-3 frame che si sovrappone in 2D con obs:
                    # verifica che non stia andando in direzione opposta e riagganciala!
                    v_dir = t.ocm_direction()
                    counter_flow = False
                    if v_dir is not None:
                        d_vec = obs.bev_center - t.last_obs_pos
                        dn = float(np.linalg.norm(d_vec))
                        if dn > 0.12 and (float(np.dot(d_vec, v_dir)) / dn) < -0.25:
                            counter_flow = True
                    if not counter_flow and (iou_curr >= best_recover_iou or (obs.cls == 0 and dist_b <= 0.25)):
                        best_recover_iou = iou_curr
                        best_recover_t = t

            if dup:
                continue
            if best_recover_t is not None:
                best_recover_t.update(obs, frame_idx)
                continue

            self.tracks.append(BEVKalmanTrack(obs, frame_idx, self.dt, img_wh=img_wh))

        # Start a high-confidence 2D-only track for a clipped vehicle. Its
        # footprint stays absent until a complete cuboid is observed.
        for obs in lo_obs:
            if (obs.cls == 0 or not obs.border_truncated
                    or obs.conf < self.high_conf):
                continue
            duplicate = any(
                int(t.cls) == int(obs.cls)
                and t.time_since_update <= 4
                and self._box_iou_2d(t.box_xyxy, obs.box_xyxy) >= 0.30
                for t in self.tracks
            )
            if not duplicate:
                self.tracks.append(BEVKalmanTrack(
                    obs, frame_idx, self.dt, img_wh=img_wh))

        # 6. Prune dead or border-exited tracks
        kept = []
        for t in self.tracks:
            if t.time_since_update > self.max_age:
                continue
            if self._check_border_exit(t, img_wh):
                continue
            if t.hits < self.min_hits and t.time_since_update >= 3:
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
                if getattr(v, "bev_visible", True):
                    v.bev_quad = trk_obj.get_bev_footprint(
                        unit, center_override=sm_pos, heading_override=heading_override)
                else:
                    v.bev_quad = None
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
