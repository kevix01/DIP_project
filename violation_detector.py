import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon, Point


class ZebraCrossingROI:
    def __init__(self, roi_file_path: str, bev_unit: float):
        """
        Carica le coordinate della ROI dal file JSON.
        Il JSON deve contenere le chiavi 'bev_points' e 'image_points'.
        """
        self.polygon = None
        self.bev_points = None
        self.image_points = None
        self.bev_unit = float(bev_unit) if bev_unit and bev_unit > 1e-9 else 1.0
        self._load_roi(roi_file_path)

    def _load_roi(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File ROI non trovato: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "bev_points" not in data or "image_points" not in data:
            raise ValueError("Il file JSON della ROI deve contenere le chiavi 'bev_points' e 'image_points'.")

        self.bev_points = np.array(data["bev_points"], dtype=np.float64)
        self.image_points = np.array(data["image_points"], dtype=np.int32)

        if len(self.bev_points) >= 3 and len(self.image_points) >= 3:
            self.polygon = Polygon(self.bev_points)
        else:
            raise ValueError("La ROI deve contenere almeno 3 punti sia per BEV che per Immagine.")

    def contains_point(self, point: np.ndarray) -> bool:
        if point is None or not np.all(np.isfinite(point)):
            return False
        return self.polygon.contains(Point(float(point[0]), float(point[1])))

    def overlaps_polygon(self, quad: np.ndarray) -> bool:
        """
        Verifica se l'impronta a terra del veicolo (bev_quad) interseca la ROI.
        Nota: t.bev_quad in V2Pipeline è in coordinate omografiche grezze (BEU * bev_unit),
        mentre self.polygon è in coordinate BEU normalizzate. Dividiamo per bev_unit.
        """
        if quad is None or len(quad) < 3 or not np.all(np.isfinite(quad)):
            return False
        quad_beu = np.asarray(quad, dtype=np.float64) / self.bev_unit
        veh_poly = Polygon(quad_beu)
        if not veh_poly.is_valid:
            veh_poly = veh_poly.buffer(0)
        return self.polygon.intersects(veh_poly)


class ViolationDetector:
    def __init__(self, video_name: str, roi_file: str, bev_unit: float,
                 ttc_threshold: float = 3.0, timeout_frames: int = 30,
                 conflict_radius: float = 1.1, save_files: bool = True):
        self.ttc_threshold = float(ttc_threshold)
        self.timeout_frames = int(timeout_frames)
        self.conflict_radius = float(conflict_radius)  # Distanza max (in BEU) al punto di impatto (d_CPA)
        self.bev_unit = float(bev_unit) if bev_unit and bev_unit > 1e-9 else 1.0
        self.save_files = bool(save_files)

        # Inizializza ROI
        self.roi = ZebraCrossingROI(roi_file, self.bev_unit)

        # Setup cartella di output
        v_name = os.path.splitext(os.path.basename(video_name))[0]
        self.out_dir = os.path.join("violations_" + v_name)
        if self.save_files:
            os.makedirs(self.out_dir, exist_ok=True)

        # State tracking
        self.history_violators = set()  # ID dei veicoli che hanno già commesso l'infrazione
        self.active_violations = {}     # {veh_id: {'start_frame': int, 'json_path': str, ...}}

        print(f"[ViolationDetector] Inizializzato | Output: {self.out_dir} (save={self.save_files}) | TTC Thresh: {self.ttc_threshold}s")

    @staticmethod
    def compute_ttc(p_veh: np.ndarray, v_veh: np.ndarray,
                    p_ped: np.ndarray, v_ped: np.ndarray,
                    conflict_radius: float = 1.1) -> float:
        """
        Calcola il TTC 2D reale in BEV verificando che esista una rotta di collisione.
        Codici di ritorno per non-violazione (utili per la tabella di debug):
          -1.0 : Traiettorie divergenti (distanza in aumento)
          -2.0 : Pedone già passato / in allontanamento dalla corsia del veicolo
          -3.0 : Traiettorie non incidenti (d_CPA > conflict_radius, corsie diverse)
           inf : Velocità relativa o del veicolo quasi nulla
        """
        p_v = np.asarray(p_veh, dtype=np.float64)
        v_v = np.asarray(v_veh, dtype=np.float64)
        p_p = np.asarray(p_ped, dtype=np.float64)
        v_p = np.asarray(v_ped, dtype=np.float64)

        veh_speed = float(np.linalg.norm(v_v))
        if veh_speed < 0.15:
            return float('inf')  # Veicolo fermo

        # Vettore dal veicolo verso il pedone: r = p_ped - p_veh
        r = p_p - p_v
        u_veh = v_v / veh_speed  # Versore direzione di marcia del veicolo

        # 1. Controllo Longitudinale: il pedone è già dietro al veicolo?
        d_long = float(np.dot(r, u_veh))
        if d_long < -0.4:
            return -2.0  # Il veicolo ha già superato la linea del pedone

        # 2. Controllo Laterale: il pedone ha già superato la corsia e si sta allontanando?
        r_lat = r - d_long * u_veh
        d_lat = float(np.linalg.norm(r_lat))
        ped_speed = float(np.linalg.norm(v_p))
        if d_lat > 0.6 and ped_speed > 0.10:
            u_lat = r_lat / d_lat
            v_ped_away = float(np.dot(v_p, u_lat))
            # Se v_ped_away > 0.08, il pedone si trova a lato dell'auto e cammina allontanandosi da essa
            if v_ped_away > 0.08:
                return -2.0

        # 3. Calcolo t_CPA (Time to Closest Point of Approach)
        dp = p_v - p_p
        dv = v_v - v_p
        dv2 = float(np.dot(dv, dv))
        if dv2 < 1e-6:
            return float('inf')

        dot_pv = float(np.dot(dp, dv))
        if dot_pv >= 0.0:
            return -1.0  # Traiettorie divergenti

        t_cpa = -dot_pv / dv2

        # 4. Verifica Distanza Minima d_CPA all'istante t_cpa
        # Se a t = t_cpa i due oggetti restano distanti più di conflict_radius, non c'è collisione!
        d_cpa_vec = dp + dv * t_cpa
        d_cpa = float(np.linalg.norm(d_cpa_vec))
        if d_cpa > conflict_radius:
            return -3.0  # Sfilano su corsie diverse senza intersecarsi

        return float(t_cpa)

    def bev_to_perspective(self, bev_pos: np.ndarray, H_inv: np.ndarray) -> list:
        """
        Riproietta un punto BEV (normalizzato per bev_unit) nella visuale prospettica.
        """
        p_scaled = np.asarray(bev_pos, dtype=np.float64) * self.bev_unit
        p_hom = np.array([p_scaled[0], p_scaled[1], 1.0], dtype=np.float64)
        p_persp = H_inv @ p_hom
        w = p_persp[2] if abs(p_persp[2]) > 1e-12 else 1e-12
        p_persp = p_persp / w
        return [round(float(p_persp[0])), round(float(p_persp[1]))]

    def process_frame(self, frame: np.ndarray, tracks: list, frame_idx: int, H_inv: np.ndarray) -> dict:
        """
        Logica principale chiamata ad ogni frame.
        Ritorna un dizionario `debug_info` con tutte le metriche del frame corrente
        per la visualizzazione nella tabella di debug e la colorazione delle impronte.
        """
        pedestrians_in_roi = []
        vehicles_in_roi = []

        # 1. Smistamento tracce e check geometrici sulla ROI
        for t in tracks:
            if t.cls == 0:
                if self.roi.contains_point(t.bev_pos):
                    pedestrians_in_roi.append(t)
            elif t.cls == 2:
                if getattr(t, 'bev_quad', None) is not None and self.roi.overlaps_polygon(t.bev_quad):
                    vehicles_in_roi.append(t)

        # Ordina per ID per stabilità visiva nella tabella di debug
        pedestrians_in_roi.sort(key=lambda x: x.id)
        vehicles_in_roi.sort(key=lambda x: x.id)

        ttc_matrix = {}         # (veh_id, ped_id) -> float
        min_ttc_per_veh = {}    # veh_id -> float (min TTC positivo)
        newly_triggered = set() # veh_id multati in questo esatto frame

        # 2. Calcolo matrice completa TTC e controllo infrazioni
        for veh in vehicles_in_roi:
            min_pos_ttc = float('inf')
            trigger_ped = None
            trigger_ttc = None

            for ped in pedestrians_in_roi:
                ttc = self.compute_ttc(
                    veh.bev_pos, veh.bev_vel,
                    ped.bev_pos, ped.bev_vel,
                    conflict_radius=self.conflict_radius
                )
                ttc_matrix[(veh.id, ped.id)] = ttc

                if ttc >= 0.0 and ttc < min_pos_ttc:
                    min_pos_ttc = ttc

                if 0.0 <= ttc <= self.ttc_threshold:
                    if trigger_ttc is None or ttc < trigger_ttc:
                        trigger_ttc = ttc
                        trigger_ped = ped

            min_ttc_per_veh[veh.id] = min_pos_ttc

            # Se c'è almeno un pedone con 0 <= TTC <= soglia e il veicolo non è già stato multato
            if trigger_ped is not None and veh.id not in self.history_violators:
                self._trigger_violation(frame, frame_idx, veh, trigger_ped, trigger_ttc, H_inv)
                newly_triggered.add(veh.id)

        # 3. Gestione Uscita Veicoli Multati (Salvataggio 2° Immagine)
        active_ids = list(self.active_violations.keys())
        veh_track_dict = {t.id: t for t in tracks if t.cls == 2}

        for veh_id in active_ids:
            state = self.active_violations[veh_id]
            frames_elapsed = frame_idx - state['start_frame']

            exited_roi = False
            track_lost = False

            if veh_id in veh_track_dict:
                veh_t = veh_track_dict[veh_id]
                if getattr(veh_t, 'bev_quad', None) is not None:
                    if not self.roi.overlaps_polygon(veh_t.bev_quad):
                        exited_roi = True
                else:
                    if not self.roi.contains_point(veh_t.bev_pos):
                        exited_roi = True
            else:
                track_lost = True

            reason = None
            if exited_roi:
                reason = "exited_roi"
            elif track_lost and frames_elapsed >= self.timeout_frames:
                reason = f"timeout_track_lost_{frames_elapsed}_frames"
            elif frames_elapsed >= self.timeout_frames * 2:
                reason = "absolute_timeout"

            if reason is not None:
                self._close_violation(veh_id, frame, frame_idx, reason)

        return {
            "frame_idx": frame_idx,
            "pedestrians_in_roi": pedestrians_in_roi,
            "vehicles_in_roi": vehicles_in_roi,
            "vehicles_in_roi_ids": {v.id for v in vehicles_in_roi},
            "violating_vehicle_ids": set(self.history_violators),
            "newly_triggered_ids": newly_triggered,
            "ttc_matrix": ttc_matrix,
            "min_ttc_per_veh": min_ttc_per_veh,
            "ttc_threshold": self.ttc_threshold,
        }

    def _trigger_violation(self, frame, frame_idx, veh, ped, ttc, H_inv):
        self.history_violators.add(veh.id)

        veh_persp = self.bev_to_perspective(veh.bev_pos, H_inv) if H_inv is not None else [0, 0]
        ped_persp = self.bev_to_perspective(ped.bev_pos, H_inv) if H_inv is not None else [0, 0]

        prefix = f"violation_v{veh.id}_f{frame_idx}"
        img_path = os.path.join(self.out_dir, f"{prefix}_start.jpg")
        json_path = os.path.join(self.out_dir, f"{prefix}.json")

        data = {
            "video_frame": frame_idx,
            "vehicle_id": veh.id,
            "pedestrian_id": ped.id,
            "ttc_seconds": round(ttc, 3),
            "vehicle_centroid_perspective": veh_persp,
            "pedestrian_centroid_perspective": ped_persp,
            "images": {
                "start_infraction": f"{prefix}_start.jpg",
                "exit_infraction": None,
                "exit_reason": "pending"
            }
        }

        if self.save_files:
            cv2.imwrite(img_path, frame)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

        self.active_violations[veh.id] = {
            "start_frame": frame_idx,
            "json_path": json_path,
            "prefix": prefix,
            "data": data
        }
        print(f"\n🚨 INFRAZIONE: Veicolo {veh.id} vs Pedone {ped.id} (TTC: {ttc:.2f}s) al frame {frame_idx}")

    def _close_violation(self, veh_id, frame, frame_idx, reason):
        state = self.active_violations.pop(veh_id)

        img_filename = f"{state['prefix']}_end.jpg"
        img_path = os.path.join(self.out_dir, img_filename)

        data = state['data']
        data['images']['exit_infraction'] = img_filename
        data['images']['exit_reason'] = reason
        data['exit_frame'] = frame_idx

        if self.save_files:
            cv2.imwrite(img_path, frame)
            with open(state['json_path'], 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

        print(f"\n✅ INFRAZIONE CHIUSA: Veicolo {veh_id} al frame {frame_idx} (Motivo: {reason})")