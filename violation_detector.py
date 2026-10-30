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
        self.bev_unit = bev_unit
        self._load_roi(roi_file_path)

    def _load_roi(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File ROI non trovato: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
            
        if "bev_points" not in data or "image_points" not in data:
            raise ValueError("Il file JSON della ROI deve contenere le chiavi 'bev_points' e 'image_points'.")
            
        # Carica le coordinate come numpy array
        self.bev_points = np.array(data["bev_points"], dtype=np.float64)
        self.image_points = np.array(data["image_points"], dtype=np.int32)
        
        if len(self.bev_points) >= 3 and len(self.image_points) >= 3:
            # Creiamo il poligono Shapely esclusivamente con i punti BEV per la logica matematica
            self.polygon = Polygon(self.bev_points)
        else:
            raise ValueError("La ROI deve contenere almeno 3 punti sia per BEV che per Immagine.")

    def contains_point(self, point: np.ndarray) -> bool:
        return self.polygon.contains(Point(point[0], point[1]))

    def overlaps_polygon(self, quad: np.ndarray) -> bool:
        if quad is None or len(quad) < 3:
            return False
        veh_poly = Polygon(quad)
        return self.polygon.intersects(veh_poly)

class ViolationDetector:
    def __init__(self, video_name: str, roi_file: str, bev_unit: float, 
                 ttc_threshold: float = 3.0, timeout_frames: int = 30):
        self.ttc_threshold = float(ttc_threshold)
        self.timeout_frames = int(timeout_frames)
        self.bev_unit = float(bev_unit)
        
        # Inizializza ROI
        self.roi = ZebraCrossingROI(roi_file, self.bev_unit)
        
        # Setup cartella di output
        v_name = os.path.splitext(os.path.basename(video_name))[0]
        self.out_dir = os.path.join("violations_" + v_name)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # State tracking
        self.history_violators = set() # ID dei veicoli che hanno già commesso l'infrazione
        self.active_violations = {}    # {veh_id: {'start_frame': int, 'json_path': str}}
        
        print(f"[ViolationDetector] Inizializzato. Output dir: {self.out_dir} | TTC Thresh: {self.ttc_threshold}s")

    @staticmethod
    def compute_ttc(p_veh: np.ndarray, v_veh: np.ndarray, p_ped: np.ndarray, v_ped: np.ndarray) -> float:
        """
        Calcola il Time-to-Collision (TTC).
        Ritorna -1 se le traiettorie sono divergenti, o float('inf') se parallele/ferme.
        """
        dp = p_veh - p_ped
        dv = v_veh - v_ped
        
        dv2 = np.dot(dv, dv)
        if dv2 < 1e-6:
            return float('inf') # Velocità relative nulle
        
        dot_pv = np.dot(dp, dv)
        if dot_pv >= 0:
            return -1.0 # Traiettorie divergenti (distanza in aumento)
            
        ttc = -dot_pv / dv2
        return float(ttc)

    def bev_to_perspective(self, bev_pos: np.ndarray, H_inv: np.ndarray) -> list:
        """
        Riporietta un punto BEV (scalato per bev_unit) nella visuale prospettica originaria.
        """
        p_scaled = bev_pos * self.bev_unit
        p_hom = np.array([p_scaled[0], p_scaled[1], 1.0])
        p_persp = H_inv @ p_hom
        p_persp = p_persp / p_persp[2]
        return [round(float(p_persp[0])), round(float(p_persp[1]))]

    def process_frame(self, frame: np.ndarray, tracks: list, frame_idx: int, H_inv: np.ndarray):
        """
        Logica principale chiamata ad ogni frame.
        `tracks` deve essere la lista di oggetti _TrackView (quelli emessi dalla V2Pipeline).
        """
        pedestrians_in_roi = []
        vehicles_in_roi = []
        
        # 1. Smistamento tracce e check geometrici
        for t in tracks:
            # Pedoni
            if t.cls == 0: 
                if self.roi.contains_point(t.bev_pos):
                    pedestrians_in_roi.append(t)
            # Veicoli
            elif t.cls == 2:
                # Controlla overlap se ha un quad valido
                if getattr(t, 'bev_quad', None) is not None and self.roi.overlaps_polygon(t.bev_quad):
                    vehicles_in_roi.append(t)

        # 2. Controllo Infrazioni (TTC)
        if len(pedestrians_in_roi) > 0:
            for veh in vehicles_in_roi:
                # Evita di multare lo stesso veicolo due volte
                if veh.id in self.history_violators:
                    continue
                    
                for ped in pedestrians_in_roi:
                    ttc = self.compute_ttc(veh.bev_pos, veh.bev_vel, ped.bev_pos, ped.bev_vel)
                    
                    if 0.0 <= ttc <= self.ttc_threshold:
                        # INFRAZIONE RILEVATA!
                        self._trigger_violation(frame, frame_idx, veh, ped, ttc, H_inv)
                        break # Passa al prossimo veicolo

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
                # E' uscito dalle strisce?
                if getattr(veh_t, 'bev_quad', None) is not None:
                    if not self.roi.overlaps_polygon(veh_t.bev_quad):
                        exited_roi = True
                else:
                    # Se ha perso temporaneamente il quad, consideriamo l'uscita in base al centroide
                    if not self.roi.contains_point(veh_t.bev_pos):
                        exited_roi = True
            else:
                track_lost = True

            # Condizioni per salvare l'immagine finale e chiudere la violazione
            reason = None
            if exited_roi:
                reason = "exited_roi"
            elif track_lost and frames_elapsed >= self.timeout_frames:
                reason = f"timeout_track_lost_{frames_elapsed}_frames"
            elif frames_elapsed >= self.timeout_frames * 2: # Failsafe assoluto
                reason = "absolute_timeout"

            if reason is not None:
                self._close_violation(veh_id, frame, frame_idx, reason)

    def _trigger_violation(self, frame, frame_idx, veh, ped, ttc, H_inv):
        self.history_violators.add(veh.id)
        
        # Calcolo coordinate prospettiche
        veh_persp = self.bev_to_perspective(veh.bev_pos, H_inv)
        ped_persp = self.bev_to_perspective(ped.bev_pos, H_inv)
        
        prefix = f"violation_v{veh.id}_f{frame_idx}"
        img_path = os.path.join(self.out_dir, f"{prefix}_start.jpg")
        json_path = os.path.join(self.out_dir, f"{prefix}.json")
        
        # Salva Immagine Iniziale
        cv2.imwrite(img_path, frame)
        
        # Compila JSON
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
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        # Aggiunge alla macchina a stati per l'uscita
        self.active_violations[veh.id] = {
            "start_frame": frame_idx,
            "json_path": json_path,
            "prefix": prefix,
            "data": data
        }
        print(f"🚨 INFRAZIONE: Veicolo {veh.id} vs Pedone {ped.id} (TTC: {ttc:.2f}s). Salvato in {json_path}")

    def _close_violation(self, veh_id, frame, frame_idx, reason):
        state = self.active_violations.pop(veh_id)
        
        img_filename = f"{state['prefix']}_end.jpg"
        img_path = os.path.join(self.out_dir, img_filename)
        
        # Salva Immagine Finale
        cv2.imwrite(img_path, frame)
        
        # Aggiorna il JSON
        data = state['data']
        data['images']['exit_infraction'] = img_filename
        data['images']['exit_reason'] = reason
        data['exit_frame'] = frame_idx
        
        with open(state['json_path'], 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"✅ INFRAZIONE CHIUSA: Veicolo {veh_id} al frame {frame_idx} (Motivo: {reason})")