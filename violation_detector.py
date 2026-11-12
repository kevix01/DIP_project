import os
import json
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from shapely.geometry import Polygon, Point
from shapely.prepared import prep


class ZebraCrossingROI:
    def __init__(self, roi_file_path: str, bev_unit: float):
        """
        Carica le coordinate della ROI dal file JSON.
        Il JSON deve contenere le chiavi 'bev_points' e 'image_points'.
        """
        self.polygon = None
        self._prep = None
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
            self._prep = prep(self.polygon)
        else:
            raise ValueError("La ROI deve contenere almeno 3 punti sia per BEV che per Immagine.")

    def contains_point(self, point: np.ndarray) -> bool:
        if point is None or not np.all(np.isfinite(point)):
            return False
        return self._prep.contains(Point(float(point[0]), float(point[1])))

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
        return self._prep.intersects(veh_poly)


# --------------------------------------------------------------------------- #
# Intento di movimento del pedone
# --------------------------------------------------------------------------- #

INTENT_UNKNOWN = "UNKNOWN"
INTENT_MOVING = "MOVING"        # marcia coerente
INTENT_SLOW = "SLOW"            # moto lento
INTENT_STOPPED = "STOPPED"      # fermo sul posto
INTENT_TURNING = "TURNING"      # sta cambiando direzione (non ancora inversione)
INTENT_REVERSED = "REVERSED"    # sta tornando indietro rispetto alla marcia


@dataclass
class PedIntent:
    """
    Modello dell'intento di movimento di un pedone, basato sullo spostamento
    netto su finestre temporali (non sulla velocità istantanea, che per i
    pedoni è rumore: EMA della posizione + Kalman su posizione grezza +
    smoother Savitzky-Golay zero-phase).

    - finestra "long": direzione di marcia stabile (intento impegnato)
    - finestra "short": direzione del moto recente (possibile inversione)

    Lo stato REVERSED/TURNING serve a distinguere "il pedone si sta
    allontanando dalla traiettoria del veicolo" (falso positivo) da
    "il pedone è fermo sulla striscia" (infrazione reale).
    """
    ped_id: int
    long_frames: int
    short_frames: int
    step_frames: int = 8           # finestra breve (~0.25s) per il veto di ritirata
    moving_speed: float = 0.25     # BEU/s sopra il quale il pedone è "in marcia"
    stopped_speed: float = 0.08    # BEU/s sotto il quale il pedone è "fermo"
    flip_deg: float = 45.0         # angolo oltre il quale si considera un cambio direzione
    retreat_cos: float = -0.5      # coseno sotto il quale il moto recente è "in ritirata"
    retreat_dist: float = 0.05     # BEU di spostamento minimo sulla finestra step
    hist: deque = field(default_factory=lambda: deque(maxlen=256))
    raw_hist: deque = field(default_factory=lambda: deque(maxlen=4))
    last_seen_frame: int = -1

    state: str = INTENT_UNKNOWN
    speed_long: float = 0.0
    speed_short: float = 0.0
    speed_step: float = 0.0
    flip_angle: float = 0.0
    retreating: bool = False       # veto sticky: una volta in retirata resta fino a nuovo moto netto
    dir_long: object = None
    dir_short: object = None
    dir_step: object = None
    effective_vel: object = None

    def update(self, frame_idx: int, pos: np.ndarray, fps: float):
        raw = np.asarray(pos, dtype=np.float64).copy()
        self.last_seen_frame = int(frame_idx)
        # Outlier rejection adattivo. La mediana di riferimento e' calcolata
        # SOLO sui campioni RAW: se si usasse la history filtrata, la mediana
        # dei campioni respinti resterebbe ferma (med_step -> 0) e ogni frame
        # successivo verrebbe respinto a sua volta, bloccando la traccia su
        # una posizione fantasma ("latching") e facendo apparire fermo un
        # pedone che invece si muove.
        filtered = raw
        self.raw_hist.append((int(frame_idx), raw))
        if len(self.raw_hist) >= 4:
            window = np.stack([p for _, p in self.raw_hist], axis=0)
            steps = np.linalg.norm(np.diff(window, axis=0), axis=1)
            med_step = float(np.median(steps)) if len(steps) > 0 else 0.0
            thresh = max(0.06, 3.0 * med_step)
            med = np.median(window, axis=0)
            if float(np.linalg.norm(raw - med)) > thresh:
                filtered = med
        self.hist.append((int(frame_idx), filtered))
        self._recompute(fps)

    def _window(self, cur_frame: int, n_frames: int):
        """Restituisce (frame_ref, pos_ref) del campione più vecchio entro la finestra."""
        target = cur_frame - int(n_frames)
        ref = None
        for fi, pos in reversed(self.hist):
            if fi <= target:
                ref = (fi, pos)
                break
        return ref

    def _recompute(self, fps: float):
        fps = float(fps) if fps and fps > 1e-6 else 30.0
        cur_frame, cur_pos = self.hist[-1]

        def window_vec(n_frames):
            ref = self._window(cur_frame, n_frames)
            if ref is None:
                return None
            ref_frame, ref_pos = ref
            dt = (cur_frame - ref_frame) / fps
            if dt <= 1e-6:
                return None
            delta = cur_pos - ref_pos
            dist = float(np.linalg.norm(delta))
            if dist <= 1e-6:
                return (None, 0.0, dist)
            return (delta / dist, dist / dt, dist)

        long_v = window_vec(self.long_frames)
        short_v = window_vec(self.short_frames)
        step_v = window_vec(self.step_frames)

        self.dir_long = long_v[0] if long_v else None
        self.speed_long = long_v[1] if long_v else 0.0
        self.dir_short = short_v[0] if short_v else None
        self.speed_short = short_v[1] if short_v else 0.0
        self.dir_step = step_v[0] if step_v else None
        self.speed_step = step_v[1] if step_v else 0.0

        zero = np.zeros(2, dtype=np.float64)

        if short_v is None:
            self.state = INTENT_UNKNOWN
            self.flip_angle = 0.0
            self.effective_vel = zero
            return

        if self.dir_long is not None and self.dir_short is not None:
            cosf = float(np.clip(np.dot(self.dir_long, self.dir_short), -1.0, 1.0))
            self.flip_angle = float(np.degrees(np.arccos(cosf)))
        else:
            self.flip_angle = 0.0

        # Veto di retirata: su una finestra breve lo spostamento netto è
        # opposto alla direzione di marcia impegnata (long). Sticky: resta
        # attivo finché non c'è un nuovo moto netto nella direzione di marcia.
        if self.dir_long is not None and step_v is not None:
            step_dist = step_v[2]
            if (step_dist >= self.retreat_dist and self.dir_step is not None
                    and float(np.dot(self.dir_step, self.dir_long)) < self.retreat_cos):
                self.retreating = True
            elif (step_dist >= self.retreat_dist and self.dir_step is not None
                  and float(np.dot(self.dir_step, self.dir_long)) > 0.5):
                self.retreating = False

        if self.retreating:
            self.state = INTENT_REVERSED
            self.effective_vel = (self.dir_step * self.speed_step
                                  if self.dir_step is not None else zero)
        elif self.speed_short >= self.moving_speed and self.speed_long >= self.moving_speed:
            # Il pedone sta camminando: l'intento è la direzione di marcia.
            # La direzione breve prevale solo se lo spostamento netto sulla
            # finestra short è significativo: uno scatto breve del tracker
            # non deve ridisegnare la traiettoria d'intento.
            # L'intento di moto è SEMPRE la direzione della finestra long:
            # solo un cambio di direzione sostenuto per tutta la finestra
            # lunga (>>1s) può ridisegnare la traiettoria. Le finestre short
            # e step servono solo a classificare lo stato e il veto di
            # retirata, mai a sovrascrivere la direzione impegnata.
            if self.flip_angle > self.flip_deg:
                self.state = INTENT_TURNING
            else:
                self.state = INTENT_MOVING
            if self.dir_long is not None:
                self.effective_vel = self.dir_long * self.speed_long
            elif self.dir_short is not None:
                self.effective_vel = self.dir_short * self.speed_short
            else:
                self.effective_vel = zero
        elif self.speed_short < self.stopped_speed:
            # Fermo: resta occupante statico della zona di conflitto.
            self.state = INTENT_STOPPED
            self.effective_vel = zero
        else:
            self.state = INTENT_SLOW
            self.effective_vel = (self.dir_short * self.speed_short
                                  if self.dir_short is not None else zero)

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "speed_long": round(float(self.speed_long), 3),
            "speed_short": round(float(self.speed_short), 3),
            "speed_step": round(float(self.speed_step), 3),
            "flip_angle": round(float(self.flip_angle), 1),
            "retreating": bool(self.retreating),
        }


class ViolationDetector:
    def __init__(self, video_name: str, roi_file: str, bev_unit: float,
                 ttc_threshold: float = 3.0,
                 clip_track_timeout_s: float = 1.5,
                 clip_absolute_timeout_s: float = 3.0,
                 conflict_radius: float = 1.1, save_files: bool = True,
                 fps: float = 30.0, confirm_frames: int = 3,
                 intent_long_s: float = 1.2, intent_short_s: float = 0.5,
                 intent_moving_speed: float = 0.25,
                 intent_stopped_speed: float = 0.08,
                  intent_flip_deg: float = 45.0,
                  intent_retreat_cos: float = -0.5,
                  intent_step_s: float = 0.25,
                  intent_retreat_dist: float = 0.05,
                  lateral_pass_dlat: float = 0.45,
                  lateral_pass_cpa: float = 0.75,
                  lateral_pass_away: float = 0.05,
                   conflict_dlong: float = 0.2,
                   lateral_gate_factor: float = 1.20,
                   clip_video: bool = True,
                  clip_fourcc: str = "mp4v"):
        self.ttc_threshold = float(ttc_threshold)
        self.fps = float(fps) if fps and fps > 1e-6 else 30.0

        # Timeout della clip in SECONDI (coerenti con il framerate):
        #  - clip_track_timeout_s   : attesa dopo la perdita del tracking del veicolo
        #  - clip_absolute_timeout_s : durata massima assoluta di un'infrazione
        self.clip_track_timeout_s = max(0.0, float(clip_track_timeout_s))
        self.clip_absolute_timeout_s = max(0.0, float(clip_absolute_timeout_s))
        self.clip_video = bool(clip_video)
        self.clip_fourcc = str(clip_fourcc)
        self.conflict_radius = float(conflict_radius)  # Distanza max (in BEU) al punto di impatto (d_CPA)
        # Gate laterale (-6.0): fattore moltiplicativo applicato a
        # conflict_radius per la soglia d_lat "fuori corsia".
        self.lateral_gate_radius = self.conflict_radius * float(lateral_gate_factor)
        self.bev_unit = float(bev_unit) if bev_unit and bev_unit > 1e-9 else 1.0
        self.save_files = bool(save_files)
        self.confirm_frames = max(1, int(confirm_frames))
        self.intent_retreat_cos = float(intent_retreat_cos)

        # A fps bassi le finestre in secondi diventano pochissimi frame e uno
        # scatto di detection le domina: imponiamo un minimo di frame.
        self.intent_long_frames = max(15, int(round(intent_long_s * self.fps)))
        self.intent_short_frames = max(6, int(round(intent_short_s * self.fps)))
        self.intent_step_frames = max(3, int(round(intent_step_s * self.fps)))
        self.intent_retreat_dist = float(intent_retreat_dist)
        self.intent_moving_speed = float(intent_moving_speed)
        self.intent_stopped_speed = float(intent_stopped_speed)
        self.intent_flip_deg = float(intent_flip_deg)

        # Veto "superato lateralmente": soglie del controllo laterale che
        # distingue un pedone che marcia verso/oltre la corsia (conflitto
        # reale) da un pedone che cammina parallelo sul bordo, gia' fuori
        # dalla linea di marcia del veicolo e in allontanamento.
        self.lateral_pass_dlat = float(lateral_pass_dlat)
        self.lateral_pass_cpa = float(lateral_pass_cpa)
        self.lateral_pass_away = float(lateral_pass_away)

        # Gate geometrico della zona di conflitto: il pedone deve trovarsi
        # dentro il raggio di conflitto lateralmente (d_lat) e deve essere
        # ancora AVANTI al veicolo (d_long > conflict_dlong). Se il pedone
        # e' gia' stato superato dal veicolo, il CPA puo' comunque cadere
        # vicino perche' proietta la traiettoria intenzionale del pedone
        # sulla corsia, ma il conflitto e' solo geometrico, non fisico.
        self.conflict_dlong = float(conflict_dlong)

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
        self.ped_intents = {}           # {ped_id: PedIntent}
        self.pending_conflicts = {}     # {veh_id: {'ped_id': int, 'count': int, 'last_frame': int}}

        print(f"[ViolationDetector] Inizializzato | Output: {self.out_dir} (save={self.save_files}) | "
              f"TTC Thresh: {self.ttc_threshold}s | Conferma: {self.confirm_frames} frame | "
              f"Intent: long={self.intent_long_frames}f short={self.intent_short_frames}f")

    # ------------------------------------------------------------------ #
    # Valutazione conflitto "intent-gated"
    # ------------------------------------------------------------------ #
    def evaluate_conflict(self, veh, ped, intent: PedIntent):
        """
        Valuta il conflitto veicolo/pedone usando l'INTENTO di movimento del
        pedone invece della sua velocità istantanea.

        Restituisce (ttc, label). ttc < 0 indica non-conflitto con codice:
          -1.0 : traiettorie divergenti
          -2.0 : pedone già superato / alle spalle del veicolo
          -3.0 : nessuna rotta di collisione (d_CPA > conflict_radius)
          -4.0 : il pedone sta cambiando direzione / torna indietro
                  (NON è un falso positivo: il veicolo non lo raggiunge)
          -5.0 : veto laterale (pedone fuori corsia e in allontanamento)
          -6.0 : gate laterale (pedone fuori dal raggio di conflitto)
          -7.0 : gate longitudinale (pedone già superato dal veicolo)
        """
        p_v = np.asarray(veh.bev_pos, dtype=np.float64)
        v_v = np.asarray(veh.bev_vel, dtype=np.float64)
        # Usa la posizione filtrata (outlier rejection) se disponibile:
        # il jitter di detection non deve spostare il pedone verso la corsia.
        if intent is not None and len(intent.hist) > 0:
            p_p = np.asarray(intent.hist[-1][1], dtype=np.float64)
        else:
            p_p = np.asarray(ped.bev_pos, dtype=np.float64)

        veh_speed = float(np.linalg.norm(v_v))
        if veh_speed < 0.15:
            return float('inf'), "VEICOLO FERMO"

        u_veh = v_v / veh_speed

        r = p_p - p_v
        d_long = float(np.dot(r, u_veh))

        # Il pedone è già alle spalle del veicolo: nessun conflitto futuro.
        if d_long < -0.4:
            return -2.0, "OLTRE VEICOLO"

        # Geometria laterale condivisa dai controlli successivi:
        # r_lat/d_lat (distanza del pedone dalla linea di marcia del veicolo),
        # u_lat (versore laterale verso il pedone) e away_lat (quota della
        # velocita' INTENZIONALE del pedone lungo tale verso).
        r_lat = r - d_long * u_veh
        d_lat = float(np.linalg.norm(r_lat))
        u_lat = r_lat / d_lat if d_lat > 1e-9 else None

        v_int = intent.effective_vel if intent is not None else None
        if v_int is None:
            v_int = np.zeros(2, dtype=np.float64)
        v_int = np.asarray(v_int, dtype=np.float64)
        int_speed = float(np.linalg.norm(v_int))
        away_lat = float(np.dot(v_int, u_lat)) if u_lat is not None else 0.0

        # Pedone lateralmente fuori dalla linea del veicolo e in allontanamento:
        # ha gia' attraversato la corsia, nessun conflitto futuro.
        # Pre-filtro economico: la verifica rigorosa e' il veto laterale qui sotto.
        if d_lat > 0.6 and int_speed > 0.05 and away_lat > 0.08:
            return -2.0, "PASSATO (oltre)"

        if int_speed > 0.05:
            # Pedone in moto: CPA tra veicolo e traiettoria INTENZIONALE del pedone.
            dp = p_v - p_p
            dv = v_v - v_int
            dv2 = float(np.dot(dv, dv))
            if dv2 < 1e-6:
                return float('inf'), "VEL RELATIVA NULLA"
            dot_pv = float(np.dot(dp, dv))
            if dot_pv >= 0.0:
                return -1.0, "DIVERGENTI"

            t_cpa = -dot_pv / dv2
            d_cpa_vec = dp + dv * t_cpa
            d_cpa = float(np.linalg.norm(d_cpa_vec))
            if d_cpa > self.conflict_radius:
                return -3.0, "CORSIE DIVERSE"

            # Gate laterale: il pedone e' piu' lontano dalla linea di marcia
            # del veicolo del raggio di conflitto. Il CPA puo' comunque
            # cadere vicino perche' la traiettoria INTENZIONALE del pedone
            # taglia la corsia, ma il veicolo raggiunge la zona di conflitto
            # ben prima che il pedone ci arrivi: il conflitto e' solo
            # geometrico, non fisico.
            if d_lat > self.lateral_gate_radius:
                return -6.0, "PEDONE FUORI CORSIA"

            # Gate longitudinale: il pedone e' gia' stato superato dal
            # veicolo (o e' sullo stesso punto): qualunque conflitto con la
            # sua traiettoria intenzionale e' un falso positivo geometrico.
            if d_long <= self.conflict_dlong:
                return -7.0, "PEDONE SUPERATO"

            # Veto laterale: il pedone e' gia' ben fuori dalla linea del
            # veicolo e si allontana lateralmente. Se anche il punto di
            # conflitto (CPA) cade a piu' di lateral_cpa_min dal centro
            # della corsia veicolare, le traiettorie non si incrociano:
            # il TTC e' un falso positivo geometrico.
            n_lat = np.array([-u_veh[1], u_veh[0]], dtype=np.float64)
            lat_cpa = abs(float(np.dot(d_cpa_vec, n_lat)))
            if (d_lat >= self.lateral_pass_dlat
                    and lat_cpa >= self.lateral_pass_cpa
                    and away_lat > self.lateral_pass_away):
                return -5.0, "SUPERATO LATERALE"

            # Veto di retirata: il pedone sta tornando indietro rispetto alla
            # propria marcia (non verso il veicolo) -> falso positivo soppresso.
            if intent is not None and intent.retreating:
                return -4.0, "PEDONE IN RITIRATA"

            return float(t_cpa), "IN MOTO"

        # Pedone fermo: resta occupante statico della zona di conflitto.
        # Il veicolo lo raggiungerà se la sua traiettoria futura passa entro
        # conflict_radius dalla sua posizione.
        if d_lat > self.conflict_radius:
            return -3.0, "CORSIE DIVERSE"
        t_arrive = d_long / veh_speed
        if t_arrive > self.ttc_threshold:
            return float('inf'), "TROPPO LONTANO"
        return float(t_arrive), "PEDONE FERMO"

    def _update_intents(self, tracks: list, frame_idx: int):
        """Aggiorna lo stato di intento di tutti i pedoni tracciati (anche fuori ROI)."""
        seen = set()
        for t in tracks:
            if t.cls != 0:
                continue
            seen.add(t.id)
            intent = self.ped_intents.get(t.id)
            if intent is None:
                intent = PedIntent(
                    ped_id=t.id,
                    long_frames=self.intent_long_frames,
                    short_frames=self.intent_short_frames,
                    step_frames=self.intent_step_frames,
                    retreat_cos=self.intent_retreat_cos,
                    retreat_dist=self.intent_retreat_dist,
                    moving_speed=self.intent_moving_speed,
                    stopped_speed=self.intent_stopped_speed,
                    flip_deg=self.intent_flip_deg,
                )
                self.ped_intents[t.id] = intent
            intent.update(frame_idx, t.bev_pos, self.fps)

        # Pulizia delle intenzioni dei pedoni non più tracciati
        stale_after = max(30, int(round(3.0 * self.fps)))
        for pid in list(self.ped_intents.keys()):
            if pid not in seen and (frame_idx - self.ped_intents[pid].last_seen_frame) > stale_after:
                del self.ped_intents[pid]

    def _format_timestamp(self, frame_idx: int) -> str:
        """Converte il numero di frame nel minutaggio del video (es. 01m23s450ms)."""
        total_sec = max(0.0, float(frame_idx - 1) / max(self.fps, 1e-6))
        minutes = int(total_sec // 60)
        seconds = int(total_sec % 60)
        millis = int(round((total_sec - int(total_sec)) * 1000))
        if millis >= 1000:
            seconds += 1
            millis -= 1000
            if seconds >= 60:
                minutes += 1
                seconds -= 60
        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours:02d}h{minutes:02d}m{seconds:02d}s{millis:03d}ms"
        return f"{minutes:02d}m{seconds:02d}s{millis:03d}ms"

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

        # 0. Aggiorna l'intento di movimento dei pedoni in ROI (serve al gate sui falsi positivi)
        self._update_intents(tracks, frame_idx)

        # 1. Smistamento tracce e check geometrici sulla ROI
        #    (per i pedoni usa la posizione filtrata dell'intento, così il
        #    jitter di detection non fa entrare/uscire il pedone dalla ROI)
        for t in tracks:
            if t.cls == 0:
                it = self.ped_intents.get(t.id)
                pos = it.hist[-1][1] if (it is not None and len(it.hist) > 0) else t.bev_pos
                if self.roi.contains_point(pos):
                    pedestrians_in_roi.append(t)
            elif t.cls == 2:
                if getattr(t, 'bev_quad', None) is not None and self.roi.overlaps_polygon(t.bev_quad):
                    vehicles_in_roi.append(t)

        # Intenti solo per i pedoni effettivamente in ROI (usati da evaluate_conflict)
        roi_ped_intents = {p.id: self.ped_intents[p.id] for p in pedestrians_in_roi if p.id in self.ped_intents}

        # Ordina per ID per stabilità visiva nella tabella di debug
        pedestrians_in_roi.sort(key=lambda x: x.id)
        vehicles_in_roi.sort(key=lambda x: x.id)

        ttc_matrix = {}         # (veh_id, ped_id) -> float
        pair_status = {}       # (veh_id, ped_id) -> label leggibile
        min_ttc_per_veh = {}    # veh_id -> float (min TTC positivo)
        confirm_counts = {}     # veh_id -> frame di conferma consecutivi
        newly_triggered = set() # veh_id multati in questo esatto frame

        # 2. Valutazione conflitto intent-gated e controllo infrazioni
        for veh in vehicles_in_roi:
            min_pos_ttc = float('inf')
            trigger_ped = None
            trigger_ttc = None

            for ped in pedestrians_in_roi:
                intent = roi_ped_intents.get(ped.id)
                ttc, label = self.evaluate_conflict(veh, ped, intent)
                ttc_matrix[(veh.id, ped.id)] = ttc
                pair_status[(veh.id, ped.id)] = label

                if ttc >= 0.0 and ttc < min_pos_ttc:
                    min_pos_ttc = ttc

                if 0.0 <= ttc <= self.ttc_threshold:
                    if trigger_ttc is None or ttc < trigger_ttc:
                        trigger_ttc = ttc
                        trigger_ped = ped

            min_ttc_per_veh[veh.id] = min_pos_ttc

            # 2b. Persistenza: il conflitto deve essere confermato su N frame
            # consecutivi e riferirsi allo stesso pedone.
            pending = self.pending_conflicts.get(veh.id)
            if trigger_ped is None:
                self.pending_conflicts.pop(veh.id, None)
                confirm_counts[veh.id] = 0
            else:
                if (pending is not None
                        and pending['ped_id'] == trigger_ped.id
                        and (frame_idx - pending['last_frame']) == 1):
                    pending['count'] += 1
                    pending['last_frame'] = frame_idx
                else:
                    pending = {'ped_id': trigger_ped.id, 'count': 1, 'last_frame': frame_idx}
                    self.pending_conflicts[veh.id] = pending
                confirm_counts[veh.id] = pending['count']

                # Se il veicolo non è già stato multato e il conflitto è confermato
                if (pending['count'] >= self.confirm_frames
                        and veh.id not in self.history_violators):
                    self._trigger_violation(frame, frame_idx, veh, trigger_ped, trigger_ttc, H_inv)
                    newly_triggered.add(veh.id)

        # 3. Gestione infrazioni attive: prosecuzione/terminazione della clip
        active_ids = list(self.active_violations.keys())
        veh_track_dict = {t.id: t for t in tracks if t.cls == 2}
        ped_track_dict = {t.id: t for t in tracks if t.cls == 0}  # <--- AGGIUNTO

        for veh_id in active_ids:
            # Evita di scrivere due volte il primo frame appena registrato in _trigger_violation
            if veh_id in newly_triggered:
                continue

            state = self.active_violations[veh_id]
            veh_t = veh_track_dict.get(veh_id)
            ped_t = ped_track_dict.get(state.get('pedestrian_id'))  # <--- AGGIUNTO

            if veh_t is not None:
                # Il veicolo e' ancora tracciato: la clip continua.
                state['last_track_frame'] = frame_idx
                state['track_lost_frame'] = None
                self._append_clip_frame(state, frame, veh_t, ped_t, H_inv)  # <--- MODIFICATO
            else:
                # Veicolo perso: si continua a registrare finche' non scade
                # la finestra di fallback (in secondi).
                if state.get('track_lost_frame') is None:
                    state['track_lost_frame'] = frame_idx
                self._append_clip_frame(state, frame, None, ped_t, H_inv)   # <--- MODIFICATO
                lost_for = frame_idx - state['track_lost_frame']
                if lost_for >= self.clip_track_timeout_s * self.fps:
                    state['close_reason'] = f"timeout_track_lost_{lost_for}_frames"
                    self._close_violation(veh_id, frame, frame_idx, state['close_reason'])
                    continue

            frames_elapsed = frame_idx - state['start_frame']
            if veh_t is not None:
                exited_roi = False
                if getattr(veh_t, 'bev_quad', None) is not None:
                    exited_roi = not self.roi.overlaps_polygon(veh_t.bev_quad)
                else:
                    exited_roi = not self.roi.contains_point(veh_t.bev_pos)
                if exited_roi:
                    self._close_violation(veh_id, frame, frame_idx, "exited_roi")
                    continue

            if frames_elapsed >= self.clip_absolute_timeout_s * self.fps:
                self._close_violation(veh_id, frame, frame_idx, "absolute_timeout")

        return {
            "frame_idx": frame_idx,
            "pedestrians_in_roi": pedestrians_in_roi,
            "vehicles_in_roi": vehicles_in_roi,
            "vehicles_in_roi_ids": {v.id for v in vehicles_in_roi},
            "violating_vehicle_ids": set(self.history_violators),
            "newly_triggered_ids": newly_triggered,
            "ttc_matrix": ttc_matrix,
            "pair_status": pair_status,
            "min_ttc_per_veh": min_ttc_per_veh,
            "ttc_threshold": self.ttc_threshold,
            "confirm_frames_required": self.confirm_frames,
            "confirm_counts": confirm_counts,
            "ped_intents": {pid: it.as_dict() for pid, it in roi_ped_intents.items()},
        }

    @staticmethod
    def _draw_centroid_markers(img, markers):
        """Disegna un marker (cerchio + etichetta) su ogni centroid in prospettiva.

        `markers` è una lista di (punto [x, y], etichetta, colore BGR).
        """
        out = img
        for point, label, color in markers:
            if point is None:
                continue
            x, y = int(point[0]), int(point[1])
            if not (0 <= x < out.shape[1] and 0 <= y < out.shape[0]):
                continue
            cv2.circle(out, (x, y), 12, color, 3, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(out, label, (x + 16, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, lineType=cv2.LINE_AA)
        return out

    def _open_clip(self, path, frame):
        """Apre il writer video per la clip dell'infrazione (risoluzione nativa)."""
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.clip_fourcc)
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            writer.release()
            return None
        return writer

    def _append_clip_frame(self, state, frame, veh_t, ped_t, H_inv):
        """Appende un frame alla clip, marcando il centroide del veicolo e del pedone coinvolto."""
        writer = state.get('writer')
        if writer is None:
            return

        markers = []
        if H_inv is not None:
            if veh_t is not None:
                veh_persp = self.bev_to_perspective(veh_t.bev_pos, H_inv)
                markers.append((veh_persp, f"V{state['vehicle_id']}", (0, 0, 255)))      # Rosso (BGR)

            if ped_t is not None:
                ped_persp = self.bev_to_perspective(ped_t.bev_pos, H_inv)
                markers.append((ped_persp, f"P{state['pedestrian_id']}", (0, 255, 255))) # Giallo (BGR)

        if not markers:
            # Nessun marker da disegnare: si scrive il frame originale senza copia.
            writer.write(frame)
        else:
            marked = self._draw_centroid_markers(frame.copy(), markers)
            writer.write(marked)

        state['clip_frames'] = state.get('clip_frames', 0) + 1

    def _trigger_violation(self, frame, frame_idx, veh, ped, ttc, H_inv):
        self.history_violators.add(veh.id)

        veh_persp = self.bev_to_perspective(veh.bev_pos, H_inv) if H_inv is not None else [0, 0]
        ped_persp = self.bev_to_perspective(ped.bev_pos, H_inv) if H_inv is not None else [0, 0]

        # --- MODIFICA 1: Minutaggio nel nome della clip anziché il numero di frame ---
        ts_str = self._format_timestamp(frame_idx)
        prefix = f"violation_v{veh.id}_t{ts_str}"
        json_path = os.path.join(self.out_dir, f"{prefix}.json")
        clip_name = f"{prefix}.mp4"

        intent = self.ped_intents.get(ped.id)

        data = {
            "video_frame": frame_idx,
            "video_timestamp": ts_str,
            "vehicle_id": veh.id,
            "pedestrian_id": ped.id,
            "ttc_seconds": round(ttc, 3),
            "vehicle_centroid_perspective": veh_persp,
            "pedestrian_centroid_perspective": ped_persp,
            "pedestrian_intent": intent.as_dict() if intent is not None else None,
            "clip": {
                "file": clip_name,
                "fps": round(self.fps, 3),
                "start_frame": frame_idx,
                "start_timestamp": ts_str,
                "end_frame": None,
                "frames": 0,
                "exit_reason": "pending"
            }
        }

        state = {
            "start_frame": frame_idx,
            "json_path": json_path,
            "prefix": prefix,
            "data": data,
            "vehicle_id": veh.id,
            "pedestrian_id": ped.id,  # <--- MODIFICA 2: salva l'ID del pedone coinvolto
            "writer": None,
            "clip_frames": 0,
            "last_track_frame": frame_idx,
            "track_lost_frame": None,
        }
        self.active_violations[veh.id] = state

        if self.save_files and self.clip_video:
            writer = self._open_clip(os.path.join(self.out_dir, clip_name), frame)
            if writer is not None:
                state['writer'] = writer

        # Il frame di inizio entra nella clip (marca sia il veicolo che il pedone).
        self._append_clip_frame(state, frame, veh, ped, H_inv)

        if self.save_files:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

        print(f"\n🚨 INFRAZIONE: Veicolo {veh.id} vs Pedone {ped.id} (TTC: {ttc:.2f}s) a {ts_str} (frame {frame_idx})")

    def close_all(self, reason: str = "end_of_stream"):
        """Chiude tutte le infrazioni ancora attive (rilascia i writer delle clip)."""
        for veh_id in list(self.active_violations.keys()):
            state = self.active_violations.get(veh_id)
            frame_idx = state['start_frame'] if state else 0
            self._close_violation(veh_id, None, frame_idx, reason)

    def _close_violation(self, veh_id, frame, frame_idx, reason, tracks=None, H_inv=None):
        state = self.active_violations.pop(veh_id)

        data = state['data']
        clip = data['clip']
        clip['exit_reason'] = reason
        clip['end_frame'] = frame_idx
        clip['frames'] = state.get('clip_frames', 0)
        data['exit_frame'] = frame_idx

        writer = state.get('writer')
        if writer is not None:
            writer.release()
            state['writer'] = None
        elif not (self.save_files and self.clip_video):
            # Clip non generata (--no-clip o salvataggio disabilitato).
            clip['file'] = None

        if self.save_files:
            with open(state['json_path'], 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

        print(f"\n✅ INFRAZIONE CHIUSA: Veicolo {veh_id} al frame {frame_idx} (Motivo: {reason})")
