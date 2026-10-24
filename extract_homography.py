import cv2
import numpy as np
import argparse
import sys
import os

# Assicura che i moduli locali siano raggiungibili
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bev_realtime_v2 as bev_v2
import v2.hybrid_v2  
from ultralytics import YOLO

class SafeArgs:
    """Intercetta le richieste di parametri legacy per evitare AttributeError."""
    def __init__(self):
        self.smooth = 11
        self.smooth_poly = 2
        self.homography = "auto"
        
    def __getattr__(self, item):
        # Se lo script legacy cerca un attributo non definito, restituisce None
        return None

def main():
    parser = argparse.ArgumentParser(description="Estrae la matrice di omografia globale.")
    parser.add_argument("--input", required=True, help="Video di input per la calibrazione")
    parser.add_argument("--kp-model", required=True, help="Checkpoint del modello (.pt o .engine)")
    parser.add_argument("--output", default="homography.npy", help="Nome del file di output (.npy)")
    parser.add_argument("--imgsz", type=int, default=640, help="Risoluzione di inferenza")
    parser.add_argument("--device", default="cuda:0", help="Device di inferenza")
    parser.add_argument("--max-frames", type=int, default=1500, help="Stop automatico a 1500 frame (~50 sec a 30fps). Imposta a 0 per leggere tutto il video.")
    args = parser.parse_args()

    print(f"Caricamento modello {args.kp_model}...")
    model = YOLO(args.kp_model, task="pose")
    
    bev_v2.V2Pipeline.CONFIG.update(
        kp_vis=0.5, 
        suppress_nested=0.85, 
        class_conf={0: 0.4, 1: 0.5, 3: 0.5}
    )
    bev_v2.POST.update(
        refine=True, 
        world_lock=True,
        vel_heading=True,
        hold_still=True,
        bev_fit=True
    )
    
    pipe = bev_v2.V2Pipeline(pose_model=model, device=args.device, kp_imgsz=args.imgsz)
    
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video {args.input}")
        return

    pipe.src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit_frames = min(total_frames, args.max_frames) if args.max_frames > 0 else total_frames
    
    states = []
    frame_count = 0
    
    print(f"Analisi in background in corso (Limite impostato a {limit_frames} frame per ottimizzare i tempi)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        state = pipe.step(frame)
        states.append(state)
        
        frame_count += 1
        print(f"\rElaborazione frame {frame_count}/{limit_frames} ({(frame_count/limit_frames)*100:.1f}%)", end="", flush=True)
        
        # Stop anticipato se superiamo i frame richiesti per calibrare
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break
            
    cap.release()
    print(f"\nEstrazione completata ({len(states)} frame). Avvio ottimizzazione globale del piano stradale...")

    if not states or not any(s.tracks for s in states):
        print("Errore: Nessun veicolo rilevato nel video. Impossibile calcolare l'omografia.")
        return

    # Inietta l'oggetto blindato per scavalcare ogni check legacy di argparse
    dummy_args = SafeArgs()
    
    bev_v2._stabilize_offline_v2(pipe, states, fixed=True, traj_C=None, traj_steps=None, args=dummy_args)

    H_matrix = None
    for s in states:
        if getattr(s, 'H', None) is not None:
            H_matrix = np.asarray(s.H, dtype=np.float64)
            break
    
    if H_matrix is not None and np.all(np.isfinite(H_matrix)):
        # estrazione della bev_unit
        bev_unit = pipe._bev_unit_locked
        if bev_unit is None or not np.isfinite(bev_unit) or bev_unit <= 1e-9:
            raise SystemExit(
                "bev_unit non calibrato: servono almeno 128 footprint "
                "veicolo valide nel video."
            )

        bev_unit = float(bev_unit)
        unit_output = os.path.splitext(args.output)[0] + "_bev_unit.txt"

        # salvataggio su file della bev_unit
        with open(unit_output, "w", encoding="utf-8") as f:
            f.write(f"{bev_unit:.12g}\n")

        print(f"[SUCCESSO] bev_unit: {bev_unit:.12g}")
        print(f"[SUCCESSO] Valore salvato in: {unit_output}")

        # salvataggio matrice di omografia
        np.save(args.output, H_matrix)
        
        # Genera anche un file testuale comodamente leggibile da umano per verifiche veloci
        txt_output = args.output.replace('.npy', '.txt')
        np.savetxt(txt_output, H_matrix, fmt='%.6e')
        
        print(f"\n[SUCCESSO] Matrice salvata in formato numpy in: {args.output}")
        print(f"[SUCCESSO] Matrice salvata in formato testo in: {txt_output}")
        print("\nValori dell'omografia:")
        print(H_matrix)
    else:
        print("\n[ERRORE] L'ottimizzatore non è riuscito a convergere su un piano stradale valido.")
        print("Prova a fornire un video con traffico meglio distribuito su tutta la scena.")

if __name__ == "__main__":
    main()