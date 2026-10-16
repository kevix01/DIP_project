import os
import sys

# Assicura che la directory corrente sia nel path di sistema
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Importa il modulo custom *prima* di caricare YOLO per registrare la classe del modello
import v2.hybrid_v2

from ultralytics import YOLO

def main():
    print("Caricamento del modello PyTorch...")
    model = YOLO("checkpoints/urbanomnidetect_v2_x_640.pt")
    
    print("Avvio esportazione TensorRT (potrebbe richiedere alcuni minuti)...")
    model.export(format="engine", imgsz=640, half=True, device=0)
    print("Esportazione completata!")

if __name__ == "__main__":
    main()