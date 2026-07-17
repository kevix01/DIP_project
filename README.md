# README operativo

Breve guida pratica per eseguire detection YOLO su BEV e tracking nella pipeline `trafcam_3d`, basata sulla repository del paper *Monocular 3D Vehicle Detection Using Uncalibrated Traffic Cameras through Homography*.

## Panoramica

La pipeline lavora in due fasi principali:

1. **Detection su BEV**: YOLO legge `bev.avi` e salva le rotated bounding boxes nel file `bev.txt`.
2. **Tracking**: lo script `rbox_tracking_BrnoCompSpeed` legge `bev.avi`, `ori.avi`, `bev.txt` e la calibrazione, trasforma le detection in world space, esegue tracking e produce output video/txt/json.

## Struttura dati attesa

Per una sessione in stile BrnoCompSpeed, la repo si aspetta una struttura simile:

```text
trafcam_3d/
└── data/
    └── BrnoCompSpeed_session5_left/
        ├── videos/
        │   ├── bev.avi
        │   └── ori.avi
        ├── outputs/
        │   └── bev.txt
        └── calibs/
            └── system_dubska_optimal_calib.json
```

Note utili:

- `ori.avi` è il video originale della camera.
- `bev.avi` è il video trasformato in bird's-eye view.
- `bev.txt` contiene le detection YOLO per frame nel formato `frame_id x y w h r class_id conf`.
- `system_dubska_optimal_calib.json` contiene i parametri di calibrazione usati dalla parte BEV/tracking.

## Setup ambiente

Installare i requisiti dai file dei requirements.

## Detection YOLO su BEV

Eseguire la detection dalla cartella `yolov3`.

```bash
cd trafcam_3d/yolov3

python detect.py \
  --rotated --rotated-anchor --half-angle --tail --dual-view \
  --weights weights/best_ra_tail_180_dv.pt \
  --names data/bev_network.names \
  --single-cls \
  --img-size 448 \
  --conf-thres 0.2 \
  --iou-thres 0.2 \
  --source ../data/BrnoCompSpeed_session5_left/videos/bev.avi \
  --output ../data/BrnoCompSpeed_session5_left/outputs \
  --save-txt \
  --use-mask
```

Osservazioni:

- `--source` deve puntare al video **BEV**, non al video originale.
- `--save-txt` è necessario per generare il file di predizioni usato poi dal tracking.
- `--use-mask` va usato solo se sono disponibili i file mask per quella scena; in caso contrario va rimosso.
- `--img-size` deve essere compatibile con il modello e in pratica è tipicamente un multiplo di 32.

## Tracking

Eseguire il tracking dalla root del progetto:

```bash
cd trafcam_3d
python -m bev.tool.rbox_tracking_BrnoCompSpeed \
  --video-root /data \
  --video-tag 5_left \
  --fps 25 \
  --no-vis
```

Significato dei parametri principali:

- `--video-root`: cartella che contiene le sessioni (`session5_left`, ecc.).
- `--video-tag`: identificatore della sessione/camera, ad esempio `5_left`.
- `--fps`: numero di frame per secondo **logico** usato dal tracker; influenza `max_age`, `min_hits_init` e `min_hits_recover` del tracker SORT modificato.
- `--no-vis`: evita la visualizzazione live con `imshow`, ma non impedisce il salvataggio video.

## Output del tracking

Tipicamente il tracking produce:

- `outputs/bev_tracking.avi`: video BEV con box e ID.
- `outputs/ori_tracking.avi`: video originale con proiezione dei punti tracciati.
- `outputs/bev_tracking.txt`: log testuale delle tracce.
- `outputs/system_ori_tracking.json`: esportazione JSON finale.

Nota: nel video originale `ori_tracking.avi` lo script base disegna soprattutto **punti/circles** proiettati sulle auto, mentre gli ID testuali sono mostrati in modo più esplicito nel video BEV.

## Note operative

- Il tracking usa come input i video `bev.avi` e `ori.avi`, mentre usa `bev.txt` solo come sorgente di detection per frame.
- Se la detection è stata eseguita solo su una parte del video, i `frame_id` in `bev.txt` devono restare coerenti con i frame letti dal tracker.
- Se il file AVI di output risulta quasi vuoto, una causa comune è che `VideoWriter.write()` sia stato eseguito solo nel ramo di visualizzazione; conviene tenere separati salvataggio video e `imshow()`.
- Se OpenCV non supporta bene `x264`, può essere necessario usare un codec più semplice come `MJPG` nel `VideoWriter`.

## Flusso minimo consigliato

1. Preparare `ori.avi`, `bev.avi` e il file di calibrazione.
2. Lanciare YOLO su `bev.avi` per generare `bev.txt`.
3. Lanciare il tracking con `rbox_tracking_BrnoCompSpeed`.
4. Verificare `bev_tracking.avi`, `ori_tracking.avi` e `bev_tracking.txt`.
