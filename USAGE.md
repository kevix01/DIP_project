# Command reference

All tools run from the repository root. The v2 checkpoints need `v2/hybrid_v2.py` importable
(`bev_realtime_v2.py` arranges that itself; in your own code `import v2.hybrid_v2` before
`YOLO(...)`). Never run Python from a directory that contains an `ultralytics/` source clone.

## `debug_violations.py` — violation flow (video → BEV → TTC)

Detection + 3D cuboids from one v2 checkpoint, tracking, the orthogonality-homography
bird's-eye view, and the pedestrian-crossing violation detector. This is the only
entry point; `bev_realtime_v2.py` / `bev_realtime.py` are libraries it drives.

```bash
python debug_violations.py --input clip.mp4 --kp-model checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine \
    --kp-imgsz 640 --kp-conf 0.1 --kp-vis 0.6 --class-conf 0:0.4,1:0.5,3:0.5 \
    --smooth 11 --device cuda:0 --output out.mp4 \
    --bev-unit-file homography_bangkok_bev_unit.txt --roi-file roi_bev.json --ttc-thresh 2.5 --confirm-frames 6
```

| Flag | Default | Meaning |
|---|---|---|
| `--kp-vis F` | 0.5 | min per-corner cuboid confidence (3rd keypoint channel) to draw a cuboid; below it the object is tracked box-only |
| `--class-conf 'id:conf,...'` | none | per-class minimum confidence, COCO ids (`0:0.4,1:0.5,3:0.5` keeps traffic lights from becoming bikes) |
| `--layout side|dashboard` | side | `side`: camera and BEV side by side; `dashboard`: 16:9 composite with info strip (landscape sources) |
| `--bev-canvas WxH` | source size | BEV panel size (dashboard: 640x1080) |
| `--caption TEXT` | | scene caption in the UI; the static/moving camera tag is appended automatically |
| `--title s,in,hold,out` | | opening titles over this clip, timings in source seconds |
| `--object-px PX` | 40 | BEV zoom: fixed px per median footprint |
| `--label-min-h PX` | 26 | no id chip on boxes shorter than this; the cuboid is still drawn |
| `--fps F` | from source | frame rate the intent/timeout thresholds assume (in seconds) |
| `--source-type auto|video|stream` | auto | input kind |
| `--track-high-conf F` | | high-confidence track gate |
| `--track-ocm-weight F` | | OCM weight in the tracker |
| `--bev-unit-file PATH` | | text file holding the bev_unit (meters per BEV pixel) |
| `--show-bev-ids` | off | draw track ids on the BEV radar |

Recommended on aerial footage: `--kp-imgsz 960` (or 1280). The models are trained at 640; a
larger input recovers far and small traffic and the cuboids stay tight.

### TensorRT

`--export tensorrt --half` exports an FP16 engine next to the checkpoint on first use (about
20 s for the x model) and loads it on every later run; the filename records imgsz and precision,
so an engine built for another resolution is never silently reused. `--export onnx` works the
same way. Requires **TensorRT 10.x** (`pip install "tensorrt>=10,<11"`) — 11.x removed
`BuilderFlag.FP16` and the Ultralytics exporter fails on it.

Loading an exported engine yourself needs the task passed explicitly:

```python
from ultralytics import YOLO
model = YOLO("urbanomnidetect_v2_x_640.engine", task="pose")   # without task=, keypoints are None
```

The exporter records the task as `detect` in the engine metadata, so without `task="pose"` the
pose predictor never runs: boxes still come back and `results.keypoints` is `None`, which is a
quiet failure. `load_detector(..., task="pose")` in `uod/model.py` does this for you.

Measured on the x model at 960 px, one RTX PRO 6000, 60 frames:

| | ms/frame | fps | cuboid corner agreement vs PyTorch |
|---|---|---|---|
| PyTorch FP32 | 7.7 | 130 | reference |
| TensorRT FP16 | 4.6 | 217 | median 0.9 px, p95 4.3 px |

End to end (detector + tracking + ground solve + refinement + render) the same clip runs at
48 fps with TensorRT against 45 with PyTorch: past this point the detector is not the bottleneck.

## `export_engine.py` — standalone TensorRT/ONNX export

See `EXPORT_ENGINE.md` for the full guide. Short version:

```bash
python export_engine.py --weights checkpoints/urbanomnidetect_v2_x_640.pt --imgsz 640 --half   # -> ..._640_fp16.engine
python export_engine.py --weights checkpoints/urbanomnidetect_v2_x_640.pt --imgsz 960 --format onnx
```

## Infrazioni sulle strisce: clip e JSON

Quando si passa `--roi-file roi_bev.json`, la pipeline valuta i conflitti veicolo/pedone nella ROI
(strisce pedonali) e salva per ogni infrazione un **mini filmato** nella cartella
`violations_<nome_video>/`.

```bash
python debug_violations.py --input clip.mp4 --kp-model checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine \
    --device cuda:0 --roi-file roi_bev.json --ttc-thresh 2.5 --confirm-frames 6 \
    --output out.mp4
```

| Flag | Default | Meaning |
|---|---|---|
| `--roi-file PATH` | required | JSON con i punti BEV (`bev_points`) e immagine (`image_points`) della zona strisce |
| `--ttc-thresh S` | 2.5 | soglia TTC (secondi) per considerare il conflitto un'infrazione |
| `--confirm-frames N` | 3 | frame consecutivi di conflitto richiesti per confermare l'infrazione |
| `--clip-track-timeout-s S` | 1.5 | secondi di registrazione della clip **dopo** la perdita del tracking del veicolo; la clip continua anche senza il veicolo tracciato |
| `--clip-absolute-timeout-s S` | 3.0 | durata massima (secondi) della clip di un'infrazione |
| `--no-clip` | off | non genera il mini filmato: salva solo il JSON |

**La clip** (`violation_v<veicolo>_f<frame>.mp4`) viene scritta in streaming durante
l'elaborazione: contiene esattamente i frame dal trigger dell'infrazione alla sua chiusura, alla
**risoluzione nativa del video sorgente**, con il **solo centroide del veicolo infractor**
(marker rosso + etichetta `V<id>`) sovrapposto in ogni frame in cui il veicolo è tracciato.
Non viene generato nessun JPG.

**Chiusura dell'infrazione** (e quindi fine della clip):

| Causa | `clip.exit_reason` | Quando |
|---|---|---|
| veicolo uscito dalla ROI | `exited_roi` | normale: il veicolo ha superato le strisce |
| tracking perso | `timeout_track_lost_<N>_frames` | il veicolo non è più tracciato da `--clip-track-timeout-s` |
| durata massima | `absolute_timeout` | durata superiore a `--clip-absolute-timeout-s` |
| fine stream | `end_of_stream` | il video è terminato con l'infrazione ancora attiva |

**JSON** (`violation_v<veicolo>_f<frame>.json`): frame del trigger, id veicolo e pedone, TTC,
centroide prospettico del veicolo e del pedone **al frame dell'infrazione**, stato e velocità
dell'intento del pedone, e il blocco `clip` con nome file, fps, frame iniziale/finale, numero di
frame e motivo di chiusura.

## `bev_realtime.py` / `bev_realtime_v2.py` — shared pipeline flags (libraries of `debug_violations.py`)

```
--input PATH|DIR|N          video path, image directory or webcam index
--kp-model PATH             checkpoint
--kp-imgsz N  --kp-conf F   inference size and confidence
--homography ortho|adam|none   BEV solver
--device DEV  --export none|engine|tensorrt|onnx  --half
--ground-indices 0,1,2,3    force the ground keypoint indices (auto-detected by vote otherwise)
--no-3d  --no-snap-rect  --no-trails  --trail-len N  --no-fov
--no-stabilize  --bev-window N     homography smoothing / rolling footprint buffer
--max-age N  --min-hits N     tracker
--smooth N   offline zero-phase smoothing of every track and the homography (odd N, e.g. 11; 0 = live)
--output PATH  --display  --max-frames N
```

## ROI / homography tools

```bash
python select_roi_bev.py     # pick the zebra-crossing ROI on a frame -> roi_bev.json
python extract_homography.py # solve the image->ground homography -> homography_bangkok.npy
python homography_rt.py      # runtime homography helper
```

## Webapp di debug

```bash
# unico comando, identico su Linux/macOS e Windows
.venv/bin/python webapp/run_server.py          # Linux/macOS
.venv\Scripts\python.exe webapp\run_server.py  # Windows

.venv/bin/python webapp/run_server.py --foreground  # server in primo piano
.venv/bin/python webapp/run_server.py --stop       # ferma il server (pidfile)
```

Il launcher usa l'interprete della venv del progetto (`.venv/bin/python` o
`.venv/Scripts/python.exe`) e avvia uvicorn in background; i processi figli
(export, omografia, selettore ROI) usano la stessa venv.
