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
    --homography-file homography_bangkok.npy \
    --bev-unit-file homography_bangkok_bev_unit.txt --roi-file roi_bev.json --ttc-thresh 2.5 --confirm-frames 4
```

`debug_violations.py` unisce due parser: quello della pipeline (`bev_realtime.py`) e
quello v2 (`bev_realtime_v2.py`). Le tabelle seguenti elencano **tutte** le flag
accettate, con il default effettivo.

#### Flag della pipeline base

| Flag | Default | Meaning |
|---|---|---|
| `--input PATH\|DIR\|N` | obbligatorio | video path, image directory, or webcam index |
| `--kp-model PATH` | obbligatorio | checkpoint pose (Ultralytics YOLO) |
| `--kp-imgsz N` | 640 | inference size |
| `--kp-conf F` | 0.25 | minimum detection confidence |
| `--homography ortho\|adam\|none` | ortho | BEV solver ('adam' è alias CLI del solver ortogonale veloce) |
| `--device DEV` | cpu | device di inferenza (`cuda:0`, `cpu`, …) |
| `--export none\|engine\|tensorrt\|onnx` | none | esporta il checkpoint a TensorRT/ONNX e lo ricarica |
| `--half` | off | FP16 per l'engine esportato |
| `--ground-indices 0,1,2,3` | auto | indici keypoint a terra forzati (altrimenti votati) |
| `--no-3d` | off | disegna solo i quadrilateri a terra, senza cuboidi |
| `--no-snap-rect` | off | mostra le footprint BEV grezze, non rettangularizzate |
| `--no-trails` | off | disabilita le scie di movimento in BEV |
| `--trail-len N` | 16 | frame di scia BEV mantenuti |
| `--no-stabilize` | off | risolve l'omografia BEV per frame invece che sul buffer rolling |
| `--bev-window N` | 512 | dimensione del buffer rolling (footprint) per un asse BEV stabile |
| `--no-fov` | off | non disegna il campo visivo della telecamera nella BEV |
| `--max-age N` | 30 | frame persi tollerati prima di considerare una traccia persa |
| `--min-hits N` | 2 | hit minimi per considerare una traccia confermata |
| `--smooth N` | 0 | finestra di smoothing offline zero-phase (0 = off, streaming live); si consiglia un numero dispari (es. 11) |
| `--output PATH` | nessuno | scrive il video completo annotato |
| `--display` | off | mostra una finestra live |
| `--max-frames N` | 0 | limite di frame; 0 = tutti |

#### Flag v2 (pipeline e rendering)

| Flag | Default | Meaning |
|---|---|---|
| `--kp-vis F` | 0.5 | min per-corner cuboid confidence (3rd keypoint channel) to draw a cuboid; below it the object is tracked box-only |
| `--class-conf 'id:conf,...'` | none | per-class minimum confidence, COCO ids (`0:0.4,1:0.5,3:0.5` keeps traffic lights from becoming bikes) |
| `--layout side\|dashboard` | side | `side`: camera and BEV side by side; `dashboard`: 16:9 composite with info strip (landscape sources) |
| `--bev-canvas WxH` | source size | BEV panel size (dashboard: 640x1080) |
| `--caption TEXT` | | scene caption in the UI; the static/moving camera tag is appended automatically |
| `--title s,in,hold,out` | | opening titles over this clip, timings in source seconds |
| `--object-px PX` | 40 | BEV zoom: fixed px per median footprint |
| `--label-min-h PX` | 26 | no id chip on boxes shorter than this; the cuboid is still drawn |
| `--fps F` / `--refine-fps F` | 0 (da sorgente) | frame rate the intent/timeout thresholds assume (in seconds) |
| `--source-type auto\|video\|stream` | auto | input kind |
| `--track-high-conf F` | `max(--kp-conf, 0.35)` | high-confidence track gate |
| `--track-ocm-weight F` | 0.25 | OCM weight in the tracker |
| `--show-bev-ids` | off | draw track ids on the BEV radar |
| `--homography-file PATH` | `homography_bangkok.npy` | file `.npy` con l'omografia statica calibrata |
| `--bev-unit-file PATH` | nessuno (stima automatica) | file `.txt` con la `bev_unit` calibrata (metri per pixel BEV) |

#### Flag del rilevatore di infrazioni

| Flag | Default | Meaning |
|---|---|---|
| `--roi-file PATH` | obbligatorio | JSON con i punti BEV (`bev_points`) e immagine (`image_points`) della zona strisce |
| `--ttc-thresh S` | 2.5 | soglia TTC (secondi) per considerare il conflitto un'infrazione |
| `--confirm-frames N` | 4 | frame consecutivi di conflitto TTC richiesti per confermare l'infrazione |
| `--clip-track-timeout-s S` | 1.5 | secondi di registrazione della clip **dopo** la perdita del tracking del veicolo; la clip continua anche senza il veicolo tracciato |
| `--clip-absolute-timeout-s S` | 3.0 | durata massima (secondi) della clip di un'infrazione |
| `--no-clip` | off | non genera il mini filmato: salva solo il JSON |

#### Flag specifiche del debug

| Flag | Default | Meaning |
|---|---|---|
| `--table-height PX` | 280 | altezza in pixel della tabella TTC in basso |
| `--no-save-violations` | off | disabilita il salvataggio su disco di JSON e clip |

**Abbreviazioni disabilitate**: i parser usano `allow_abbrev=False`. Il flag accettato
per la scala BEV è **solo `--bev-unit-file`** (file di testo con la `bev_unit`);
`--bev-unit` e `--bev-un` vengono rifiutati con un errore argparse.


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
    --device cuda:0 --roi-file roi_bev.json --ttc-thresh 2.5 --confirm-frames 4 \
    --output out.mp4
```

| Flag | Default | Meaning |
|---|---|---|
| `--roi-file PATH` | required | JSON con i punti BEV (`bev_points`) e immagine (`image_points`) della zona strisce |
| `--ttc-thresh S` | 2.5 | soglia TTC (secondi) per considerare il conflitto un'infrazione |
| `--confirm-frames N` | 4 | frame consecutivi di conflitto richiesti per confermare l'infrazione |
| `--clip-track-timeout-s S` | 1.5 | secondi di registrazione della clip **dopo** la perdita del tracking del veicolo; la clip continua anche senza il veicolo tracciato |
| `--clip-absolute-timeout-s S` | 3.0 | durata massima (secondi) della clip di un'infrazione |
| `--no-clip` | off | non genera il mini filmato: salva solo il JSON |

**Flag con abbreviazione disabilitata**: i parser di `debug_violations.py` usano
`allow_abbrev=False`. Il flag accettato è **solo `--bev-unit-file`** (file di testo
con la `bev_unit`); `--bev-unit` e `--bev-un` vengono rifiutati con un errore argparse.

**La clip** (`violation_v<veicolo>_t<MMmSSsMMMms>.mp4`; per video di almeno
un'ora il minutaggio è preceduto da `HHh`) viene scritta in streaming durante
l'elaborazione: contiene esattamente i frame dal trigger dell'infrazione alla sua chiusura, alla
**risoluzione nativa del video sorgente**, con i **centroidi dei soggetti coinvolti**
sovrapposti in ogni frame in cui sono tracciati: veicolo infractor (marker rosso + etichetta `V<id>`)
e pedone (marker giallo + etichetta `P<id>`). Non viene generato nessun JPG.

**Chiusura dell'infrazione** (e quindi fine della clip):

| Causa | `clip.exit_reason` | Quando |
|---|---|---|
| veicolo uscito dalla ROI | `exited_roi` | normale: il veicolo ha superato le strisce |
| tracking perso | `timeout_track_lost_<N>_frames` | il veicolo non è più tracciato da `--clip-track-timeout-s` |
| durata massima | `absolute_timeout` | durata superiore a `--clip-absolute-timeout-s` |
| fine stream | `end_of_stream` | il video è terminato con l'infrazione ancora attiva |

**JSON** (`violation_v<veicolo>_t<MMmSSsMMMms>.json`): frame del trigger, id veicolo e pedone, TTC,
centroide prospettico del veicolo e del pedone **al frame dell'infrazione**, stato e velocità
dell'intento del pedone, e il blocco `clip` con nome file, fps, frame iniziale/finale, numero di
frame e motivo di chiusura.

## `bev_realtime.py` / `bev_realtime_v2.py` — shared pipeline flags (libraries driven by `debug_violations.py`)

These two modules are libraries, not standalone entry points: they expose the
argparse fragments used by `debug_violations.py` (`build_argparser()` and
`build_v2_argparser()`) and the pipeline/rendering classes it drives. The full
flag tables with defaults are in the "Flag della pipeline base" and "Flag v2"
sections above; the v2 fragment adds `--roi-file`, `--ttc-thresh`,
`--confirm-frames`, `--clip-track-timeout-s`, `--clip-absolute-timeout-s`,
`--no-clip`, `--homography-file` and `--bev-unit-file` to the base ones.

## ROI / homography tools

```bash
python select_roi_bev.py     # pick the zebra-crossing ROI on a frame -> roi_bev.json
python extract_homography.py # solve the image->ground homography -> homography_bangkok.npy
python homography_rt.py      # runtime homography helper
```

`extract_homography.py` and `select_roi_bev.py` require the YOLO pose checkpoint
(`--kp-model`) and the homography/`bev_unit` pair; see `README.md` for the full
flag tables and `docs/extract_homography.md` / `docs/select_roi_bev.md` for details.

## Installazione (PyTorch con CUDA)

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` abilita l'indice NVIDIA CUDA 12.6 tramite
`--extra-index-url https://download.pytorch.org/whl/cu126`. Se la vostra GPU
richiede una CUDA diversa, modificate quella URL (es. `cu128`, `cu130`) prima
di installare. Verificate la build con
`python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`.

## Webapp di debug

```bash
# unico comando, identico su Linux/macOS e Windows
.venv/bin/python webapp/run_server.py          # Linux/macOS
.venv\Scripts\python.exe webapp\run_server.py  # Windows

.venv/bin/python webapp/run_server.py --foreground  # server in primo piano
.venv/bin/python webapp/run_server.py --stop       # ferma il server (pidfile)
```

Il launcher usa l'interprete con cui è stato invocato: se la webapp viene avviata
da dentro un ambiente virtuale (qualsiasi venv: `.venv`, `venv`, `env`, conda),
server, export, calibrazione, selettore ROI e debug live usano **quella** venv.
La cartella `.venv` del progetto è usata solo come fallback quando il comando
viene lanciato con l'interprete di sistema. I processi figli (export, omografia,
selettore ROI) usano lo stesso interprete; il debug live gira in-process.

### Dove la webapp salva i file

Ogni sessione di debug crea una cartella sotto `.webapp/sessions/{session_id}/`
(configurabile con `DIP_WEB_WORKDIR`):

| Output | Percorso | Abilitato da |
|---|---|---|
| JSON infrazioni | `.webapp/sessions/{session_id}/violations/violation_v<veicolo>_t<MMmSSsMMMms>.json` | "salva JSON + clip video infrazioni" |
| clip video infrazioni | `.webapp/sessions/{session_id}/violations/violation_v<veicolo>_t<MMmSSsMMMms>.mp4` | "salva JSON + clip video infrazioni" |
| video annotato | `.webapp/sessions/{session_id}/annotated.mp4` | "salva video annotato" |

Nella tab **4 · Debug** il pannello "File salvati in (sessione corrente)" mostra
questi percorsi in tempo reale: le voci disattivate sono marcate "(non attivo)".

### Parametri della sessione debug (API `POST /api/debug/sessions`)

La tab Debug invia questi campi JSON all'endpoint; i valori indicati sono i
default applicati dal server (`webapp/main.py`) e dal `DebugConfig`
(`webapp/debug_runner.py`).

| Campo | Default | Meaning |
|---|---|---|
| `input_path` | obbligatorio | video/stream da elaborare |
| `engine_path` | `DIP_WEB_ENGINE` / `config.DEFAULT_ENGINE` | checkpoint pose |
| `roi_path` | obbligatorio | JSON della ROI |
| `homography_path` | `DIP_WEB_HOMOGRAPHY` / `config.DEFAULT_HOMOGRAPHY` | file `.npy` dell'omografia statica calibrata |
| `bev_unit_file` | `null` (la UI precompila `DIP_WEB_BEV_UNIT` / `config.DEFAULT_BEV_UNIT_FILE`) | file `.txt` con la `bev_unit` calibrata |
| `device` | `DIP_WEB_DEVICE` / `config.DEFAULT_DEVICE` | device di inferenza |
| `source_type` | `auto` | `auto`, `video` o `stream` |
| `kp_imgsz` | 640 | dimensione di inferenza |
| `kp_conf` | 0.1 | confidenza minima del detector |
| `kp_vis` | 0.6 | confidenza per-cuboide |
| `class_conf` | `0:0.4,1:0.5,3:0.5` | soglie per classe COCO |
| `object_px` | 80 | zoom BEV (px per footprint mediana) |
| `smooth` | 11 | finestra di smoothing offline (0 = live) |
| `track_ocm_weight` | 2.0 | peso della direzione nel tracker |
| `track_high_conf` | 0.45 | soglia per avviare una nuova traccia |
| `ttc_thresh` | 2.5 | soglia TTC (secondi) |
| `confirm_frames` | 4 | frame consecutivi di conflitto TTC per confermare l'infrazione |
| `table_height` | 280 | altezza in px della tabella TTC |
| `max_frames` | 0 | limite di frame; 0 = tutti |
| `save_violations` | true | salva JSON + clip delle infrazioni |
| `no_clip` | false | se true, salva solo il JSON (senza clip) |
| `save_video` | false | salva il video completo annotato |
| `no_3d`, `no_snap_rect`, `no_trails`, `no_fov` | false | disattivano cuboidi, rettangularizzazione, scie BEV e campo visivo |
| `ground_indices` | `null` | indici keypoint a terra forzati (es. `0,1,2,3`) |
| `label_min_h` | 26 | altezza minima (px) per mostrare l'id chip |
| `bev_canvas` | `null` | dimensione pannello BEV `WxH` (default: dimensione sorgente) |
| `caption` | `""` | didascalia della scena |

Endpoint di lettura: `GET /api/debug/sessions` (elenco), `GET /api/debug/sessions/{id}`
(stato), `GET /api/debug/sessions/{id}/violations` (eventi con `json_path_url` e
`clip_path_url`), `GET /api/debug/outputs` (percorsi della sessione corrente) e
`GET /api/files/{name}` (download dei file prodotti).

