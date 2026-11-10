# Guida alla consegna: estrazione dell’omografia e rilevamento delle violazioni

Questa guida descrive gli script usati nel progetto:

- **extract_homography.py** stima la trasformazione dal piano immagine al piano stradale BEV e la relativa scala.
- **select_roi_bev.py** permette di disegnare e salvare la regione stradale d’interesse (ROI).
- **violation_detector.py** contiene la logica che valuta i conflitti veicolo-pedone e registra le violazioni.
- **bev_realtime_v2.py** esegue la pipeline di inferenza, tracking, BEV e rilevamento delle violazioni.
 - **debug_violations.py** esegue la pipeline con visualizzazioni diagnostiche, inclusa la tabella TTC.
 - **export_engine.py** esporta il checkpoint `.pt` in un engine TensorRT/ONNX ottimizzato (vedi `EXPORT_ENGINE.md`).

## Flusso di lavoro

La sequenza tipica è: calibrare una scena, disegnare la ROI per quella stessa scena, quindi elaborare il video con la pipeline di rilevamento. I checkpoint del modello e i video sorgente sono file esterni: non sono inclusi in questa cartella.

1. Estrarre omografia e scala dal video di calibrazione:

    python extract_homography.py --input "video_calibrazione.mp4" --kp-model "weights/pose.pt" --output homography_bangkok.npy --device cuda:0

2. Selezionare la ROI sul video, usando la coppia appena estratta:

    python select_roi_bev.py --frame "video_calibrazione.mp4" --homography homography_bangkok.npy --bev-unit homography_bangkok_bev_unit.txt --output roi_bev.json

3. Avviare il debug sul video da analizzare. Il valore numerico di bev_unit deve essere quello scritto nel file omonimo prodotto al punto 1:

    python debug_violations.py --input "video_demo.mp4" --kp-model "weights/pose.pt" --roi-file roi_bev.json --bev-unit 0.31147524796 --device cuda:0 --output "demo_debug.mp4"

Prima di elaborare, è consigliabile esportare il modello in engine FP16 (una volta sola):

    python export_engine.py --weights weights/pose.pt --imgsz 640 --half
    # poi usare --kp-model weights/pose_640_fp16.engine

Per elaborare senza la visualizzazione diagnostica, si può eseguire la pipeline normale:

    python debug_violations.py --input "video_demo.mp4" --kp-model "weights/pose.pt" --roi-file roi_bev.json --bev-unit 0.31147524796 --device cuda:0 --output "demo.mp4"

Eseguire gli ultimi due comandi dalla cartella del progetto: la pipeline carica l’omografia statica dal percorso relativo **homography_bangkok.npy**. Per un’altra scena, mettere in quel percorso la matrice della calibrazione corrispondente e creare/aggiornare una ROI con la medesima calibrazione. Non abbinare una matrice H a una bev_unit o a una ROI prodotte con una calibrazione diversa.

## Cosa fa ogni script

### extract_homography.py

Legge un video e inferisce le pose dei veicoli con un checkpoint YOLO pose. Filtra le footprint a terra attendibili, identifica i keypoint a terra, quindi stima una singola omografia globale con vincolo di ortogonalità. Non esegue tracking e non produce un video annotato.

**Input:** video di calibrazione leggibile da OpenCV e checkpoint pose compatibile (.pt o .engine). È preferibile una telecamera fissa, con veicoli ben visibili distribuiti sulla scena.

**Output:** con **--output homography_bangkok.npy** salva la matrice 3×3 in **homography_bangkok.npy**, una copia testuale in **homography_bangkok.txt** e la scala in **homography_bangkok_bev_unit.txt**. H e bev_unit sono una coppia: mantenerle insieme.

| Opzione | Default | Descrizione |
|---|---:|---|
| **--input** | obbligatorio | Video da calibrare. |
| **--kp-model** | obbligatorio | Checkpoint del modello pose (.pt o .engine). |
| **--output** | homography.npy | Nome/percorso della matrice .npy; usare un nome che termini in .npy per ottenere anche il gemello .txt. |
| **--imgsz** | 640 | Dimensione d’inferenza. |
| **--device** | cuda:0 | Dispositivo, ad esempio cuda:0 o cpu. |
| **--max-frames** | 1500 | Numero massimo di frame; 0 analizza tutto il video. |
| **--no-robust** | disattivato | Disattiva la pesatura robusta degli outlier; usarlo solo per confronto diagnostico. |

### select_roi_bev.py

Avvia un’interfaccia web locale nel browser. L’utente carica o indica un’immagine/video, sceglie i vertici della regione sul frame e salva la ROI. L’applicazione calcola anche i vertici corrispondenti in coordinate BEV.

**Input:** immagine o video (facoltativo da CLI, altrimenti caricabile nell’interfaccia), matrice H 3×3 e file di testo contenente un singolo valore bev_unit. Può anche caricare un JSON ROI esistente.

**Output:** JSON (predefinito roi_bev.json) con almeno image_points e bev_points, oltre ai metadati della sorgente e della calibrazione. Il salvataggio avviene dal pulsante dell’interfaccia; **--output** imposta il nome/percorso suggerito.

| Opzione | Default | Descrizione |
|---|---:|---|
| **--frame** | nessuno | File iniziale: immagine o video. |
| **--homography** | nessuno | File H 3×3 in formato .npy o .txt. |
| **--bev-unit** | nessuno | Percorso del file di testo con la bev_unit (in questo script non è un numero). |
| **--roi** | nessuno | JSON ROI da caricare/modificare. |
| **--output** | roi_bev.json | Nome/percorso suggerito per il JSON salvato. |
| **--frame-index** | 1 | Frame iniziale del video, indice a partire da 1. |
| **--port** | 8080 | Porta dell’interfaccia locale (127.0.0.1). |
| **--no-browser** | disattivato | Avvia il server senza aprire automaticamente il browser. |
| **--reset-calibration-from-roi** | disattivato | Opzione dichiarata dal parser; nella versione corrente il ripristino della calibrazione della ROI è già il comportamento predefinito. |
| **--keep-session-calibration** | disattivato | Mantiene la calibrazione fornita nella sessione invece di ripristinare quella registrata nella ROI. |

### violation_detector.py

Questo file **non è un comando CLI autonomo**: definisce ZebraCrossingROI e ViolationDetector, usati da **bev_realtime_v2.py** e **debug_violations.py**. Riceve le tracce e i frame già elaborati dalla pipeline, valuta i conflitti nella ROI tramite TTC e informazioni sul moto/intento dei pedoni, quindi salva gli eventi confermati.

La ROI JSON deve contenere image_points e bev_points (almeno tre punti per ciascun poligono). Le opzioni per configurare il rilevatore sono passate al programma che lo richiama:

| Opzione della pipeline | Default | Descrizione |
|---|---:|---|
| **--roi-file** | obbligatorio | JSON della ROI da analizzare. |
| **--ttc-thresh** | 2.5 secondi | Soglia TTC di conferma del conflitto. |
| **--confirm-frames** | 3 | Frame consecutivi richiesti per confermare l’evento. |
| **--clip-track-timeout-s** | 1.5 secondi | Attesa massima dopo la perdita del tracking del veicolo. |
| **--clip-absolute-timeout-s** | 3.0 secondi | Durata massima assoluta della clip dell’evento. |
| **--no-clip** | disattivato | Non scrive il filmato del singolo evento, ma mantiene il JSON. |
| **--bev-unit** | stimata se omessa | Valore numerico che deve essere coerente con H e ROI. |

Gli eventi sono scritti nella cartella **violations_<nome_video>/**, relativa alla cartella di lavoro. Per ciascun evento vengono prodotti un JSON con frame, timestamp, ID, TTC e dati della clip, e un MP4 con i marker dei soggetti coinvolti — il centroide del veicolo (rosso, `V<id>`) e quello del pedone (giallo, `P<id>`) — salvo l’uso di **--no-clip**. Il nome di clip e JSON contiene il minutaggio del video (`violation_v<id>_t<MMmSSsMMMms>`). Il programma principale può inoltre salvare un video annotato completo tramite **--output**.

### debug_violations.py

Esegue la pipeline BEV v2 e il rilevatore, aggiungendo un pannello diagnostico TTC. Richiede video/stream, checkpoint pose e ROI JSON; per la logica delle violazioni richiede anche una H invertibile e una bev_unit coerente con la ROI.

**Output:** se si specifica **--output**, salva il video completo con annotazioni e pannello debug. Salva inoltre gli eventi nella cartella **violations_<nome_video>/**. Per il video completo è necessario FFmpeg con encoder libx264 disponibile nel PATH. Le clip dei singoli eventi sono scritte con OpenCV.

Opzioni specifiche del debug:

| Opzione | Default | Descrizione |
|---|---:|---|
| **--table-height** | 280 px | Altezza del pannello/tabella TTC. |
| **--no-save-violations** | disattivato | Non salva gli artefatti degli eventi (JSON e clip); non disabilita il video debug passato a --output. |

Oltre a queste due opzioni, lo script accetta le opzioni della pipeline base e quelle v2 riportate nelle tabelle seguenti. Il parser riconosce l’interfaccia completa del runtime, ma il ciclo di debug non applica ogni opzione avanzata: in particolare refinement offline, pseudo-label e detector ausiliario non fanno parte del flusso debug. Le opzioni di violazione elencate sopra valgono anche quando si lancia **bev_realtime_v2.py** direttamente.

#### Opzioni base della pipeline (accettate da debug_violations.py)

| Opzioni | Default / significato |
|---|---|
| **--input**, **--kp-model** | Obbligatorie: sorgente video/stream e checkpoint pose. |
| **--kp-imgsz**, **--kp-conf** | 640, 0.25: dimensione e confidenza del detector pose. |
| **--device**, **--export**, **--half** | cpu; none (engine, tensorrt, onnx disponibili); FP16 per l’export accelerato. |
| **--ground-indices** | Automatico; altrimenti indici keypoint a terra separati da virgole, ad es. 0,1,2,3. |
| **--homography** | ortho; modalità ortho, adam o none. |
| **--no-3d**, **--no-snap-rect** | Disattivano cuboidi o rettangolarizzazione delle footprint. |
| **--no-trails**, **--trail-len** | Scie BEV disattivate o lunghezza (16 frame). |
| **--no-fov** | Nasconde il campo visivo della telecamera nella BEV. |
| **--no-stabilize**, **--bev-window**, **--solver-ema** | Controllo stabilizzazione; finestra 512 footprint, EMA 0 (disattivata). |
| **--track-alpha**, **--track-beta** | Parametri del tracker: 0.5, 0.08. |
| **--max-age**, **--min-hits**, **--no-cmc** | Tracker: 30, 2; disabilita la compensazione del moto camera. |
| **--output**, **--display**, **--max-frames** | Nessun file completo se --output è omesso; mostra la finestra se richiesto; 0 elabora tutto. |
| **--smooth**, **--smooth-poly** | Finestra offline 0 (disattivata), ordine Savitzky-Golay 2. |
| **--bev-freeze**, **--scale-lock**, **--fixed-camera** | Opzioni BEV offline; camera off di default (auto/on/off). |
| **--online** | Attiva la stabilizzazione causale per stream. |
| **--sync**, **--render-procs**, **--queue** | Rendering sincrono; processi 1 (valore 0 usa un thread); coda 4. |
| **--aux-model**, **--aux-conf**, **--aux-every**, **--aux-footprint**, **--no-aux-box**, **--aux-overlap**, **--aux-width-scale**, **--aux-gate**, **--aux-gate-iou** | Flag riconosciuti dal parser base per il detector ausiliario (default: yolo26x.pt, confidenza 0.30, intervallo 1, overlap 0.12, scala 0.9, IoU 0.3); il percorso debug imposta il detector ausiliario a none e non lo usa. |

#### Opzioni v2 e rendering (accettate da debug_violations.py)

| Opzioni | Default / significato |
|---|---|
| **--kp-vis**, **--suppress-nested**, **--class-conf** | 0.5; 0 (disattivato); soglie per classe, ad es. 0:0.4,1:0.5,3:0.5. |
| **--bev-canvas**, **--layout** | Dimensione pannello WxH; layout side oppure dashboard (default side). |
| **--caption**, **--title** | Testo scena; tempi del titolo di apertura in secondi sorgente. |
| **--object-px**, **--bev-fit**, **--bev-fit-min-px**, **--bev-fit-pct** | Zoom BEV 40 px per oggetto mediano; autofit disattivato, minimo 22 px, percentile 90. |
| **--label-min-h**, **--coast-mark**, **--show-bev-ids** | Altezza minima etichetta 26 px; aggiunge ? alle tracce in coast; mostra gli ID BEV. |
| **--source-type** | auto; selezionare video o stream per forzare il tipo. |
| **--track-high-conf**, **--track-ocm-weight** | Soglia automatica; peso direzione tracker 0.25. |
| **--refine** / **--track-cleanup** | Abilita refinement offline delle tracce nella pipeline v2. |
| **--no-fill-box**, **--no-vel-heading**, **--no-rigid-cuboids** | Disattivano le rispettive correzioni geometriche del refinement. |
| **--refine-fps**, **--speed-ratio**, **--max-reproj**, **--max-gap**, **--min-fill** | Parametri refinement: FPS sorgente automatico (0), rapporto velocità 3, reproiezione 0.10, gap 45, riempimento minimo 0.5. |
| **--no-dedupe**, **--no-shape-smooth**, **--shape-win**, **--no-world-lock**, **--no-hold-still** | Controllano deduplicazione, smoothing (finestra 31), world lock e blocco dei veicoli fermi. |
| **--min-track**, **--coast-tail**, **--merge-gap** | Refinement: 12, 8, 30 frame. |
| **--labels**, **--labels-from**, **--labels-source** | Esportazione facoltativa geometria pseudo-label; frame iniziale 0, sorgente originale opzionale. |
| **--bev-unit** | Valore numerico (default automatico): usare quello della calibrazione della ROI. |

Le opzioni refinement in questa tabella sono quelle del parser generale e sono effettive nel normale runtime v2 quando il relativo percorso offline viene usato; il ciclo di **debug_violations.py** non esegue quel refinement. Le opzioni principali effettive per il rilevamento nel debug sono input, checkpoint, ROI, bev_unit, soglia TTC e conferma frame, oltre ai controlli di visualizzazione esplicitamente applicati.

### bev_realtime_v2.py

È il programma principale per eseguire il modello v2 su un video o uno stream. Rileva persone e veicoli, stima i cuboidi 3D dai keypoint, mantiene le tracce nel tempo, visualizza la scena in prospettiva e in BEV e, con una ROI, applica il rilevatore di violazioni. Non include la tabella diagnostica TTC di **debug_violations.py**.

**Input:** video o stream supportato da OpenCV, checkpoint YOLO26 Pose26 compatibile con v2, file JSON ROI e omografia statica. Il file **homography_bangkok.npy** deve essere presente nella cartella di lavoro: il codice lo carica con quel nome. **--bev-unit**, se specificato, è un valore numerico e deve appartenere alla stessa calibrazione della ROI.

**Output:** con **--output** salva il video completo annotato; con **--display** mostra la finestra durante l’elaborazione. Il default non salva il video completo. Il rilevatore salva comunque gli eventi confermati in **violations_<nome_video>/**: JSON e clip MP4, a meno che si specifichi **--no-clip**. Il salvataggio del video completo richiede FFmpeg con libx264 nel PATH.

| Opzione | Default | Descrizione |
|---|---:|---|
| **--input** | obbligatorio | Percorso del video/stream in ingresso. |
| **--kp-model** | obbligatorio | Checkpoint del modello pose v2. |
| **--roi-file** | obbligatorio | JSON con la ROI e i suoi punti immagine/BEV. |
| **--device** | cpu | Dispositivo per l’inferenza, per esempio cpu o cuda:0. |
| **--kp-imgsz**, **--kp-conf** | 640, 0.25 | Dimensione d’inferenza e confidenza minima del modello. |
| **--bev-unit** | automatica | Scala BEV numerica; passare quella usata per creare la ROI. |
| **--ttc-thresh**, **--confirm-frames** | 2.5 s, 3 | Soglia TTC e numero di frame consecutivi necessari a confermare una violazione. |
| **--clip-track-timeout-s**, **--clip-absolute-timeout-s** | 1.5 s, 3.0 s | Tempi di chiusura della clip dopo la perdita del track e durata massima dell’evento. |
| **--no-clip** | disattivato | Non salva le clip dei singoli eventi; i JSON vengono comunque salvati. |
| **--output** | nessuno | Percorso del video completo annotato. |
| **--display** | disattivato | Mostra la finestra di output durante l’elaborazione. |
| **--max-frames** | 0 | Limite di frame; 0 elabora tutta la sorgente. |
| **--source-type** | auto | Forza la sorgente come video o stream, se necessario. |

Il programma accetta inoltre le opzioni base e v2 elencate nelle due tabelle CLI della sezione **debug_violations.py** sopra. Sono gli stessi parser condivisi; **--table-height** e **--no-save-violations** sono invece specifici dello script di debug.

Esempio:

    python debug_violations.py --input "video_demo.mp4" --kp-model "weights/pose.pt" --roi-file roi_bev.json --bev-unit 0.31147524796 --device cuda:0 --output "demo.mp4"

## Dipendenze e file necessari all’esecuzione

Installare le dipendenze Python con **python -m pip install -r requirements.txt**. L’estrazione e la pipeline di violazioni richiedono un checkpoint YOLO pose compatibile, non contenuto nel repository. Una GPU CUDA è consigliata; con un checkpoint e un ambiente compatibili si può usare **--device cpu**. Per salvare il video completo con **--output** serve anche FFmpeg con libx264.

Per la calibrazione standalone servono **homography_rt.py** e il package **uod** (in particolare il modulo keypoints). Per l’applicazione e il debug v2 servono inoltre **bev_realtime.py**, **bev_realtime_v2.py**, **refine_v2.py**, **violation_detector.py**, **v2/__init__.py**, **v2/hybrid_v2.py** e **uod/**. **select_roi_bev.py** usa OpenCV, NumPy e la libreria standard Python.

Il runtime v2 legge **homography_bangkok.npy** dalla cartella di lavoro, mentre la ROI è indicata con **--roi-file**. La coppia leggibile **homography_bangkok.txt** e **homography_bangkok_bev_unit.txt** è utile per controllo e per caricare la calibrazione nel selettore ROI; il valore passato alla pipeline con **--bev-unit** è invece numerico.

## Nota sul README generale

La sezione seguente conserva la documentazione generale del progetto UrbanOmniDetect da cui derivano i moduli runtime.

<p align="center">
  <h1 align="center">UrbanOmniDetect</h1>
  <h3 align="center">Calibration-Free View-Agnostic Monocular 3D Object Detection for Urban Scenes</h3>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=306TgWoAAAAJ">Mehmet Kerem Turkcan</a>&emsp;
    <a href="https://scholar.google.com/citations?user=GP7T1fgAAAAJ">Devika Gumaste</a>&emsp;
    <a href="https://scholar.google.com/citations?user=TlPI8yIAAAAJ">Zoran Kostic</a>
    <br/>
    <a href="https://www.aidl.ee.columbia.edu/">AIDL Lab</a>, <a href="https://www.ee.columbia.edu/">Department of Electrical Engineering</a>
    <br/>
    <b>Columbia University</b>
    <br/><br/>
    <a href="https://openaccess.thecvf.com/content/CVPR2026W/DriveX/papers/Turkcan_Calibration-Free_View-Agnostic_Monocular_3D_Object_Detection_for_Urban_Scenes_CVPRW_2026_paper.pdf"><img src="https://img.shields.io/badge/Paper-CVF%20Open%20Access-b31b1b.svg" alt="Paper"></a>
    <a href="https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20UrbanOmniDetect--2-Models-blue" alt="UrbanOmniDetect-2 Models"></a>
    <a href="https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20UrbanOmniDetect-Models-blue" alt="UrbanOmniDetect Models"></a>
    <a href="https://huggingface.co/datasets/mehmetkeremturkcan/UrbanOmniView"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20UrbanOmniView-Dataset-green" alt="UrbanOmniView Dataset"></a>
    <a href="https://drivex-workshop.github.io/cvpr2026/"><img src="https://img.shields.io/badge/CVPR%202026-DriveX%20Workshop%20(Oral)-4b44ce.svg" alt="CVPR 2026 DriveX Workshop"></a>
    <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL%203.0-orange.svg" alt="License"></a>
  </p>
  <p align="center">
    <b><a href="#urbanomnidetect-2">UrbanOmniDetect-2</a></b>, the current release, is developed and maintained by <a href="https://scholar.google.com/citations?user=306TgWoAAAAJ">Mehmet Kerem Turkcan</a>.
  </p>
</p>

<p align="center">
  <img src="https://github.com/mkturkcan/urbanomnidetect/raw/main/assets/urbanomniview.png" width="100%" alt="UrbanOmniView examples across ego-vehicle, infrastructure, and aerial viewpoints"/>
</p>

This repository contains two generations of calibration-free monocular 3D detection for urban scenes:

- **[UrbanOmniDetect-2](#urbanomnidetect-2)** (September 2026) is one network for COCO-80 detection and 3D cuboids, with tracking, a bird's-eye view, and offline trajectory refinement. Checkpoints are at [mehmetkeremturkcan/UrbanOmniDetect-2](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2).
- **[UrbanOmniDetect](#urbanomnidetect-cvpr-2026-drivex)** is the pose-only model from the CVPR 2026 DriveX paper, trained on the UrbanOmniView dataset. Checkpoints are at [mehmetkeremturkcan/UrbanOmniDetect](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect).

In file names and code, `v2` refers to UrbanOmniDetect-2 and `v1` to the paper model. The two sets of checkpoints are not interchangeable.

## UrbanOmniDetect-2

*Released September 2026. Developed and maintained by [Mehmet Kerem Turkcan](https://scholar.google.com/citations?user=306TgWoAAAAJ).*

UrbanOmniDetect-2 replaces the pose-only paper model with a single **hybrid detection and pose** network: a YOLO26 with a Pose26 head that is a full COCO-80 detector and, for every road user, also regresses the eight projected corners of its 3D box. It runs on any viewpoint without calibration, and the same forward pass feeds the tracker, the bird's-eye view, and the offline refinement that turns per-frame detections into rigid, physically consistent trajectories.

 - **Detection and 3D cuboids from one forward pass.** All 80 COCO classes, with cuboids for person, bicycle, car, motorcycle, bus, and truck. No auxiliary detector is needed.
 - **Trained on a mixture of 2D and 3D data.** COCO and VisDrone (boxes only) ground the detector. Real keypoint data (KITTI, DAIR-V2X), CDrone, and rendered vehicles (MeshFleet, Objaverse) teach the cuboids through a masked pose loss, so box-only images never push the keypoint head.
 - **Pedestrian-crossing violation detection.** `debug_violations.py` tracks road users, projects them to BEV, and flags vehicle/pedestrian conflicts inside a zebra-crossing ROI with intent-aware TTC.
- **Five scales at 640 px.** Run them at 960 px on aerial footage to recover far, small traffic while the cuboids stay tight.

### Models

Every checkpoint is the epoch of its run with the highest road-class pose mAP, followed by a class-branch-only fine-tune on real box data (COCO + VisDrone) that corrects class boundaries learned from renders. The pose head is untouched by that step. Metrics are on the mixed validation set (COCO, VisDrone, KITTI and DAIR-V2X, CDrone, renders) at 640 px.

| Model | Params (M) | COCO AP | VisDrone AP @960 | KITTI 2D AP (Mod.) | KITTI AP<sub>3D</sub> E / M / H | KITTI AP<sub>BEV</sub> E / M / H | Download |
| --- | --: | --: | --: | --: | --: | --: | --- |
| v2-N | 2.6 | 14.3 | 14.6 | 91.5 | 31.5 / 22.2 / 18.8 | 37.0 / 26.5 / 22.4 | [urbanomnidetect_v2_n_640.pt](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/main/checkpoints/urbanomnidetect_v2_n_640.pt) |
| v2-S | 9.9 | 21.3 | 19.1 | 94.6 | 42.9 / 32.5 / 27.6 | 49.7 / 37.5 / 31.0 | [urbanomnidetect_v2_s_640.pt](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/main/checkpoints/urbanomnidetect_v2_s_640.pt) |
| v2-M | 21.3 | 28.1 | 25.1 | 95.4 | 46.3 / 34.8 / 30.7 | 50.7 / 39.6 / 35.3 | [urbanomnidetect_v2_m_640.pt](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/main/checkpoints/urbanomnidetect_v2_m_640.pt) |
| v2-L | 25.7 | 28.7 | 25.1 | 96.5 | **53.6 / 39.4 / 33.1** | **56.7 / 42.5 / 36.9** | [urbanomnidetect_v2_l_640.pt](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/main/checkpoints/urbanomnidetect_v2_l_640.pt) |
| v2-X | 57.6 | **30.4** | **26.7** | **96.6** | 47.5 / 37.1 / 32.9 | 52.1 / 41.1 / 36.4 | [urbanomnidetect_v2_x_640.pt](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2/resolve/main/checkpoints/urbanomnidetect_v2_x_640.pt) |

### Quick start

```bash
pip install -r requirements.txt          # ultralytics 8.4.61 (tested), torch, opencv, scipy, ...
huggingface-cli download mehmetkeremturkcan/UrbanOmniDetect-2 \
    checkpoints/urbanomnidetect_v2_x_640.pt --local-dir .
```

The checkpoints unpickle a class defined in `v2/hybrid_v2.py`, so run from the repository root (or put it on `PYTHONPATH`) and import it before loading. Do not run Python from a directory that contains an `ultralytics/` source clone, because it shadows the installed package.

```python
import v2.hybrid_v2                      # registers HybridPoseModel26 for unpickling
from ultralytics import YOLO

model = YOLO("checkpoints/urbanomnidetect_v2_x_640.pt")
r = model.predict("frame.jpg", imgsz=960, conf=0.1, device="cuda:0")[0]
boxes, classes = r.boxes.xyxy, r.boxes.cls          # COCO-80 ids
kpts = r.keypoints.data                              # (N, 8, 3): x, y, cuboid confidence
```

Indices 0 to 3 are the ground-contact corners and indices 4 to 7 are the top corners, the opposite of the paper model.

Video, with tracking, the bird's-eye view, and the violation detector:

```bash
python debug_violations.py --input clip.mp4 --kp-model checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine \
    --kp-imgsz 640 --kp-conf 0.1 --kp-vis 0.6 --class-conf 0:0.4,1:0.5,3:0.5 \
    --smooth 11 --device cuda:0 --output out.mp4 \
    --bev-unit 0.31147524796 --roi-file roi_bev.json --ttc-thresh 2.5 --confirm-frames 6
```

 - `--kp-imgsz 640` matches the engine input. On drone footage a larger input (960/1280) recovers far, small traffic and the cuboids stay tight.
 - `--export tensorrt --half` builds an FP16 engine on first use and reuses it afterwards. On the X model at 960 px on one RTX PRO 6000, this measures **4.6 ms per frame (217 fps) against 7.7 ms (130 fps) in PyTorch FP32**, a 1.7x speed-up, with cuboid corners agreeing to a median 0.9 px. TensorRT 10.x is required, since 11.x removed the builder-flag API that Ultralytics uses.
 - Everything else is documented in `USAGE.md` (`python debug_violations.py --help`).

### ROI / homography tools

```bash
python select_roi_bev.py     # pick the zebra-crossing ROI on a frame -> roi_bev.json
python extract_homography.py # solve the image->ground homography
python homography_rt.py      # orthogonality-constrained BEV homography solver
```

### Repository structure

```
urbanomnidetect/
  debug_violations.py    # entry point: video -> BEV -> zebra-crossing violation detection
  bev_realtime_v2.py     # v2 pipeline library: detection, tracking, BEV, dashboard
  bev_realtime.py        # shared pipeline library (the v2 adapter subclasses it)
  violation_detector.py  # intent-aware TTC conflict evaluation on the ROI
  export_engine.py       # standalone TensorRT/ONNX engine exporter (see EXPORT_ENGINE.md)
  homography_rt.py       # orthogonality-constrained BEV homography solver
  select_roi_bev.py      # ROI picker -> roi_bev.json
  extract_homography.py  # homography solver
  uod/                   # shared runtime: keypoints, BEV, tracking (OCSort), visualization, HUD
  v2/                    # UrbanOmniDetect-2 model registration: hybrid_v2.py (unpickling)
  USAGE.md               # command reference for the inference tools
```

## UrbanOmniDetect (CVPR 2026 DriveX)

The paper model, presented as an oral at the [CVPR 2026 DriveX workshop](https://drivex-workshop.github.io/cvpr2026/) by Mehmet Kerem Turkcan, Devika Gumaste, and Zoran Kostic.

### Highlights

- **Calibration-free 3D detection.** A single model predicts 3D bounding box keypoints from a raw RGB image without camera intrinsics, depth estimation, or ground-plane priors.
- **View-agnostic.** One architecture works across ego-vehicle, infrastructure, and aerial drone viewpoints.
- **Strongest on the harder KITTI splits.** AP<sub>3D</sub> of 30.71 and AP<sub>BEV</sub> of 35.19 on the Moderate split at IoU ≥ 0.7, ahead of calibration-dependent baselines on Moderate and Hard.
- **Real-time.** Under 11 ms per image on an A100 with TensorRT at 640 × 640.
- **Robust to calibration error.** Calibration-dependent methods lose more than 80% of their accuracy with a 5% focal-length error. UrbanOmniDetect takes no intrinsics as input and is invariant to such errors by construction.

### Key results

#### Architecture comparison

Keypoint mAP<sub>50:95</sub> on the UrbanOmniView validation set at 640 × 640. Adding the P2 feature level improves every scale.

| Backbone | n | s | m | l | x |
|:---------|:---:|:---:|:---:|:---:|:---:|
| YOLO11 | 0.547 | 0.644 | 0.699 | 0.703 | 0.719 |
| YOLO11 + P6 | 0.548 | 0.639 | 0.693 | 0.698 | 0.718 |
| **YOLO11 + P2** | **0.559** | **0.656** | **0.717** | **0.729** | **0.751** |
| YOLO12 | 0.470 | 0.580 | 0.651 | 0.654 | 0.684 |
| YOLOv9 | 0.545 | 0.634 | 0.688 | 0.701 | 0.716 |
| YOLOv8 | 0.549 | 0.607 | 0.662 | 0.682 | 0.693 |

#### KITTI benchmark

Monocular 3D detection on KITTI at IoU ≥ 0.7. The baselines use camera calibration. UrbanOmniDetect does not.

| Method | AP<sub>3D</sub> Easy | AP<sub>3D</sub> Mod. | AP<sub>3D</sub> Hard | AP<sub>BEV</sub> Easy | AP<sub>BEV</sub> Mod. | AP<sub>BEV</sub> Hard |
|:-------|:---:|:---:|:---:|:---:|:---:|:---:|
| MonoDGP | **30.76** | 22.34 | 19.02 | **39.40** | 28.20 | 24.42 |
| MonoCon | 26.33 | 19.01 | 15.98 | 34.65 | 25.39 | 21.93 |
| MonoLSS | 25.91 | 18.29 | 15.94 | 34.70 | 25.36 | 21.84 |
| DEVIANT | 24.63 | 16.54 | 14.52 | 32.60 | 23.04 | 19.99 |
| **UrbanOmniDetect** | 29.61 | **30.71** | **27.76** | 33.86 | **35.19** | **31.38** |

### Quick start

#### Prerequisites

A working CUDA and PyTorch environment is required. To set one up from scratch on Windows or Ubuntu, see [CUDA2025](https://github.com/mkturkcan/CUDA2025) for a step-by-step guide based on Miniconda.

Then install the remaining dependencies:

```bash
pip install ultralytics scipy scikit-learn opencv-python matplotlib
```

#### Download a pretrained model

The trained checkpoints are hosted on [Hugging Face](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect). The following command downloads the recommended YOLO11x model with the P2 feature-pyramid head.

```bash
huggingface-cli download mehmetkeremturkcan/UrbanOmniDetect \
    checkpoints/urbanomnidetect_yolo11x-p2_1920.pt \
    --local-dir .
```

To download every checkpoint at once:

```bash
huggingface-cli download mehmetkeremturkcan/UrbanOmniDetect --local-dir .
```

The full set covers YOLOv8, YOLOv9, YOLO11, and YOLO12 across all scales and head configurations.

#### Single-image inference

The model uses the standard Ultralytics prediction API. Point it at an image from any viewpoint and it predicts 3D bounding box keypoints without camera parameters.

```python
from ultralytics import YOLO

model = YOLO("checkpoints/urbanomnidetect_yolo11x-p2_1920.pt")
results = model.predict("your_image.jpg", imgsz=1920, conf=0.1, device="cuda:0")
```

Each detection has eight ordered keypoints, the 2D projections of its 3D bounding box corners. Indices 0 to 3 are the top corners and indices 4 to 7 are the ground-contact corners.

#### Bird's-eye view

Ground-contact keypoints are mapped to a top-down plane through an orthogonality-constrained homography (`homography_rt.py`). No camera calibration is needed. See `debug_violations.py` for the full video pipeline that renders the BEV view.

#### Real-time BEV on video

The video pipeline detects objects, tracks them across frames, solves the BEV homography, and renders the camera view next to a bird's-eye view. Tracked objects keep a stable identity and hold their last position when a detection is briefly missed, so the layout stays steady. Run it through `debug_violations.py`; see `USAGE.md` for the full command reference.

```bash
python debug_violations.py --input drone_manhattan.mp4 \
    --kp-model checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine \
    --kp-imgsz 640 --device cuda:0 --export tensorrt --output bev.mp4
```

Add `--display` for a live window. See `USAGE.md` for the full command reference.

### Training

Training code and experiment definitions (`cfg/`, `train.py`, `v2/` training scripts) are not part of this trimmed repository; only `v2/hybrid_v2.py` is kept, for checkpoint unpickling.

## Citation

If you use UrbanOmniDetect or the UrbanOmniView dataset in your research, please cite:

```bibtex
@inproceedings{turkcan2026urbanomnidetect,
  title     = {Calibration-Free View-Agnostic Monocular {3D} Object Detection for Urban Scenes},
  author    = {Turkcan, Mehmet Kerem and Gumaste, Devika and Kostic, Zoran},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages     = {786--795},
  year      = {2026}
}
```

If you use UrbanOmniDetect-2, please cite:

```bibtex
@software{turkcan2026urbanomnidetect2,
  author = {Turkcan, Mehmet Kerem},
  title  = {{UrbanOmniDetect-2}: One Network for Detection and Calibration-Free {3D} Cuboids from Any Viewpoint},
  year   = {2026},
  month  = {9},
  url    = {https://github.com/mkturkcan/urbanomnidetect}
}
```

## Acknowledgements

The UrbanOmniDetect paper and the UrbanOmniView dataset began while the first author was a postdoc in the [Department of Electrical Engineering](https://www.ee.columbia.edu/) ([AIDL Lab](https://www.aidl.ee.columbia.edu/)) at Columbia University. This work was supported by the NSF Engineering Research Center for Smart Streetscapes under Award EEC-2133516, NSF Grants CNS-2450567 and CNS-2038984, and by computing resources from the NVIDIA Academic Grant Program and the Empire AI Consortium.

## License

This project is released under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0).
