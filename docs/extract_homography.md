# Estrazione Omografia e bev_unit

Questo documento descrive come usare `extract_homography.py` per calibrare la
**matrice di omografia globale** (image → piano stradale BEV) e la **bev_unit**
(scala media delle footprint in BEV) a partire da un video di calibrazione.

Lo script è **standalone**: non usa la pipeline real-time (`bev_realtime_v2`),
non fa tracking completo né rendering. Esegue solo l'inferenza pose, filtra le
footprint affidabili dei veicoli, raggruppa le detection in tracce minime per
IoU (serve al raffinamento rigido, §5.6) e risolve una singola omografia
globale con l'ottimizzatore a vincolo di ortogonalità (`homography_rt.py`).

---

## Indice

1. [Dipendenze](#1-dipendenze)
2. [Utilizzo](#2-utilizzo)
3. [Parametri CLI](#3-parametri-cli)
4. [File di output](#4-file-di-output)
5. [Come funziona internamente](#5-come-funziona-internamente)
6. [Interpretazione dei risultati](#6-interpretazione-dei-risultati)
7. [Uso dei file generati nella pipeline](#7-uso-dei-file-generati-nella-pipeline)
8. [Risoluzione dei problemi](#8-risoluzione-dei-problemi)

---

## 1. Dipendenze

### Moduli locali del progetto

| Modulo | Ruolo |
|---|---|
| `homography_rt.py` | Solver di omografia a vincolo di ortogonalità (solo numpy) |
| `uod/keypoints.py` | `GroundIndexResolver`: identifica i 4 keypoint a terra |

### Librerie esterne

| Libreria | Ruolo |
|---|---|
| `ultralytics` | Inferenza del modello pose (YOLO); richiede PyTorch a runtime |
| `torch` | Runtime di PyTorch per l'inferenza Ultralytics (non importato direttamente dallo script) |
| `opencv-python` | Lettura video e test di convessità |
| `numpy` | Calcolo numerico |

Lo script **non dipende più** da `bev_realtime_v2`, `bev_realtime` né da altri
moduli della pipeline real-time (il refinement rigido è implementato dentro
lo script stesso). Non richiede nemmeno un
file di omografia pre-esistente in input (il vecchio script terminava con
`sys.exit(1)` se `homography_bangkok.npy` mancava).

---

## 2. Utilizzo

### Comando base

```bash
.venv/bin/python extract_homography.py \
  --input video_dataset_bangkok/video_041.avi \
  --kp-model checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine \
  --max-frames 0 \
  --output homography_bangkok.npy \
  --reproject-roi
```

`--max-frames 0` legge **tutto** il video: il vincolo di rigidità per-traccia
(§5.6) ha bisogno di più osservazioni dello stesso oggetto, e il risultato
migliora nettamente con più dati (residuo ROI 0.074 BEU su 900 frame contro
0.032 BEU su 9000 frame sul video Bangkok).

`--reproject-roi` aggiorna anche `roi_bev.json` con la nuova calibrazione
(§6.5): senza, il pannello BEV e quello in prospettiva mostrerebbero
l'attraversamento pedonale in due posti diversi.

### Requisiti del video di calibrazione

Per una calibrazione affidabile il video deve avere:

- **Telecamera fissa** (o quasi): l'omografia globale assume un piano stradale
  statico rispetto alla camera.
- **Traffico ben distribuito su tutta la scena**: ogni zona del piano stradale
  dovrebbe essere attraversata da qualche veicolo durante il video. Zone della
  scena mai toccate da footprint affidabili degradano la rettifica locale.
- **Almeno 128 footprint veicolo valide** (consigliato): sotto questa soglia lo
  script emette un `[ATTENZIONE]` e la bev_unit può risultare poco stabile.
- **Video più lungo possibile**: il raffinamento rigido per-traccia (§5.6)
  ha bisogno di più osservazioni dello stesso oggetto. Su un video Bangkok da
  900 frame il residuo sulla ROI è 0.074 BEU, su 9000 frame scende a
  0.032 BEU.
- Veicoli con keypoint 3D ben visibili: telecamere troppo angolate o con
  occlusioni pesanti riducono il numero di footprint affidabili.

---

## 3. Parametri CLI

| Parametro | Default | Descrizione |
|---|---|---|
| `--input` | *richiesto* | Video di input per la calibrazione |
| `--kp-model` | *richiesto* | Checkpoint del modello pose (`.pt` o `.engine`) |
| `--output` | `homography.npy` | Nome del file `.npy` di output (genera anche il gemello `.txt`) |
| `--imgsz` | `640` | Risoluzione di inferenza (deve corrispondere alla scala del checkpoint) |
| `--device` | `cuda:0` | Device di inferenza |
| `--max-frames` | `1500` | Stop automatico dopo N frame (`0` = tutto il video) |
| `--refine-iters` | `5` | Iterazioni di raffinamento rigido per-traccia (§5.6). `0` = disabilitato |
| `--rigid-pct` | `65` | Percentile del residuo di forma rigida sotto il quale un'osservazione viene conservata |
| `--roi-file` | `roi_bev.json` | File ROI usato come riferimento diagnostico di coerenza (§6.4). Se assente, il check viene saltato |
| `--reproject-roi` | *off* | Ricalcola i `bev_points` di `--roi-file` con la nuova calibrazione e li scrive in `--roi-out` (§6.5) |
| `--roi-out` | = `--roi-file` | Dove scrivere la ROI ri-proiettata |
| `--no-robust` | *off* | Disattiva i pesi IRLS Cauchy (`OrthoHomographySolver(robust=False)`): i corner outlier non vengono rigettati. Solo per confronto diagnostico — vedi §6.3 |

---

## 4. File di output

Con `--output homography.npy` vengono prodotti **tre file**:

| File | Contenuto |
|---|---|
| `homography.npy` | Matrice 3×3 `float64`, mappa image → piano BEV, da caricare con `np.load` |
| `homography.txt` | Stessa matrice in testo leggibile (`%.6e`), per verifiche a occhio |
| `homography_bev_unit.txt` | Una riga: la `bev_unit` (dimensione mediana della footprint in pixel-BEV, nella gauge della H estratta) |

I nomi sono derivati da `--output`: la bev_unit è sempre
`<output_senza_estensione>_bev_unit.txt` e il testo è `<output>` con
estensione `.txt`.

---

## 5. Come funziona internamente

Il flusso, in ordine di esecuzione:

1. **Inferenza pose** (YOLO) su ogni frame del video.
2. **Filtro qualità dei keypoint** — replica fedelmente i controlli della
   pipeline v2 (`V2Pipeline._detect_pose`):
   - soglia di confidenza effettiva 0.25 per le classi ammesse alla calibrazione
     (person 0.4, bike 0.5, moto 0.5 sono i valori nominali, ma per le classi
     veicolo la soglia effettiva resta `KP_CONF = 0.25`);
   - visibilità dei keypoint (`kp_vis = 0.5`): accettata se tutti i keypoint
     hanno visibilità ≥ 0.5, oppure se la media ≥ 0.5, i keypoint a terra ≥ 0.425
     e il minimo ≥ 0.35;
   - keypoint dentro la box 2D (tolleranza 0.75) e non collassati a (0,0);
   - **convessità** della footprint (un quad auto-intersecante scarta la detection);
   - area minima della footprint (≥ max(16 px², 0.3% della box 2D));
   - keypoint **tutti dentro il frame** (gli estremi estrapolati non sono affidabili);
   - scarto del centro dei keypoint dalla box ≤ 0.5 × diagonale della box;
   - box non tagliata dal bordo immagine (margine `max(15 px, 1% della diagonale)`);
   - rapporto area keypoint/box tra 0.12 e 2.5.
3. **Selezione classi** — entrano in calibrazione solo veicoli con ground
   affidabile: car (2), bus (5), truck (7), bicycle (1), motorcycle (3). I
   pedoni sono esclusi (la pipeline li rappresenta con un quad sintetico
   dalla box 2D, privo di informazione geometrica utile alla rettifica).
4. **GroundIndexResolver** — determina quali 4 degli 8 keypoint sono i
   vertici a terra: vota per 5 frame osservati (frame in cui sono state
   accettate detection di calibrazione) e si blocca a maggioranza stretta.
5. **Raggruppamento in tracce** — le detection sono raggruppate per IoU in
   immagine (`_group_tracks`, IoU minimo 0.2, max 12 frame persi), scartando
   le tracce con meno di 4 osservazioni, così da sapere quali osservazioni
   appartengono allo **stesso oggetto rigido**. È l'unico uso del tracking:
   un veicolo ha UNA sola forma di footprint.
6. **Solve globale di bootstrap** — `OrthoHomographySolver`
   (Levenberg-Marquardt su 4 DoF essenziali, residui = coseni degli angoli
   delle footprint + parallelismo opposto, pesi IRLS Cauchy) su tutte le
   footprint raccolte: una sola omografia per tutto il video, come fa
   `_apply_global_homography` offline nella pipeline. Il flag `--no-robust`
   disattiva i pesi IRLS Cauchy (vedi §6.3).
7. **Raffinamento rigido per-traccia** (vedi §5.6) — ad ogni iterazione le
   footprint proiettate nella gauge corrente vengono allineate alla **forma
   rigida** della propria traccia (stima MLE della forma a mediana con
   allineamento tipo Procrustes: `_shape_mle` + `_best_alignment`), e le
   osservazioni il cui disaccordo con quella forma supera `--rigid-pct`
   percentile vengono scartate. Il solve viene poi ripetito sull'insieme
   pulito: a partire dalla prima iterazione efficace, il dataset usato dal
   solver e per la `bev_unit` finale contiene **solo le footprint tracciate**
   (quelle raggruppate in §5). Di default 5 iterazioni.
8. **bev_unit** — mediana di √(larghezza×lunghezza) delle footprint proiettate
   con la H appena estratta (formula di `_class_footprint_dims`).
9. **Salvataggio** dei tre file descritti sopra.

### 5.6 Perché serve il vincolo di rigidità per-traccia

Un veicolo è un oggetto rigido: la sua footprint a terra ha sempre la stessa
forma. Ma una singola osservazione non lo dimostra — un'auto parzialmente
occlusa o tagliata dal bordo del frame produce un quadrilatero più piccolo
e deformato, e quel campio "infora" il piano stradale.

Il vincolo è applicabile solo con più osservazioni dello **stesso** oggetto.
Il bootstrap della H porta ogni footprint nel piano rettificato, dove la
forma di ciascuna traccia può essere stimata (MLE Procrustes) e ogni
osservazione confrontata con essa. Lo scarto fra la forma rigida e
l'osservazione misura quanto quella vista è "pulita".

Le osservazioni pulite formano un dataset omogeneo per il solver, che a
quel punto vede solo quadrilateri realmente rettangolari. Questo è il
contributo che la vecchia pipeline completa traeva dal refinement offline
(`_shape_mle` + `_best_alignment`); qui è espresso come **filtro**, così lo
script resta standalone e non deve ricostruire la geometria 3D per riposare
i footprint.

La forma stimata **non** viene proiettata sul rettangolo migliore: il vincolo
di rigidità serve a giudicare un'osservazione, non a correggerla, e
proiettare sul rettangolo falserebbe il riferimento con cui le osservazioni
sono confrontate. Misurato sull'intero video Bangkok, togliere lo snap fa
scendere il residuo sulla ROI da 0.0336 a 0.0322 BEU.

Misure sul video Bangkok `video_041.avi` (9000 frame, 36314 footprint,
confronto similarity-invariant rispetto alla calibrazione in uso nel repo):

| Configurazione | residuo ROI (BEU) | residuo footprint (BEU) | rettangularità `mean\|cos\|` |
|---|---|---|---|
| solve singolo (senza raffinamento) | 0.0508 | 0.1671 | 0.0933 |
| **raffinamento rigido, 5 iterazioni** | **0.0322** | **0.1136** | **0.0812** |

La calibrazione di riferimento già presente nel repo ha rettangularità
0.0933: la nuova è oggettivamente più rettificata, non solo più vicina a
quella esistente.

---

## 6. Interpretazione dei risultati

### 6.1 La gauge dell'omografia

Il solver ortogonale stima la rettifica del piano stradale **a meno di una
similarity** (rotazione + scala uniforme + traslazione in BEV): questi gradi
di libertà sono invisibili all'obiettivo di rettangolarità. Conseguenze:

- Due solve sullo stesso video (o su versioni diverse dello script) producono
  H **numericamente diverse** ma **geometricamente equivalenti**: una
  verifica sui dati Bangkok (§6.2) mostra che la H in uso nel repo e quella
  appena estratta differiscono per una scala di gauge ≈ 1.24 e una
  rotazione di ≈ −2.5°.
- **La bev_unit è coerente solo con la H che l'ha generata.** È espresso
  nella stessa gauge. Usare H di una estrazione con bev_unit di un'altra è
  un errore di gauge (le footprint sarebbero scalate/ruotate male).

### 6.2 Valori di riferimento misurati (video Bangkok)

Estrazione eseguita su `video_dataset_bangkok/video_041.avi` (video integrale,
9000 frame, **36314 footprint** veicolo valide) con
`--kp-model checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine`
e `--max-frames 0`:

| Configurazione | `robust` | loss | iterazioni | `bev_unit` |
|---|---|---|---|---|
| Default (IRLS Cauchy + raffinamento rigido) | `True` | `137.83` | 29 | **`0.335545853435`** |
| `--no-robust` | `False` | — | — | — |

Su `video_041_tagliato_2.avi` (900 frame, 6526 footprint) il default produce
`bev_unit = 0.380562035749` con loss `19.06`: il risultato peggiora sensibilmente
con meno dati, perché il vincolo di rigidità per-traccia ha bisogno di più
osservazioni dello stesso oggetto. **Calibrare sul video più lungo disponibile.**

I valori storici `0.373092117369` (default) e `0.367480188935`
(`--no-robust`) citati in §6.3 appartengono a esecuzioni con versioni
precedenti dello script e non descrivono la coppia attualmente presente nel
repo.

### 6.2bis La coppia attualmente nel repo

| Configurazione | `bev_unit` |
|---|---|
| Coppia in uso nel repo (`homography_bangkok.npy` + `homography_bangkok_bev_unit.txt`) | `0.335545853435` |
| Stesso valore in `roi_bev.json` | `0.33554585343482757` |

La coppia nel repo è **già coerente**: `roi_bev.json` è stata ri-proiettata con
l'omografia e la `bev_unit` attualmente presenti (residuo
similarity-invariant sui vertici della ROI ≈ 3e-7 BEU, scala ≈ 1.000,
rotazione ≈ 0°). Non va mescolata con calibrazioni provenienti da
estrazioni diverse: come ricorda la regola d'oro sotto, **H e bev_unit vanno
sempre usate insieme**, perché una gauge diversa introduce un errore di
scala sulle dimensioni delle footprint in BEV, che si traduce in errori sulle
distanze di conflitto usate dal rilevatore TTC.

### 6.3 Effetto di `--no-robust`

Il solve non robusto **non rigetta i corner outlier**: su una esecuzione di
riferimento la loss passa da `29.90` a `189.25` (× 6.3) e la convergenza
avviene in 7 iterazioni invece di 30 (la LM si ferma su un minimo peggiore).
La `bev_unit` cambia da `0.373092117369` a `0.367480188935` (−1.5 %): la
differenza di gauge è piccola, ma la rettifica è peggiore. Questi numeri
provengono da una versione precedente dello script (la coppia oggi nel repo
ha `bev_unit = 0.335545853435`, §6.2bis): vanno usati come ordine di
grandezza, non come riferimento assoluto.

**Uso corretto**: `--no-robust` serve solo per confronto diagnostico
("quanto guadagno il rigetto degli outlier?"). Non va usato per produrre
la coppia `H` + `bev_unit` destinata alla pipeline, e i suoi numeri **non
sono confrontabili** con quelli di un solve robusto (loss su insiemi di
quad diversi, scale diverse).

### 6.4 Check di coerenza con la ROI esistente

Al termine dell'estrazione lo script ri-proietta i `image_points` della ROI
(`roi_bev.json`) con la H appena estratta e li confronta con i `bev_points`
memorizzati nel file. Poiché un'omografia è definita a meno di una similitudine
del piano BEV, il confronto usa il **residuo similarity-invariant**: si stima la
similitudine (scala + rotazione + traslazione) che porta la BEV candidata su
quella memorizzata e si misura il residuo dei punti. Il gauge stimato (scala e
rotazione) viene stampato a fianco, così le due calibrazioni sono confrontabili
sullo stesso piano. Un residuo grande resta una campanella di allarme: la nuova
calibrazione si discosta da quella su cui la ROI è stata disegnata.

Con la coppia attualmente nel repo (H e `bev_unit` coerenti) il residuo
similarity-invariant è nullo per costruzione (i `bev_points` di
`roi_bev.json` sono stati ri-proiettati con quella stessa calibrazione).
Un residuo grande resta una campanella di allarme: la nuova calibrazione si
discosta da quella su cui la ROI è stata disegnata.

Quando si cambia la calibrazione, la ROI in BEV va ri-proiettata con la nuova
H: `roi_bev.json` memorizza infatti i `bev_points` assoluti calcolati con la
calibrazione del momento in cui la ROI è stata disegnata.

### 6.5 Ri-proiettare la ROI dopo un cambio di calibrazione

`roi_bev.json` memorizza la ROI due volte:

- `image_points` — la selezione fatta sul frame del video;
- `bev_points` — gli stessi punti proiettati nel piano BEV **con la calibrazione
  del momento in cui la ROI è stata salvata**.

Cambiando calibrazione, i `bev_points` memorizzati restano quelli vecchi: il
pannello BEV disegnerebbe l'attraversamento pedonale nella posizione
sbagliata mentre il pannello in prospettiva lo disegnerebbe correctly — i due
pannelli non concorderebbero, e il controllo "pedone fuori corsia" (che
confronta la posizione BEV del pedone con la ROI) darebbe risultati arbitrari.

La **selezione** della ROI non cambia: cambia solo la sua espressione nel
piano BEV. Perciò basta ri-proiettare i `image_points` con la nuova H:

```bash
.venv/bin/python extract_homography.py \
  --input video_dataset_bangkok/video_041.avi \
  --kp-model checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine \
  --max-frames 0 --output homography_bangkok.npy \
  --reproject-roi
```

Questo scrive `roi_bev.json` aggiornato (o nel file indicato da `--roi-out`)
senza toccare nessun altro file del flusso. `select_roi_bev.py` applica la
stessa regola quando salva una ROI: i `bev_points` vengono **sempre**
ricalcolati dai `image_points` con la calibrazione in uso.

### Regola d'oro

> H e bev_unit **devono sempre viaggiare insieme**, come coppia prodotta
> dalla stessa esecuzione dello script. Mai mescolare file di estrazioni
> diverse.

---

## 7. Uso dei file generati nella pipeline

La pipeline real-time consuma la coppia in due punti:

1. **Omografia statica** — `V2Pipeline.__init__` in `bev_realtime_v2.py`
   carica la matrice:

   ```python
   self.static_H = np.load("homography_bangkok.npy")
   ```

   Per usare una nuova calibrazione: sostituire quel file (o adattare il
   percorso) con il proprio `--output`.

2. **bev_unit** — indicata via CLI come file di testo:

   ```bash
   .venv/bin/python bev_realtime_v2.py ... --bev-unit-file homography_bangkok_bev_unit.txt
   ```

   (il valore viene letto dal file `*_bev_unit.txt`, **dello stesso `--output`**
   con cui è stata prodotta la H in uso).

   La coppia **attualmente in uso nel repo** è:

   | File | Valore |
   |---|---|
   | `homography_bangkok.npy` | H in gauge "repo" |
   | `homography_bangkok_bev_unit.txt` | `0.335545853435` |

   Se si estrae una nuova coppia (es. `--output homography.npy` →
   `homography_bev_unit.txt`), vanno **sostituiti entrambi i file insieme** e
   `--bev-unit-file` deve puntare al nuovo file `*_bev_unit.txt`; anche la
   ROI va ri-proiettata con `--reproject-roi` (§6.5).

   Il confronto con una coppia in una gauge diversa è
   invariante: la pipeline rende la BEV adattando il viewport alla gauge, quindi
   coppie coerenti diverse producono la stessa vista, a scala di gauge ribaltata.

---

## 8. Risoluzione dei problemi

| Sintomo | Causa probabile | Rimedio |
|---|---|---|
| `Errore: Impossibile aprire il video` | Percorso errato o codec non supportato | Verificare il percorso; provare a convertire in `.mp4`/`.avi` standard |
| `Errore: troppe poche footprint affidabili` | Video senza traffico, keypoint poco visibili, camera troppo angolata | Usare un video con più veicoli ben visibili; verificare il checkpoint |
| `[ATTENZIONE] Solo N footprint valide (< 128)` | Clip corta o traffico scarso | Aumentare `--max-frames` o usare un video più lungo; la bev_unit può essere poco stabile |
| Loss finale alta (`info.loss` grande nel log) | Troppi outlier, scene miste, camera che si muove | Preferire camera fissa e traffico distribuito; la loss cresce col numero di quad: confrontarla **per quad** |
| Checkpoint `.engine` fallisce | TensorRT non compatibile con la GPU/driver | Usare il checkpoint `.pt` equivalente |
| Distanze/TTC anomali dopo aver cambiato omografia | Gauge incoerente: H nuova con `bev_unit` vecchio (o viceversa) | Sostituire **sempre** `homography_bangkok.npy` e `homography_bangkok_bev_unit.txt` insieme e aggiornare `--bev-unit-file` (§6.2, §7) |
| `loss` molto più alta del previsto con `--no-robust` | Comportamento atteso: gli outlier non sono rigettati | Usare il solve robusto (default); `--no-robust` è solo diagnostico (§6.3) |

### Verifica rapida di una estrazione

```bash
# la matrice deve essere finita e ben condizionata
.venv/bin/python - <<'EOF'
import numpy as np
H = np.load("homography.npy")
print(H)
print("finita:", np.all(np.isfinite(H)), "| det(H[:2,:2]):",
      np.linalg.det(H[:2, :2]))
with open("homography_bev_unit.txt") as f:
    print("bev_unit:", float(f.read().strip()))
EOF
```

Una H sana ha valori finiti, `det` del blocco affine non nullo e una
bev_unit positiva e dell'ordine delle footprint proiettate attese (tipicamente
0.1–1 in BEV, dipende dalla risoluzione e dalla scena).
