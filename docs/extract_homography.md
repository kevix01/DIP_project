# Estrazione Omografia e bev_unit

Questo documento descrive come usare `extract_homography.py` per calibrare la
**matrice di omografia globale** (image → piano stradale BEV) e la **bev_unit**
(scala media delle footprint in BEV) a partire da un video di calibrazione.

Lo script è **standalone**: non usa la pipeline real-time (`bev_realtime_v2`),
non fa tracking né rendering. Esegue solo l'inferenza pose, filtra le
footprint affidabili dei veicoli e risolve una singola omografia globale con
l'ottimizzatore a vincolo di ortogonalità (`homography_rt.py`).

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
| `ultralytics` | Inferenza del modello pose (YOLO) |
| `opencv-python` | Lettura video e test di convessità |
| `numpy` | Calcolo numerico |

Lo script **non dipende più** da `bev_realtime_v2`, `bev_realtime`, `refine_v2`
né da nessun altro modulo della pipeline real-time. Non richiede nemmeno un
file di omografia pre-esistente in input (il vecchio script terminava con
`sys.exit(1)` se `homography_bangkok.npy` mancava).

---

## 2. Utilizzo

### Comando base

```bash
.venv/bin/python extract_homography.py \
  --input video_dataset_bangkok/video_041_tagliato_2.avi \
  --kp-model checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine \
  --output homography.npy
```

### Requisiti del video di calibrazione

Per una calibrazione affidabile il video deve avere:

- **Telecamera fissa** (o quasi): l'omografia globale assume un piano stradale
  statico rispetto alla camera.
- **Traffico ben distribuito su tutta la scena**: ogni zona del piano stradale
  dovrebbe essere attraversata da qualche veicolo durante il video. Zone della
  scena mai toccate da footprint affidabili degradano la rettifica locale.
- **Almeno 128 footprint veicolo valide** (consigliato): sotto questa soglia lo
  script emette un `[ATTENZIONE]` e la bev_unit può risultare poco stabile.
  Con 1500 frame a 30 fps (~50 s di traffico) si superano tipicamente le
  5000+ footprint.
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
   - soglia di confidenza per classe (`class_conf`: person 0.4, bike 0.5, moto 0.5);
   - visibilità dei keypoint (`kp_vis = 0.5`);
   - keypoint dentro la box 2D (tolleranza 0.75) e non collassati a (0,0);
   - **convessità** della footprint (un quad auto-intersecante scarta la detection);
   - area minima della footprint (≥ max(16 px², 0.3% della box 2D));
   - keypoint **tutti dentro il frame** (gli estremi estrapolati non sono affidabili);
   - box non tagliata dal bordo immagine;
   - rapporto area keypoint/box tra 0.12 e 2.5.
3. **Selezione classi** — entrano in calibrazione solo veicoli con ground
   affidabile: car (2), bus (5), truck (7), bicycle (1), motorcycle (3). I
   pedoni sono esclusi (la pipeline li rappresenta con un quad sintetico
   dalla box 2D, privo di informazione geometrica utile alla rettifica).
4. **GroundIndexResolver** — determina quali 4 degli 8 keypoint sono i
   vertici a terra (voto maggioritario sui primi frame, come in pipeline).
5. **Solve globale** — `OrthoHomographySolver` (Levenberg-Marquardt su 4 DoF
   essenziali, residui = coseni degli angoli delle footprint + parallelismo
   opposto, pesi IRLS Cauchy) su **tutte** le footprint raccolte: una sola
   omografia per tutto il video, come fa `_apply_global_homography` offline
   nella pipeline. Il flag `--no-robust` disattiva i pesi IRLS Cauchy
   (vedi §6.3).
6. **bev_unit** — mediana di √(larghezza×lunghezza) delle footprint proiettate
   con la H appena estratta (formula di `_class_footprint_dims`).
7. **Salvataggio** dei tre file descritti sopra.

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

Estrazione eseguita su `video_dataset_bangkok/video_041_tagliato_2.avi`
(900 frame, **6526 footprint** veicolo valide) con
`--kp-model checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine`:

| Solve | `robust` | loss | iterazioni | `bev_unit` |
|---|---|---|---|---|
| Default (IRLS Cauchy) | `True` | `29.90` | 30 | **`0.373092117369`** |
| `--no-robust` | `False` | `189.25` | 7 | `0.367480188935` |

La `bev_unit` della coppia già in uso nel repo
(`homography_bangkok.npy` + `homography_bangkok_bev_unit.txt`) è
**`0.31147524796`**; ricalcolata sulle stesse 6526 footprint dà `0.308080`
(≈ −1 % rispetto al valore memorizzato, che era stato calcolato su un
insieme di quad leggermente diverso).

Il rapporto fra le due gauge è **0.373092117369 / 0.31147524796 = 1.1978**:
le due omografie sono geometricamente equivalenti ma espresse in gauge
diverse (differiscono per una similitudine di scala ≈ 1.20–1.24 e una
rotazione di pochi gradi). Come ricorda la regola d'oro, **H e bev_unit
devono essere usate insieme**: passare `--bev-unit 0.31147524796` con la H
nuova (o viceversa) introduce un errore di gauge del ~20 % sulle dimensioni
delle footprint in BEV, che si traduce in errori sulle distanze di conflitto
usate dal rilevatore TTC.

### 6.3 Effetto di `--no-robust`

Il solve non robusto **non rigetta i corner outlier**: la loss passa da
`29.90` a `189.25` (× 6.3) e la convergenza avviene in 7 iterazioni invece
di 30 (la LM si ferma su un minimo peggiore). La `bev_unit` cambia da
`0.373092117369` a `0.367480188935` (−1.5 %): la differenza di gauge è
piccola, ma la rettifica è peggiore.

**Uso corretto**: `--no-robust` serve solo per confronto diagnostico
("quanto guadagno il rigetto degli outlier?"). Non va usato per produrre
la coppia `H` + `bev_unit` destinata alla pipeline, e i suoi numeri **non
sono confrontabili** con quelli di un solve robusto (loss su insiemi di
quad diversi, scale diverse).

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

2. **bev_unit** — passata via CLI come parametro:

   ```bash
   .venv/bin/python bev_realtime_v2.py ... --bev-unit 0.31147524796
   ```

   (valore letto dal file `*_bev_unit.txt`, **dello stesso `--output`** con cui
   è stata prodotta la H in uso).

   La coppia **attualmente in uso nel repo** è:

   | File | Valore |
   |---|---|
   | `homography_bangkok.npy` | H in gauge "repo" |
   | `homography_bangkok_bev_unit.txt` | `0.31147524796` |

   Se si estrae una nuova coppia (es. `--output homography.npy` →
   `bev_unit = 0.373092117369`), vanno **sostituiti entrambi i file insieme**
   e il valore di `--bev-unit` deve passare a quello nuovo.

Il confronto con la vecchia coppia Bangkok (`H` con `bev_unit ≈ 0.311`) è
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
| Distanze/TTC anomali dopo aver cambiato omografia | Gauge incoerente: H nuova con `bev_unit` vecchio (o viceversa) | Sostituire **sempre** `homography_bangkok.npy` e `homography_bangkok_bev_unit.txt` insieme e aggiornare `--bev-unit` (§6.2, §7) |
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
