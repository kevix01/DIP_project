# select_roi_bev.py

Selettore di ROI (regione di interesse) su un frame video o su un'immagine,
con anteprima BEV e salvataggio in JSON.

La ROI BEV delimita l'area della strada "di interesse" su cui il pipeline
conta le violazioni. I punti vengono selezionati in coordinate immagine e
proiettati nel piano BEV (bird's-eye view) tramite l'omografia e la
`bev_unit`.

Il file prodotto (`roi_bev.json`) viene poi consumato da
`violation_detector.py` (`ZebraCrossingROI`), che lo usa per decidere se
un veicolo ha invaso la ROI.

---

## Indice

- [Come funziona](#come-funziona)
- [Utilizzo](#utilizzo)
- [Parametri CLI](#parametri-cli)
- [L'interfaccia](#linterfaccia)
- [Il file di output](#il-file-di-output)
  - [Riepilogo dei campi](#riepilogo-dei-campi)
  - [Campi di calibrazione](#campi-di-calibrazione)
  - [Campi geometrici](#campi-geometrici)
  - [Campi di provenienza della sorgente](#campi-di-provenienza-della-sorgente)
- [Coerenza calibrazione / ROI](#coerenza-calibrazione--roi)
- [Provenienza della sorgente](#provenienza-della-sorgente)
- [Recupero del frame di riferimento](#recupero-del-frame-di-riferimento)
- [Consumo downstream](#consumo-downstream)
- [Comportamento dei vertici](#comportamento-dei-vertici)
- [Troubleshooting](#troubleshooting)

---

## Come funziona

1. **Sorgente** — un video o un'immagine viene caricata (da CLI con
   `--frame`, oppure dalla UI). Per un video l'applicazione mostra uno
   slider per spostarsi fra i frame.

2. **Calibrazione** — l'omografia 3x3 e la `bev_unit` definiscono la
   proiezione dal piano immagine al piano BEV. La `bev_unit` è la scala
   che converte i pixel BEV in metri.

3. **Selezione dei vertici** — l'utente clicca sul canvas per definire un
   poligono. Ogni vertice vive in coordinate immagine (pixel).

4. **Proiezione** — al salvataggio ogni vertice immagine viene proiettato
   nel piano BEV con l'omografia e diviso per la `bev_unit`:

   ```
   bev_point = apply_homography(image_point, H) / bev_unit
   ```

5. **Salvataggio** — il JSON contiene sia i vertici immagine (per
   eventuali sovrapposizioni in video) sia i vertici BEV (per il conteggio
   delle violazioni), più tutti i metadati di provenienza.

### Come viene salvato il file

Il salvataggio **non** usa il download del browser: il server HTTP locale
scrive fisicamente il file JSON su disco e poi il browser lo scarica dalla
rotta `/download-roi`. Questo è il comportamento introdotto per supportare
Windows, dove il download del browser può fallire silenziosamente.

- **Standalone** (`python select_roi_bev.py ...`): il file viene scritto
  nella cartella di `--output` (per `--output roi_bev.json` è la cartella
  di lavoro), con il nome scelto nel campo "Nome file JSON".
- **Tramite webapp**: il file viene scritto nel percorso indicato da
  `--save-path` (usato dal frontend per il flusso "estrai ROI dalla
  webapp").

In entrambi i casi il nome è sanitizzato: vengono rimossi eventuali
percorsi e il nome viene forzato con estensione `.json`.

### La gauge: perché la calibrazione è così importante

Una ROI vale solo nella **gauge** in cui è stata creata, cioè nella specifica
combinazione di scala e orientamento definita dalla coppia
`(homography, bev_unit)`. I `bev_points` sono numeri che hanno senso solo
all'interno di quella gauge.

Se si ricalcola la stessa ROI con una omografia diversa (per esempio
un'estrazione più recente, o un video diverso), i valori numerici dei
`bev_points` cambiano — anche solo di un fattore di scala, se la
geometria della scena è la stessa ma la calibrazione è stata ri-estratta.

Per questo la coppia `(homography, bev_unit)` deve provenire **sempre dalla
stessa estrazione**: vedi `docs/extract_homography.md`.

---

## Utilizzo

```bash
# ROI nuova su un video, con calibrazione già estratta
.venv/bin/python select_roi_bev.py \
    --frame video_dataset_bangkok/video_041_tagliato_2.avi \
    --homography homography_bangkok.npy \
    --bev-unit-file homography_bangkok_bev_unit.txt \
    --output roi_bev.json

# Modifica di una ROI esistente: la calibrazione di provenienza viene
# ripristinata automaticamente e la sorgente viene verificata.
.venv/bin/python select_roi_bev.py \
    --frame video_dataset_bangkok/video_041_tagliato_2.avi \
    --roi roi_bev.json
```

### Flusso tipico

```bash
# 1. Estrarre la calibrazione (produce omografia + bev_unit coerenti)
.venv/bin/python extract_homography.py \
    --video video_dataset_bangkok/video_041_tagliato_2.avi \
    --out homography_bangkok

# 2. Selezionare la ROI usando quella calibrazione
.venv/bin/python select_roi_bev.py \
    --frame video_dataset_bangkok/video_041_tagliato_2.avi \
    --homography homography_bangkok.npy \
    --bev-unit-file homography_bangkok_bev_unit.txt \
    --output roi_bev.json

# 3. Se la ROI va ritoccata: riaprire con --roi
.venv/bin/python select_roi_bev.py \
    --frame video_dataset_bangkok/video_041_tagliato_2.avi \
    --roi roi_bev.json
```

---

## Parametri CLI

| Parametro | Default | Descrizione |
|---|---|---|
| `--frame` | – | Video o immagine iniziale. Può anche essere caricata dalla UI. Se è un video, il percorso assoluto viene salvato nel JSON come `source_video`. |
| `--homography` | – | File dell'omografia 3x3. Accettati `.npy` e `.txt` (3 righe di 3 valori). Se non viene dato e c'è una ROI, la calibrazione di provenienza viene ripristinata. |
| `--bev-unit-file` | – | File di testo con la `bev_unit` (un solo numero). Stessa logica dell'omografia. |
| `--roi` | – | ROI JSON da pre-caricare. Vengono caricati i vertici, la calibrazione di provenienza e la posizione della sorgente. |
| `--output` | `roi_bev.json` | Nome/percorso del file di output. In modalità standalone il server scrive il file salvato nella cartella di `--output`; il nome è anche precompilato nel campo "Nome file JSON" dell'interfaccia. |
| `--frame-index` | `1` | Frame iniziale per i video, **1-based** (stessa base del JSON: 1 = primo frame). Con `--roi`, se non viene indicato un valore esplicito, l'app si posiziona sul frame di riferimento della ROI. |
| `--port` | `8080` | Porta del server HTTP locale. |
| `--no-browser` | – | Non aprire il browser automaticamente. |
| `--keep-session-calibration` | – | Quando si pre-carica una ROI, **non** ripristinare la calibrazione di provenienza: si usa quella passata a `--homography`/`--bev-unit-file`. |

---

## L'interfaccia

L'applicazione avvia un server HTTP locale (`127.0.0.1:<port>`) e apre il
browser su quella pagina.

| Sezione | Contenuto |
|---|---|
| **Sorgente** | File caricato, slider e input numerico del frame (solo video). |
| **Calibrazione** | Omografia e `bev_unit` caricate, con il nome del file. |
| **Vista** | Zoom, reset, adatta allo schermo. |
| **ROI** | Rettangolo di default, cancella punti, carica ROI JSON esistente. |
| **Salvataggio** | Campo "Nome file JSON" (default `roi_bev.json`), pulsante Salva ROI, Esci, e la barra di stato (verde = ok, rosso = warning/errore). |
| **Punti BEV** | Elenco dei vertici proiettati con le coordinate. |

### Comandi mouse

| Azione | Effetto |
|---|---|
| Click su sfondo | Aggiunge un vertice |
| Click su maniglia + drag | Sposta il vertice |
| Right-click su maniglia | Rimuove il vertice |
| Click su un lato | Inserisce un vertice su quel lato |
| Rotella | Zoom |
| Alt + drag sinistro | Pan della vista |
| Ctrl + drag sinistro | Sposta tutta la ROI |

### Comandi tastiera

| Tasto | Effetto |
|---|---|
| `+` / `=` | Zoom in |
| `−` / `_` | Zoom out |
| `0` | Reset zoom e vista |
| `Canc` | Rimuove il vertice selezionato |
| `S` | Salva la ROI |

---

## Il file di output

### Riepilogo dei campi

| Campo | Tipo | Obbligatorio | Significato |
|---|---|---|---|
| `homography_file` | string | sì | Nome del file dell'omografia usata per la proiezione. |
| `bev_unit_file` | string | sì | Nome del file da cui proviene la `bev_unit`. |
| `bev_unit` | float | sì | Scala metri/pixel applicata ai pixel BEV. |
| `image_points` | list[[x, y]] | sì | Vertici del poligono in **pixel immagine**. |
| `bev_points` | list[[x, y]] | sì | Vertici del poligono in **metri**, nel piano BEV. |
| `calibration_source` | object | no | Da dove arriva la calibrazione usata (`"roi"` o `"session"`). |
| `source_video` | string | video | Percorso assoluto del video da cui la ROI è stata ricavata. |
| `source_video_frame_id` | int | video | Frame di riferimento, **1-based**. |
| `source_frame_index_base` | int | no | Base dell'indice frame dichiarata (`1`). Rende esplicita la convenzione. |
| `source_frame` | string | immagine | Percorso assoluto dell'immagine sorgente. |
| `source_width` | int | no | Larghezza in pixel della sorgente. |
| `source_height` | int | no | Altezza in pixel della sorgente. |
| `source_original_name` | string | no | Nome originale del file, utile per le sorgenti caricate da UI (il path è content-addressed). |
| `source_fingerprint` | object | no | Identificativo del file sorgente per **contenuto**. |

Esempio reale (abbreviato):

```json
{
  "homography_file": "homography_bangkok.npy",
  "bev_unit_file": "homography_bangkok_bev_unit.txt",
  "bev_unit": 0.31147524796,
  "image_points": [[29.87, 196.86], [1023.02, 192.31], "..."],
  "bev_points": [[-4.12, -3.20], [3.03, -3.36], "..."],
  "calibration_source": {"homography": "roi", "bev_unit": "roi"},
  "source_video": "/home/kevin/Scrivania/DIP_project/video_dataset_bangkok/video_041_tagliato_2.avi",
  "source_video_frame_id": 1,
  "source_frame_index_base": 1,
  "source_width": 1200,
  "source_height": 1100,
  "source_original_name": "video_041_tagliato_2.avi",
  "source_fingerprint": {
    "size": 62207670,
    "mtime": 1792156855,
    "sha1_head": "bae01a9880069e0170954bf66b9d3a85a3292080"
  }
}
```

### Campi di calibrazione

#### `homography_file`

Nome del file dell'omografia usata per proiettare i vertici. È solo il
**nome**, non il percorso: il file viene cercato (al caricamento di una ROI)
nella directory del ROI, in quella di lavoro e in quella del progetto.

Se l'omografia è stata caricata da UI con drag & drop, il nome è quello del
file originale.

#### `bev_unit_file`

Nome del file da cui è stata letta la `bev_unit`. Stessa logica di ricerca
dell'omografia.

#### `bev_unit`

La scala della calibrazione: quanti metri corrisponde un pixel del piano BEV.
In pratica `bev_unit = metri_per_pixel`.

È il valore **numerico effettivamente usato** per dividere i pixel BEV, non
un riferimento a un file. Per questo motivo `bev_unit` e `homography_file`
devono provenire dalla stessa estrazione: una `bev_unit` diversa con la
stessa omografia (o viceversa) produce `bev_points` numericamente sbagliati
pur restando geometricamente simili.

#### `calibration_source`

Registra **da dove** è arrivata la calibrazione usata per il salvataggio:

```json
"calibration_source": {"homography": "roi", "bev_unit": "roi"}
```

| Valore | Significato |
|---|---|
| `"roi"` | La componente è stata ripristinata dal file ROI in editing (è la calibrazione con cui la ROI era stata creata). |
| `"session"` | La componente proviene dalla sessione corrente: caricata da `--homography`/`--bev-unit-file` o da drag & drop nella UI. |

I due campi sono indipendenti: se l'utente sostituisce solo l'omografia,
si avrà `{"homography": "session", "bev_unit": "roi"}`.

**A cosa serve.** In passato, se si apriva una ROI con una calibrazione
diversa da quella in sessione, la UI non avvisava e il JSON non permetteva
di sapere quale delle due fosse stata usata per la ricalcolazione.
Ora, combinando `calibration_source` (da quale parte è arrivata),
`homography_file`/`bev_unit_file` (quale file) e `bev_unit` (il valore), la
ricalcolazione è **auditabile**: se i `bev_points` non corrispondono a
quelli attesi, il JSON dice immediatamente se il file è stato rigenerato con
una calibrazione diversa.

La calibrazione di provenienza ha la precedenza per default; con
`--keep-session-calibration` si inverte la priorità.

### Campi geometrici

#### `image_points`

Lista di coppie `[x, y]` in **pixel immagine**, con l'origine in alto a
sinistra. Sono i vertici esattamente come li vede l'utente sul canvas.

Sono utili per:
- sovrapporre la ROI sul frame in video (la ROI "accompagna" la scena);
- rigenerare i `bev_points` se la calibrazione cambia;
- verificare che i vertici siano dentro l'immagine.

Il downstream (`violation_detector.py`) li converte in `int32` e li usa per
il disegno della ROI prospettica sul video.

#### `bev_points`

Lista di coppie `[x, y]` in **metri**, ottenute come:

```
bev_point = apply_homography(image_point, H) / bev_unit
```

Sono i vertici del poligono nel piano BEV, e sono i valori su cui il
pipeline conta le violazioni. Il poligono è chiuso: l'ultimo vertice si
riconnette al primo.

I `bev_points` vengono **sempre ricalcolati** dai `image_points` con la
calibrazione in uso al momento del salvataggio: non vengono mai copiati dal
file caricato. Il file salvato è quindi sempre internamente coerente.

Il sistema di riferimento BEV ha origine nel punto fisso della calibrazione
(corner del rettangolo di riferimento): le coordinate possono quindi essere
negative. Non è un errore.

### Campi di provenienza della sorgente

#### `source_video`

Percorso **assoluto** del video da cui la ROI è stata ricavata. Presente
solo se la sorgente era un video.

Quando il video è stato caricato dalla UI, il percorso è quello della cache
content-addressed (vedi [Sorgenti caricate da UI](#sorgenti-caricate-da-ui)),
non il path temporaneo originale.

#### `source_video_frame_id`

Il frame su cui la ROI è stata definita, in base **1-based** (1 = primo
frame, come mostrato nella UI).

È il frame "di riferimento": i `image_points` sono validi rispetto a
quello specifico frame.

#### `source_frame_index_base`

Vale `1` e dichiara esplicitamente la base dell'indice
`source_video_frame_id`. Serve a evitare che un consumer interpreti
l'indice come 0-based (come è invece nell'API interna del server, che
parla in indici 0-based).

#### `source_frame`

Percorso assoluto dell'immagine sorgente. Presente solo se la sorgente era
un'immagine fissa (in quel caso non esiste un "frame" e il campo
`source_video_frame_id` non viene scritto).

#### `source_width` / `source_height`

Dimensioni in pixel della sorgente. Permettono di verificare che i
`image_points` (che sono in pixel assoluti) siano compatibili con la
sorgente su cui si sta lavorando, e servono da fallback quando il
fingerprint non è disponibile (ROI create prima di questa versione).

#### `source_original_name`

Nome originale del file sorgente. Per le sorgenti caricate da UI il path
su disco è un hash, quindi questo campo conserva l'informazione leggibile
("video_041_tagliato_2.avi").

#### `source_fingerprint`

Identificativo del file sorgente basato sul **contenuto**, non sul percorso:

```json
"source_fingerprint": {
  "size": 62207670,
  "mtime": 1792156855,
  "sha1_head": "bae01a9880069e0170954bf66b9d3a85a3292080"
}
```

| Campo | Significato |
|---|---|
| `size` | Dimensione del file in byte. |
| `mtime` | Data di ultima modifica (epoch Unix, secondi). |
| `sha1_head` | SHA1 calcolato su testa, centro e coda del file (1 MiB ciascuno). |

Lo `sha1_head` non è lo SHA1 dell'intero file: è un hash di un sottoinsieme
del file, scelto per essere **veloce** su video grandi pur restando
abbastanza discriminante. In pratica è l'equivalente di un "content hash"
rapido.

**A cosa serve.** Permette di verificare che la sorgente caricata sia
*davvero* lo stesso file, anche se il percorso è cambiato o il file è stato
ricopiato. Al caricamento di una ROI viene confrontato con la sorgente
corrente:

- coincide → la UI conferma "provenienza e calibrazione verificate";
- non coincide → la UI avvisa in rosso che i `image_points` non sono più
  validi per la sorgente caricata.

---

## Coerenza calibrazione / ROI

Una ROI vale solo nella gauge in cui è stata creata. Il caricamento di una
ROI esistente **ripristina quella coppia** `(homography, bev_unit)`,
cercando i file indicati nel JSON in quest'ordine:

1. il percorso così com'è (relativo alla working directory);
2. la directory del file ROI;
3. la directory del progetto (dove vive lo script).

Se l'omografia non è trovabile ma la `bev_unit` sì (o viceversa), la
componente recuperata viene usata e l'altra resta quella di sessione; la UI
segnala il mismatch. Se nessuna delle due è recuperabile, si usa la
calibrazione di sessione e l'utente viene avvisato.

**Fallback della `bev_unit`.** Se il file `bev_unit_file` non è
ritrovabile, ma il JSON contiene il valore numerico `bev_unit`, quel valore
viene usato direttamente. In questo modo la ROI resta nella sua gauge
originale anche senza accesso al file, e `bev_unit_file` conserva comunque
il nome del file di origine per riferimento.

Per l'omografia non esiste un fallback simile: la matrice 3x3 non è salvata
nel JSON (i file restano la fonte di verità), quindi senza il file
l'omografia di sessione è l'unica opzione.

In ogni caso i `bev_points` vengono ricalcolati con la calibrazione
effettivamente in uso, quindi il file salvato è sempre internamente
coerente — cambiare calibrazione non produce mai un file "mezzo vecchio,
mezzo nuovo".

---

## Provenienza della sorgente

### Verifica al caricamento

Quando si carica una ROI, la sorgente corrente viene confrontata con quella
della ROI:

1. **fingerprint** (se presente) — confronto di dimensione e `sha1_head`;
2. **dimensioni** (fallback per ROI vecchie) — confronto di
   `source_width`/`source_height`;
3. nessuna delle due → la verifica non è possibile e viene dichiarata
   "sconosciuta".

### Sorgenti caricate da UI

I file caricati tramite browser vengono copiati in una cache persistente
content-addressed:

```
~/.cache/roi_bev_uploads/<sha1>.<estensione>
```

Il percorso resta valido dopo la chiusura dell'applicazione, quindi il JSON
salvato non punta mai a un file temporaneo cancellato (che sarebbe il
problema di una cache in `/tmp`). Il nome originale è comunque conservato in
`source_original_name`.

Ricaricando lo stesso file la cache lo riutilizza: nessun duplicato.

---

## Recupero del frame di riferimento

```bash
# frame 1 (1-based) della sorgente della ROI
.venv/bin/python select_roi_bev.py --frame <video.avi> --frame-index 1
```

Attenzione alla base degli indici: `--frame-index` è **1-based** come il
JSON, mentre l'API interna del server (`/frame?index=`) è 0-based. Il
traduzione è automatica.

Il seek usa `CAP_PROP_POS_FRAMES` e **verifica** la posizione ottenuta: su
backend keyframe-based (tipico di H.264) il frame viene riletto
sequenzialmente per essere esatto.

Con `--roi`, se non viene indicato un `--frame-index` esplicito,
l'applicazione si posiziona automaticamente sul frame di riferimento della
ROI.

---

## Consumo downstream

`violation_detector.py` carica il JSON:

```python
from violation_detector import ZebraCrossingROI

roi = ZebraCrossingROI("roi_bev.json", bev_unit=0.31147524796)
```

`ZebraCrossingROI` richiede esattamente due chiavi: **`bev_points`** e
**`image_points`**. Se mancano solleva un `ValueError`; se hanno meno di 3
punti solleva un errore esplicito.

Tutti gli altri campi del JSON sono **metadati**: non influenzano il
conteggio delle violazioni, ma servono a mantenere tracciabilità e
riproducibilità.

---

## Comportamento dei vertici

- I vertici caricati da una ROI JSON **sopravvivono** al cambio frame: il
  rettangolo di default non viene ricreato.
- "Cancella punti" svuota la ROI e il cambio frame **non** ricrea nulla: si
  riparte da zero.
- Aggiungere un vertice (click su sfondo o su un lato) riattiva
  l'editing manuale; "Rettangolo" ripristina i 4 vertici di default.
- Il salvataggio richiede **almeno 4 punti** e un'omografia caricata.

---

## Troubleshooting

| Sintomo | Causa | Rimedio |
|---|---|---|
| "Servono almeno 4 punti" | ROI con meno di 4 vertici | Aggiungi vertici o premi "Rettangolo" |
| "Carica prima un'omografia" | Omografia assente | Caricala dalla UI o riapri la ROI (la calibrazione di provenienza viene ripristinata) |
| "SORGENTE DIVERSA" | La sorgente caricata non è quella della ROI | Usa il file originale indicato in `source_video`, oppure procedi consapevolmente: i `image_points` vanno ricalcolati |
| "Calibrazione: omografia non recuperata" | Il file omografia indicato nel JSON non è stato trovato | Copialo nella directory del progetto, oppure carica l'omografia manualmente |
| I `bev_points` non corrispondono a quelli attesi | La calibrazione usata è diversa da quella di creazione | Controlla `calibration_source` nel JSON; ricalcola con la coppia originale |
| Il video non si carica | Codec non supportato da OpenCV | Convertire il video (es. con ffmpeg) in un formato supportato |
| "Errore caricamento frame" | Frame non raggiungibile | Verificare `--frame-index` e il numero totale di frame |

---

## Note

- L'anteprima BEV usa la calibrazione in uso; con una ROI caricata la
  calibrazione di provenienza ha la precedenza.
- La coppia `(homography, bev_unit)` deve provenire dalla **stessa**
  estrazione: vedi `docs/extract_homography.md`.
- L'omografia non è salvata *dentro* il JSON (solo il nome del file): il
  file resta la fonte di verità, così ogni ricalcolazione usa gli stessi
  artefatti. La `bev_unit` invece è salvata anche come valore numerico, per
  poter essere recuperata anche senza il file (vedi
  [Coerenza calibrazione / ROI](#coerenza-calibrazione--roi)).
