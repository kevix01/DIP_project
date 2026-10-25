# Changelog: Ottimizzazione e Stabilizzazione del Tracking BEV OC-SORT

Questo documento illustra le modifiche architetturali apportate al meccanismo di tracking (`uod/bev_ocsort.py`) e alla pipeline di deduplicazione (`bev_realtime_v2.py`) per risolvere criticità legate a scambi di identità (ID switch), sdoppiamento delle tracce e instabilità geometriche dei bounding box 3D. Lo script va eseguito con i seguenti argomenti:
python bev_realtime_v2.py --input ..\Dataset_bangkok\videos\video_018_tagliato.avi --source-type video --kp-model ..\omnidetect_v2\checkpoints\urbanomnidetect_v2_x_640.engine --kp-imgsz 640 --kp-conf 0.1 --kp-vis 0.6 --class-conf 0:0.4,1:0.5,3:0.5 --suppress-nested 0.85 --object-px 80 --smooth 11 --device cuda:0 --output output_videos\out_018_stabilize_7_.avi --track-ocm-weight 2 --bev-unit 0.31147524796 --show-bev-ids --track-high-conf 0.45 

## 🎯 Problemi Risolti

1. **Scambio di ID tra pedoni in direzioni opposte:** Risolto il paradosso per cui due pedoni che si incrociano scambiavano la traccia a causa di una falla nel calcolo del vettore OCM (Observation-Centric Momentum) e della mancanza di un "hard veto" nello Stage 3 del tracker.
2. **Salto di ID da veicoli leggeri (motorini) ad auto in ingresso:** Risolto il bug per cui una traccia persa al centro della scena (es. un motorino) "saltava" su un'auto appena entrata ai bordi dell'inquadratura, a causa dell'eccessiva espansione del gate BEV e della mancanza di controlli 2D.
3. **Sdoppiamento delle tracce (Doppie Detection):** Eliminata la comparsa di doppie bounding box prospettiche sullo stesso pedone o veicolo causata dall'assenza di NMS intra-classe sulle detection prive di keypoint validi (`kp_ok = False`).
4. **Instabilità ID per rotazioni repentine del Bounding Box 3D:** Risolta l'assegnazione di nuovi ID causata dai "salti" del centro BEV quando la rete neurale ruotava bruscamente il cuboide 3D di pedoni e motorini.

---

## 🛠️ Modifiche Architetturali (Core Tracking)

### 1. Matching Cascade e Hard Veto OCM (Observation-Centric Momentum)
* **Matching Cascade in Stage 1:** Le tracce attive (viste nel frame precedente, `time_since_update == 1`) competono ora per le detection *prima* delle tracce in coasting (`time_since_update > 1`). Questo impedisce a una traccia "fantasma" con un gate molto espanso di rubare una detection a una traccia solida.
* **Doppio Veto Direzionale:** La logica OCM è stata riscritta. Ora valuta la coerenza direzionale non solo rispetto all'ultima posizione osservata (`last_obs_pos`), ma anche rispetto alla **posizione predetta dal filtro di Kalman (`pred_pos`)** e a un'**ancora storica** (3-4 frame precedenti). Questo applica un veto assoluto (`INF_COST`) alle associazioni controcorrente in *tutti* gli stage (1, 2 e 3), risolvendo il paradosso pre-incrocio.

### 2. Veto "Interno ➝ Bordo" e Gate di Distanza 2D
* È stato introdotto un controllo incrociato sulle coordinate 2D (`box_xyxy`). Se l'ultima posizione nota di una traccia era all'interno della scena, non le è permesso associarsi a una detection che compare improvvisamente sul bordo estremo del frame.
* Aggiunto un **gate di distanza massima in pixel 2D** relativo alla diagonale del bounding box. Questo previene salti visivamente assurdi che risultavano possibili in BEV a causa della distorsione prospettica.

### 3. Logica di Fallback 2D per `kp_ok = False` e Anti-Jitter 3D
* Le osservazioni con keypoint 3D inaffidabili (`kp_ok = False`) non vengono più scartate né usate per creare tracce fittizie. Vengono ora relegate allo **Stage 2** e usate **esclusivamente per mantenere in vita le tracce esistenti** (se c'è una forte sovrapposizione 2D).
* Quando un veicolo leggero subisce una rotazione anomala del BB 3D (identificata da un salto spaziale BEV incoerente col moto del box 2D), il tracker scarta i keypoint 3D per quel frame e **trasla rigidamente** l'impronta a terra e il cuboide del frame precedente, mantenendo l'ID e la stabilità visiva.
* I pedoni ricevono un cuboide sintetico non ruotato fin dal costruttore della traccia (`__init__`), eliminando le oscillazioni dovute ai keypoint grezzi.

### 4. Gestione Veicoli: Scala, Famiglia e Riordino Ciclico
* Implementata la separazione per "Famiglia Veicolo" (2 ruote vs 4+ ruote) basata sul voto di maggioranza della classe COCO originale (`source_cls_votes`).
* Ridotto il gate massimo in BEV per i veicoli (`0.85` base) e stretta la tolleranza massima di variazione di scala (da `1.8` a `1.42`), impedendo a una traccia "bici" di saltare su un "camion".
* Aggiunto un **riordino ciclico anti-farfalla** su 4 permutazioni (Kabsch alignment) per prevenire flip di 90° o 180° dell'orientamento del cuboide.

---

## 🧹 Modifiche alla Deduplicazione (`bev_realtime_v2.py`)

* **Raggruppamento per Classe Unificata:** La deduplicazione pre-tracking (`_deduplicate_unified`) raggruppa ora correttamente le istanze per classe unificata (`0` = pedoni, `2` = veicoli), permettendo di sopprimere i duplicati generati quando il detector classifica lo stesso oggetto fisico sia come `car` che come `motorcycle`.
* **NMS su Fallback 2D:** Le detection classificate come `without_ground` (`kp_ok = False`) vengono ora deduplicate tramite IoU 2D sia contro le detection "sane" sia al proprio interno. Questo elimina del tutto le sdoppiature prospettiche sui pedoni isolate.

---

## 🎛️ Soglie Principali che Puoi Regolare (CLI e Codice)

La pipeline è altamente parametrizzabile per adattarsi a diversi dataset. Di seguito i parametri chiave esposti tramite CLI o cablati nel codice per il fine-tuning del tracker:

| Parametro / Variabile | Dove si trova | Valore Consigliato | Effetto della modifica |
| :--- | :--- | :--- | :--- |
| `--suppress-nested` | Flag CLI (`bev_realtime_v2.py`) | **`0.85`** | Sopprime i box veicolo annidati (es. cabina dentro un camion) quando le footprint BEV si sovrappongono. Se a `0.0` (default) non esegue questa pulizia. |
| `--track-high-conf` | Flag CLI (`bev_realtime_v2.py`) | **`0.40` – `0.45`** | Alza la confidenza minima richiesta per far nascere una **nuova** traccia. Le tracce già attive continuano ad agganciare detection fino a `--kp-conf` (es. `0.25`) per il mantenimento. |
| `--track-ocm-weight`| Flag CLI (`bev_realtime_v2.py`) | **`2.0` – `5.0`** | Peso della penalità direzionale. Con l'attuale sdoppiamento `total_cost`/`gate_cost`, valori alti impediscono gli scambi durante gli incroci senza causare l'uccisione delle tracce isolate per micro-jitter. |
| `--min-hits` | Flag CLI (`bev_realtime_v2.py`) | **`3`** | Frame consecutivi richiesti per confermare una traccia. Abbassarlo velocizza la stabilizzazione dell'ID appena un oggetto entra in scena. |
| `--kp-vis` | Flag CLI (`bev_realtime_v2.py`) | **`0.60`** | Soglia di visibilità dei keypoint. `0.6` evita che l'occlusione parziale di un singolo vertice disattivi continuamente la modalità 3D dei motorini. |
| `dup_2d` | `_deduplicate_unified` (Codice) | **`0.35` – `0.40`** | Soglia IoU 2D oltre la quale due detection (es. pedone) nello stesso frame vengono fuse prima del tracker. Se noti ancora tracce sdoppiate, abbassala a `0.30`. |
| `dup_iou_thr` | Step 5 `BEVOCSortTracker` (Codice) | **`0.35` (pedoni)<br>`0.50` (veicoli)** | Impedisce di creare una nuova traccia se un'osservazione *non associata* si sovrappone visivamente a una traccia attiva (evita ghost track). |
```eof

Spero ti sia d'aiuto per chiudere il cerchio su queste sessioni di debug! Se in futuro avrai tempo di integrare anche la "Modifica 2" per l'occlusione dal basso (il birillo) e il controllo di convessità, potrai riprendere gli snippet che ti ho fornito e ampliare ulteriormente la robustezza. Ottimo lavoro con l'integrazione finora!