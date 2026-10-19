# Changelog — Stabilizzazione pedoni (BEV + vista camera)

## Sintesi

Sono state introdotte modifiche per stabilizzare le bounding box e i cubetti 3D dei pedoni (classe COCO 0), sia nella vista BEV (omografia, pannello di destra) che nella vista camera (pannello di sinistra). Il problema principale era che i pedoni hanno footprint quasi quadrati e keypoints rumorosi, il che causava rotazioni di 90°, espansione/contrazione della box e balzi improvvisi nella BEV.

---

## Modifiche per file

### `bev_realtime.py`

1. **Nuovo parametro `_rigid_lock_after_ped`** (riga 159): i pedoni bloccano la forma dopo 12 osservazioni invece di 20, poiché il quadrato è più semplice del rettangolo veicolo.

2. **Nuovo dizionario `_rigid_ped`** (riga 160): archivio separato per le impronte rigide dei pedoni, per evitare conflitti di schema con `_rigid` (veicoli).

3. **Nuovo metodo `_pedestrian_rigid_footprint()`** (riga 675): stima un'impronta a terra **quadrata costante** per i pedoni:
   - Forza forma quadrata (nessun axis-flip tra lunghezza/larghezza)
   - Tiene l'orientamento fisso dopo il lock (nessun salto di 90° da Kabsch)
   - Applica un'**EMA causale sul centroide** (alpha=0.2 dopo lock, 0.4 prima) per evitare balzi improvvisi
   - Aggiorna la scala incrementalmente fino al lock

4. **Fix bug `np.cross` su vettori 2D**: sostituito con prodotto vettoriale 2D manuale (`a[0]*b[1] - a[1]*b[0]`) nel metodo `_pedestrian_rigid_footprint`.

5. **Guardia `bev_unit <= 1e-9`**: evita divisione per zero quando la scala BEV non è ancora valida.

6. **Modifica `_apply_global_homography()`** (riga 1163): quando `--fixed-camera on`, i pedoni ora usano `_pedestrian_rigid_footprint` invece di `_rigid_footprint` (vehicle-style), evitando sovrascrittura del quadrato rigido con forma rettangolare instabile.

---

### `bev_realtime_v2.py`

1. **Nuovo stato `self._ped_box_ema`** (riga ~407): dizionario `tid -> (w_ema, h_ema)` per l'EMA causale sulle dimensioni della 2D box dei pedoni.

2. **Nuovo stato `self._ped_height`** (riga ~408): dizionario `tid -> altezza` per l'altezza costante del cubetto 3D sintetico dei pedoni.

3. **Modifica `V2Pipeline.step()`** (riga ~536-618):
   - I pedoni con `kp_ok=True` ora usano `_pedestrian_rigid_footprint` per `bev_quad` invece di `apply_homography(v.ground, state.H)` (proiezione raw)
   - I pedoni con `kp_ok=False` ma `ground` valido (`elif`) ricevono lo stesso trattamento (prima erano esclusi)
   - **Nuovo blocco di stabilizzazione camera** (prima di `return state`):
     - EMA causale più aggressivo sulla 2D box (0.25 nuovo / 0.75 precedente)
     - Box ricostruita attorno al bottom-centre (contatto a terra, punto più stabile)
     - Cubetto 3D sintetico: 4 ground corners dal rettangolo smussato + 4 roof corners sollevati di altezza costante (nessuna rotazione)
     - `v.bev_quad` ricalcolato dallo stesso ground sintetico per coerenza camera|BEV

4. **Modifica `_stabilize_offline_v2()`** (riga ~1161): nel blocco "PROIEZIONE MIRATA IN BEV", i pedoni usano `pipe._pedestrian_rigid_footprint` invece di `apply_homography(v.ground, H_mat)`.

5. **Modifica `_stabilize_offline_v2()`** (riga ~1206): nel blocco finale "SBLOCCO DEI BBOX", i pedoni vengono saltati (`if v.cls == 0: continue`) per non sovrascrivere il loro footprint rigido quadrato.

---

### `uod/smoothing.py`

1. **`smooth_footprints()`**: i pedoni (cls==0) vengono saltati completamente. Il loro heading è ill-definito (simmetria 90°) e ridecomporlo in centroide+heading reintroduce rotazioni.

2. **`smooth_states()`**: i pedoni (cls==0) vengono saltati nel blocco di smoothing del footprint BEV, per lo stesso motivo sopra.

---

## Flusso di esecuzione

### Senza `--smooth` (tempo reale)
`V2Pipeline.step()` → `super().step()` (tracker + rigid footprint) → blocco camera (EMA box + cubetto sintetico) → `v.bev_quad` ricalcolato dal ground sintetico → render

### Con `--smooth 11 --fixed-camera on` (offline)
Pass 1: `pipe.step()` per ogni frame → Pass 2: `_stabilize_offline_v2`:
- `_orig_stabilize` → `smooth_states` (skip pedoni per footprint) + `_apply_global_homography` (pedoni con `_pedestrian_rigid_footprint`)
- Blocco v2: `H = static_H` + ricalcolo `bev_quad` per pedoni con `_pedestrian_rigid_footprint`
- Blocco finale SBLOCCO: skip pedoni

In entrambi i percorsi, il `bev_quad` finale dei pedoni proviene sempre da `_pedestrian_rigid_footprint`, garantendo coerenza tra pannello sinistro (camera) e destro (BEV).

---

## Effetti attesi

| Elemento | Prima | Dopo |
|----------|-------|------|
| Box 2D pedone (camera) | Si allarga/restringe frame a frame | Dimensioni smussate EMA, ancorate al bottom-centre |
| Cubetto 3D pedone (camera) | Ruota e cambia dimensioni | Footprint quadrato rigido + altezza costante, nessuna rotazione |
| Impronta BEV pedone | Balzi, rotazioni 90°, espansione/contrazione | Quadrato costante, centro smussato, posizione stabile |
| Coerenza camera|BEV | Non garantita | Stesso ground sintetico per entrambi i pannelli |


## Tracker OC-SORT in BEV e buffer fixed-lag

**Confronto:** branch `pedestrian_stabilize_2`, commit di base `cc78858` (`Removed duplicated call to _pedestrian_stabilize...`).  
**Ramo di lavoro previsto:** `codex/ocsort-realtime-changelog`.

Questa sezione documenta il codice presente nel ramo di lavoro rispetto al branch di base. Il nuovo percorso V2 sostituisce il tracking d’immagine del precedente loop con associazione direttamente sul piano BEV e un buffer circolare causale con look-ahead limitato.

### Modifiche principali

- `uod/bev_ocsort.py` introduce `BEVOCSortTracker`, `BEVKalmanTrack`, `BEVObservation`, `FastSavGol` e `FixedLagBuffer`.
- `bev_realtime_v2.py` proietta le osservazioni nel BEV calibrato, aggiorna OC-SORT, applica il buffer fixed-lag e gestisce l’acquisizione sia da file sia da stream.
- `uod/keypoints.py` aggiunge `Detection.source_cls`, che conserva la classe COCO originale anche quando `d.cls` viene unificata per la visualizzazione.
- `export_engine.py` imposta `imgsz=960` come dimensione predefinita per l’esportazione TensorRT.

### Principio di funzionamento di OC-SORT

Il tracker mantiene per ogni oggetto una posizione e una velocità nel piano BEV. Il filtro di Kalman predice la posizione al frame successivo e corregge la stima quando arriva una nuova osservazione. L’associazione globale fra tracce e rilevamenti è risolta con l’algoritmo Hungarian.

Il costo considera la distanza BEV normalizzata rispetto a un gate di moto. OC-SORT aggiunge l’**Observation-Centric Momentum (OCM)**: confronta la direzione ricavata dagli ultimi rilevamenti con la direzione tra l’ultima osservazione e il candidato. La direzione osservata riduce gli scambi quando le traiettorie si incrociano.

Il matching è eseguito in tre passaggi:

1. associazione delle tracce alle osservazioni ad alta confidenza;
2. recupero delle tracce ancora senza match con osservazioni a bassa confidenza, fino a quattro frame di coasting;
3. recupero OC-SORT basato sull’ultima posizione osservata per le tracce ancora non associate.

Le nuove tracce vengono create solo dai rilevamenti ad alta confidenza non associati. Dopo un’interruzione, l’**Observation-Centric Re-Update (ORU)** riporta lo stato all’ultima osservazione e riproduce aggiornamenti virtuali lineari tra questa e la nuova osservazione, prima di correggere lo stato con il nuovo rilevamento.

Le osservazioni veicolo sono ottenute proiettando i quattro keypoint a terra con l’omografia statica e normalizzandoli per `bev_unit`. Le auto e i veicoli pesanti usano una forma rigida stimata con Procrustes; i pedoni usano il centro BEV ricavato dal bottom-centre del box e una footprint quadrata adattiva.

### Protezione dagli scambi auto–moto e dalle variazioni di scala

Prima dell’unificazione della classe grafica, `source_cls` conserva la classe COCO. Il deduplicatore raggruppa i veicoli anche per classe originale, evitando che una rilevazione di moto sopprima un’auto solo perché le footprint si sovrappongono. Il tracker divide poi le classi in due famiglie: bici/moto (`1`, `3`) e auto/bus/camion (`2`, `5`, `7`). Un’associazione tra famiglie diverse viene esclusa.

La dimensione osservata `obs_scale` è la radice dell’area della footprint BEV. Il costo di matching:

- scarta una coppia se il rapporto fra le scale supera `1.8`;
- aggiunge una penalità `0.15 × min(abs(log(scale_obs / scale_track)), 1)` alle coppie rimaste;
- aggiorna la scala della traccia con un’EMA `0.9 × scala_precedente + 0.1 × scala_osservata` solo se il rapporto resta entro `1.8`.

Questi vincoli rendono meno probabile che la traccia dell’auto rosa adotti la footprint del motorino. La soglia `1.8` è un valore interno al codice, non un’opzione CLI.

### Buffer circolare fixed-lag: `W=11`, `L=5`

Il buffer conserva gli ultimi `W` stati elaborati, ciascuno con frame e geometrie delle tracce. Per impostazione predefinita `W=11` e `L=W//2=5`. Quando il buffer è pieno, viene emesso il frame centrale `t-5`, usando i frame da `t-10` a `t`. La latenza temporale di cinque frame fornisce look-ahead senza attendere l’intero video.

Le logiche implementate nel buffer sono:

- **Savitzky–Golay zero-phase e velocità:** `FastSavGol` adatta un polinomio di secondo grado alla traiettoria BEV della traccia. Sul frame centrale restituisce posizione smussata `(X,Y)` e derivata analitica `(Vx,Vy)` del polinomio.
- **Interpolazione dei micro-gap:** se il frame da emettere è in coasting e la traccia ha un’osservazione valida sia prima sia dopo nella finestra, interpola linearmente box 2D, keypoint, ground quad e posizione BEV. Il footprint BEV viene poi ricostruito dalla forma rigida della traccia e dalla posizione smussata.
- **Soppressione flicker e code di coasting:** una traccia con meno di `min_hits` osservazioni nel buffer non viene emessa. Un frame in coasting senza una nuova osservazione nel look-ahead viene rimosso dall’output. Questa implementazione usa `min_hits` e il buffer; non applica i parametri offline `min_track` e `coast_tail`.
- **Veicoli fermi e heading da traiettoria:** sotto `0.35` unità BEV/s e con spostamento inferiore a `0.18`, il buffer conserva un anchor di posizione e angolo. Per un moto sufficientemente rettilineo, con velocità almeno `0.8`, spostamento almeno `0.35` e rapporto spostamento/percorso almeno `0.92`, allinea l’asse lungo del rettangolo alla velocità stimata.

Il filtro Savitzky–Golay precalcola le pseudo-inverse delle matrici di Vandermonde per le lunghezze e gli indici della finestra. Per ogni emissione, posizione e velocità sono ottenute con prodotti matriciali sui pesi precalcolati.

### Acquisizione da file e da streaming

Il buffer fixed-lag di 11 stati è usato in entrambi i modi. La sorgente in ingresso è gestita separatamente da `FrameSource`:

- **File video:** `read()` delega a `VideoCapture.read()` e consegna i frame sequenzialmente. A fine file `flush_remaining()` emette i cinque stati finali che non hanno più look-ahead.
- **Stream / telecamera:** un thread daemon legge continuamente da OpenCV, con `CAP_PROP_BUFFERSIZE=1`, e sovrascrive uno slot contenente il frame più recente. Lo slot è protetto da `threading.Lock`: non è un’implementazione lock-free. Se l’elaborazione è più lenta della telecamera, i frame intermedi possono essere scartati per evitare una coda crescente. Quando non c’è ancora un frame, il loop attende brevemente senza inserire uno stato vuoto nel buffer fixed-lag. Lo stream non viene flushato a fine acquisizione.
- `--source-type auto` seleziona lo stream per indici numerici di camera e URL; `video` e `stream` forzano esplicitamente la modalità.

### Ottimizzazioni del tracker

- Il filtro di Kalman usa una covarianza `2×2` condivisa per le coordinate BEV: `X` e `Y` vengono aggiornate insieme su vettori, con divisione scalare per `s=P00+r`, senza inversione di matrici `4×4`.
- `FastSavGol` precalcola i pesi di posizione e derivata, evitando di ricostruire il filtro a ogni frame.
- Lo slot singolo dello stream limita la coda di acquisizione; il buffer fixed-lag mantiene una finestra temporale finita.

### Parametri modificabili e soglie interne

| Parametro | Valore attivo | Effetto |
|---|---:|---|
| `--smooth N` | predefinito `11` nel loop V2 | dimensione `W`; deve essere almeno `3`, viene resa dispari; `L=W//2` |
| `--kp-conf` | `0.25` | confidenza minima del detector; la soglia high-confidence del tracker è `max(kp_conf, 0.35)` |
| `--kp-vis` | `0.5` | visibilità minima dei keypoint per accettare la footprint veicolo |
| `--class-conf` | vuoto | soglie di confidenza per classe COCO |
| `--max-age` | `30` | frame massimi di coasting prima di eliminare una traccia |
| `--min-hits` | `2` | osservazioni minime per confermare/emetterne una traccia |
| `--suppress-nested` | `0` | soglia per sopprimere box annidati; `0` disabilita la funzione |
| `--source-type` | `auto` | scelta tra acquisizione automatica, file sequenziale e stream |
| `--refine-fps` | metadata della sorgente | override del frame rate usato per `dt`; `0` usa il frame rate letto dalla sorgente |

Altre soglie sono impostate nel codice: peso OCM `0.25`, costo massimo Hungarian `1.25`, gate di base `0.45` pedoni / `1.25` veicoli, velocità massime `1.8` / `9.0` unità BEV/s, espansione del gate fino a otto frame, rapporto massimo di scala `1.8`, lock forma veicolo dopo venti osservazioni e margine bordo `0.006` della diagonale immagine.

### Differenze e limiti da tenere presenti

- Il voto `class_votes` attuale vota `obs.cls`. Poiché le classi dei veicoli vengono unificate a `2`, il voto stabilizza pedone/veicolo ma **non** distingue `car` da `truck`. `source_cls` è preservata per il gate di famiglia, ma non viene ancora stabilizzata con un istogramma di maggioranza per singola classe COCO.
- La chiusura al bordo non è immediata e non verifica la velocità verso l’esterno: `_check_border_exit` elimina una traccia non aggiornata solo quando il box è vicino al bordo da almeno tre frame e la traccia ha almeno due osservazioni. È una regola di culling al bordo, non una replica completa di `_exit_state`.
- `--min-track`, `--coast-tail`, `--smooth-poly`, `--no-hold-still` e `--no-vel-heading` sono ancora presenti in parte dei parser storici, ma il nuovo `main()` non li passa al `FixedLagBuffer`. La finestra Savitzky–Golay usa `poly=2`; le logiche hold-still e heading sono attive con i valori predefiniti del buffer.
- L’associazione BEV usa famiglia e scala della footprint; non include un termine di IoU dei box 2D.

## Verifica puntuale della descrizione allegata

Questa verifica statica confronta le funzionalità descritte con il percorso effettivamente eseguito da `bev_realtime_v2.py`, `bev_realtime.py` e `uod/bev_ocsort.py`. Le etichette **Implementata**, **Parziale** e **Non implementata** distinguono il comportamento presente nel codice dalle parti proposte nella descrizione ma non ancora collegate.

### 1. Filtro pre-tracking `kp_ok` e trattamento dei pedoni

**Filtro delle detection.** `V2Pipeline._detect_pose()` accetta le classi COCO strada `0, 1, 2, 3, 5, 7` (persona, bicicletta, auto, moto, bus, camion) e applica `--class-conf` alla classe COCO originale. Il detector riceve inoltre `--kp-conf` come soglia globale. Una detection ottiene `kp_ok=True` quando sono disponibili almeno otto keypoint, tutti i valori sono finiti e non nulli, la visibilità minima supera `--kp-vis` e i keypoint restano entro il box con un margine pari a `0.75` volte larghezza e altezza. Per le detection affidabili viene costruita `ground` usando gli indici del resolver, oppure `[0, 1, 2, 3]` come fallback.

Prima del tracker, `_deduplicate_unified()` conserva `source_cls` e unifica `d.cls` delle classi veicolo a `2` per la pipeline e la visualizzazione. Per le detection con footprint valida, la deduplicazione le raggruppa anche per classe COCO originale, così bici/moto e auto non vengono soppresse solo perché le loro footprint si sovrappongono; le detection senza footprint vengono invece lasciate passare come box-only. La classe originale resta disponibile al tracker.

**`_suppress_nested`: implementata, ma con differenze rispetto alla sequenza proposta.** La funzione viene eseguita in `_detect_pose()` dopo la deduplicazione e prima del filtro `kp_ok` applicato in `V2Pipeline.step()`. La CLI la lascia disattivata per impostazione predefinita (`--suppress-nested 0`; `0.85` è il valore consigliato nell’help). Quando è attiva, verifica l’annidamento dei box 2D e l’overlap dei quadrilateri a terra usando i keypoint in coordinate immagine; il rapporto di overlap deve essere almeno `0.3`. Richiede keypoint con forma sufficiente per entrambe le detection, ma non richiede `kp_ok=True`. Inoltre, poiché le classi sono già state unificate, tratta come veicoli tutte le classi unificate a `2`, incluse bici e moto. Per aderire alla descrizione, il filtro dovrebbe operare solo su footprint affidabili, oppure non dovrebbe permettere a una detection con `kp_ok=False` di sopprimere una detection valida.

**Veicoli: filtro `kp_ok` prima del tracker implementato, ma avviene dopo `_suppress_nested`.** Nel ciclo `V2Pipeline.step()`, un veicolo viene proiettato e passato a `BEVOCSortTracker` soltanto se `d.kp_ok` è vero e sono presenti almeno otto keypoint. Se perde l’affidabilità dei keypoint, quel frame non produce un’osservazione BEV: una traccia già esistente può restare in coasting e venire riagganciata in seguito. Il secondo stadio a bassa confidenza riguarda invece la confidenza della detection, non recupera un veicolo scartato perché `kp_ok=False`.

**Pedoni: indipendenza da `kp_ok` implementata nel passaggio BEV.** Ogni detection `cls == 0` viene accettata anche se i keypoint della rete sono inaffidabili. La pipeline proietta con `static_H` il bottom-centre `(cx, y2)` del box 2D, normalizza per `bev_unit`, stima `obs_scale` proiettando i due estremi della larghezza del box e crea una footprint quadrata. La `BEVObservation` del pedone viene marcata `kp_ok=True`, quindi la qualità dei keypoint del cuboide non lo esclude dal tracker.

La stabilizzazione 2D del pedone non è però completata prima della proiezione: il primo punto a terra usa il box grezzo del detector; dopo una corrispondenza, `BEVKalmanTrack._update_pedestrian()` applica EMA a larghezza e altezza (`0.25` osservazione, `0.75` box precedente), ancorando il box al bordo inferiore, e sintetizza keypoint/cuboide regolari. Il centro BEV usa un’EMA con coefficiente `0.4` durante l’apprendimento e `0.15` dopo il lock. Alla creazione della traccia `__init__` copia ancora i keypoint originali: per il primo stato, se il detector non li ha forniti, la geometria sintetica non è ancora stata costruita. Questo è un comportamento parziale rispetto alla proposta di creare subito il box stabilizzato e il punto di contatto prima del tracker.

**Riordino angolare dei quattro vertici veicolo: non implementato in questo percorso.** Il codice seleziona `raw_ground = d.kpts[gi]` e lo proietta direttamente; il riordino anti-farfalla indicato nella descrizione è stato rimosso e non è sostituito qui da un ordinamento angolare. Forma e angolo vengono poi stimati da Procrustes. L’orientamento iniziale resta quindi dipendente dall’ordine dei keypoint forniti dal resolver/modello, un punto da correggere se si vuole eliminare la causa di footprint ruotate o instabili.

### 2. Pipeline real-time, stato e ciclo di vita delle tracce

Il `main()` attuale esegue una pipeline sincrona: legge un frame, chiama `V2Pipeline.step()` (rilevamento, pre-filtro, proiezione BEV, aggiornamento OC-SORT), inserisce lo snapshot nel buffer e renderizza il frame quando il buffer lo emette. Il percorso `main()` non invoca `_stabilize_offline_v2`; le funzioni di raffinamento offline, l’import di `refine_v2` e relativi flag storici restano comunque nel repository/script. In particolare `--refine` viene ancora parsato, ma questo nuovo `main()` non lo applica al risultato.

**Proiezione immediata prima del tracker: implementata.** Le osservazioni valide sono trasformate con l’omografia statica e normalizzate in coordinate BEV prima di `BEVOCSortTracker.update()`. Per i pedoni il centro è il bottom-centre proiettato; per i veicoli è la media dei quattro vertici a terra proiettati. Il tracker opera quindi sulle coordinate BEV, non sui box 2D dell’immagine.

**Stato di Kalman descritto come sette dimensioni: non implementato.** Lo stato filtrato è composto da posizione e velocità nel piano, `pos=(X,Y)` e `vel=(Vx,Vy)`, equivalente a `[X,Y,Vx,Vy]`. La covarianza condivisa è `2×2` per asse posizione/velocità. Scala, classe, footprint rigida e heading sono mantenuti in campi separati; non esistono nello stato di Kalman `S`, aspect ratio `R` o velocità della scala `V_S`.

**OC-SORT e riaggancio: implementati con tre passaggi di associazione.** `BEVOCSortTracker` predice le tracce, associa prima le osservazioni ad alta confidenza con costo Hungarian e penalità OCM, poi tenta il matching delle osservazioni a bassa confidenza per tracce con al massimo quattro frame senza aggiornamento, infine esegue il recupero sulle osservazioni ad alta confidenza usando l’ultima posizione osservata. Nuove tracce nascono solo da osservazioni high-confidence non associate. Al riaggancio dopo un gap, `BEVKalmanTrack.update()` applica ORU riproducendo aggiornamenti con osservazioni virtuali lineari tra l’ultima osservazione e quella corrente.

**Nomi di stato e integrazione TTC: non implementati.** Il codice non definisce stati enumerati `TENTATIVE`, `CONFIRMED`, `COASTING`, `DELETED` e non collega le tracce a un modulo TTC o di verifica infrazioni. Gestisce la stessa parte essenziale tramite contatori: `hits`, `age` e `time_since_update`. `min_hits` è usato per la soppressione nel buffer e per eliminare una traccia ancora poco osservata se resta senza aggiornamento; `time_since_update` rappresenta il coasting; il tracker elimina una traccia quando supera `max_age`. Posizione e velocità smussate del buffer sono applicate agli snapshot destinati al rendering, non retroalimentate allo stato di Kalman del frame successivo.

**Uscita dal bordo: parziale rispetto a `_exit_state`.** `_check_border_exit()` elimina una traccia non aggiornata che tocca il margine dell’immagine da almeno tre frame, se ha almeno due osservazioni. Non verifica il verso della velocità e non chiude immediatamente l’oggetto quando si muove verso l’esterno.

**Apprendimento della forma: implementato.** Le auto/bus/camion usano una forma rigida rettangolare aggiornata incrementalmente con Procrustes e bloccata dopo 20 osservazioni (`lock_after_veh`); il lock congela la forma/dimensione, mentre l’heading continua a essere stimato dalle osservazioni veicolo. I pedoni usano una footprint quadrata adattiva: fase di burn-in fino a 12 osservazioni (`lock_after_ped`), poi EMA verso un prior fisico `0.19` in unità BEV. La box 2D del pedone è aggiornata con pesi `0.25` osservazione e `0.75` stato precedente. Le bici e le moto condividono la famiglia di associazione a due ruote; auto, bus e camion condividono quella dei veicoli a motore.

Per prevenire lo scambio auto–motorino, il tracker conserva `source_cls` prima dell’unificazione: non associa famiglie diverse (`1,3` contro `2,5,7`). Per i veicoli rifiuta inoltre un’associazione se il rapporto fra le scale delle footprint BEV supera `1.8`; per le coppie ammesse aggiunge una penalità di scala e aggiorna `veh_scale` con EMA. Il voto `class_votes` resta invece sulle classi unificate (`persona`/`veicolo`), non è un voto di maggioranza tra `car`, `motorcycle`, `truck` ecc. La famiglia del veicolo è quella della classe originale della traccia al momento della sua creazione.

### 3. Buffer circolare fixed-lag e funzionalità temporali

**Opzione B implementata; opzione A non implementata.** `FixedLagBuffer` è costruito con finestra predefinita `W=11` e ritardo `L=W//2=5`. A 30 FPS il ritardo è circa 167 ms; la finestra di 11 campioni copre dieci intervalli fra frame, circa 333 ms. Una volta pieno, il buffer emette il campione centrale, corrispondente a `t-5`. Non esiste un’opzione alternativa a latenza zero che emetta subito il bordo destro.

- **Savitzky–Golay centrato e velocità analitica: implementati.** `FastSavGol` precalcola le pseudo-inverse di Vandermonde e valuta posizione BEV e prima derivata del polinomio all’indice richiesto. Con una finestra completa, l’emissione centrale usa i campioni precedenti e successivi per lo smoothing zero-phase; la velocità restituita è in unità BEV al secondo secondo `dt=1/fps`. Il filtro è configurato con grado `2` nel nuovo `main()`. Il flag storico `--smooth-poly` viene parsato dal parser base ma non è passato alla costruzione del `FixedLagBuffer`, che imposta `poly=2`.
- **Interpolazione dei gap: implementata con condizioni meno restrittive della descrizione.** Per il frame da emettere, `_interpolate_window_gaps()` interviene se il track-view è in coasting e trova un’osservazione reale precedente e una successiva nella finestra. Interpola linearmente box 2D, posizione BEV, keypoint (se entrambi hanno forma `8×2`) e ground quad (se entrambi hanno forma `4×2`), poi azzera `time_since_update` nello snapshot. Non c’è un controllo esplicito `gap <= 5`: il limite effettivo deriva dal fatto che gli estremi devono stare nella finestra corrente, quindi un intervallo fra osservazioni può arrivare fino a dieci frame. Non interpola direttamente `bev_quad`; dopo lo smoothing la footprint viene ricostruita dalla forma conservata nel tracker. Questa interpolazione di rendering è distinta dall’ORU, che corregge lo stato di Kalman quando la traccia viene riassociata.
- **Flicker e coda coasting: implementati, con ambito limitato al frame emesso.** Il buffer scarta la traccia se il massimo `hits` osservato nella finestra è minore di `min_hits`. Inoltre non emette uno snapshot in coasting se non trova un’osservazione reale dal frame target in avanti nella parte futura della finestra. Non applica i parametri offline `--min-track` e `--coast-tail`, che non sono passati a `FixedLagBuffer`.
- **Veicoli fermi e heading da traiettoria: implementati con soglie interne.** Il buffer considera fermo un veicolo con velocità sotto `0.35` unità BEV/s e spostamento sotto `0.18`; mantiene un anchor se la posizione filtrata resta entro `0.25`. L’heading da traiettoria è valutato con velocità almeno `0.8`, spostamento almeno `0.35` e rapporto spostamento/percorso almeno `0.92`; l’override è limitato a una correzione massima di 45°. I flag `--no-hold-still` e `--no-vel-heading` non sono trasferiti al `FixedLagBuffer` dal nuovo `main()`, perciò queste logiche restano attive ai valori predefiniti.

### 4. Sorgenti video e svuotamento del buffer

- **File video: implementato.** `FrameSource.read()` usa `VideoCapture.read()` sequenzialmente e non scarta frame deliberatamente. A fine file il `main()` invoca `FixedLagBuffer.flush_remaining()` e renderizza gli stati rimasti nel buffer, salvo che sia stato raggiunto il limite `--max-frames`.
- **Stream/camera: implementato come slot latest-frame protetto da lock.** Un thread daemon legge da OpenCV dopo aver impostato `CAP_PROP_BUFFERSIZE=1`; conserva un solo frame e lo sovrascrive con il più recente. `read()` preleva e svuota lo slot sotto `threading.Lock`. Non è lock-free: se inferenza/rendering sono più lenti della camera, i frame intermedi possono essere saltati e non si accumula una coda di frame vecchi. Quando lo slot è vuoto, il loop aspetta 1 ms senza inserire stati vuoti nel buffer fixed-lag. Lo stream non esegue `flush_remaining()` alla fine. Il ritardo del buffer è di cinque frame processati; dato che lo slot può scartare acquisizioni intermedie, non equivale a una latenza temporale rigida rispetto ai frame originali della camera.
- **Selezione sorgente: implementata.** `--source-type auto` tratta indici numerici e URL (`rtsp`, `rtmp`, `http`, `https`) come stream; `video` e `stream` consentono di forzare il modo. La frequenza usata per `dt` proviene da `--refine-fps` se diverso da zero, altrimenti dai metadati della sorgente, con fallback a 30 FPS.

### 5. Parametri effettivi e differenze dalla descrizione

Nel percorso `main()` corrente i valori CLI predefiniti sono: `--kp-conf 0.25`; `--kp-vis 0.5`; `--class-conf` vuoto (nessuna soglia aggiuntiva per classe); `--suppress-nested 0` (disattivato); `--max-age 30`; `--min-hits 2`; `--smooth 0` tradotto dal wrapper V2 nella finestra `W=11`; `--source-type auto`; `--refine-fps 0` (usa gli FPS della sorgente). L’alta confidenza del tracker è `max(kp_conf, 0.35)`, quindi nel caso predefinito vale `0.35`. Una finestra `--smooth` pari o superiore a 3 viene resa dispari da `FixedLagBuffer`; i valori inferiori a 3 ricadono su 11 nel wrapper.

Le principali soglie non esposte come opzioni CLI sono: peso OCM `0.25`; costo massimo di assegnazione `1.25`; gate di base `0.45` per pedoni e `1.25` per veicoli, ampliati in coasting fino a otto frame secondo la velocità massima configurata (`1.8` e `9.0` unità BEV/s); rapporto massimo fra scale veicolo `1.8` con penalità `0.15`; margine bordo `0.006` della diagonale immagine e uscita dopo tre frame sul bordo. Le soglie del filtro fermo/heading sono riportate nella sezione buffer.

In sintesi, il codice implementa già la separazione persona/veicolo rispetto a `kp_ok`, la proiezione BEV prima del tracker, OC-SORT con ORU/OCM, il gate di famiglia e scala per ridurre lo scambio auto–motorino e il fixed-lag smoothing. La descrizione allegata non corrisponde ancora al codice per il gate `kp_ok` prima di `_suppress_nested`, la creazione preventiva della geometria sintetica del pedone, il riordino angolare dei vertici, lo stato Kalman a sette variabili, gli stati nominali/TTC, la chiusura immediata direzionale ai bordi e il limite esplicito di cinque frame per l’interpolazione.
