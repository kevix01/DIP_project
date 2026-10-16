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
