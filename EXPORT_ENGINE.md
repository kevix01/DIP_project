# Export engine (TensorRT / ONNX)

Come creare il file `.engine` ottimizzato da usare poi con `--kp-model` in
`debug_violations.py` (o `bev_realtime_v2.py`), senza passare per il flusso live.

## Script standalone: `export_engine.py`

```bash
# FP16 a 640 px (default) -> checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine
python export_engine.py --weights checkpoints/urbanomnidetect_v2_x_640.pt --imgsz 640

# FP32 explícito
python export_engine.py --weights checkpoints/urbanomnidetect_v2_x_640.pt --imgsz 640 --fp32

# Altra risoluzione
python export_engine.py --weights checkpoints/urbanomnidetect_v2_x_640.pt --imgsz 960 --half

# ONNX invece di TensorRT
python export_engine.py --weights checkpoints/urbanomnidetect_v2_x_640.pt --imgsz 640 --format onnx
```

| Flag | Default | Significato |
|---|---|---|
| `--weights PATH` | (required) | checkpoint `.pt` di partenza |
| `--imgsz N` | 640 | risoluzione per cui l'engine è valido |
| `--format engine\|tensorrt\|onnx` | engine | backend di export |
| `--half` / `--fp32` | half | FP16 (default) o FP32 |
| `--dynamic` | off | shape dinamiche |
| `--device` | cuda:0 | device di export |

Il nome del file prodotto codifica i parametri (`<base>_<imgsz>_fp16.engine`,
`..._fp32.engine`, `..._dyn.engine`, `.onnx`), quindi un engine costruito per
una risoluzione/precisione diversa non viene mai riutilizzato per errore.
Se il file esiste già, l'export viene saltato (cache su disco).

## Uso del file esportato

```bash
python debug_violations.py --input video.mp4 \
    --kp-model checkpoints/urbanomnidetect_v2_x_640_640_fp16.engine \
    --kp-imgsz 640 --device cuda:0 --roi-file roi_bev.json ...
```

Nota: caricare un `.engine` con task pose richiede `task="pose"`; lo fa già
`load_detector(...)` in `uod/model.py`, usato da entrambe le pipeline.

## Export dal flusso live (alternativa equivalente)

Le stesse opzioni restano disponibili nel flujo: `--export engine|tensorrt|onnx --half`
(viene chiamato `uod.model.maybe_export` + `load_detector`, identici allo script
standalone). Utile quando non serve un file persistente separato.

## Requisiti

- TensorRT 10.x (`pip install "tensorrt>=10,<11"`): la 11.x ha rimosso
  `BuilderFlag.FP16` e l'exporter Ultralytics fallisce.
- CUDA funzionante per l'export (`--device cuda:0`).
