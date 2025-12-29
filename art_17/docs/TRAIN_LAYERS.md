# Entrenamiento de Capas 3–5 (endpoint `/train_layers_3_5`) 🧠

Este documento explica cómo usar el endpoint **/train_layers_3_5** para entrenar las capas 3–5 (Asociativa + Atención + Ejecutiva) usando el extractor ART ONNX o el fallback PyTorch.

## Endpoints

- POST /train_layers_3_5
  - Query params: `epochs` (int, por defecto 1)
  - JSON body: `{"samples": [{"input_data": [1600 floats], "anomaly_label": 0|1}, ...]}`
  - Response: `{ "status": "queued", "job_id": "job_1" }`

- GET /train_layers_3_5/status/{job_id}
  - Response: job state: `queued | running | done | error` y campos adicionales `path`, `loss` o `error`.

## Comportamiento

- El endpoint ejecuta el entrenamiento en segundo plano (Thread) para no bloquear el reactor principal.
- El extractor de features usado es `models/art_17.onnx` (ONNX). Si onnxruntime falla, el sistema carga el checkpoint PyTorch (`modelos_guardados/last_checkpoint.pth`) y usa el modelo ART en PyTorch como fallback.
- Al finalizar, exporta `models/art_17_capa3_5.onnx` y devuelve la ruta en el job status.

## Ejemplo (curl)

```
curl -X POST "http://localhost:8000/train_layers_3_5?epochs=1" -H 'Content-Type: application/json' -d '{"samples": [{"input_data": [0.0, ..., 0.0], "anomaly_label": 0}]}'

# Check status
curl http://localhost:8000/train_layers_3_5/status/job_1
```

## Notas

- El job exporta un ONNX con opset 18 cuando es posible.
- Para ejecuciones de prueba use `epochs=0` para forzar export rápido sin bucle de entrenamiento.

