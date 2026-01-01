# ONNX: listado, propósito y uso

Este documento describe los ONNX generados durante la campaña y indica exactamente qué script los consume y cómo usarlos.

## Archivos ONNX en `calibracion/models`
- `best_omega21.onnx` — Candidate promoted ONNX (la versión final de la campaña). Uso: inference en producción/servicio y punto de partida de la validación/calibración.
- `best_omega21_headtuned.onnx` — Head-only fine-tuned candidate (experimento quirúrgico, no promovido).
- `best_omega21_readout.onnx` — Readout trained candidate (pequeño linear readout sobre features medias).
- `best_omega21_aucsurrogate*.onnx` — Variantes entrenadas con AUC surrogate (experimentos, no promovidos).
- `best_omega21_hybrid.onnx` — Readout + mix-loss + augment (experimento híbrido).
- `best_omega21_last2.onnx`, `best_omega21_lastblock.onnx` — versiones con entrenamientos conservadores en las últimas capas.

> Nota: los experimentos están guardados para reproducibilidad y auditoría. El artefacto promovido es `best_omega21.onnx`.

## Qué scripts usan cada ONNX y para qué
- `calibrate_and_retest.py` — toma un ONNX (idealmente `best_omega21.onnx`) y ejecuta barridas de `scale`/`offset` aplicadas a la señal de entrada para encontrar calibraciones que maximizan AUC. Produce: JSON de grid, heatmap PNG, `_meta.json`.

- `validate_omega21.py` — valida un ONNX corriendo una generación sintética de normales vs anómalos, calcula stats: `mean_std`, `high_corr_frac`, `auc`, `frac_sig`. Uso: gate de promoción (vital-signs).

- `diagnose_alucinacion.py` — usa checkpoints (p. ej. `modelos_guardados/multi_seed_seed_2.pth`) para generar estímulos senoidales y analizar la correlación atom/logit; produce atom heatmaps y report JSON.

- `tune_head_seed.py` — re-entrena solo la cabeza o readout y exporta ONNX (varias variantes generadas: `best_omega21_readout.onnx`, `best_omega21_headtuned.onnx`, etc.).

- `scripts/ci/check_calibration.py` — consume `analysis/release_omega21.json` y el ONNX listado para comprobar que la AUC calibrada es reproducible dentro de tolerancia (usa `validate` o evaluación calibrada con `onnxruntime`).

## Contenido y firmas internas esperadas
- Entradas: algunos ONNX esperan `input_tokens` (int64 tokens, shape [1,32]) y otros esperan `x` (float waveform, shape [1,1600]) — tus scripts detectan automáticamente por introspección de la sesión ONNX y el tipo del primer input.
- Salidas: en producción usamos el output `features` o `feats` (mean-pooled logits) o `score` (en readout exported ONNX). Verifica `sess.get_inputs()[0].type` y `sess.get_outputs()[0].shape` para prevenir mismatch.

## Cómo usar un ONNX para inferencia rápida (ejemplo):

```python
import onnxruntime as ort
import numpy as np
from scripts.diagnose_art_features import to_tokens_single

sess = ort.InferenceSession('calibracion/models/best_omega21.onnx')
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]
print('input type', inp.type, 'name', inp.name)

# waveform input example
wave = np.random.normal(0,0.02, size=(1600,)).astype(np.float32)
if 'int64' in inp.type:
    toks = to_tokens_single(wave).reshape(1,32).astype(np.int64)
    logits = sess.run([out.name], {inp.name: toks})[0]
else:
    waveform = wave.reshape(1,1600).astype(np.float32)
    logits = sess.run([out.name], {inp.name: waveform})[0]

print('output shape', getattr(logits,'shape', None))
```

## Ejemplo de salida de introspección ONNX (esperada)

Al inspeccionar un ONNX con `onnxruntime` se espera ver algo como:

```
ONNX input: x tensor(float) output: features [1, 1600]
```

Y si el ONNX devuelve un `score` (readout exported), el output shape será `[1, 1]`.

## Ejemplo: JSON corto de validación (esperado)

```json
{
  "passed": false,
  "stats": {
    "mean_std": 0.5579,
    "high_corr_frac": 0.0,
    "auc": 0.160775,
    "frac_sig": 0.8056
  },
  "reasons": ["auc 0.160775 < 0.85"]
}
```

(Estos fragmentos ayudan a revisar si el ONNX y la validación están produciendo resultados con la misma estructura que usamos en CI.)
## Recomendaciones de despliegue
- Valida todas las rutas `input_type`/`output_shape` antes de integrar en producción. Usa `calibrate_and_retest.py` para encontrar los parámetros `scale`/`offset` apropiados para la entrada si trabajas con trazas de sensores reales.

## Auditoría y trazabilidad
- Para cada ONNX promovido mantén `release_omega21.json` que incluye `sha256` y la calibración recomendada; los artefactos y scripts reproducibles están en `calibracion/`.

### Bundle core (nueva)
- Hemos creado un bundle *core* con los ONNX principales en `models/core/v1.1-Omega-Calibrated` y un `manifest.json` con hashes (`sha256`) y tamaños. Este bundle es la referencia inmutable para futuros proyectos que quieran basarse en la campaña `v1.1-Omega-Calibrated`.
- La rama y tag git asociadas son **`core/v1.1-Omega-Calibrated`** y el `manifest.json` dentro del bundle indica exactamente qué archivos y firmas contiene.
- Uso: clona esta rama o apunta a `models/core/v1.1-Omega-Calibrated` para obtener el conjunto canónico de ONNX (promovido + variantes experimentales).
