# Calibracion — Omega 21

Resumen de la calibración y artefactos generados durante la campaña de diagnóstico y calibración.

## Objetivo
Recopilar el sistema calibrado y la documentación asociada: ONNX promovido, archivos de diagnóstico (atoms, sweep, heatmaps), scripts reproducibles y metadatos de la promoción.

## Contenido
- `models/` — ONNX exportados (incluye `best_omega21.onnx` y variantes experimentales)
- `analysis/` — resultados de calibración y diagnósticos (JSON, PNG, heatmaps)
- `scripts/` — scripts clave para reproducir la calibración y diagnósticos (`calibrate_and_retest.py`, `tune_head_seed.py`, `validate_omega21.py`, etc.)
- `docs/README.md` — este archivo

## Calibración promovida
- scale: -0.38
- offset: 0.42
- calibrated AUC (synthetic): 0.841634
- metadata: `analysis/release_omega21.json`

## Reproducción rápida
1. Reproducir la barrida ultra-fina (ejemplo):

```bash
python3 calibracion/scripts/calibrate_and_retest.py calibracion/models/best_omega21.onnx --scales -0.42 -0.41 -0.405 -0.4 -0.395 -0.39 -0.38 --offsets 0.38 0.39 0.4 0.41 0.42 --n-samples 2000 --out calibracion/analysis/calibration_negscale_ultra_fine.json
```

2. Validar un ONNX:

```bash
python3 calibracion/scripts/validate_omega21.py calibracion/models/best_omega21.onnx --n-samples 2000
```

## Notas
- No se promovió un modelo que superara 0.85 en AUC de forma nativa: la solución operativa es la calibración (lente externa). Se documenta todo el proceso y los artefactos en `analysis/release_omega21.json`.

## Contacto
Para dudas sobre reproducibilidad o entrada de datos reales, abrir issue o PR en este repo.
