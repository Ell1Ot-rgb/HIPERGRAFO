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

3. Diagnóstico de "alucinación" (átomos vs logits) usando checkpoint:

```bash
python3 scripts/diagnose_alucinacion.py modelos_guardados/multi_seed_seed_2.pth --out-prefix seed2_diag
# Salidas: heatmaps y report JSON en analysis/diagnosis
```

4. Reentrenamiento quirúrgico (Head / Readout) – ejemplo:

```bash
python3 scripts/tune_head_seed.py --checkpoint modelos_guardados/multi_seed_seed_2.pth --train-readout --unfreeze-head --epochs 3 --steps-per-epoch 200 --batch-size 64 --lr 1e-4 --out calibracion/models/best_omega21_readout_plus_head.onnx
```

5. Verificar las 5 capas (script de auditoría):

```bash
python3 scripts/ci/check_5capas.py --checkpoint modelos_guardados/multi_seed_seed_2.pth --onnx calibracion/models/best_omega21.onnx
```

6. Ejecutar la verificación corta de calibración en CI (localmente):

```bash
python3 scripts/ci/check_calibration.py --release-file analysis/release_omega21.json --n-samples 200 --tolerance 0.06
```

---

## Archivos adicionales
- `calibracion/docs/onnx.md` — detalles por ONNX y su uso en scripts
- `scripts/ci/check_5capas.py` — verificación '5 capas'
- `scripts/ci/check_calibration.py` — smoke check de calibración usado por CI

## Notas
- No se promovió un modelo que superara 0.85 en AUC de forma nativa: la solución operativa es la calibración (lente externa). Se documenta todo el proceso y los artefactos en `analysis/release_omega21.json`.

## Contacto
Para dudas sobre reproducibilidad o entrada de datos reales, abrir issue o PR en este repo.
