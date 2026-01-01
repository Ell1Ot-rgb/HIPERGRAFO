Core ONNX bundle — v1.1-Omega-Calibrated

Contenido:
- ONNX promovido: `best_omega21.onnx` (sha256 incluida en `manifest.json`)
- Variantes experimentales incluidas para reproducibilidad: headtuned, readout, hybrid, aucsurrogate, last2, lastblock, etc.

Propósito:
Este directorio es un bundle inmutable pensado como punto de partida para proyectos que necesiten el modelo promovido y las variantes auditadas. Mantiene un `manifest.json` con `sha256` y `size` para trazabilidad.

Uso rápido:
- Citar la rama/tag `core/v1.1-Omega-Calibrated` en tu proyecto para referenciar el conjunto canónico.
- Verifica `manifest.json` antes de usar los binarios para asegurar integridad.

Fecha: 2026-01-01