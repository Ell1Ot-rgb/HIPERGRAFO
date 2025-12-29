# Art 17: Model Package

This folder contains the artifacts required to reproduce and test the model and layers 0–5 locally.

Quick start:
1. Create a Python virtualenv and install requirements: `pip install -r requirements.txt` (use the project's /docker/ or requirements)
2. Use `scripts/train_capa3_5_with_onnx.py` to run the smoke training for layers 3–5 (it will use ONNX extractor or fallback to PyTorch ART checkpoint).
3. Use `bucle_entrenamiento.sh` to run the continuous training loop that generates 1600D synthetic data.

Files:
- docs/: documentation
- scripts/: runnable scripts
- code/: reference code of layers and generators
- art_17.onnx, art_17_fixed.onnx, art_17_capa3_5.onnx: exported ONNX models
- METADATA.sha256: SHA‑256 checksums for important files (verify with `sha256sum -c METADATA.sha256`)

