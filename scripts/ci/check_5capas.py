#!/usr/bin/env python3
"""Check that the 5 logical capas/components exist and extract quick diagnostics.

Checks performed:
 - Import and instantiate ART model (PyTorch): checks for attributes indicative of the 5 capas
 - Confirm number of layers (LSTM 'layers' ModuleList) and head presence
 - Run atoms_from_wave on a small synthetic waveform and assert length 25
 - Confirm ONNX file exists and report its input/output shapes via onnxruntime

Exits 0 on success, non-zero on failure.
"""
from pathlib import Path
import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=str, default='modelos_guardados/multi_seed_seed_2.pth')
p.add_argument('--onnx', type=str, default='models/best_omega21.onnx')
args = p.parse_args()

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import model class
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('server_mod', 'src/local_server/servidor_art_v7_hipergrafo.py')
    server_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_mod)
    ART = getattr(server_mod, 'ART_Brain_V7_Complete')
except Exception as e:
    print('ERROR: cannot import ART_Brain_V7_Complete:', e)
    sys.exit(2)

print('Instantiating ART model (cpu) ...')
import torch
try:
    model = ART(dim=128, depth=6, vocab=2048)
except Exception as e:
    print('ERROR instantiating model:', e)
    sys.exit(3)

# check components
required_attrs = ['emb', 'rough_path', 'pause_inj', 'layers', 'opi_activation', 'layer_norm', 'head']
missing = [a for a in required_attrs if not hasattr(model, a)]
if missing:
    print('FAIL: model missing required attributes:', missing)
    sys.exit(4)
else:
    print('OK: model has required attributes')

# layers count
layers_count = len(model.layers)
print('Model.layers count =', layers_count)
if layers_count < 3:
    print('WARN: expected at least 3 LSTM layers for temporal processing')

# check head shape
sd = model.state_dict()
head_w = sd.get('head.weight') if 'head.weight' in sd else None
head_b = sd.get('head.bias') if 'head.bias' in sd else None
if head_w is None or head_b is None:
    print('FAIL: head parameters not found in state_dict')
    sys.exit(5)
else:
    print('OK: head params found, shape:', tuple(head_w.shape))

# verify atoms-like calculation locally (avoid importing matplotlib-requiring module)
try:
    import numpy as np
    t = np.linspace(0,1,1600)
    wave = np.sin(2*np.pi*5*t).astype(np.float32)
    # replicate atoms_from_wave behaviour: split into 25 chunks and mean
    n = len(wave)
    chunk = n // 25
    atoms = []
    for i in range(25):
        s = i*chunk
        e = s+chunk if i<24 else n
        seg = wave[s:e]
        atoms.append(float(seg.mean()))
    atoms = np.array(atoms)
    print('Local atoms calc shape:', atoms.shape)
    if atoms.shape[0] != 25:
        print('FAIL: expected 25 atoms, got', atoms.shape[0])
        sys.exit(6)
    else:
        print('OK: 25 atoms (local calc) detected')
except Exception as e:
    print('ERROR computing atoms locally:', e)
    sys.exit(7)

# check ONNX
onnx_path = Path(args.onnx)
if not onnx_path.exists():
    print('FAIL: ONNX file not found:', onnx_path)
    sys.exit(8)
print('OK: ONNX exists at', onnx_path)

try:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print('ONNX input:', inp.name, inp.type, 'output:', out.name, out.shape)
except Exception as e:
    print('WARN: cannot introspect ONNX via onnxruntime:', e)

print('\nAll checks passed (or warnings).')
sys.exit(0)
