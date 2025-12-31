#!/usr/bin/env python3
"""CI check: verify calibration artefact exists and reproduces calibrated AUC within tolerance.

Usage:
  python3 scripts/ci/check_calibration.py --release-file analysis/release_omega21.json --n-samples 200 --tolerance 0.02

Exits with code 0 on success, non-zero otherwise.
"""
from pathlib import Path
import json
import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument('--release-file', type=str, default='analysis/release_omega21.json')
p.add_argument('--n-samples', type=int, default=200)
p.add_argument('--tolerance', type=float, default=0.02)
args = p.parse_args()

release = Path(args.release_file)
if not release.exists():
    print(f"ERROR: release file not found: {release}")
    sys.exit(2)

meta = json.loads(release.read_text())
onnx = meta.get('onnx')
if not onnx:
    print('ERROR: no "onnx" key in release metadata')
    sys.exit(2)
onnx_path = Path(onnx)
if not onnx_path.exists():
    print(f"ERROR: ONNX file not found: {onnx_path}")
    sys.exit(2)

expected_auc = meta.get('calibrated_auc')
if expected_auc is None:
    print('ERROR: no "calibrated_auc" in release metadata')
    sys.exit(2)

print('Running smoke validation:')
print('  onnx:', onnx_path)
print('  expected calibrated_auc:', expected_auc)
print('  n_samples:', args.n_samples)

# Ensure repo root on path and import validator function
ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from scripts.validate_omega21 import compute_feat_stats_from_onnx
except Exception as e:
    print('ERROR importing validator:', e)
    sys.exit(3)

# If calibration parameters available, perform calibrated evaluation
calib = meta.get('calibration') or {}
scale = calib.get('scale')
offset = calib.get('offset')

if scale is not None and offset is not None:
    print(f'Using calibration scale={scale} offset={offset} for the smoke eval')
    try:
        import onnxruntime as ort
        from scripts.diagnose_art_features import to_tokens_single
        from scripts.validate_omega21 import make_dataset
        from sklearn.metrics import roc_auc_score
        import numpy as np
    except Exception as e:
        print('ERROR importing dependencies for calibrated eval:', e)
        sys.exit(7)

    try:
        sess = ort.InferenceSession(str(onnx_path))
        inp_meta = sess.get_inputs()[0]
        inp_name = inp_meta.name
        out_name = sess.get_outputs()[0].name
        X = make_dataset(args.n_samples//2, signal=1.0)
        feats = []
        labels = []
        for s in X:
            arr = np.array(s['input_data'], dtype=np.float32)
            arr2 = arr * float(scale) + float(offset)
            # choose input format
            if 'int64' in inp_meta.type or 'tensor(int64)' in str(inp_meta.type):
                toks = to_tokens_single(arr2).reshape(1,32).astype(np.int64)
                logits = sess.run([out_name], {inp_name: toks})[0]
            else:
                waveform = arr2.reshape(1,1600).astype(np.float32)
                logits = sess.run([out_name], {inp_name: waveform})[0]
            if getattr(logits, 'ndim', None) == 3:
                feat = logits.mean(axis=1).reshape(-1)
            else:
                feat = logits.reshape(-1)
            feats.append(feat.astype(np.float32))
            labels.append(s['anomaly_label'])
        F = np.vstack(feats)
        try:
            auc = float(roc_auc_score(np.array(labels), np.mean(F, axis=1)))
        except Exception:
            auc = None
    except Exception as e:
        print('ERROR during calibrated eval:', e)
        sys.exit(8)
else:
    # fallback to generic validator
    try:
        stats = compute_feat_stats_from_onnx(str(onnx_path), n_samples=args.n_samples)
    except Exception as e:
        print('ERROR running validator:', e)
        sys.exit(4)
    auc = stats.get('auc')

print('  observed_auc:', auc)

if auc is None:
    print('ERROR: observed AUC is None')
    sys.exit(5)

if auc + args.tolerance >= expected_auc:
    print(f'PASS: observed AUC {auc:.6f} within tolerance of expected {expected_auc:.6f} (tol={args.tolerance})')
    sys.exit(0)
else:
    print(f'FAIL: observed AUC {auc:.6f} below expected {expected_auc:.6f} - tol {args.tolerance}')
    sys.exit(6)
