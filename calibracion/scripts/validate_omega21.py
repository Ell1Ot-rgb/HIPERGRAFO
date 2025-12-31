#!/usr/bin/env python3
"""Validate an ONNX model for promotion using vital-sign checks.

Checks performed:
- Compute feature stats (mean_std, high_corr_frac) using synthetic dataset
- Compute AUC on a small holdout (generated) to approximate detection ability
- Compute frac_sig (fraction of dims with significant mean difference between classes)

Exit 0 if all thresholds pass, non-zero otherwise. Writes JSON summary to stdout/file if requested.
"""
from pathlib import Path
import numpy as np
import onnxruntime as ort
import json
import argparse
from scripts.diagnose_art_features import to_tokens_single
from sklearn.metrics import roc_auc_score

DEFAULT_THRESH = {'auc':0.85,'high_corr_frac':0.10,'frac_sig':0.5}


def make_dataset(n_per_class=100, signal=1.0):
    t = np.linspace(0,1,1600)
    normals=[]; anoms=[]
    rng = np.random.RandomState(0)
    for _ in range(n_per_class):
        n_components = rng.randint(3,6)
        sig = np.zeros_like(t)
        for _c in range(n_components):
            freq = rng.uniform(1,10)
            phase = rng.uniform(0,2*np.pi)
            amp = rng.uniform(0.05,0.5)
            sig += amp*np.sin(2*np.pi*freq*t + phase)
        sig += rng.normal(0,0.02,size=t.shape)
        normals.append(sig.astype(np.float32))
    for _ in range(n_per_class):
        n_components = rng.randint(3,6)
        sig = np.zeros_like(t)
        for _c in range(n_components):
            freq = rng.uniform(1,10)
            phase = rng.uniform(0,2*np.pi)
            amp = rng.uniform(0.05,0.5)
            sig += amp*np.sin(2*np.pi*freq*t + phase)
        sig += signal * np.exp(-5*(t-0.5)**2)  # localized anomaly bump
        sig += rng.normal(0,0.02,size=t.shape)
        anoms.append(sig.astype(np.float32))
    samples=[]
    for s in normals:
        samples.append({'input_data': s.tolist(), 'anomaly_label':0})
    for s in anoms:
        samples.append({'input_data': s.tolist(), 'anomaly_label':1})
    return samples


def compute_feat_stats_from_onnx(onnx_path, n_samples=200):
    sess = ort.InferenceSession(str(onnx_path))
    inp_meta = sess.get_inputs()[0]
    inp_name = inp_meta.name
    out = sess.get_outputs()[0].name
    X = make_dataset(n_samples//2, signal=1.0)
    feats=[]
    labels=[]
    for s in X:
        arr = np.array(s['input_data'], dtype=np.float32)
        # choose input format depending on ONNX input type
        input_type = inp_meta.type
        try:
            if 'int64' in input_type or 'tensor(int64)' in str(input_type):
                toks = to_tokens_single(arr).reshape(1,32).astype(np.int64)
                logits = sess.run([out], {inp_name: toks})[0]
            else:
                # feed raw waveform as float [1,1600]
                waveform = arr.reshape(1,1600).astype(np.float32)
                logits = sess.run([out], {inp_name: waveform})[0]
        except Exception as e:
            raise
        # interpret output: prefer features vector (>=2D), else treat scalar as 1D
        if getattr(logits, 'ndim', None) == 3:
            feat = logits.mean(axis=1).reshape(-1)
        elif getattr(logits, 'ndim', None) == 2:
            feat = logits.reshape(-1)
        elif getattr(logits, 'ndim', None) == 1:
            feat = logits.reshape(-1)
        else:
            raise RuntimeError(f'Unexpected ONNX output shape: {getattr(logits, "shape", None)}')
        feats.append(feat.astype(np.float32))
        labels.append(s['anomaly_label'])
    F = np.vstack(feats)
    stds = F.std(axis=0)
    mean_std = float(stds.mean())
    frac_low = float((stds < 0.05).mean())
    D = F.shape[1]
    if D < 2:
        high_corr_frac = 0.0
    else:
        M = min(200,D)
        idx = np.random.choice(D, M, replace=False)
        sub = F[:, idx]
        corr = np.corrcoef(sub.T)
        high_corr_frac = float((np.abs(corr[np.triu_indices(M, k=1)]) > 0.9).mean())
    # auc
    try:
        auc = float(roc_auc_score(np.array(labels), np.mean(F, axis=1)))
    except Exception:
        auc = None
    # frac_sig: fraction of dims with significant mean diff
    lbls = np.array(labels)
    if lbls.sum()>0 and (lbls==0).sum()>0:
        mdiff = np.abs(F[lbls==0].mean(axis=0) - F[lbls==1].mean(axis=0))
        frac_sig = float((mdiff > 0.1).sum()/mdiff.size)
    else:
        frac_sig = 0.0
    return {'mean_std':mean_std, 'frac_low_var':frac_low, 'high_corr_frac':high_corr_frac, 'auc':auc, 'frac_sig':frac_sig}


def validate(onnx_path, thresholds=DEFAULT_THRESH, n_samples=200):
    stats = compute_feat_stats_from_onnx(onnx_path, n_samples=n_samples)
    passed = True
    reasons = []
    if thresholds.get('auc') is not None:
        if stats['auc'] is None or stats['auc'] < thresholds['auc']:
            passed = False
            reasons.append(f"auc {stats['auc']} < {thresholds['auc']}")
    if stats['high_corr_frac'] > thresholds['high_corr_frac']:
        passed = False
        reasons.append(f"high_corr_frac {stats['high_corr_frac']} > {thresholds['high_corr_frac']}")
    if stats['frac_sig'] < thresholds['frac_sig']:
        passed = False
        reasons.append(f"frac_sig {stats['frac_sig']} < {thresholds['frac_sig']}")
    return {'passed': passed, 'stats': stats, 'reasons': reasons}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('onnx', type=str)
    p.add_argument('--auc', type=float, default=DEFAULT_THRESH['auc'])
    p.add_argument('--high-corr', type=float, default=DEFAULT_THRESH['high_corr_frac'])
    p.add_argument('--frac-sig', type=float, default=DEFAULT_THRESH['frac_sig'])
    p.add_argument('--n-samples', type=int, default=200)
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()
    res = validate(args.onnx, thresholds={'auc':args.auc,'high_corr_frac':args.high_corr,'frac_sig':args.frac_sig}, n_samples=args.n_samples)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    if not res['passed']:
        raise SystemExit(2)
