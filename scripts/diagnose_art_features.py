#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort
from pathlib import Path
from collections import defaultdict
import random

VOCAB = 2048

# tokenization helper
def to_tokens_single(vec):
    toks = []
    for s in range(0, 1600, 50):
        chunk = vec[s:s+50]
        toks.append(int((chunk.mean() + 1) * 1024) % VOCAB)
    return np.array(toks, dtype=np.int64)

# generate physics-like samples
def make_dataset(n):
    t = np.linspace(0, 1, 1600)
    X = []
    for _ in range(n):
        n_components = np.random.randint(3, 6)
        sig = np.zeros_like(t)
        for _c in range(n_components):
            freq = np.random.uniform(1, 10)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.05, 0.5)
            sig += amp * np.sin(2 * np.pi * freq * t + phase)
        sig += np.random.normal(0, 0.02, size=t.shape)
        X.append(sig.astype(np.float32))
    return X


def compute_stats(feats):
    feats = np.vstack(feats)  # (N, D)
    stds = feats.std(axis=0)
    mean_std = stds.mean()
    frac_low_var = (stds < 0.05).mean()
    # correlation metrics (sample a subset if large)
    D = feats.shape[1]
    # compute correlation matrix on a random subset of dims if D large
    M = min(200, D)
    idx = np.random.choice(D, M, replace=False)
    sub = feats[:, idx]
    corr = np.corrcoef(sub.T)
    high_corr_frac = (np.abs(corr[np.triu_indices(M, k=1)]) > 0.9).mean()
    # svd decay
    try:
        s = np.linalg.svd(feats, compute_uv=False)
        svd_ratio = float(s[0] / s.sum())
    except Exception:
        svd_ratio = None
    return {
        'mean_std': float(mean_std),
        'frac_low_var': float(frac_low_var),
        'high_corr_frac': float(high_corr_frac),
        'svd_top_frac': svd_ratio,
        'D': int(D),
        'N': int(feats.shape[0])
    }


def main():
    random.seed(0); np.random.seed(0)
    sess = ort.InferenceSession(str(Path('models/art_17.onnx')))
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    X = make_dataset(200)
    feats = []
    for vec in X:
        toks = to_tokens_single(vec).reshape(1,32)
        logits = sess.run([out], {inp: toks})[0]  # shape (1,32,2048)
        feat = logits.mean(axis=1).reshape(-1)
        feats.append(feat.astype(np.float32))
    stats = compute_stats(feats)
    print('Diagnosis for models/art_17.onnx:')
    for k,v in stats.items():
        print(f'  {k}: {v}')
    Path('analysis/art_17_diagnosis.json').write_text(str(stats))

if __name__ == '__main__':
    main()
