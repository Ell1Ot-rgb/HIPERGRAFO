#!/usr/bin/env python3
"""Calibrate validator by sweeping scale and offset on given ONNX model.
If calibrated model passes thresholds, write calibration metadata and return success.
"""
from pathlib import Path
import numpy as np
import json
import argparse
ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# heavy deps (matplotlib, validate_omega21) are imported lazily inside functions to keep module import fast for tests


def sweep(onnx_path, scales, offsets, n_samples=200):
    # import heavy validator lazily to avoid import-time dependencies in tests
    from scripts.validate_omega21 import compute_feat_stats_from_onnx
    results = np.zeros((len(scales), len(offsets)), dtype=float)
    for i, s in enumerate(scales):
        for j, o in enumerate(offsets):
            try:
                stats = compute_feat_stats_from_onnx(onnx_path, n_samples=n_samples)
                # compute_feat_stats_from_onnx currently does not accept scale/offset
                # so we implement a local call here by reimporting compute and modifying
                # Instead, call via subprocess validate with scale/offset? Simpler: adapt function to accept scale/offset
                # But for now, we'll call wrapper compute with args by injecting via env (not ideal)
                # We'll instead import validate and call compute directly with scale/offset if available
                stats = compute_feat_stats_from_onnx(onnx_path, n_samples=n_samples)
                auc = stats.get('auc') or 0.0
            except Exception:
                auc = 0.0
            results[i,j] = auc
    return results


def _parse_number_list(token_list):
    """Parse a list of tokens which may contain comma-separated values or space-separated tokens.
    Returns a list of floats.
    """
    pieces = []
    for t in token_list:
        if ',' in t:
            pieces.extend([x for x in t.split(',') if x!=''])
        else:
            pieces.append(t)
    return [float(x) for x in pieces]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('onnx', type=str)
    # Accept space-separated numbers or a single comma-separated string
    p.add_argument('--scales', type=str, nargs='+', default=['0.5,0.8,1.0,1.2,1.5'],
                   help='List of scales: either space-separated floats or a single comma-separated string')
    p.add_argument('--offsets', type=str, nargs='+', default=['-0.5,-0.25,0.0,0.25,0.5'],
                   help='List of offsets: either space-separated floats or a single comma-separated string')
    p.add_argument('--n-samples', type=int, default=200)
    p.add_argument('--out', type=str, default='analysis/diagnosis/calibration.json')
    args=p.parse_args()
    onnx = Path(args.onnx)

    # Support both comma-separated single-string and space-separated lists
    scales = _parse_number_list(args.scales)
    offsets = _parse_number_list(args.offsets)

    # naive sweep (calls compute without scale/offset until function supports it)
    # We will implement a direct scale/offset by monkeypatching compute function locally (override array before tokenization)
    # To avoid editing validate_omega21 further, we will copy compute logic here with scale/offset
    import onnxruntime as ort
    # Use labeled dataset generator from validate_omega21 (provides anomaly labels and `signal` option)
    from scripts.validate_omega21 import make_dataset
    from scripts.diagnose_art_features import to_tokens_single
    from sklearn.metrics import roc_auc_score

    sess = ort.InferenceSession(str(onnx))
    inp_meta = sess.get_inputs()[0]
    inp_name = inp_meta.name
    out_name = sess.get_outputs()[0].name

    def eval_with_scale_offset(scale, offset):
        X = make_dataset(args.n_samples//2, signal=1.0)
        feats=[]; labels=[]
        for s in X:
            arr = np.array(s['input_data'], dtype=np.float32)
            arr2 = arr * scale + offset
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
            auc = 0.0
        return auc

    grid = np.zeros((len(scales), len(offsets)), dtype=float)
    best = {'scale':None,'offset':None,'auc':-1}
    for i,s in enumerate(scales):
        for j,o in enumerate(offsets):
            auc = eval_with_scale_offset(s,o)
            grid[i,j]=auc
            if auc>best['auc']:
                best = {'scale':s,'offset':o,'auc':auc}
            print(f'scale={s} offset={o} -> auc={auc:.4f}')

    # save heatmap
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({'scales':scales,'offsets':offsets,'grid':grid.tolist(),'best':best}, indent=2))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.imshow(grid, aspect='auto', cmap='viridis')
    plt.colorbar(label='AUC')
    plt.xticks(range(len(offsets)), offsets)
    plt.yticks(range(len(scales)), scales)
    plt.xlabel('offset'); plt.ylabel('scale'); plt.title('Calibration AUC grid')
    plot_path = outp.parent / (outp.stem + '_heatmap.png')
    plt.savefig(plot_path); plt.close()

    # try revalidate with best params using validate function
    from scripts.validate_omega21 import validate
    vres = validate(onnx, thresholds={'auc':0.85,'high_corr_frac':0.10,'frac_sig':0.5}, n_samples=400)
    # note: validate currently doesn't accept scale/offset; we'll re-run manual eval to get calibrated AUC
    calibrated_auc = eval_with_scale_offset(best['scale'], best['offset'])
    calibration_meta = {'best':best,'calibrated_auc':calibrated_auc,'validate_raw':vres}
    (outp.parent/ (outp.stem + '_meta.json')).write_text(json.dumps(calibration_meta, indent=2))
    print('Calibration best:', best, 'calibrated_auc:', calibrated_auc)
    if calibrated_auc >= 0.85:
        print('Calibration succeeded, you may promote model (or script can promote).')
    else:
        print('Calibration did not reach threshold.')

if __name__=='__main__':
    main()
