#!/usr/bin/env python3
import json, time, random
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
import onnxruntime as ort

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
client = TestClient(__import__('src.local_server.servidor_art_v7_hipergrafo', fromlist=['app']).app)

OUT_JSON = Path('analysis_5capas_detailed.json')
OUT_CSV = Path('analysis_5capas_summary.csv')
PLOTS_DIR = Path('analysis/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

signal_levels = [0.0, 0.5, 1.0, 2.0, 5.0]
sample_sizes = [20, 50, 100]
EPOCHS_LAYERS = 3  # number of epochs to train Capa3-5 per experiment (reduced for speed)
NORM_MODES = ['global', 'per_sample']  # comparison A/B: global vs per-sample normalization

sess_art = ort.InferenceSession('models/art_17.onnx')
art_inp = sess_art.get_inputs()[0].name
art_out = sess_art.get_outputs()[0].name

random.seed(0); np.random.seed(0)

# Helper functions

results = []
if OUT_JSON.exists():
    try:
        with open(OUT_JSON) as f:
            results = json.load(f)
    except Exception:
        results = []

_done = set((r.get('n'), r.get('signal'), r.get('norm_mode')) for r in results if 'n' in r and 'signal' in r and 'norm_mode' in r)

def make_dataset(n_per_class, signal):
    normals = [np.random.normal(0,0.05,1600).astype(np.float32) for _ in range(n_per_class)]
    anoms = []
    for _ in range(n_per_class):
        arr = np.random.normal(0,0.05,1600)
        start = random.randint(0,1550)
        arr[start:start+50] += signal
        arr += signal*0.1
        anoms.append(arr.astype(np.float32))
    # stratified split later
    return normals, anoms


def _trapz(y, x):
    y = np.asarray(y); x = np.asarray(x)
    if y.size < 2:
        return 0.0
    return float(((y[:-1] + y[1:]) * (x[1:] - x[:-1]) / 2.0).sum())

def roc_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.sum()==0 or labels.sum()==len(labels):
        return None
    desc = np.argsort(-scores)
    labels_sorted = labels[desc]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1-labels_sorted)
    tpr = tp / labels.sum()
    fpr = fp / (len(labels)-labels.sum())
    auc = _trapz(tpr, fpr)
    return float(auc)

for n in sample_sizes:
    for signal in signal_levels:
        for norm_mode in NORM_MODES:
            if (n, signal, norm_mode) in _done:
                print(f'Skipping already-done: n={n}, signal={signal}, norm_mode={norm_mode}')
                continue
            print(f'Experiment: n_per_class={n}, signal={signal}, norm_mode={norm_mode}')
            normals, anoms = make_dataset(n, signal)
            # build stratified train/test
            train_n = max(2, int(0.6*n))
            test_n = n - train_n
            train = [{'input_data':normals[i].tolist(),'anomaly_label':0} for i in range(train_n)] + [{'input_data':anoms[i].tolist(),'anomaly_label':1} for i in range(train_n)]
            test = [{'input_data':normals[i].tolist(),'anomaly_label':0} for i in range(train_n, n)] + [{'input_data':anoms[i].tolist(),'anomaly_label':1} for i in range(train_n, n)]
            # train reactor once
            samples_for_reactor = train + test
            random.shuffle(samples_for_reactor)
            r = client.post('/train_reactor', json={'samples': samples_for_reactor, 'epochs': 1})
            hip = r.json().get('hipergrafo',{}) if r.status_code==200 else {'error':'train_reactor_failed'}
            # queue layers training (include norm_mode)
            # pass Seq-VCR and spectral params (defaults tuned as suggested)
            q = f'/train_layers_3_5?epochs={EPOCHS_LAYERS}&norm_mode={norm_mode}&vcr_var_weight=25.0&vcr_cov_weight=1.0&spectral_alpha=0.001&kb_weight=0.1&seq_len=4&edge_threshold=0.5'
            r = client.post(q, json={'samples': train})
            if r.status_code!=200:
                results.append({'n':n,'signal':signal,'norm_mode':norm_mode,'error':'queue_failed','status_code':r.status_code})
                with open(OUT_JSON,'w') as f: json.dump(results, f, indent=2)
                continue
            jid = r.json()['job_id']
            deadline = time.time()+120
            status = None
            job=None
            while time.time()<deadline:
                s = client.get(f'/train_layers_3_5/status/{jid}').json()
                job = s
                status = s.get('status')
                if status in ('done','error'):
                    break
                time.sleep(1)
            if status!='done':
                results.append({'n':n,'signal':signal,'norm_mode':norm_mode,'error':'job_failed','status':job})
                with open(OUT_JSON,'w') as f: json.dump(results, f, indent=2)
                continue
            path = job.get('path')
            hip_job = job.get('hipergrafo', hip)
            # evaluate on test
            import onnxruntime as ort
            s2 = ort.InferenceSession(path)
            scores=[]; labels=[]
            for s_ in test:
                arr = np.array(s_['input_data'], dtype=np.float32)
                toks=[]
                for i in range(0,1600,50):
                    chunk = arr[i:i+50]
                    toks.append(int((chunk.mean()+1)*1024)%2048)
                toks_np = np.array(toks, dtype=np.int64).reshape(1,32)
                logits = sess_art.run([art_out], {art_inp: toks_np})[0]
                feat = logits.mean(axis=1).astype(np.float32)
                res = s2.run(None, {'features': feat})
                score = float(np.array(res[0]).reshape(-1)[0])
                scores.append(score)
                labels.append(s_['anomaly_label'])
            scores=np.array(scores); labels=np.array(labels)
            mean_norm = float(scores[labels==0].mean()) if (labels==0).sum()>0 else None
            mean_anom = float(scores[labels==1].mean()) if (labels==1).sum()>0 else None
            acc = float(((scores>0.5)==labels).mean())
            try:
                auc = roc_auc(labels, scores)
            except Exception:
                # fallback simple None if computation fails
                auc = None
            # feature stats on test
            feats_all=[]
            for s_ in test:
                arr = np.array(s_['input_data'], dtype=np.float32)
                toks=[]
                for i in range(0,1600,50):
                    chunk = arr[i:i+50]
                    toks.append(int((chunk.mean()+1)*1024)%2048)
                toks_np = np.array(toks, dtype=np.int64).reshape(1,32)
                logits = sess_art.run([art_out], {art_inp: toks_np})[0]
                feat = logits.mean(axis=1).astype(np.float32).reshape(-1)
                feats_all.append(feat)
            feats_all = np.vstack(feats_all) if feats_all else np.zeros((0,2048))
            lbls = np.array([s_['anomaly_label'] for s_ in test]) if test else np.array([])
            if feats_all.size:
                mdiff = np.abs(feats_all[lbls==0].mean(axis=0) - feats_all[lbls==1].mean(axis=0)) if lbls.sum()>0 and (lbls==0).sum()>0 else np.zeros(feats_all.shape[1])
                frac_sig = float((mdiff>0.1).sum()/mdiff.size)
            else:
                frac_sig = 0.0
            rec = {'n':n,'signal':signal,'norm_mode':norm_mode,'hipergrafo':hip_job,'mean_norm':mean_norm,'mean_anom':mean_anom,'accuracy':acc,'auc':auc,'frac_sig_dims_test':frac_sig,'job_path':path}
            results.append(rec)
            with open(OUT_JSON,'w') as f: json.dump(results, f, indent=2)
            print('Saved interim result for', n, signal, norm_mode)

# produce CSV summary
import csv
with open(OUT_CSV,'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['n','signal','norm_mode','mean_norm','mean_anom','accuracy','auc','frac_sig_dims_test','job_path','hipergrafo'])
    for r in results:
        writer.writerow([r.get('n'), r.get('signal'), r.get('norm_mode'), r.get('mean_norm'), r.get('mean_anom'), r.get('accuracy'), r.get('auc'), r.get('frac_sig_dims_test'), r.get('job_path'), json.dumps(r.get('hipergrafo', {}))])

print('Done experiments; saved JSON and CSV')

# Generate plots (matplotlib only)
try:
    import matplotlib.pyplot as plt
    import csv
    # read CSV
    rows = []
    with open(OUT_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'n': int(r['n']),
                'signal': float(r['signal']),
                'norm_mode': r.get('norm_mode', 'unknown'),
                'accuracy': float(r['accuracy']) if r['accuracy']!='' else None,
                'auc': float(r['auc']) if r['auc'] not in ('','None','nan') else None,
                'frac_sig': float(r['frac_sig_dims_test']) if r['frac_sig_dims_test']!='' else None
            })
    if not rows:
        raise RuntimeError('No CSV rows to plot')
    ns = sorted(set(r['n'] for r in rows))
    modes = sorted(set(r['norm_mode'] for r in rows))
    for n in ns:
        for metric, y_label, fname_suffix in [('accuracy','accuracy','acc'), ('auc','AUC','auc'), ('frac_sig','frac_sig_dims','frac')]:
            plt.figure()
            for mode in modes:
                sub = [r for r in rows if r['n']==n and r['norm_mode']==mode]
                if not sub:
                    continue
                sub = sorted(sub, key=lambda x: x['signal'])
                x = [r['signal'] for r in sub]
                y = [r[metric] if r[metric] is not None else 0 for r in sub]
                plt.plot(x, y, marker='o', label=mode)
            plt.title(f'{y_label} vs Signal (n={n})')
            plt.xlabel('signal')
            plt.ylabel(y_label)
            plt.grid(True)
            plt.legend()
            plt.savefig(PLOTS_DIR/f'{fname_suffix}_vs_signal_n{n}.png')
            plt.close()
    print('Plots saved in', PLOTS_DIR)
except Exception as e:
    print('Plot generation failed:', e)

# Create brief markdown summary
md = Path('docs/analysis_5capas.md')
md.parent.mkdir(parents=True, exist_ok=True)
with open(md,'w') as f:
    f.write('# Análisis 5‑capas (grid experiments)\n\n')
    f.write('Resumen rápido de las ejecuciones — ver `analysis_5capas_summary.csv` y `analysis_5capas_detailed.json`.\n\n')
    f.write('## Plots\n')
    for n in sorted(set([r['n'] for r in results])):
        f.write(f'- ![acc](../analysis/plots/acc_vs_signal_n{n}.png)  ![auc](../analysis/plots/auc_vs_signal_n{n}.png)  ![frac](../analysis/plots/frac_sig_vs_signal_n{n}.png)\n')

print('Report written to', md)
