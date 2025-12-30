#!/usr/bin/env python3
"""Micro-grid to test aggressive Seq-VCR params as suggested.
Runs combinations: sample_sizes x norm_modes for signal=5.0 and collects metrics.
"""
import time, random, json
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
import onnxruntime as ort

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
client = TestClient(__import__('src.local_server.servidor_art_v7_hipergrafo', fromlist=['app']).app)

SAMPLE_SIZES = [20, 50]
NORM_MODES = ['global', 'per_sample']
SIGNAL = 5.0
EPOCHS_LAYERS = 20
VCR_VAR = 100.0
VCR_COV = 10.0
EDGE_THRESH = 0.05
SEQ_LEN = 4
SPECTRAL_ALPHA = 0.001
KB_WEIGHT = 0.1

OUT = Path('analysis/microgrid_seqvcr_results.json')
results = []
# resume-awareness: load previous results and skip completed successful combos
if OUT.exists():
    try:
        with open(OUT) as f:
            results = json.load(f)
    except Exception:
        results = []
_done = set((r.get('n'), r.get('signal'), r.get('norm_mode')) for r in results if 'n' in r and 'signal' in r and 'norm_mode' in r and not r.get('error'))

random.seed(0); np.random.seed(0)

def make_dataset(n_per_class, signal):
    normals = [np.random.normal(0,0.05,1600).astype(np.float32) for _ in range(n_per_class)]
    anoms = []
    for _ in range(n_per_class):
        arr = np.random.normal(0,0.05,1600)
        start = random.randint(0,1550)
        arr[start:start+50] += signal
        arr += signal*0.1
        anoms.append(arr.astype(np.float32))
    return normals, anoms

for n in SAMPLE_SIZES:
    for norm_mode in NORM_MODES:
        signal = SIGNAL
        print(f'Running micro-experiment: n={n}, signal={signal}, norm_mode={norm_mode}')
        normals, anoms = make_dataset(n, signal)
        train_n = max(2, int(0.6*n))
        train = [{'input_data':normals[i].tolist(),'anomaly_label':0} for i in range(train_n)] + [{'input_data':anoms[i].tolist(),'anomaly_label':1} for i in range(train_n)]
        test = [{'input_data':normals[i].tolist(),'anomaly_label':0} for i in range(train_n, n)] + [{'input_data':anoms[i].tolist(),'anomaly_label':1} for i in range(train_n, n)]

        r = client.post('/train_reactor', json={'samples': train + test, 'epochs': 1})
        hip = r.json().get('hipergrafo',{}) if r.status_code==200 else {'error':'train_reactor_failed'}

        q = (f'/train_layers_3_5?epochs={EPOCHS_LAYERS}&norm_mode={norm_mode}'
             f'&vcr_var_weight={VCR_VAR}&vcr_cov_weight={VCR_COV}&spectral_alpha={SPECTRAL_ALPHA}'
             f'&kb_weight={KB_WEIGHT}&seq_len={SEQ_LEN}&edge_threshold={EDGE_THRESH}')
        r = client.post(q, json={'samples': train})
        if r.status_code!=200:
            results.append({'n':n,'signal':signal,'norm_mode':norm_mode,'error':'queue_failed','status_code':r.status_code})
            OUT.write_text(json.dumps(results, indent=2))
            continue
        jid = r.json()['job_id']
        deadline = time.time() + 1800
        status=None; job=None
        while time.time()<deadline:
            s = client.get(f'/train_layers_3_5/status/{jid}').json()
            job = s; status = s.get('status')
            print('status', n, norm_mode, status)
            if status in ('done','error'):
                break
            time.sleep(2)
        if status!='done':
            results.append({'n':n,'signal':signal,'norm_mode':norm_mode,'error':'job_failed','status':job})
            OUT.write_text(json.dumps(results, indent=2))
            continue
        path = job.get('path')
        hip_job = job.get('hipergrafo')
        # evaluate
        art = ort.InferenceSession('models/art_17_finetuned.onnx') if Path('models/art_17_finetuned.onnx').exists() else ort.InferenceSession('models/art_17.onnx')
        art_in = art.get_inputs()[0].name
        art_out = art.get_outputs()[0].name
        s2 = ort.InferenceSession(path)
        scores=[]; labels=[]
        feats_all=[]
        for s_ in test:
            arr = np.array(s_['input_data'], dtype=np.float32)
            toks=[]
            for i in range(0,1600,50):
                chunk = arr[i:i+50]
                toks.append(int((chunk.mean()+1)*1024)%2048)
            toks_np = np.array(toks, dtype=np.int64).reshape(1,32)
            logits = art.run([art_out], {art_in: toks_np})[0]
            # handle both (1,32,2048) -> mean over tokens OR (1,2048) (pre-averaged feats)
            if getattr(logits, 'ndim', None) == 3:
                feat = logits.mean(axis=1).astype(np.float32).reshape(1, -1)
            else:
                feat = np.asarray(logits).astype(np.float32).reshape(1, -1)
            # ensure input shape matches ONNX fixed batch dim [1, 2048]
            feat_in = np.asarray(feat, dtype=np.float32).reshape(1, -1)
            try:
                res = s2.run(None, {'features': feat_in})
            except Exception:
                # try transposed fallback
                res = s2.run(None, {'features': feat_in.T})
            score = float(np.array(res[0]).reshape(-1)[0])
            scores.append(score)
            labels.append(s_['anomaly_label'])
            feats_all.append(feat.reshape(-1))
        scores=np.array(scores); labels=np.array(labels)
        mean_norm = float(scores[labels==0].mean()) if (labels==0).sum()>0 else None
        mean_anom = float(scores[labels==1].mean()) if (labels==1).sum()>0 else None
        acc = float(((scores>0.5)==labels).mean())
        # auc
        def _trapz(y,x):
            y=np.asarray(y); x=np.asarray(x)
            if y.size<2: return 0.0
            return float(((y[:-1]+y[1:])*(x[1:]-x[:-1])/2.0).sum())
        def roc_auc(labels, scores):
            labels = np.asarray(labels); scores=np.asarray(scores)
            if labels.sum()==0 or labels.sum()==len(labels): return None
            desc = np.argsort(-scores); ls = labels[desc]
            tp=np.cumsum(ls); fp=np.cumsum(1-ls)
            tpr = tp/labels.sum(); fpr = fp/(len(labels)-labels.sum())
            return float(_trapz(tpr,fpr))
        auc = roc_auc(labels, scores)
        feats_all = np.vstack(feats_all) if feats_all else np.zeros((0,2048))
        if feats_all.size:
            mdiff = np.abs(feats_all[labels==0].mean(axis=0) - feats_all[labels==1].mean(axis=0)) if labels.sum()>0 and (labels==0).sum()>0 else np.zeros(feats_all.shape[1])
            frac_sig = float((mdiff>0.1).sum()/mdiff.size)
        else:
            frac_sig = 0.0
        rec = {'n':n,'signal':signal,'norm_mode':norm_mode,'mean_norm':mean_norm,'mean_anom':mean_anom,'accuracy':acc,'auc':auc,'frac_sig':frac_sig,'job_path':path,'hipergrafo':hip_job}
        results.append(rec)
        OUT.write_text(json.dumps(results, indent=2))
        print('Saved result:', rec)

print('Micro-grid finished; results saved to', OUT)
