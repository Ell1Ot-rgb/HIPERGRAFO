#!/usr/bin/env python3
"""Stimulus sweep and automatic calibration (sign-flip) for a given seed checkpoint.

- Runs sweeps over frequencies, amplitudes and offsets recording corr(atom_mean, logits)
- If majority of responses show negative correlation, attempts head sign-flip and validates via validate_omega21
- Writes detailed report to analysis/diagnosis/sweep_{seed}.json and plots
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.diagnose_alucinacion import generate_sine, atoms_from_wave, load_checkpoint, HeadWrapper
from scripts.validate_omega21 import validate


def sweep_and_calibrate(ck_path: Path, freqs=[1.0,3.0,7.0,13.0], amps=[0.2,0.5,1.0,2.0], offsets=[-0.5,0.0,0.5], phases=[0.0,1.0], out_prefix='sweep'):
    report_dir = Path('analysis/diagnosis')
    report_dir.mkdir(parents=True, exist_ok=True)

    S_state, c2_state, head_state = load_checkpoint(ck_path)
    # instantiate models
    from src.local_server.servidor_local import Capa2EspacioTemporal
    c2 = Capa2EspacioTemporal(input_dim=1600, hidden_dim=512)
    c2.load_state_dict(c2_state)
    # load head flexible
    if head_state is None:
        head = torch.nn.Linear(1600,1)
    else:
        keys=list(head_state.keys())
        if any(k.startswith('h.') for k in keys):
            class HMod(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.h = torch.nn.Linear(1600,1)
            hm=HMod()
            hm.load_state_dict(head_state)
            head = hm.h
        else:
            head = torch.nn.Linear(1600,1)
            head.load_state_dict(head_state)
    head_wr = HeadWrapper(head)
    c2.eval(); head_wr.eval()

    all_entries=[]
    for amp in amps:
        for off in offsets:
            for f in freqs:
                for ph in phases:
                    w = generate_sine(f, phase=ph, amp=amp)
                    if off != 0.0:
                        w = w + off
                    atoms = atoms_from_wave(w)
                    with torch.no_grad():
                        x_in = torch.from_numpy(w.astype(np.float32)).unsqueeze(0).unsqueeze(1)
                        c2out, _ = c2(x_in)
                        logit = float(head_wr(c2out).squeeze().item())
                    all_entries.append({'freq':f,'amp':amp,'offset':off,'phase':ph,'atom_mean':float(atoms.mean()),'logit':logit})

    # aggregate
    atom_means = np.array([e['atom_mean'] for e in all_entries])
    logits = np.array([e['logit'] for e in all_entries])
    corr = float(np.corrcoef(atom_means, logits)[0,1])

    # plot atom_mean vs logit scatter and per-condition heatmap
    fig1 = report_dir / f'{out_prefix}_scatter.png'
    plt.figure()
    plt.scatter(atom_means, logits)
    plt.xlabel('atom_mean'); plt.ylabel('logit'); plt.title('atom_mean vs logit scatter')
    plt.grid(True); plt.savefig(fig1); plt.close()

    # heatmap by amp/offset
    # create matrix of mean logits per (amp,offset)
    amps_sorted = sorted(amps); offs_sorted = sorted(offsets)
    mat = np.zeros((len(amps_sorted), len(offs_sorted)))
    counts = np.zeros_like(mat)
    for e in all_entries:
        i = amps_sorted.index(e['amp']); j = offs_sorted.index(e['offset'])
        mat[i,j] += e['logit']; counts[i,j] += 1
    mat = mat / np.maximum(counts, 1)
    fig2 = report_dir / f'{out_prefix}_heatmap_meanlogit.png'
    plt.figure(figsize=(6,4))
    plt.imshow(mat, aspect='auto', cmap='RdBu')
    plt.colorbar(); plt.xticks(range(len(offs_sorted)), offs_sorted); plt.yticks(range(len(amps_sorted)), amps_sorted)
    plt.xlabel('offset'); plt.ylabel('amp'); plt.title('mean logit per amp/offset')
    plt.savefig(fig2); plt.close()

    res = {'ck': str(ck_path), 'corr_atom_logit': corr, 'n_conditions': len(all_entries), 'scatter': str(fig1), 'heatmap': str(fig2)}
    report_path = report_dir / f'{out_prefix}_report.json'
    report_path.write_text(json.dumps(res, indent=2))

    # decide whether to attempt sign-flip: if corr < -0.3 (strong negative) attempt flip
    attempt_flip = corr < -0.3
    res['attempt_flip'] = attempt_flip

    if attempt_flip:
        # apply sign flip to head: multiply weights and bias by -1 and export ONNX and validate
        flipped_head = torch.nn.Linear(1600,1)
        flipped_head.weight.data = - head.weight.data.clone()
        flipped_head.bias.data = - head.bias.data.clone()
        # create wrapper for full pipeline (waveform -> logit)
        class FullWrapper(torch.nn.Module):
            def __init__(self, c2, head):
                super().__init__()
                self.c2 = c2
                self.head = head
            def forward(self, x):
                x_seq = x.unsqueeze(1)
                c2out,_ = self.c2(x_seq)
                logits = self.head(c2out)
                return logits
        wrapper = FullWrapper(c2, flipped_head).eval()
        tmp_onnx = Path('models') / f'flipped_{Path(ck_path).stem}.onnx'
        try:
            dummy = torch.zeros(1,1600,dtype=torch.float32)
            torch.onnx.export(wrapper, dummy, str(tmp_onnx), opset_version=14, input_names=['x'], output_names=['logit'])
            res['flipped_onnx'] = str(tmp_onnx)
            # validate
            vres = validate(tmp_onnx, thresholds={'auc':0.85,'high_corr_frac':0.10,'frac_sig':0.5}, n_samples=400)
            res['validation_after_flip'] = vres
            if vres['passed']:
                # promote
                dst = Path('models') / 'best_omega21_validated.onnx'
                dst.write_bytes(tmp_onnx.read_bytes())
                res['promoted_validated'] = str(dst)
            else:
                res['promoted_validated'] = None
        except Exception as e:
            res['flip_error'] = str(e)
    report_path.write_text(json.dumps(res, indent=2))
    return res

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('ck', type=str)
    p.add_argument('--out-prefix', type=str, default='sweep')
    args=p.parse_args()
    r = sweep_and_calibrate(Path(args.ck), out_prefix=args.out_prefix)
    print(json.dumps(r, indent=2))
