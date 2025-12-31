#!/usr/bin/env python3
"""Diagnóstico de Alucinación
- Carga checkpoint de un seed (multi_seed_seed_{i}.pth)
- Inyecta señales seno (varias frecuencias y fases)
- Calcula 25 "átomos" por chunk (mean, std, energy)
- Pasa onda a Capa2 + Head del checkpoint y registra salida
- Genera plots: atoms vs logits (per freq), heatmap de pesos del head (25x64)
- Guarda informe JSON con métricas y paths a figuras
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
from src.local_server.servidor_local import Capa2EspacioTemporal


def generate_sine(freq, phase=0.0, amp=1.0, length=1600, sr=1600):
    t = np.arange(length) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def atoms_from_wave(wave):
    # split into 25 chunks, return array (25,)
    n = len(wave)
    chunk = n // 25
    atoms = []
    for i in range(25):
        s = i*chunk
        e = s+chunk if i<24 else n
        seg = wave[s:e]
        atoms.append(float(seg.mean()))
    return np.array(atoms)


class HeadWrapper(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.h = head
    def forward(self, x):
        return self.h(x)


def load_checkpoint(ck_path):
    ck = torch.load(str(ck_path), map_location='cpu')
    S_state = ck.get('sensor_state')
    c2_state = ck.get('c2_state')
    head_state = ck.get('head_state')
    return S_state, c2_state, head_state


def run_diagnosis(ck_path, out_prefix, freqs=[1.0,3.0,7.0,13.0], phases=[0.0,1.0], amp=1.0):
    out_dir = Path('analysis/diagnosis')
    out_dir.mkdir(parents=True, exist_ok=True)
    S_state, c2_state, head_state = load_checkpoint(ck_path)
    # instantiate models
    c2 = Capa2EspacioTemporal(input_dim=1600, hidden_dim=512)
    c2.load_state_dict(c2_state)
    # Load head, support both formats saved with prefix 'h.' or plain linear
    if head_state is None:
        head = torch.nn.Linear(1600,1)
    else:
        sample_keys = list(head_state.keys())
        if any(k.startswith('h.') for k in sample_keys):
            class HeadWithH(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.h = torch.nn.Linear(1600,1)
            head = HeadWithH()
            head.load_state_dict(head_state)
            # unwrap
            head = head.h
        else:
            head = torch.nn.Linear(1600,1)
            head.load_state_dict(head_state)
    head_wr = HeadWrapper(head)
    c2.eval(); head_wr.eval()

    results = {'seed_ck': str(ck_path), 'freqs': [], 'summary':{}}
    all_atoms = []
    all_logits = []

    for f in freqs:
        for ph in phases:
            w = generate_sine(f, phase=ph, amp=amp)
            atoms = atoms_from_wave(w)
            # model forward
            x_in = torch.from_numpy(w.astype(np.float32)).unsqueeze(0).unsqueeze(1)  # [1,1,1600]
            with torch.no_grad():
                c2out, _ = c2(x_in)
                logits = head_wr(c2out).squeeze(1).squeeze(0).item()
            all_atoms.append(atoms)
            all_logits.append(logits)
            results['freqs'].append({'freq':float(f),'phase':float(ph),'logit':float(logits),'atoms':atoms.tolist()})

    all_atoms = np.stack(all_atoms)  # (N,25)
    all_logits = np.array(all_logits)

    # Plot atoms heatmap (N x 25)
    plt.figure(figsize=(8,4))
    plt.imshow(all_atoms, aspect='auto', cmap='RdBu', interpolation='nearest')
    plt.colorbar(label='atom_mean')
    plt.xlabel('atom index')
    plt.ylabel('stimulus #')
    path1 = out_dir / f'{out_prefix}_atoms_heatmap.png'
    plt.title('Atoms activation (mean) per stimulus')
    plt.savefig(path1)
    plt.close()

    # Plot logits vs stimulus
    plt.figure()
    plt.plot(all_logits, marker='o')
    plt.title('Capa2+Head logits per stimulus')
    plt.xlabel('stimulus #')
    plt.ylabel('logit')
    path2 = out_dir / f'{out_prefix}_logits.png'
    plt.savefig(path2)
    plt.close()

    # Correlation between mean atom activation and logits
    atom_mean = all_atoms.mean(axis=1)
    corr = float(np.corrcoef(atom_mean, all_logits)[0,1])

    # Heatmap of head weights reshaped to 25 x 64
    w = head.weight.detach().cpu().numpy().reshape(-1)
    if w.size == 1600:
        W25 = w.reshape(25,64)
        plt.figure(figsize=(6,4))
        plt.imshow(W25, aspect='auto', cmap='viridis')
        plt.colorbar(); plt.title('Head weights (25 x 64)')
        path3 = out_dir / f'{out_prefix}_head_weights.png'
        plt.savefig(path3)
        plt.close()
    else:
        path3 = None

    # Save report
    report = {
        'seed_ck': str(ck_path),
        'n_stimuli': int(all_atoms.shape[0]),
        'atom_mean_mean': float(atom_mean.mean()),
        'atom_mean_std': float(atom_mean.std()),
        'logits_mean': float(all_logits.mean()),
        'logits_std': float(all_logits.std()),
        'corr_atom_logits': corr,
        'plots': {'atoms_heatmap': str(path1), 'logits': str(path2), 'head_weights': str(path3) if path3 else None}
    }
    report_path = out_dir / f'{out_prefix}_report.json'
    report_path.write_text(json.dumps(report, indent=2))
    print('Wrote report', report_path)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('ck', type=str, help='checkpoint path (modelos_guardados/multi_seed_seed_{i}.pth)')
    p.add_argument('--out-prefix', type=str, default='diagnose')
    p.add_argument('--freqs', type=str, default='1.0,3.0,7.0,13.0')
    p.add_argument('--phases', type=str, default='0.0,1.0')
    p.add_argument('--amp', type=float, default=1.0)
    args = p.parse_args()
    freqs = [float(x) for x in args.freqs.split(',')]
    phases = [float(x) for x in args.phases.split(',')]
    r = run_diagnosis(Path(args.ck), args.out_prefix, freqs=freqs, phases=phases, amp=args.amp)
    print(json.dumps(r, indent=2))
