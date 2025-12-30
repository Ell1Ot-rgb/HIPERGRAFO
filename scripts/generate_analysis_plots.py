#!/usr/bin/env python3
import json
from pathlib import Path
import csv
import matplotlib.pyplot as plt

IN = Path('analysis_5capas_detailed.json')
OUT_CSV = Path('analysis_5capas_summary.csv')
PLOTS_DIR = Path('analysis/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MD = Path('docs/analysis_5capas.md')
MD.parent.mkdir(parents=True, exist_ok=True)

with open(IN) as f:
    data = json.load(f)

# write CSV
with open(OUT_CSV,'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['n','signal','mean_norm','mean_anom','accuracy','auc','frac_sig_dims_test','job_path'])
    for r in data:
        writer.writerow([r.get('n'), r.get('signal'), r.get('mean_norm'), r.get('mean_anom'), r.get('accuracy'), r.get('auc'), r.get('frac_sig_dims_test'), r.get('job_path')])

# group by n
by_n = {}
for r in data:
    n = r['n']
    by_n.setdefault(n, []).append(r)

for n, rows in by_n.items():
    rows = sorted(rows, key=lambda x: x['signal'])
    x = [r['signal'] for r in rows]
    acc = [r.get('accuracy') if r.get('accuracy') is not None else 0 for r in rows]
    auc = [r.get('auc') if r.get('auc') is not None else 0 for r in rows]
    frac = [r.get('frac_sig_dims_test') if r.get('frac_sig_dims_test') is not None else 0 for r in rows]
    plt.figure(); plt.plot(x, acc, marker='o'); plt.title(f'Accuracy vs Signal (n={n})'); plt.xlabel('signal'); plt.ylabel('accuracy'); plt.grid(True); plt.savefig(PLOTS_DIR/f'acc_vs_signal_n{n}.png'); plt.close()
    plt.figure(); plt.plot(x, auc, marker='o'); plt.title(f'AUC vs Signal (n={n})'); plt.xlabel('signal'); plt.ylabel('AUC'); plt.grid(True); plt.savefig(PLOTS_DIR/f'auc_vs_signal_n{n}.png'); plt.close()
    plt.figure(); plt.plot(x, frac, marker='o'); plt.title(f'FracSigDims vs Signal (n={n})'); plt.xlabel('signal'); plt.ylabel('frac_sig_dims'); plt.grid(True); plt.savefig(PLOTS_DIR/f'frac_sig_vs_signal_n{n}.png'); plt.close()

# write a small markdown report
with open(MD,'w') as f:
    f.write('# Análisis 5‑capas — Resultados parciales\n\n')
    f.write('Estos resultados provienen de una ejecución del grid (señales y tamaños). ')
    f.write('Se guardó `analysis_5capas_detailed.json` con cada experimento.\n\n')
    for n in sorted(by_n.keys()):
        f.write(f'## n = {n}\n')
        f.write(f'![acc](../analysis/plots/acc_vs_signal_n{n}.png)\n')
        f.write(f'![auc](../analysis/plots/auc_vs_signal_n{n}.png)\n')
        f.write(f'![frac](../analysis/plots/frac_sig_vs_signal_n{n}.png)\n\n')

print('Plots generated and report written')