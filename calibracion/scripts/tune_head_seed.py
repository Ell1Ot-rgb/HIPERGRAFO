#!/usr/bin/env python3
"""Train only the head (linear output layer) of ART from a given checkpoint.

Protocol (surgical):
- Load checkpoint (seed 2 by default)
- Freeze all parameters except `model.head`
- Train head for a few epochs on calibrated synthetic data (scale/offset applied to waveform)
- Export ONNX and run validation; if validated and --promote, write analysis/release_omega21.json and optionally create a GH release (if `--auto-release` and `gh` available)
"""
import argparse
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from scripts.validate_omega21 import validate, make_dataset
from scripts.diagnose_art_features import to_tokens_single

# import model
import importlib.util
spec = importlib.util.spec_from_file_location('server_mod', 'src/local_server/servidor_art_v7_hipergrafo.py')
server_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_mod)
ART = getattr(server_mod, 'ART_Brain_V7_Complete')


def train_head(
    checkpoint_path: Path,
    scale: float = -0.38,
    offset: float = 0.42,
    epochs: int = 3,
    steps_per_epoch: int = 100,
    batch_size: int = 64,
    lr: float = 1e-5,
    out_onnx: Path = Path('models/best_omega21_headtuned.onnx'),
    device: str = 'cpu',
    unfreeze_last_n: int = 1,
    lr_last: float = 5e-6,
    auc_surrogate: bool = False
):
    device = torch.device(device)
    model = ART(dim=128, depth=6, vocab=2048).to(device)
    if Path(checkpoint_path).exists():
        ck = torch.load(str(checkpoint_path), map_location=device)
        # Support different checkpoint layouts: full dict under 'model_state_dict' or parted keys
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            model.load_state_dict(ck['model_state_dict'])
            print('Loaded model_state_dict from', checkpoint_path)
        elif isinstance(ck, dict) and 'head_state' in ck:
            # Load head_state into model.head if available; keep other params from continuous checkpoint already loaded inside module
            hs = ck['head_state']
            sd = model.state_dict()
            mapped = {}
            for k, v in hs.items():
                # map e.g. 'h.weight' -> 'head.weight'
                newk = k.replace('h.', 'head.')
                if newk in sd and sd[newk].shape == v.shape:
                    mapped[newk] = v
            if mapped:
                sd.update(mapped)
                model.load_state_dict(sd)
                print('Loaded head_state into model from', checkpoint_path)
            else:
                print('Found head_state but no compatible keys to load')
        else:
            print('Checkpoint found but structure not recognized; skipping explicit load (model may already have a continuous checkpoint)')
    model.train()

    # Freeze everything then selectively unfreeze
    for name, p in model.named_parameters():
        p.requires_grad = False

    # always unfreeze head
    for name, p in model.head.named_parameters():
        p.requires_grad = True
        print('Trainable param (head):', name, p.shape)

    # optionally unfreeze the last N LSTM layers
    if unfreeze_last_n and unfreeze_last_n > 0:
        to_unfreeze = model.layers[-unfreeze_last_n:]
        for idx, layer in enumerate(to_unfreeze, start=1):
            for name, p in layer.named_parameters():
                p.requires_grad = True
                print(f'Trainable param (last_layer_{idx}):', name, p.shape)

    # build optimizer with parameter groups for separate learning rates
    param_groups = []
    head_params = [p for n,p in model.head.named_parameters() if p.requires_grad]
    if head_params:
        param_groups.append({'params': head_params, 'lr': lr})

    if unfreeze_last_n and unfreeze_last_n > 0:
        last_params = []
        for layer in model.layers[-unfreeze_last_n:]:
            last_params.extend([p for n,p in layer.named_parameters() if p.requires_grad])
        if last_params:
            param_groups.append({'params': last_params, 'lr': lr_last})

    optimizer = torch.optim.Adam(param_groups)

    # batch tokenization helper
    def batch_to_tokens(batch_waveform):
        # batch_waveform: (B, 1600)
        B = batch_waveform.shape[0]
        toks = np.zeros((B, 32), dtype=np.int64)
        for i in range(B):
            toks[i,:] = to_tokens_single(batch_waveform[i]).reshape(-1)
        return toks

    # If auc_surrogate flag is set, we will use pairwise hinge loss approximating AUC
    def _auc_surrogate_loss(scores, labels, margin=1.0):
        # scores: tensor [B], labels: tensor [B] with 0/1
        pos = scores[labels==1]
        neg = scores[labels==0]
        if pos.numel() == 0 or neg.numel() == 0:
            # fallback to small loss
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        # compute pairwise margin: max(0, 1 - (s_pos - s_neg))
        # shape (P, N)
        diff = 1.0 - (pos.unsqueeze(1) - neg.unsqueeze(0))
        loss = torch.clamp(diff, min=0.0).mean()
        return loss

    total_steps = epochs * steps_per_epoch
    step = 0
    for ep in range(epochs):
        ep_loss = 0.0
        for s in range(steps_per_epoch):
            X = make_dataset(batch_size//2, signal=1.0)
            # apply calibration transform (scale/offset)
            batch = []
            for sample in X:
                arr = np.array(sample['input_data'], dtype=np.float32)
                arr2 = arr * scale + offset
                batch.append(arr2)
            batch = np.stack(batch, axis=0)
            toks = batch_to_tokens(batch)
            toks_t = torch.from_numpy(toks).long().to(device)

            optimizer.zero_grad()
            logits, states, h = model(toks_t)
            if isinstance(logits, tuple):
                logits = logits[0]
            # produce features and scalar scores
            feats = logits.mean(dim=1)  # [B, vocab]
            scores = feats.mean(dim=1)  # [B] scalar score
            labels = torch.tensor([s['anomaly_label'] for s in X], dtype=torch.float32, device=device)

            if auc_surrogate:
                loss = _auc_surrogate_loss(scores, labels, margin=1.0)
                # decorrelation penalty on features to avoid collapse of head (encourage diverse output dims)
                feats_centered = feats - feats.mean(dim=0, keepdim=True)
                N = feats_centered.shape[0]
                cov = (feats_centered.t() @ feats_centered) / float(max(N,1))
                off_diag = cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()
                cov_pen = 1e-3 * off_diag  # small weight
                loss = loss + cov_pen
            else:
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(scores, labels)

            # small l2 penalty on head weights to avoid runaway magnitudes
            l2 = 0.0
            for pn, p in model.head.named_parameters():
                l2 = l2 + p.pow(2).sum() * 1e-7
            loss = loss + l2
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item())
            step += 1
        print(f'Epoch {ep+1}/{epochs} avg_loss {ep_loss/steps_per_epoch:.6f}')

    # Save a checkpoint for traceability
    out_ck = Path('modelos_guardados') / 'head_tuned_seed2.pth'
    out_ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, out_ck)
    print('Wrote head-tuned checkpoint to', out_ck)

    # Export ONNX: wrap to return averaged features as previous
    model.eval()
    class _WrapFeats(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            l = self.base(x)[0]
            if isinstance(l, tuple):
                l = l[0]
            return l.mean(dim=1)
    w = _WrapFeats(model).to(device)
    try:
        dummy = torch.randint(0, 2048, (1,32)).long().to(device)
        torch.onnx.export(w, dummy, str(out_onnx), input_names=['input_tokens'], output_names=['feats'], opset_version=12)
        print('Exported head-tuned ONNX to', out_onnx)
    except Exception as e:
        print('Failed to export ONNX:', e)

    return out_onnx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='modelos_guardados/multi_seed_seed_2.pth')
    p.add_argument('--scale', type=float, default=-0.38)
    p.add_argument('--offset', type=float, default=0.42)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--steps-per-epoch', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--out', type=str, default='models/best_omega21_headtuned.onnx')
    p.add_argument('--validate-n', type=int, default=2000)
    p.add_argument('--promote', action='store_true', help='if validation passes, write release_omega21.json and optionally create GH release')
    p.add_argument('--auto-release', action='store_true', help='if set and `--promote`, try to create GH release using `gh` CLI')
    p.add_argument('--train-readout', action='store_true', help='instead of training head, train a small readout linear layer on top of averaged features')
    p.add_argument('--unfreeze-head', action='store_true', help='when training readout, also unfreeze and train model.head with a smaller lr')
    p.add_argument('--unfreeze-last-n', type=int, default=1, help='unfreeze the last N LSTM layers (default 1)')
    p.add_argument('--lr-last', type=float, default=5e-6, help='learning rate for the last LSTM layer(s) when unfrozen')
    p.add_argument('--auc-surrogate', action='store_true', help='train the head using an AUC surrogate (pairwise hinge) loss')
    p.add_argument('--mix-loss', action='store_true', help='mix BCE and AUC surrogate losses (default on for readout training)')
    p.add_argument('--auc-weight', type=float, default=0.5, help='weight for AUC loss when --mix-loss is used')
    p.add_argument('--aug-scale-noise', type=float, default=0.02, help='stddev for random scale augmentation per batch')
    p.add_argument('--aug-offset-noise', type=float, default=0.02, help='stddev for random offset augmentation per batch')
    p.add_argument('--cov-pen-weight', type=float, default=1e-4, help='weight for covariance off-diagonal penalty during readout training')
    p.add_argument('--score-var-weight', type=float, default=1e-3, help='weight for score variance penalty during readout training')
    args = p.parse_args()

    if args.train_readout:
        # train a small linear readout on top of averaged feats
        from torch import nn
        device='cpu'
        model = ART(dim=128, depth=6, vocab=2048).to(device)
        # try to load checkpoint same as train_head does
        if Path(args.checkpoint).exists():
            ck = torch.load(str(args.checkpoint), map_location=device)
            if isinstance(ck, dict) and 'head_state' in ck:
                # skip
                pass
        # create readout
        readout = nn.Linear(model.vocab, 1).to(device)
        # freeze model
        for p in model.parameters():
            p.requires_grad = False
        param_groups = []
        param_groups.append({'params': readout.parameters(), 'lr': args.lr})
        if args.unfreeze_head:
            for name, p in model.head.named_parameters():
                p.requires_grad = True
            param_groups.append({'params': model.head.parameters(), 'lr': max(1e-6, args.lr * 0.1)})
        opt = torch.optim.Adam(param_groups)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        # augmentation and mix-loss defaults (can be set via CLI)
        if not hasattr(args, 'aug_scale_noise'):
            args.aug_scale_noise = 0.02
        if not hasattr(args, 'aug_offset_noise'):
            args.aug_offset_noise = 0.02
        if not hasattr(args, 'mix_loss'):
            args.mix_loss = True
        if not hasattr(args, 'auc_weight'):
            args.auc_weight = 0.5
        if not hasattr(args, 'cov_pen_weight'):
            args.cov_pen_weight = 1e-4
        if not hasattr(args, 'score_var_weight'):
            args.score_var_weight = 1e-3
        def _auc_surrogate_readout(scores, labels, margin=1.0):
            pos = scores[labels==1]
            neg = scores[labels==0]
            if pos.numel()==0 or neg.numel()==0:
                return torch.tensor(0.0, device=scores.device, requires_grad=True)
            diff = 1.0 - (pos.unsqueeze(1) - neg.unsqueeze(0))
            loss = torch.clamp(diff, min=0.0).mean()
            return loss

        steps = args.epochs * args.steps_per_epoch
        for ep in range(args.epochs):
            ep_loss = 0.0
            for step in range(args.steps_per_epoch):
                # augmentation: sample per-batch small noise for scale/offset
                s_noise = float(np.random.normal(0.0, args.aug_scale_noise)) if args.aug_scale_noise>0 else 0.0
                o_noise = float(np.random.normal(0.0, args.aug_offset_noise)) if args.aug_offset_noise>0 else 0.0
                cur_scale = args.scale + s_noise
                cur_offset = args.offset + o_noise

                X = make_dataset(args.batch_size//2, signal=1.0)
                batch=[]
                labels=[]
                for s in X:
                    arr = np.array(s['input_data'], dtype=np.float32)
                    arr2 = arr * cur_scale + cur_offset
                    batch.append(arr2)
                    labels.append(s['anomaly_label'])
                batch = np.stack(batch, axis=0)
                # tokenize
                toks = np.zeros((batch.shape[0], 32), dtype=np.int64)
                for i in range(batch.shape[0]):
                    toks[i,:] = to_tokens_single(batch[i]).reshape(-1)
                toks_t = torch.from_numpy(toks).long().to(device)
                with torch.no_grad():
                    logits, states, h = model(toks_t)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    feats = logits.mean(dim=1)  # [B, vocab]
                scores = readout(feats).reshape(-1)
                labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

                # combine losses if mix-loss is requested
                if args.mix_loss:
                    bce = loss_fn(scores, labels_t)
                    auc_l = _auc_surrogate_readout(scores, labels_t)
                    loss = (1.0 - args.auc_weight) * bce + args.auc_weight * auc_l
                elif args.auc_surrogate:
                    loss = _auc_surrogate_readout(scores, labels_t)
                else:
                    loss = loss_fn(scores, labels_t)

                # regularizers
                score_var_pen = args.score_var_weight * (scores.var() * 1.0)
                feats_centered = feats - feats.mean(dim=0, keepdim=True)
                N = feats_centered.shape[0]
                cov = (feats_centered.t() @ feats_centered) / float(max(N,1))
                off_diag = cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()
                cov_pen = args.cov_pen_weight * off_diag
                loss = loss + score_var_pen + cov_pen

                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += float(loss.item())
            print(f'Readout Epoch {ep+1}/{args.epochs} avg_loss {ep_loss/args.steps_per_epoch:.6f}')
        # export ONNX wrapper combining model + readout -> output scalar score
        model.eval(); readout.eval()
        class _WrapScore(torch.nn.Module):
            def __init__(self, base, read):
                super().__init__()
                self.base = base
                self.read = read
            def forward(self, x):
                l = self.base(x)[0]
                if isinstance(l, tuple): l = l[0]
                feats = l.mean(dim=1)
                score = self.read(feats).reshape(-1,1)
                return score
        w = _WrapScore(model, readout)
        try:
            dummy = torch.randint(0, 2048, (1,32)).long()
            torch.onnx.export(w, dummy, str(args.out), input_names=['input_tokens'], output_names=['score'], opset_version=12)
            out = Path(args.out)
            print('Exported readout ONNX to', out)
        except Exception as e:
            print('Failed to export ONNX:', e)
        # Validate via validate() (it will interpret scalar outputs reasonably)
        res = validate(str(out), n_samples=args.validate_n)
        print('Validation result:', json.dumps(res, indent=2))
    else:
        out = train_head(
            checkpoint_path=Path(args.checkpoint),
            scale=args.scale,
            offset=args.offset,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            out_onnx=Path(args.out),
            unfreeze_last_n=args.unfreeze_last_n,
            lr_last=args.lr_last,
            auc_surrogate=args.auc_surrogate
        )
        res = validate(str(out), n_samples=args.validate_n)
        print('Validation result:', json.dumps(res, indent=2))

    if res['passed']:
        print('Validation passed!')
        if args.promote:
            meta = {
                'onnx': str(out),
                'scale': args.scale,
                'offset': args.offset,
                'validate': res
            }
            Path('analysis').mkdir(exist_ok=True)
            Path('analysis').joinpath('release_omega21.json').write_text(json.dumps(meta, indent=2))
            print('Wrote analysis/release_omega21.json')
            # attempt to create GH release if requested
            if args.auto_release:
                import subprocess
                tag = 'v1.1-Omega-Calibrated'
                subprocess.check_call(['git','add','analysis/release_omega21.json'])
                subprocess.check_call(['git','commit','-m',f'Promote calibrated Omega 21 {tag}'])
                subprocess.check_call(['git','tag',tag])
                subprocess.check_call(['git','push', 'origin', 'main', '--tags'])
                # create a release with gh
                subprocess.check_call(['gh','release','create',tag, str(out), '-F', str(Path('analysis')/ 'release_omega21.json')])
                print('Created GH release', tag)
    else:
        print('Validation failed; not promoting.')


if __name__ == '__main__':
    main()
