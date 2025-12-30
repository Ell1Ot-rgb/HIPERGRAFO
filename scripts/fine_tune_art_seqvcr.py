#!/usr/bin/env python3
"""Fine-tune ART PyTorch model with Seq-VCR penalties on logits/features.
Exports a new ONNX `models/art_17_finetuned.onnx` and saves checkpoint.
"""
import torch, numpy as np, random
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
import importlib.util
spec = importlib.util.spec_from_file_location('server_mod', 'src/local_server/servidor_art_v7_hipergrafo.py')
server_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_mod)
ART = getattr(server_mod, 'ART_Brain_V7_Complete')
LAST = getattr(server_mod, 'LAST_CHECKPOINT_FILE')
DEVICE = torch.device('cpu')

# Hyperparams (tunable)
EPOCHS = 6
STEPS_PER_EPOCH = 40
BATCH_SIZE = 16
LR = 1e-4
SEQ_LEN = 4
VCR_VAR_WEIGHT = 25.0
VCR_COV_WEIGHT = 1.0
VCR_TARGET_VAR = 1.0
SPECTRAL_ALPHA = 0.001
CHECKPOINT_DIR = Path('modelos_guardados')
EXPORT_ONNX = Path('models/art_17_finetuned.onnx')

# simple physics generator
def make_batch(batch_size):
    t = np.linspace(0,1,1600)
    X = []
    for _ in range(batch_size):
        n_components = np.random.randint(3,6)
        sig = np.zeros_like(t)
        for _c in range(n_components):
            freq = np.random.uniform(1,10)
            phase = np.random.uniform(0,2*np.pi)
            amp = np.random.uniform(0.05,0.5)
            sig += amp*np.sin(2*np.pi*freq*t + phase)
        sig += np.random.normal(0,0.02,size=t.shape)
        X.append(sig.astype(np.float32))
    return np.stack(X)

VOCAB = 2048
# tokenization
def to_tokens(batch_1600):
    batch_tokens = np.zeros((batch_1600.shape[0], 32), dtype=np.int64)
    for i, vec in enumerate(batch_1600):
        toks = []
        for s in range(0, 1600, 50):
            chunk = vec[s:s+50]
            toks.append(int((chunk.mean()+1)*1024)%VOCAB)
        batch_tokens[i,:] = np.array(toks)
    return batch_tokens


def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    model = ART(dim=128, depth=6, vocab=2048).to(DEVICE)
    if LAST.exists():
        ck = torch.load(str(LAST), map_location=DEVICE)
        model.load_state_dict(ck['model_state_dict'])
        print('Loaded checkpoint from', LAST)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    seq_buf = deque(maxlen=SEQ_LEN)

    for epoch in range(EPOCHS):
        ep_loss = 0.0
        for step in range(STEPS_PER_EPOCH):
            X = make_batch(BATCH_SIZE)
            toks = to_tokens(X)
            toks_t = torch.from_numpy(toks).long().to(DEVICE)
            # forward
            logits = model(toks_t)
            if isinstance(logits, tuple):
                logits = logits[0]
            # logits: [B, 32, 2048]
            feats = logits.mean(dim=1)  # [B, 2048]

            # spectral decoupling on logits
            spec = SPECTRAL_ALPHA * (logits.pow(2).mean())

            # accumulate sequential features
            seq_buf.append(feats.detach())
            var_loss = torch.tensor(0.0, device=DEVICE)
            cov_loss = torch.tensor(0.0, device=DEVICE)
            if len(seq_buf) == SEQ_LEN:
                Xseq = torch.cat(list(seq_buf), dim=0)  # [SEQ_LEN*B, D]
                var = Xseq.var(dim=0, unbiased=False)
                var_violation = (VCR_TARGET_VAR - var).clamp(min=0.0)
                var_loss = VCR_VAR_WEIGHT * var_violation.pow(2).sum()

                Xc = Xseq - Xseq.mean(dim=0, keepdim=True)
                N = Xseq.shape[0]
                covm = (Xc.t() @ Xc) / float(N)
                off_diag = covm.pow(2).sum() - torch.diagonal(covm).pow(2).sum()
                cov_loss = VCR_COV_WEIGHT * off_diag

            loss = spec + var_loss + cov_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
        print(f'Epoch {epoch+1}/{EPOCHS} loss {ep_loss/STEPS_PER_EPOCH:.6f}')

    # Save checkpoint
    ck = {'epoch': None, 'model_state_dict': model.state_dict()}
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ck_path = CHECKPOINT_DIR / 'finetuned_art_17.pth'
    torch.save(ck, ck_path)
    print('Saved checkpoint to', ck_path)

    # Export ONNX: feed integer tokens
    model.eval()
    # create a tiny wrapper that returns logits averaged (feats) to keep shape
    class _WrapFeats(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            l = self.base(x)
            if isinstance(l, tuple):
                l = l[0]
            return l.mean(dim=1)
    w = _WrapFeats(model)
    try:
        dummy = torch.randint(0, VOCAB, (1,32)).long().to(DEVICE)
        torch.onnx.export(w, dummy, str(EXPORT_ONNX), input_names=['input_tokens'], output_names=['feats'], opset_version=12)
        print('Exported fine-tuned ONNX to', EXPORT_ONNX)
    except Exception as e:
        print('Failed to export ONNX:', e)

if __name__ == '__main__':
    main()
