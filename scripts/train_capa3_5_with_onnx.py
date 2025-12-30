#!/usr/bin/env python3
"""Entrena las capas 3-5 usando el ONNX exportado como feature-extractor para la "Capa 2".
- Carga `models/art_17.onnx` y usa su salida "logits" para producir un vector por muestra
- Proyecta ese vector (2048) -> hidden*2 y aplica las capas 3/4/5 (igual que en `CortezaCognitivaV2`)
- Entrena solo las capas 3-5 (la proyección y cabezas son entrenables)
- Hace un pequeño smoke-test y guarda un ONNX resultante en `models/art_17_capa3_5.onnx`

Diseñado como prueba rápida (smoke test) para validar pipeline.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import onnxruntime as ort
from pathlib import Path

DEVICE = torch.device('cpu')
ART_ONNX = Path('models/art_17.onnx')
EXPORT_ONNX = Path('models/art_17_capa3_5.onnx')

# Config
BATCH_SIZE = 32
EPOCHS = 3
STEPS_PER_EPOCH = 30
HIDDEN = 512
VOCAB = 2048

# Seq-VCR and regularization hyperparameters (from user's recommendations)
VCR_VAR_WEIGHT = 25.0       # fuerza para evitar colapso de varianza
VCR_COV_WEIGHT = 1.0        # penalización de covarianza off-diagonal
VCR_TARGET_VAR = 1.0        # varianza mínima deseada
SEQ_LEN = 4                 # numero de pasos secuenciales para calcular cov (tiempo)
SPECTRAL_ALPHA = 0.001      # spectral decoupling sobre logits
KB_WEIGHT = 0.1             # Knuth-Bendix confluence weight
EDGE_THRESHOLD = 0.5        # umbral por defecto para crear aristas en hipergrafo

# Simple synthetic dataset generator (returns input 1600-d and anomaly label)
from collections import deque

def generate_batch(batch_size, physics=False):
    """Generador: por defecto usa osciladores sintéticos si physics=True.
    - Normales: sumas de senos (varias frecuencias/phasas)
    - Anómalos: rupturas de fase / cambios de frecuencia
    """
    X = []
    y = []
    for _ in range(batch_size):
        if physics:
            # build multi-sinusoidal signal + small noise
            t = np.linspace(0, 1, 1600)
            n_components = np.random.randint(3, 6)
            sig = np.zeros_like(t)
            for _c in range(n_components):
                freq = np.random.uniform(1, 10)
                phase = np.random.uniform(0, 2 * np.pi)
                amp = np.random.uniform(0.05, 0.5)
                sig += amp * np.sin(2 * np.pi * freq * t + phase)
            sig += np.random.normal(0, 0.02, size=t.shape)
            # anomaly with probability 0.2: a phase jump or freq shift
            if np.random.rand() < 0.2:
                cut = np.random.randint(200, 1400)
                sig[cut:] *= np.random.uniform(1.5, 3.0)  # amplitude change
                sig[cut:] += 0.5 * np.sin(2 * np.pi * (freq * 2) * t[cut:])
                lbl = 1
            else:
                lbl = 0
            vec = sig
        else:
            # fallback noisy Gaussian with occasional bump (legacy behavior)
            vec = np.random.normal(0, 0.3, size=(1600,))
            if np.random.rand() < 0.2:
                center = np.random.randint(0, 1600)
                width = np.random.randint(10, 200)
                amp = np.random.uniform(3, 6)
                xs = np.arange(1600)
                bump = amp * np.exp(-0.5 * ((xs - center) / width) ** 2)
                vec += bump
                lbl = 1
            else:
                lbl = 0
        X.append(vec.astype(np.float32))
        y.append(lbl)
    return np.stack(X), np.array(y, dtype=np.float32)

# Helper to convert 1600->32 tokens (same mapping used elsewhere)
def to_tokens(batch_1600):
    # batch_1600: np.array (batch,1600)
    batch_tokens = np.zeros((batch_1600.shape[0], 32), dtype=np.int64)
    for i, vec in enumerate(batch_1600):
        toks = []
        for s in range(0, 1600, 50):
            chunk = vec[s:s+50]
            token_val = int((chunk.mean() + 1) * 1024) % VOCAB
            toks.append(token_val)
        batch_tokens[i, :] = np.array(toks, dtype=np.int64)
    return batch_tokens

# PyTorch module implementing capa3-5 + projection
class Capa3to5(nn.Module):
    def __init__(self, feature_dim=2048, hidden=HIDDEN):
        super().__init__()
        self.proj = nn.Linear(feature_dim, hidden * 2)
        # Capa3
        self.capa3 = nn.Sequential(
            nn.Linear(hidden * 2, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, hidden)
        )
        # Capa4 attention: use simple MHA implemented via nn.MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, batch_first=True)
        # Proyección para cálculo de covarianza en espacio reducido (32 nodos)
        self.cov_proj = nn.Linear(hidden, 32)
        # Capa5 heads
        self.head_anom = nn.Linear(hidden, 1)  # produces logits (no sigmoid here)
        self.head_dend = nn.Linear(hidden, 16)

    def forward(self, features):
        # features: [batch, feature_dim]
        x = self.proj(features)  # [batch, hidden*2]
        c3 = self.capa3(x) + x.mean(dim=1, keepdim=True)  # residual-ish
        # Multihead expects sequence: we'll add a seq dim of 1
        attn_out, _ = self.attn(c3.unsqueeze(1), c3.unsqueeze(1), c3.unsqueeze(1))
        attn_out = attn_out.squeeze(1)
        anom_logits = self.head_anom(attn_out).squeeze(1)  # logits for BCEWithLogitsLoss
        anom_prob = torch.sigmoid(anom_logits)
        dend = torch.tanh(self.head_dend(attn_out))
        # cov features for Seq-VCR (project to 32 dims for efficiency and alignment with hipergrafo)
        cov_feats = self.cov_proj(attn_out)
        return anom_logits, anom_prob, dend, cov_feats


def main():
    assert ART_ONNX.exists(), f"ONNX model not found at {ART_ONNX}"

    # Prefer ONNX runtime, but fallback to PyTorch ART model if ONNX fails to run cleanly
    use_onnx = True
    sess = None
    input_name = None
    out_name = sess.get_outputs()[0].name
    try:
        sess = ort.InferenceSession(str(ART_ONNX), providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        # Quick smoke run to ensure session works
        import numpy as _np
        _sample = _np.zeros((1,32), dtype=_np.int64)
        sess.run([out_name], {input_name: _sample})
        print('ONNX session loaded and test-run ok')
    except Exception as e:
        print('ONNX session failed, will use PyTorch model as fallback:', e)
        use_onnx = False

    # If ONNX not usable, import ART_Brain model and load checkpoint
    art_model = None
    if not use_onnx:
        import importlib.util
        fn = 'src/local_server/servidor_art_v7_hipergrafo.py'
        spec = importlib.util.spec_from_file_location('server_mod', fn)
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        ART = getattr(server, 'ART_Brain_V7_Complete')
        LAST = getattr(server, 'LAST_CHECKPOINT_FILE')
        print('Loading ART checkpoint from', LAST)
        ck = torch.load(str(LAST), map_location=DEVICE)
        art_model = ART(dim=128, depth=6, vocab=2048).to(DEVICE)
        art_model.load_state_dict(ck['model_state_dict'])
        art_model.eval()
        print('PyTorch ART model loaded, will use it as feature extractor (logits averaged)')

    model = Capa3to5(feature_dim=VOCAB, hidden=HIDDEN).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Buffers for sequential statistics
    seq_buffer = deque(maxlen=SEQ_LEN)

    print('Start smoke training: epochs', EPOCHS)
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for step in range(STEPS_PER_EPOCH):
            # Use physics signals for better learning of dynamics
            X_batch, y_batch = generate_batch(BATCH_SIZE, physics=True)
            toks = to_tokens(X_batch)

            # get features via ONNX or PyTorch ART model
            if use_onnx:
                try:
                    logits = sess.run([out_name], {input_name: toks})[0]  # shape (batch,32,2048)
                    feats = logits.mean(axis=1).astype(np.float32)
                except Exception as e:
                    print('ONNX run failed during training step, error:', e)
                    raise
            else:
                # convert toks to torch tensor and feed to ART PyTorch model
                toks_t = torch.from_numpy(toks).long().to(DEVICE)
                with torch.no_grad():
                    logits = art_model(toks_t) if hasattr(art_model, '__call__') else art_model.forward(toks_t)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    feats = logits.detach().cpu().numpy().mean(axis=1).astype(np.float32)

            feats_t = torch.from_numpy(feats).to(DEVICE)  # [B, feature_dim]
            labels_t = torch.from_numpy(y_batch).to(DEVICE)

            model.train()
            optimizer.zero_grad()

            # Two forward passes for Knuth-Bendix confluence (different dropout realizations)
            logits1, anom1, dend1, cov1 = model(feats_t, return_all=True)
            logits2, anom2, dend2, cov2 = model(feats_t, return_all=True)

            # Primary anomaly loss (BCE with logits)
            bce_loss = criterion(logits1, labels_t)

            # Knuth-Bendix confluence loss (consistency under dropout)
            kb_loss = KB_WEIGHT * ( (logits1 - logits2).pow(2).mean() + (dend1 - dend2).pow(2).mean() )

            # Spectral decoupling on logits (penalize large logits)
            spec_loss = SPECTRAL_ALPHA * (logits1.pow(2).mean())

            # Seq-VCR: accumulate cov features across steps (time) and compute var/cov penalties
            seq_buffer.append(cov1.detach())  # cov1 shape [B, 32]
            cov_loss = torch.tensor(0.0, device=DEVICE)
            var_loss = torch.tensor(0.0, device=DEVICE)
            if len(seq_buffer) == SEQ_LEN:
                Xseq = torch.cat(list(seq_buffer), dim=0)  # [SEQ_LEN*B, 32]
                # variance per-dim
                var = Xseq.var(dim=0, unbiased=False)
                var_violation = (VCR_TARGET_VAR - var).clamp(min=0.0)
                var_loss = VCR_VAR_WEIGHT * (var_violation.pow(2).sum())

                # covariance matrix and off-diagonal penalty
                Xc = Xseq - Xseq.mean(dim=0, keepdim=True)
                N = Xseq.shape[0]
                cov = (Xc.t() @ Xc) / float(N)
                off_diag_norm = cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()
                cov_loss = VCR_COV_WEIGHT * off_diag_norm

            total_loss = bce_loss + kb_loss + spec_loss + var_loss + cov_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
        print(f'Epoch {epoch+1}/{EPOCHS} avg_loss: {epoch_loss/STEPS_PER_EPOCH:.6f}')

    # Quick validation on a small batch (use PyTorch ART model fallback if ONNX unavailable)
    Xv, yv = generate_batch(64)
    toks = to_tokens(Xv)
    try:
        if use_onnx:
            logits = sess.run([out_name], {input_name: toks})[0]
            feats = logits.mean(axis=1).astype(np.float32)
        else:
            toks_t = torch.from_numpy(toks).long().to(DEVICE)
            with torch.no_grad():
                logits = art_model(toks_t) if art_model is not None else None
                if isinstance(logits, tuple):
                    logits = logits[0]
                feats = logits.detach().cpu().numpy().mean(axis=1).astype(np.float32)
    except Exception as e:
        print('Validation: failed to run feature extractor, falling back to zeros:', e)
        feats = np.zeros((64, VOCAB), dtype=np.float32)

    feats_t = torch.from_numpy(feats).to(DEVICE)
    anom_pred, _ = model(feats_t)
    print('Validation anomaly mean pred:', anom_pred.mean().item())

    # Export trained capa3-5 to ONNX for later use (always attempt)
    model.eval()
    dummy = torch.randn(1, VOCAB, device=DEVICE)
    try:
        torch.onnx.export(model, dummy, str(EXPORT_ONNX), input_names=['features'], output_names=['anom','dend'], opset_version=12)
        print('Exported Capa3-5 ONNX to', EXPORT_ONNX)
    except Exception as e:
        print('Failed to export Capa3-5 ONNX:', e)

if __name__ == '__main__':
    main()
