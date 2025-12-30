# @title ⚛️ ART V7: REACTOR NEURO-SIMBÓLICO - VERSIÓN ESTABLE
# Servidor de entrenamiento optimizado para Docker (CPU, 2 cores)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
from datetime import datetime
import psutil
from collections import deque
import numpy as np

# ==============================================================================
# MÓDULOS CORE (Simplificados)
# ==============================================================================

class RoughPathEncoder(nn.Module):
    """Trayectoria continua de secuencia discreta"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        path = torch.cumsum(x * self.scale, dim=1)
        return self.proj(path)

class OPiActivation(nn.Module):
    """Activación cuántica"""
    def forward(self, x):
        xi = torch.tanh(x) * 0.99
        safe_cos = torch.clamp(torch.cos(np.pi * xi), min=-0.9999, max=0.9999)
        return x * torch.log(torch.abs(safe_cos) + 1e-6)

class SpectralDecoupling(nn.Module):
    """Anti-memorización"""
    def __init__(self, lam=0.1):
        super().__init__()
        self.lam = lam

    def forward(self, logits):
        return self.lam * torch.mean(logits**2)

# ==============================================================================
# FUNCIONES DE PÉRDIDA
# ==============================================================================

class DualIBLoss(nn.Module):
    """Sensibilidad exponencial a outliers"""
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        return torch.mean(torch.exp(torch.clamp(ce - 1.0, max=20.0)))

class TopologicalQualiaLoss(nn.Module):
    """Proxy de homología"""
    def forward_proxy(self, latent):
        if latent.shape[0] < 2:
            return torch.tensor(0.0, device=latent.device)
        b, t, d = latent.shape
        sample = latent[0]
        dist = torch.cdist(sample, sample)
        knn_dist, _ = dist.topk(k=min(5, dist.shape[0]-1), largest=False)
        return -torch.std(knn_dist)

# ==============================================================================
# MODELO ART V7 (VERSIÓN ESTABLE)
# ==============================================================================

class ART_Brain_V7_Stable(nn.Module):
    def __init__(self, dim=64, vocab=2048):
        super().__init__()
        # Input: tokens (0-2047)
        self.emb = nn.Embedding(vocab, dim)
        self.rough_path = RoughPathEncoder(dim)
        
        # 3 capas LSTM (más estable que Mamba en CPU)
        self.lstm1 = nn.LSTM(dim, dim, batch_first=True)
        self.lstm2 = nn.LSTM(dim, dim, batch_first=True)
        self.lstm3 = nn.LSTM(dim, dim, batch_first=True)
        
        self.opi = OPiActivation()
        self.norm = nn.LayerNorm(dim)
        
        # Output head
        self.head = nn.Linear(dim, vocab)
        
        # Pérdidas
        self.dual_ib = DualIBLoss()
        self.topo = TopologicalQualiaLoss()
        self.spec = SpectralDecoupling(lam=0.05)

    def forward(self, x):
        # x: [batch, seq_len] de tokens
        h = self.emb(x)  # [batch, seq_len, 64]
        h = self.rough_path(h)
        
        states = []
        
        # LSTM layers
        h1, _ = self.lstm1(h)
        h1 = self.opi(h1)
        h1 = self.norm(h1)
        states.append(h1)
        
        h2, _ = self.lstm2(h1)
        h2 = self.opi(h2)
        h2 = self.norm(h2)
        states.append(h2)
        
        h3, _ = self.lstm3(h2)
        h3 = self.opi(h3)
        h3 = self.norm(h3)
        states.append(h3)
        
        # Output
        logits = self.head(h3)  # [batch, seq_len, 2048]
        
        return logits, states, h3

# ==============================================================================
# FASTAPI SERVER
# ==============================================================================

app = FastAPI(title="ART V7 Reactor", version="7.0")

class MuestraEntrenamiento(BaseModel):
    input_data: List[float]
    anomaly_label: int

class LoteEntrenamiento(BaseModel):
    samples: List[MuestraEntrenamiento]
    epochs: int = 1

class Estadisticas:
    def __init__(self):
        self.tiempo_inicio = datetime.now()
        self.historial_loss = deque(maxlen=100)
        self.epoch = 0
        
    def registrar(self, loss):
        self.historial_loss.append(loss)
        self.epoch += 1
        
    def get_estado(self):
        return {
            "uptime_seg": (datetime.now() - self.tiempo_inicio).total_seconds(),
            "epoch": self.epoch,
            "loss_promedio": np.mean(list(self.historial_loss)) if self.historial_loss else 0,
            "memoria_mb": psutil.Process().memory_info().rss / (1024**2),
            "cpu_percent": psutil.cpu_percent()
        }

stats = Estadisticas()
device = torch.device('cpu')
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# Modelo
model = ART_Brain_V7_Stable(dim=64, vocab=2048).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

@app.post("/train_reactor")
async def train_reactor(lote: LoteEntrenamiento):
    """Entrenar el reactor"""
    try:
        if not lote.samples or len(lote.samples[0].input_data) != 1600:
            return {"status": "error", "error": "Invalid input"}
        
        # Mapeo 1600D → 32 tokens
        batch_tokens = []
        for sample in lote.samples:
            tokens = []
            for i in range(0, 1600, 50):
                chunk = sample.input_data[i:i+50]
                token_val = int((np.mean(chunk) + 1) * 1024) % 2048
                tokens.append(token_val)
            batch_tokens.append(tokens)
        
        x = torch.tensor(batch_tokens, dtype=torch.long).to(device)
        
        model.train()
        optimizer.zero_grad()
        
        with autocast():
            logits, states, latent = model(x)
            
            # Loss simple: cross entropy
            logits_flat = logits.reshape(-1, 2048)
            targets_flat = x.reshape(-1)
            
            # Usar targets desplazados para predicción siguiente token
            targets_flat = torch.roll(targets_flat, -1)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
            # Agregar regularización
            loss = loss + model.spec(logits) * 0.1
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        stats.registrar(loss.item())
        
        return {
            "status": "trained",
            "loss": float(loss.item()),
            "epoch": stats.epoch,
            "device": "CPU (2 cores)"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/status")
async def get_status():
    return {
        "status": "online",
        "reactor": "ART V7",
        "estadisticas": stats.get_estado()
    }

@app.get("/health")
async def health():
    return {"alive": True, "reactor": "ART V7", "timestamp": datetime.now().isoformat()}

@app.get("/metricas")
async def metricas():
    return {
        "loss_history": list(stats.historial_loss),
        "memoria_mb": psutil.Process().memory_info().rss / (1024**2)
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("⚛️  ART V7 REACTOR - VERSIÓN ESTABLE (LSTM)")
    print("="*70)
    print(f"   📡 http://0.0.0.0:8000")
    print(f"   💻 CPU: {torch.get_num_threads()} threads")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
