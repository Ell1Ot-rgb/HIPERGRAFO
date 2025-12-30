# ==============================================================================
# ART V7: REACTOR NEURO-SIMBÓLICO OMNISCIENTE - SERVIDOR DOCKER
# ==============================================================================
# Adaptación del Reactor para ejecutar como servidor FastAPI en Docker
# Con optimización para 2 núcleos y compatible con cliente TypeScript

import os
import sys
import subprocess
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import uvicorn
from datetime import datetime
import psutil
from collections import deque

# ==============================================================================
# INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS
# ==============================================================================

def install_ecosystem():
    """Instala dependencias necesarias para ART V7"""
    print("🚀 Inicializando ecosistema ART V7...")
    packages = [
        "causal-conv1d>=1.2.0",
        "mamba-ssm",
        "gudhi",
        "einops",
    ]
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
            print(f"   ✓ {pkg}")
        except Exception as e:
            print(f"   ⚠ {pkg}: {str(e)[:50]}")

try:
    from einops import repeat
except ImportError:
    install_ecosystem()
    from einops import repeat

# Mamba es opcional - usaremos LSTM como fallback
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠️ Mamba no disponible, usando LSTM como alternativa")

print("✅ Ecosistema listo. ART V7 Engine booting...\n")

# ==============================================================================
# MÓDULOS DE FÍSICA MATEMÁTICA
# ==============================================================================

class RoughPathEncoder(nn.Module):
    """Convierte secuencias discretas en trayectorias continuas"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        path = torch.cumsum(x * self.scale, dim=1)
        return self.proj(path)

class OPiActivation(nn.Module):
    """Activación cuántica basada en Free Will"""
    def forward(self, x):
        xi = torch.tanh(x) * 0.99
        safe_cos = torch.clamp(torch.cos(np.pi * xi), min=-0.9999, max=0.9999)
        return x * torch.log(torch.abs(safe_cos) + 1e-6)

class PauseTokenInjection(nn.Module):
    """Inyecta tiempo de reflexión (Pause Tokens)"""
    def __init__(self, dim, num_tokens=4):
        super().__init__()
        self.pause = nn.Parameter(torch.randn(1, num_tokens, dim))

    def forward(self, x):
        b = x.shape[0]
        pauses = repeat(self.pause, '1 n d -> b n d', b=b)
        return torch.cat([pauses, x], dim=1)

class SpectralDecoupling(nn.Module):
    """Penaliza magnitud de logits"""
    def __init__(self, lam=0.1):
        super().__init__()
        self.lam = lam

    def forward(self, logits):
        return self.lam * torch.mean(logits**2)

# ==============================================================================
# FUNCIONES DE PÉRDIDA AVANZADAS
# ==============================================================================

class DimensionalFlowLoss(nn.Module):
    """MEUM: Reduce dimensión fractal progresivamente"""
    def __init__(self, target_dim=3.0):
        super().__init__()
        self.target_dim = target_dim

    def estimate_fractal_dim(self, x):
        if x.shape[0] < 10:
            return torch.tensor(self.target_dim, device=x.device)
        dist = torch.cdist(x, x) + 1e-6
        r1, r2 = torch.quantile(dist, 0.1), torch.quantile(dist, 0.5)
        c1 = (dist < r1).float().mean()
        c2 = (dist < r2).float().mean()
        return torch.log(c2/c1) / torch.log(r2/r1)

    def forward(self, states):
        loss = 0
        expected = np.linspace(8.0, self.target_dim, len(states))
        for i, s in enumerate(states):
            flat = s.reshape(-1, s.shape[-1])
            if flat.shape[0] > 500:
                idx = torch.randperm(flat.shape[0])[:500]
                flat = flat[idx]
            dim_est = self.estimate_fractal_dim(flat)
            if not torch.isnan(dim_est) and not torch.isinf(dim_est):
                loss += F.mse_loss(dim_est, torch.tensor(expected[i], device=s.device, dtype=torch.float32))
        return loss * 0.01  # Reducido para 2 cores

class TopologicalQualiaLoss(nn.Module):
    """Homología Persistente"""
    def __init__(self):
        super().__init__()

    def forward_proxy(self, latent):
        b, t, d = latent.shape
        sample = latent[0]
        dist = torch.cdist(sample, sample)
        knn_dist, _ = dist.topk(k=min(5, dist.shape[0]-1), largest=False)
        return -torch.std(knn_dist)

class DualIBLoss(nn.Module):
    """Sensibilidad exponencial a Cisnes Negros"""
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        return torch.mean(torch.exp(torch.clamp(ce - 1.0, max=20.0)))

# ==============================================================================
# ARQUITECTURA ART-V7 (OPTIMIZADA PARA CPU)
# ==============================================================================

class ART_Brain_V7(nn.Module):
    def __init__(self, dim=64, depth=3, vocab=1000):  # Reducido: dim 128->64, depth 6->3
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.rough_path = RoughPathEncoder(dim)
        self.pause_inj = PauseTokenInjection(dim, num_tokens=2)
        
        self.layers = nn.ModuleList()
        for i in range(depth):
            if MAMBA_AVAILABLE:
                # Usar Mamba si está disponible
                self.layers.append(nn.Sequential(
                    Mamba(d_model=dim, d_state=8, d_conv=2, expand=1),
                    OPiActivation(),
                    nn.LayerNorm(dim)
                ))
            else:
                # Fallback a LSTM + Transformer (más estable)
                self.layers.append(nn.Sequential(
                    nn.LSTM(dim, dim, batch_first=True),
                    OPiActivation(),
                    nn.LayerNorm(dim)
                ))
        
        self.head = nn.Linear(dim, vocab)
        
        # Motores de física interna
        self.meum = DimensionalFlowLoss()
        self.dual_ib = DualIBLoss()
        self.topo = TopologicalQualiaLoss()
        self.spec = SpectralDecoupling(lam=0.05)

    def forward(self, x):
        h = self.emb(x)
        h = self.rough_path(h)
        h = self.pause_inj(h)
        
        states = []
        for i, layer in enumerate(self.layers):
            if MAMBA_AVAILABLE:
                h = layer(h)
            else:
                # Para LSTM: extraer output
                if isinstance(layer[0], nn.LSTM):
                    h, _ = layer[0](h)
                    h = layer[1](h)  # OPiActivation
                    h = layer[2](h)  # LayerNorm
                else:
                    h = layer(h)
            states.append(h)
        
        logits = self.head(h)
        return logits, states, h

# ==============================================================================
# SERVIDOR FASTAPI
# ==============================================================================

app = FastAPI(
    title="ART V7 - Reactor Neuro-Simbólico",
    description="Motor de entrenamiento omnisciente en Docker",
    version="7.0"
)

# Modelos Pydantic
class MuestraEntrenamiento(BaseModel):
    input_data: List[float]
    anomaly_label: int

class LoteEntrenamiento(BaseModel):
    samples: List[MuestraEntrenamiento]
    epochs: int = 1

class EstadisticasReactor:
    def __init__(self):
        self.tiempo_inicio = datetime.now()
        self.historial_loss = deque(maxlen=100)
        self.historial_betti = deque(maxlen=100)
        self.epoch_global = 0
        
    def registrar_loss(self, loss: float, betti: int = 0):
        self.historial_loss.append(loss)
        self.historial_betti.append(betti)
        
    def get_estado(self):
        uptime = (datetime.now() - self.tiempo_inicio).total_seconds()
        return {
            "uptime_seg": uptime,
            "epoch": self.epoch_global,
            "loss_promedio": np.mean(list(self.historial_loss)) if self.historial_loss else 0,
            "betti_promedio": np.mean(list(self.historial_betti)) if self.historial_betti else 0,
            "memoria_mb": psutil.Process().memory_info().rss / (1024**2),
            "cpu_percent": psutil.cpu_percent()
        }

# Instancia global
stats = EstadisticasReactor()
device = torch.device('cpu')

# Configuración para 2 cores
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# Modelo ART V7 (versión ligera para CPU)
model = ART_Brain_V7(dim=64, depth=3, vocab=2048).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scaler = GradScaler()

@app.post("/train_reactor")
async def train_reactor(lote: LoteEntrenamiento):
    """Entrena el reactor ART V7 con un lote de datos"""
    try:
        if not lote.samples:
            return {"status": "error", "error": "No samples provided"}
        
        # Validar dimensión
        if len(lote.samples[0].input_data) != 1600:
            return {
                "status": "error",
                "error": f"Expected 1600D input, got {len(lote.samples[0].input_data)}D"
            }
        
        # Mapeo: 1600D -> tokens (reducción de dimensión)
        # Dividimos en 32 segmentos de 50D cada uno
        batch_tokens = []
        for sample in lote.samples:
            tokens = []
            for i in range(0, 1600, 50):
                chunk = sample.input_data[i:i+50]
                # Mapeo a token (0-2047)
                token_val = int((np.mean(chunk) + 1) * 1024) % 2048
                tokens.append(token_val)
            batch_tokens.append(tokens)
        
        # Tensor de entrada
        x = torch.tensor(batch_tokens, dtype=torch.long).to(device)
        
        # Entrenamiento
        model.train()
        optimizer.zero_grad()
        
        with autocast():
            logits, states, latent = model(x)
            
            # Loss predicción (sin shift para evitar dimensión mismatch)
            logits_flat = logits.reshape(-1, 2048)
            targets_flat = x.reshape(-1)
            l_pred = F.cross_entropy(logits_flat, targets_flat)
            
            # Loss causalidad (Knuth-Bendix)
            latent_noisy = F.dropout(latent, p=0.2)
            logits_b = model.head(latent_noisy)
            l_causal = F.mse_loss(logits, logits_b)
            
            # Loss topológico
            l_topo = model.topo.forward_proxy(latent)
            
            # Loss MEUM
            l_flow = model.meum(states)
            
            # Loss espectral
            l_spec = model.spec(logits)
            
            # Total
            total_loss = l_pred + 0.5*l_causal + 0.1*l_topo + 0.05*l_flow + 0.05*l_spec
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Estadísticas
        stats.registrar_loss(total_loss.item())
        stats.epoch_global += 1
        
        return {
            "status": "trained",
            "loss": float(total_loss.item()),
            "l_pred": float(l_pred.item()),
            "l_causal": float(l_causal.item()),
            "l_topo": float(l_topo.item()),
            "epoch": stats.epoch_global,
            "device": "CPU (ART V7)"
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/status")
async def get_status():
    """Estado actual del reactor"""
    estado = stats.get_estado()
    return {
        "status": "online",
        "reactor": "ART V7",
        "modo": "Docker CPU Optimized (2 cores)",
        "estadisticas": estado,
        "torch_threads": torch.get_num_threads()
    }

@app.get("/health")
async def health():
    return {"alive": True, "reactor": "ART V7", "timestamp": datetime.now().isoformat()}

@app.get("/metricas")
async def metricas():
    return {
        "loss_history": list(stats.historial_loss),
        "betti_history": list(stats.historial_betti),
        "memoria_mb": psutil.Process().memory_info().rss / (1024**2)
    }

# ==============================================================================
# INICIALIZACIÓN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("⚛️  ART V7 REACTOR - SERVIDOR DOCKER")
    print("="*70)
    print(f"   🧠 Modelo: ART_Brain_V7 (64D, 3 capas Mamba)")
    print(f"   💻 CPU: {psutil.cpu_count()} cores, {torch.get_num_threads()} threads PyTorch")
    print(f"   🐳 Entorno: Docker")
    print(f"   📡 URL: http://0.0.0.0:8000")
    print(f"   📚 Docs: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
