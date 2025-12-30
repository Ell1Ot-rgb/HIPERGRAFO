# @title 🧠 OMEGA 21 v4.0 - SERVIDOR LOCAL (CPU OPTIMIZED)
# ESTE SERVIDOR CORRE EN TU VS CODE / PC LOCAL
# Optimizado para depuración y entrenamiento ligero sin GPU

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import uvicorn
import json
import numpy as np
from datetime import datetime
import traceback
import os
import psutil  # Para monitoreo de RAM

# ==========================================
# 1. DEFINICIÓN: Gated Multimodal Unit (GMU)
# ==========================================
class GMU(nn.Module):
    """Gated Multimodal Unit - Fusiona LSTM + Transformer con gating"""
    def __init__(self, lstm_dim: int, trans_dim: int, output_dim: int):
        super(GMU, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(lstm_dim + trans_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
        self.proj_lstm = nn.Linear(lstm_dim, output_dim) if lstm_dim != output_dim else None
        self.proj_trans = nn.Linear(trans_dim, output_dim) if trans_dim != output_dim else None
        
    def forward(self, lstm_out: torch.Tensor, trans_out: torch.Tensor) -> torch.Tensor:
        lstm_proj = self.proj_lstm(lstm_out) if self.proj_lstm else lstm_out
        trans_proj = self.proj_trans(trans_out) if self.proj_trans else trans_out
        gate_input = torch.cat([lstm_out, trans_out], dim=-1)
        gate = self.gate(gate_input)
        return lstm_proj * gate + trans_proj * (1 - gate)

# ==========================================
# 2. DEFINICIÓN: Capa 2 - Espacio-Temporal
# ==========================================
class Capa2EspacioTemporal(nn.Module):
    def __init__(self, input_dim: int = 1600, hidden_dim: int = 512):
        super(Capa2EspacioTemporal, self).__init__()
        # Reducimos dropout para convergencia más rápida en CPU
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1, # Reducido a 1 capa para velocidad en CPU
            dropout=0.0
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=4, # Reducido de 8 a 4 cabezas
            dim_feedforward=1024, # Reducido de 2048
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1) # Reducido a 1 capa
        self.gmu = GMU(hidden_dim * 2, input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]
        trans_out = self.transformer(x)
        trans_last = trans_out[:, -1, :]
        fused = self.gmu(lstm_last, trans_last)
        return fused, (h_n, c_n)

# ==========================================
# 3. DEFINICIÓN: Capa 3 - Asociativa Inferior
# ==========================================
class Capa3AsociativaInferior(nn.Module):
    def __init__(self, input_dim: int = 1600, output_dim: int = 512):
        super(Capa3AsociativaInferior, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024), # Reducido de 4096
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512), # Reducido de 2048
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        if self.skip:
            x_proj = self.skip(x)
            return out + x_proj * 0.1
        return out

# ==========================================
# 4. DEFINICIÓN: Capa 4 - Asociativa Superior
# ==========================================
class Capa4AsociativaSuper(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        super(Capa4AsociativaSuper, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 512), # Reducido
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.norm(attn_out + x)
        ffn_out = self.ffn(attn_out)
        out = self.norm(ffn_out + attn_out)
        return out[:, -1, :] if out.size(1) > 1 else out.squeeze(1)

# ==========================================
# 5. DEFINICIÓN: Capa 5 - Ejecutiva
# ==========================================
class Capa5Ejecutiva(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        super(Capa5Ejecutiva, self).__init__()
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.dendrite_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.Tanh()
        )
        self.coherence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.anomaly_head(x), self.dendrite_head(x), self.coherence_head(x)

# ==========================================
# 6. MODELO COMPLETO
# ==========================================
class CortezaCognitivaV4(nn.Module):
    def __init__(self, input_dim: int = 1600, hidden_dim: int = 512):
        super(CortezaCognitivaV4, self).__init__()
        self.capa2 = Capa2EspacioTemporal(input_dim, hidden_dim)
        self.capa3 = Capa3AsociativaInferior(input_dim, hidden_dim)
        self.capa4 = Capa4AsociativaSuper(hidden_dim)
        self.capa5 = Capa5Ejecutiva(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        c2_out, lstm_state = self.capa2(x)
        c3_out = self.capa3(c2_out)
        c4_out = self.capa4(c3_out)
        anomaly, dendrites, coherence = self.capa5(c4_out)
        return {
            'anomaly': anomaly,
            'dendrites': dendrites,
            'coherence': coherence,
            'capa2_out': c2_out,
            'capa3_out': c3_out,
            'capa4_out': c4_out
        }

# ==========================================
# 7. CONFIGURACIÓN DEL SERVIDOR LOCAL
# ==========================================

# Forzamos CPU para evitar errores si no hay CUDA configurado
device = torch.device('cpu')
# Optimización de hilos para 2 cores (evita latencia por sobre-suscripción)
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
print(f"💻 MODO LOCAL: Usando {device} con {torch.get_num_threads()} hilos")

# Instanciamos el modelo (versión ligera)
model = CortezaCognitivaV4(input_dim=1600, hidden_dim=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion_anomaly = nn.BCELoss()

app = FastAPI(
    title="OMEGA 21 v4.0 - SERVIDOR LOCAL",
    description="Versión optimizada para CPU corriendo en VS Code",
    version="4.0-LOCAL"
)

# ==========================================
# 8. MODELOS PYDANTIC
# ==========================================

class MuestraEntrenamiento(BaseModel):
    input_data: List[float]
    anomaly_label: int

class LoteEntrenamiento(BaseModel):
    samples: List[MuestraEntrenamiento]
    epochs: int = 1

# ==========================================
# 9. ENDPOINTS
# ==========================================

@app.post("/train_layer2")
async def train_layer2(lote: LoteEntrenamiento):
    try:
        # Verificar memoria disponible
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            return {"status": "error", "error": "Memoria RAM crítica (>90%)"}

        inputs = torch.tensor([s.input_data for s in lote.samples], dtype=torch.float).to(device).unsqueeze(1)
        labels = torch.tensor([s.anomaly_label for s in lote.samples], dtype=torch.float).view(-1, 1).to(device)
        
        model.train()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion_anomaly(outputs['anomaly'], labels)
        loss.backward()
        optimizer.step()
        
        return {
            "status": "trained",
            "loss": float(loss.item()),
            "device": "cpu",
            "outputs": {
                "anomaly_prob": float(outputs['anomaly'].mean().item()),
                "dendrite_adjustments": outputs['dendrites'][0].detach().tolist()
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "trace": traceback.format_exc()}

@app.get("/status")
async def get_status():
    mem = psutil.virtual_memory()
    return {
        "status": "online",
        "mode": "LOCAL_CPU",
        "ram_usage_percent": mem.percent,
        "ram_available_gb": mem.available / (1024**3)
    }

@app.get("/health")
async def health():
    return {"alive": True, "mode": "local"}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏠 SERVIDOR LOCAL OMEGA 21 - INICIANDO")
    print("="*60)
    print(f"   • URL: http://localhost:8000")
    print(f"   • CPU: {psutil.cpu_count()} cores")
    print(f"   • RAM Disponible: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
