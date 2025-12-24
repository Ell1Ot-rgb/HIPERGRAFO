"""
COLAB CAPA 2 - VERSIÓN CORREGIDA
================================
Archivo único y sin duplicaciones que ejecutar en Google Colab
Todos los endpoints están correctamente registrados

Copiar TODO este código a una celda de Colab y ejecutar
"""

# ============================================================================
# INSTALACIONES
# ============================================================================

import subprocess
import sys

def install_packages():
    packages = [
        'torch',
        'fastapi',
        'uvicorn',
        'pyngrok',
        'einops',
        'numpy'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✓ {package} installed")

install_packages()

# ============================================================================
# IMPORTS
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import threading
import uvicorn
from pyngrok import ngrok, conf
import time
import asyncio
import json
from datetime import datetime
from einops import rearrange

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📦 Device: {DEVICE}")

# Hyperparameters
input_dim = 20
seq_length = 100
d_model = 128
nhead = 4
num_layers = 2
dim_feedforward = 256
dropout = 0.1
learning_rate = 0.001
batch_size = 4

# Server
HOST = "0.0.0.0"
PORT = 8000

# ngrok (IMPORTANT: Replace with your actual token)
NGROK_AUTH_TOKEN = 'cr_37DMLjt1GZQOC3fWbGpWMgDvsip'  # Update this with your token

# ============================================================================
# ARQUITECTURA NEURAL - COMPONENTES
# ============================================================================

class InputAdapter(nn.Module):
    """Adapta entrada a dimensión del modelo"""
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)
    
    def forward(self, x):
        return self.linear(x)


class BiLSTMStateful(nn.Module):
    """Bi-LSTM con manejo explícito de estados"""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    
    def forward(self, x, h_0=None, c_0=None):
        if h_0 is None or c_0 is None:
            output, (h_n, c_n) = self.lstm(x)
        else:
            output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        return output, h_n, c_n


class TransformerEncoder(nn.Module):
    """Encoder Transformer"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, src):
        return self.transformer_encoder(src)


class GMUFusion(nn.Module):
    """Gated Multimodal Unit para fusión"""
    def __init__(self, d_model):
        super().__init__()
        self.linear_z_x = nn.Linear(d_model, d_model)
        self.linear_z_y = nn.Linear(d_model, d_model)
        self.linear_r_x = nn.Linear(d_model, d_model)
        self.linear_r_y = nn.Linear(d_model, d_model)
        self.linear_h_x = nn.Linear(d_model, d_model)
        self.linear_h_y = nn.Linear(d_model, d_model)
        self.bn_z = nn.BatchNorm1d(d_model)
        self.bn_r = nn.BatchNorm1d(d_model)
        self.bn_h = nn.BatchNorm1d(d_model)
    
    def forward(self, x, y):
        x_flat = rearrange(x, 'b s d -> (b s) d')
        y_flat = rearrange(y, 'b s d -> (b s) d')
        
        z = torch.sigmoid(self.bn_z(self.linear_z_x(x_flat) + self.linear_z_y(y_flat)))
        r = torch.sigmoid(self.bn_r(self.linear_r_x(x_flat) + self.linear_r_y(y_flat)))
        h = torch.tanh(self.bn_h(self.linear_h_x(x_flat) + self.linear_h_y(r * y_flat)))
        
        fused_output_flat = (1 - z) * x_flat + z * h
        fused_output = rearrange(fused_output_flat, '(b s) d -> b s d', b=x.shape[0])
        
        return fused_output


class Heads(nn.Module):
    """Output heads: reconstrucción y anomalía"""
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.reconstruction_head = nn.Linear(d_model, output_dim)
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        reconstruction = self.reconstruction_head(features)
        anomaly = self.anomaly_head(features)
        return reconstruction, anomaly


class HybridCognitiveLayer2(nn.Module):
    """Capa 2 Completa: LSTM + Transformer + Fusion + Heads"""
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        
        self.input_adapter = InputAdapter(input_dim, d_model)
        self.bilstm = BiLSTMStateful(d_model, d_model // 2, 2, dropout)
        self.transformer = TransformerEncoder(d_model, nhead, dim_feedforward, dropout, num_layers)
        self.gmu = GMUFusion(d_model)
        self.heads = Heads(d_model, input_dim)
    
    def forward(self, x, h_0=None, c_0=None):
        # Input: (batch, seq_len, input_dim)
        x = self.input_adapter(x)  # (batch, seq_len, d_model)
        
        lstm_out, h_n, c_n = self.bilstm(x, h_0, c_0)  # (batch, seq_len, d_model)
        
        trans_out = self.transformer(lstm_out)  # (batch, seq_len, d_model)
        
        fused = self.gmu(lstm_out, trans_out)  # (batch, seq_len, d_model)
        
        reconstruction, anomaly = self.heads(fused)  # (batch, seq_len, input_dim), (batch, seq_len, 1)
        
        return reconstruction, anomaly, h_n, c_n


print("✓ Componentes de modelo definidos")

# ============================================================================
# INICIALIZAR MODELO
# ============================================================================

model = HybridCognitiveLayer2(
    input_dim=input_dim,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

print(f"✓ Modelo HybridCognitiveLayer2 inicializado en {DEVICE}")
print(f"  Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# ESTADÍSTICAS
# ============================================================================

class EstadisticasServidor:
    def __init__(self):
        self.samples_trained = 0
        self.total_loss = 0.0
        self.batches_processed = 0
        self.start_time = datetime.now()
    
    def add_batch(self, loss, batch_size):
        self.samples_trained += batch_size
        self.total_loss += float(loss)
        self.batches_processed += 1
    
    def get_average_loss(self):
        if self.batches_processed == 0:
            return 0.0
        return self.total_loss / self.batches_processed
    
    def get_uptime(self):
        return (datetime.now() - self.start_time).total_seconds()

stats = EstadisticasServidor()

print("✓ Estadísticas inicializadas")

# ============================================================================
# DEFINIR FASTAPI
# ============================================================================

app = FastAPI(
    title="OMEGA 21 - Capa 2",
    description="Corteza Cognitiva Distribuida - Entrenamiento en Colab",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✓ FastAPI aplicación creada")

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class TrainRequest(BaseModel):
    x_train: List[List[List[float]]]
    y_reconstruction: List[List[List[float]]]
    y_anomaly: Optional[List[List[List[float]]]] = None
    learning_rate: Optional[float] = learning_rate
    epochs: Optional[int] = 1

class PredictRequest(BaseModel):
    x: List[List[List[float]]]

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raíz - confirma que el servidor está vivo"""
    return {
        "status": "online",
        "service": "OMEGA 21 - Capa 2",
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": True,
        "samples_trained": stats.samples_trained,
        "batches_processed": stats.batches_processed
    }


@app.get("/status")
async def status():
    """Status completo del servidor"""
    return {
        "status": "operational",
        "samples_trained": stats.samples_trained,
        "batches_processed": stats.batches_processed,
        "average_loss": round(stats.get_average_loss(), 6),
        "device": DEVICE,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "uptime_seconds": round(stats.get_uptime(), 2),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/info")
async def info():
    """Información del modelo y arquitectura"""
    return {
        "service": "OMEGA 21 - Capa 2 (Corteza Cognitiva)",
        "version": "1.0.0",
        "architecture": {
            "input_dim": input_dim,
            "seq_length": seq_length,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout
        },
        "components": [
            "InputAdapter",
            "BiLSTMStateful (2 layers)",
            "TransformerEncoder (multi-head attention)",
            "GMUFusion (gated multimodal unit)",
            "Heads (reconstruction + anomaly)"
        ],
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": DEVICE,
        "created_at": datetime.now().isoformat()
    }


@app.post("/train_layer2")
async def train_layer2(request: TrainRequest):
    """Entrenar la Capa 2"""
    try:
        # Convertir a tensores
        x_train = torch.tensor(request.x_train, dtype=torch.float32).to(DEVICE)
        y_recon = torch.tensor(request.y_reconstruction, dtype=torch.float32).to(DEVICE)
        
        batch_size_actual = x_train.shape[0]
        
        # Entrenar
        model.train()
        total_loss = 0.0
        
        for epoch in range(request.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            recon_pred, anomaly_pred, _, _ = model(x_train)
            
            # Loss
            loss_recon = criterion(recon_pred, y_recon)
            loss = loss_recon
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += float(loss.item())
        
        avg_loss = total_loss / request.epochs
        stats.add_batch(avg_loss, batch_size_actual)
        
        return {
            "status": "success",
            "message": "Training completed",
            "loss": round(avg_loss, 6),
            "samples_trained": stats.samples_trained,
            "batches_processed": stats.batches_processed,
            "average_loss_overall": round(stats.get_average_loss(), 6),
            "device": DEVICE,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.post("/predict")
async def predict(request: PredictRequest):
    """Realizar predicción"""
    try:
        x = torch.tensor(request.x, dtype=torch.float32).to(DEVICE)
        
        model.eval()
        with torch.no_grad():
            recon, anomaly, _, _ = model(x)
        
        return {
            "status": "success",
            "reconstruction": recon.cpu().numpy().tolist(),
            "anomaly_probability": anomaly.cpu().numpy().tolist(),
            "shape_reconstruction": list(recon.shape),
            "shape_anomaly": list(anomaly.shape),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/diagnostico")
async def diagnostico():
    """Diagnóstico completo del sistema"""
    return {
        "service": "OMEGA 21 - Capa 2",
        "status": "operational",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_training": model.training,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "statistics": {
            "samples_trained": stats.samples_trained,
            "batches_processed": stats.batches_processed,
            "average_loss": round(stats.get_average_loss(), 6),
            "uptime_seconds": round(stats.get_uptime(), 2)
        },
        "endpoints_available": [
            "GET /",
            "GET /health",
            "GET /status",
            "GET /info",
            "POST /train_layer2",
            "POST /predict",
            "GET /diagnostico",
            "GET /docs"
        ],
        "timestamp": datetime.now().isoformat()
    }


print("✓ Todos los endpoints registrados")

# ============================================================================
# EJECUTAR SERVIDOR
# ============================================================================

def run_server():
    print("\n" + "="*80)
    print("🚀 INICIANDO SERVIDOR FASTAPI")
    print("="*80)
    
    try:
        # Configurar ngrok
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        print(f"✓ ngrok token configurado")
        
        # Cerrar tunnels previos
        ngrok.kill()
        print(f"✓ Tunnels previos cerrados")
        
        # Crear tunnel
        public_url = ngrok.connect(PORT)
        print(f"✓ ngrok tunnel establecido: {public_url}")
        
        # Iniciar FastAPI en thread
        def run_fastapi():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            uvicorn.run(app, host=HOST, port=PORT, log_level="info")
        
        api_thread = threading.Thread(target=run_fastapi, daemon=True)
        api_thread.start()
        print(f"✓ FastAPI iniciado en {HOST}:{PORT}")
        
        print("\n" + "="*80)
        print("✅ SERVIDOR LISTO")
        print("="*80)
        print(f"📍 URL pública: {public_url}")
        print(f"📍 Documentación: {public_url}/docs")
        print("="*80 + "\n")
        
        # Mantener vivo
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            print("\n⛔ Servidor detenido")
            ngrok.kill()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

# EJECUTAR
if __name__ == "__main__" or True:  # True para ejecutar en Colab
    run_server()
