"""
CAPA 2 - SERVIDOR FASTAPI COMPLETO Y FUNCIONAL
Script para Google Colab - Entrena modelo HybridCognitiveLayer2

INSTRUCCIONES:
1. Copia ESTE CÓDIGO completo (no el cuadernocolab.py)
2. Pega en Google Colab
3. Ejecuta la celda
4. Espera el mensaje "ngrok tunnel established"
5. Usa el script validar_capa2_v2.py para probar

NOTA: Este archivo corrige los problemas del cuadernocolab.py original
que tenía múltiples instancias de FastAPI desorganizadas.
"""

# ============================================================================
# FASE 1: INSTALACIONES Y IMPORTS
# ============================================================================

import sys
print("📦 Instalando dependencias...")

!pip install -q fastapi uvicorn pyngrok torch einops onnx onnxruntime onnxscript

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import numpy as np
import onnx
import onnxruntime as ort
from einops import rearrange

from datetime import datetime
import os
import json

print("✅ Dependencias instaladas\n")

# ============================================================================
# FASE 2: CONFIGURACIÓN GLOBAL
# ============================================================================

# Detectar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Dispositivo: {device}")

# Hiperparámetros del modelo
input_dim = 20
sequence_length = 100
hidden_dim_half = 64
output_dim = 20
anomaly_head_dim = 1
d_model = hidden_dim_half * 2
lstm_hidden_dim = hidden_dim_half
num_lstm_layers = 2
lstm_dropout = 0.1
nhead = 4
dim_feedforward = 512
transformer_dropout = 0.1
num_transformer_layers = 2

# Configuración de entrenamiento
CHECKPOINT_DIR = '/content/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DAT_EPOCH_THRESHOLD = 10  # Delayed Attention Training

# ngrok
NGROK_AUTH_TOKEN = 'YOUR_VALID_NGROK_AUTH_TOKEN_HERE'  # ⚠️ REEMPLAZAR CON TOKEN REAL

print("✅ Configuración global inicializada\n")

# ============================================================================
# FASE 3: DEFINIR COMPONENTES DEL MODELO
# ============================================================================

class InputAdapter(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.linear(x)


class BiLSTMStateful(nn.Module):
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

    def forward(self, x, h_0, c_0):
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        return output, h_n, c_n


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        return self.transformer_encoder(src)


class GMUFusion(nn.Module):
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
    def __init__(self, d_model, output_dim, anomaly_head_dim):
        super().__init__()
        self.reconstruction_head = nn.Linear(d_model, output_dim)
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, anomaly_head_dim),
            nn.Sigmoid()
        )

    def forward(self, features):
        reconstruction_output = self.reconstruction_head(features)
        anomaly_output = self.anomaly_head(features)
        return reconstruction_output, anomaly_output


class HybridCognitiveLayer2(nn.Module):
    def __init__(self, input_dim, d_model, lstm_hidden_dim, num_lstm_layers,
                 lstm_dropout, nhead, dim_feedforward, transformer_dropout,
                 num_transformer_layers, output_dim, anomaly_head_dim):
        super().__init__()
        self.input_adapter = InputAdapter(input_dim, d_model)
        self.bilstm = BiLSTMStateful(
            input_size=d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout
        )
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            num_layers=num_transformer_layers
        )
        self.gmu_fusion = GMUFusion(d_model=d_model)
        self.heads = Heads(d_model=d_model, output_dim=output_dim, anomaly_head_dim=anomaly_head_dim)

    def forward(self, x, h_0, c_0):
        x_adapted = self.input_adapter(x)
        lstm_output, h_n_out, c_n_out = self.bilstm(x_adapted, h_0, c_0)
        transformer_output = self.transformer_encoder(lstm_output)
        fused_output = self.gmu_fusion(lstm_output, transformer_output)
        reconstruction_output, anomaly_output = self.heads(fused_output)
        return reconstruction_output, anomaly_output, h_n_out, c_n_out


print("✅ Componentes del modelo definidos\n")

# ============================================================================
# FASE 4: INICIALIZAR MODELO Y OPTIMIZADOR
# ============================================================================

model = HybridCognitiveLayer2(
    input_dim=input_dim,
    d_model=d_model,
    lstm_hidden_dim=lstm_hidden_dim,
    num_lstm_layers=num_lstm_layers,
    lstm_dropout=lstm_dropout,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    transformer_dropout=transformer_dropout,
    num_transformer_layers=num_transformer_layers,
    output_dim=output_dim,
    anomaly_head_dim=anomaly_head_dim
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001)

print(f"✅ Modelo inicializado")
print(f"   Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Dispositivo: {device}\n")

# ============================================================================
# FASE 5: DEFINIR MODELOS PYDANTIC PARA FASTAPI
# ============================================================================

class MuestraEntrenamiento(BaseModel):
    input_data: List[float]
    anomaly_label: int

class LoteEntrenamiento(BaseModel):
    samples: List[MuestraEntrenamiento]

class MuestraPrediccion(BaseModel):
    input_data: List[float]

class LotePrediccion(BaseModel):
    samples: List[MuestraPrediccion]

print("✅ Modelos Pydantic definidos\n")

# ============================================================================
# FASE 6: CREAR APLICACIÓN FASTAPI
# ============================================================================

app = FastAPI(
    title="OMEGA-21 Capa 2",
    description="Corteza Cognitiva Distribuida - Capa 2 Temporal-Espacial",
    version="3.0"
)

# Agregar CORS para permitir acceso desde cualquier lugar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ Aplicación FastAPI creada\n")

# ============================================================================
# FASE 7: DEFINIR ENDPOINTS
# ============================================================================

# Variables globales para rastrear estado
current_epoch = 0
total_samples_trained = 0
total_loss = 0.0

@app.post("/train_layer2")
async def train_layer2(batch_data: LoteEntrenamiento):
    """Endpoint para entrenar la Capa 2 con un lote de datos"""
    global current_epoch, total_samples_trained, total_loss
    
    model.train()
    optimizer.zero_grad()

    # Procesar datos de entrada
    batch_x_list = []
    batch_anomaly_labels_list = []

    if not batch_data.samples:
        return {"status": "failure", "message": "Lote vacío"}

    actual_batch_size = len(batch_data.samples)

    for sample in batch_data.samples:
        if len(sample.input_data) != sequence_length * input_dim:
            return {"status": "failure", "message": f"Dimensión incorrecta"}

        input_tensor = torch.tensor(sample.input_data, dtype=torch.float32).reshape(sequence_length, input_dim)
        batch_x_list.append(input_tensor)

        anomaly_label_tensor = torch.full((sequence_length, 1), float(sample.anomaly_label), dtype=torch.float32)
        batch_anomaly_labels_list.append(anomaly_label_tensor)

    # Stack y mover al dispositivo
    x = torch.stack(batch_x_list).to(device)
    anomaly_labels = torch.stack(batch_anomaly_labels_list).to(device)

    # Inicializar estados LSTM
    h_0 = torch.zeros(num_lstm_layers * 2, actual_batch_size, lstm_hidden_dim).to(device)
    c_0 = torch.zeros(num_lstm_layers * 2, actual_batch_size, lstm_hidden_dim).to(device)

    # Delayed Attention Training
    if current_epoch < DAT_EPOCH_THRESHOLD:
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.transformer_encoder.parameters():
            param.requires_grad = True

    # Forward pass
    reconstruction_output, anomaly_output, h_n_out, c_n_out = model(x, h_0, c_0)

    # Calcular pérdidas
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    reconstruction_loss = criterion_mse(reconstruction_output, x)
    anomaly_loss = criterion_bce(anomaly_output, anomaly_labels)
    lstm_aux_loss = criterion_mse(h_n_out, h_0) + criterion_mse(c_n_out, c_0)

    total_loss_batch = 0.6 * reconstruction_loss + 0.3 * anomaly_loss + 0.1 * lstm_aux_loss

    # Backprop y actualización
    total_loss_batch.backward()
    optimizer.step()

    # Rastrear estadísticas
    current_epoch += 1
    total_samples_trained += actual_batch_size
    total_loss = float(total_loss_batch.item())

    return {
        "status": "success",
        "epoch": current_epoch,
        "loss": total_loss,
        "reconstruction_loss": float(reconstruction_loss.item()),
        "anomaly_loss": float(anomaly_loss.item()),
        "samples_processed": actual_batch_size,
        "total_samples_trained": total_samples_trained,
        "device": str(device)
    }


@app.get("/status")
async def get_status():
    """Obtener estado del servidor"""
    return {
        "status": "operational",
        "current_epoch": current_epoch,
        "total_samples_trained": total_samples_trained,
        "average_loss": total_loss,
        "device": str(device),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/info")
async def get_info():
    """Obtener información del modelo"""
    return {
        "model_name": "HybridCognitiveLayer2",
        "version": "3.0",
        "input_dim": input_dim,
        "sequence_length": sequence_length,
        "output_dim": output_dim,
        "hidden_dim": d_model,
        "lstm_layers": num_lstm_layers,
        "transformer_layers": num_transformer_layers,
        "attention_heads": nhead,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device)
    }


@app.post("/predict_onnx")
async def predict_onnx(batch_data: LotePrediccion):
    """Endpoint de predicción (placeholder - implementar ONNX si es necesario)"""
    model.eval()
    
    batch_x_list = []
    
    if not batch_data.samples:
        return {"status": "failure", "message": "Lote vacío"}

    actual_batch_size = len(batch_data.samples)

    for sample in batch_data.samples:
        if len(sample.input_data) != sequence_length * input_dim:
            return {"status": "failure", "message": "Dimensión incorrecta"}
        input_tensor = torch.tensor(sample.input_data, dtype=torch.float32).reshape(sequence_length, input_dim)
        batch_x_list.append(input_tensor)

    x = torch.stack(batch_x_list).to(device)
    h_0 = torch.zeros(num_lstm_layers * 2, actual_batch_size, lstm_hidden_dim).to(device)
    c_0 = torch.zeros(num_lstm_layers * 2, actual_batch_size, lstm_hidden_dim).to(device)

    with torch.no_grad():
        reconstruction_output, anomaly_output, _, _ = model(x, h_0, c_0)

    return {
        "status": "success",
        "reconstruction": reconstruction_output.cpu().tolist(),
        "anomaly_scores": anomaly_output.cpu().tolist()
    }


print("✅ Endpoints definidos\n")

# ============================================================================
# FASE 8: EJECUTAR SERVIDOR CON NGROK
# ============================================================================

print("🚀 Iniciando servidor...")

import subprocess
import threading
from pyngrok import ngrok

# Configurar ngrok
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Iniciar servidor Uvicorn en thread
def run_server():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

print("⏳ Esperando que el servidor inicie...")
import time
time.sleep(3)

# Crear túnel ngrok
try:
    public_url = ngrok.connect(8000, bind_tls=True)
    print(f"\n✅ ngrok tunnel active")
    print(f"🔗 Public URL: {public_url}")
    print(f"📍 Local: http://0.0.0.0:8000")
    print(f"📚 Docs: {public_url}/docs")
    
except Exception as e:
    print(f"\n❌ Error con ngrok: {e}")
    print(f"⚠️  Asegúrate de que NGROK_AUTH_TOKEN es válido")
    print(f"    Obtén un token en: https://dashboard.ngrok.com/get-started/your-authtoken")

# Mantener vivo
print("\n🔄 Servidor ejecutándose...")
print("Presiona Ctrl+C para detener\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n⏹️  Servidor detenido")

# ============================================================================
# FIN
# ============================================================================

print("""
✅ CAPA 2 LISTA

Ahora ejecuta en tu terminal local:
  python /workspaces/HIPERGRAFO/validar_capa2_v2.py

O accede directamente a:
  https://[tu-url-ngrok]/docs
""")
