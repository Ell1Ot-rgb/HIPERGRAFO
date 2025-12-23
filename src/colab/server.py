# @title 🧠 OMEGA 21 - Servidor de Entrenamiento Distribuido (GNN)
# Copia este código en una celda de Google Colab y ejecútalo.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from pyngrok import ngrok
import json

# ==========================================
# 1. Definición del Modelo de Corteza Cognitiva (Capas 2-5)
# ==========================================

class CortezaCognitivaV2(nn.Module):
    def __init__(self, input_dim=1600, hidden_dim=512):
        super(CortezaCognitivaV2, self).__init__()
        
        # CAPA 2: Espacio-Temporal
        # 2A: Temporal (LSTM)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # 2B: Espacial (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Gating Multimodal (GMU) para fusionar LSTM y Transformer
        self.gmu_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + input_dim, 1),
            nn.Sigmoid()
        )
        
        # CAPA 3: Asociativa Inferior (MLP Residual)
        self.capa3 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, hidden_dim)
        )
        
        # CAPA 4: Asociativa Superior (Self-Attention)
        self.capa4_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        
        # CAPA 5: Ejecutiva (Decision Heads)
        self.capa5_decision = nn.Linear(hidden_dim, 1) # Predicción de anomalía
        self.capa5_dendrites = nn.Linear(hidden_dim, 16) # Ajustes para las 16 dendritas

    def forward(self, x):
        # x shape: [batch, seq_len, 1600]
        
        # 2A: Temporal
        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :] # [batch, hidden*2]
        
        # 2B: Espacial (usamos el último frame para el transformer en este ejemplo)
        trans_out = self.transformer(x)
        trans_last = trans_out[:, -1, :] # [batch, 1600]
        
        # GMU Fusion
        gate = self.gmu_gate(torch.cat([lstm_last, trans_last], dim=1))
        # Redimensionar trans_last para que coincida con lstm_last si es necesario, 
        # o usar una proyección. Aquí simplificamos:
        fused = lstm_last * gate # Simplificación de la fusión
        
        # Capa 3
        c3 = self.capa3(fused) + fused.mean() # Residual simple
        
        # Capa 4
        c4, _ = self.capa4_attention(c3.unsqueeze(0), c3.unsqueeze(0), c3.unsqueeze(0))
        c4 = c4.squeeze(0)
        
        # Capa 5
        anomaly_prob = torch.sigmoid(self.capa5_decision(c4))
        dendrite_adj = torch.tanh(self.capa5_dendrites(c4))
        
        return anomaly_prob, dendrite_adj

# ==========================================
# 2. Configuración del Servidor
# ==========================================

# Inicializar Modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_v2 = CortezaCognitivaV2().to(device)
optimizer = torch.optim.Adam(model_v2.parameters(), lr=0.001)
criterion = nn.BCELoss()

app = FastAPI()

class MuestraEntrenamiento(BaseModel):
    input_data: List[float]
    anomaly_label: int

class LoteEntrenamiento(BaseModel):
    samples: List[MuestraEntrenamiento]

@app.post("/train_layer2")
async def train_layer2(lote: LoteEntrenamiento):
    if not lote.samples:
        return {"status": "empty"}

    # Preparar tensores
    inputs = torch.tensor([s.input_data for s in lote.samples], dtype=torch.float).to(device)
    labels = torch.tensor([s.anomaly_label for s in lote.samples], dtype=torch.float).view(-1, 1).to(device)
    
    # El modelo espera [batch, seq_len, features]. 
    # Como recibimos batches de 64, podemos tratarlos como una secuencia o como 64 muestras independientes.
    # Para este entrenamiento, los tratamos como muestras independientes con seq_len=1.
    inputs = inputs.unsqueeze(1) 

    model_v2.train()
    optimizer.zero_grad()
    
    pred_anomaly, dendrite_adj = model_v2(inputs)
    
    loss = criterion(pred_anomaly, labels)
    loss.backward()
    optimizer.step()

    print(f"🧠 Capa 2-5 Entrenada. Loss: {loss.item():.6f} | Anomalías detectadas: {pred_anomaly.mean().item():.2%}")
    
    return {
        "status": "trained",
        "loss": loss.item(),
        "avg_anomaly_prob": pred_anomaly.mean().item(),
        "suggested_adjustments": dendrite_adj.mean(dim=0).detach().cpu().tolist()
    }

@app.post("/entrenar")
async def entrenar(lote: Any):
    # Endpoint antiguo para compatibilidad
    return {"status": "deprecated", "message": "Use /train_layer2 for 5-layer architecture"}


# ==========================================
# 3. Iniciar Servidor con ngrok
# ==========================================

# ngrok.set_auth_token("TU_TOKEN_AQUI") # Descomentar y poner token si es necesario
public_url = ngrok.connect(8000).public_url
print(f"\n🚀 SERVIDOR LISTO EN: {public_url}")
print("Copia esta URL en tu archivo configColab.ts en VS Code\n")

uvicorn.run(app, host="0.0.0.0", port=8000)
