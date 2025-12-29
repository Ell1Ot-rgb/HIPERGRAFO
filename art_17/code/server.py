# @title 🧠 OMEGA 21 - Servidor de Entrenamiento Distribuido (GNN)
# Copia este código en una celda de Google Colab y ejecútalo.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from pyngrok import ngrok
import json
import numpy as np
from datetime import datetime
import traceback

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
    epochs: int = 1

# ==========================================
# 3. Estadísticas Globales del Servidor
# ==========================================

class EstadisticasServidor:
    def __init__(self):
        self.total_muestras_entrenadas = 0
        self.total_batches_procesados = 0
        self.total_loss = 0.0
        self.tiempo_inicio = datetime.now()
        self.ultimas_predicciones = []
        self.historial_loss = []
        
    def registrar_entrenamiento(self, loss, batch_size):
        self.total_muestras_entrenadas += batch_size
        self.total_batches_procesados += 1
        self.total_loss += loss
        self.historial_loss.append(loss)
        
    def get_estadisticas(self):
        tiempo_transcurrido = (datetime.now() - self.tiempo_inicio).total_seconds()
        return {
            "total_muestras": self.total_muestras_entrenadas,
            "total_batches": self.total_batches_procesados,
            "loss_promedio": self.total_loss / max(self.total_batches_procesados, 1),
            "tiempo_transcurrido_segundos": tiempo_transcurrido,
            "dispositivo": "cuda" if torch.cuda.is_available() else "cpu",
            "version_pytorch": torch.__version__,
        }

estadisticas = EstadisticasServidor()

@app.post("/train_layer2")
async def train_layer2(lote: LoteEntrenamiento):
    """
    Entrena Capas 2-5: Procesamiento Temporal + Espacial + Fusión + Asociativa + Ejecutiva
    
    Input: Vector 1600D de LOCAL (expandido desde 256D)
    Output: 
    - loss: Pérdida de entrenamiento
    - avg_anomaly_prob: Probabilidad promedio de anomalía
    - suggested_adjustments: 16 ajustes dendríticos para próximo ciclo
    """
    try:
        if not lote.samples:
            return {"status": "empty", "error": "No samples provided"}
        
        # Validar tamaño de input
        esperado = 1600
        actual = len(lote.samples[0].input_data)
        if actual != esperado:
            return {
                "status": "error",
                "error": f"Input dimension mismatch: expected {esperado}, got {actual}",
                "help": "Asegúrate que el vector expandido es 1600D (256D × 25/4)"
            }

        # Preparar tensores
        inputs = torch.tensor(
            [s.input_data for s in lote.samples], 
            dtype=torch.float
        ).to(device)
        
        labels = torch.tensor(
            [s.anomaly_label for s in lote.samples], 
            dtype=torch.float
        ).view(-1, 1).to(device)
        
        # El modelo espera [batch, seq_len, features]. 
        # Tratamos cada muestra como una secuencia de longitud 1
        inputs = inputs.unsqueeze(1)  # [batch, 1, 1600]

        model_v2.train()
        optimizer.zero_grad()
        
        pred_anomaly, dendrite_adj = model_v2(inputs)
        
        loss = criterion(pred_anomaly, labels)
        loss.backward()
        optimizer.step()

        # Registrar estadísticas
        estadisticas.registrar_entrenamiento(loss.item(), len(lote.samples))
        
        # Procesamiento de resultados
        avg_anomaly = pred_anomaly.mean().item()
        avg_dendrites = dendrite_adj.mean(dim=0).detach().cpu().tolist()
        
        print(f"🧠 [Capa 2-5] Entrenada - Loss: {loss.item():.6f} | Anomalías: {avg_anomaly:.2%} | Muestras: {len(lote.samples)}")
        
        return {
            "status": "trained",
            "loss": float(loss.item()),
            "avg_anomaly_prob": float(avg_anomaly),
            "suggested_adjustments": avg_dendrites,
            "batch_size": len(lote.samples),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ ERROR en /train_layer2: {error_msg}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.get("/status")
async def get_status():
    """Obtiene estado actual del servidor y estadísticas"""
    try:
        return {
            "status": "online",
            "modelo": "CortezaCognitivaV2",
            "dispositivo": "cuda" if torch.cuda.is_available() else "cpu",
            "estadisticas": estadisticas.get_estadisticas(),
            "capacidad": {
                "capas": "2-5 (Temporal + Espacial + Asociativa + Ejecutiva)",
                "input_dim": 1600,
                "hidden_dim": 512,
                "output_anomaly": 1,
                "output_dendrites": 16,
                "parametros_entrenables": sum(p.numel() for p in model_v2.parameters() if p.requires_grad)
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check simple"""
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat(),
        "uptime": (datetime.now() - estadisticas.tiempo_inicio).total_seconds()
    }

@app.get("/info")
async def get_info():
    """Información detallada sobre el servidor y arquitectura"""
    return {
        "nombre": "OMEGA 21 - Corteza Cognitiva Distribuida",
        "version": "3.0",
        "fecha": datetime.now().isoformat(),
        "arquitectura": {
            "capas": {
                "capa_2a": {
                    "nombre": "Temporal Processing",
                    "tipo": "Bi-LSTM",
                    "input_dim": 1600,
                    "hidden_dim": 512,
                    "output_dim": 1024
                },
                "capa_2b": {
                    "nombre": "Spatial Processing",
                    "tipo": "Transformer Encoder",
                    "input_dim": 1600,
                    "num_heads": 8,
                    "num_layers": 2,
                    "output_dim": 1600
                },
                "capa_3": {
                    "nombre": "Associative Lower",
                    "tipo": "MLP Residual",
                    "input_dim": 1024,
                    "hidden_dim": 4096,
                    "output_dim": 512
                },
                "capa_4": {
                    "nombre": "Associative Upper",
                    "tipo": "Self-Attention",
                    "input_dim": 512,
                    "num_heads": 4,
                    "output_dim": 512
                },
                "capa_5": {
                    "nombre": "Executive Meta-Cognition",
                    "tipo": "Decision Head",
                    "anomaly_output": 1,
                    "dendrite_adjustments": 16
                }
            },
            "fusion": {
                "nombre": "GMU",
                "tipo": "Gated Multimodal Unit",
                "fusion_inputs": ["lstm", "transformer"],
                "activation": "Sigmoid"
            }
        },
        "entrenamiento": {
            "optimizador": "Adam",
            "lr": 0.001,
            "criterio_perdida": "Binary Cross Entropy",
            "dispositivo": "CUDA" if torch.cuda.is_available() else "CPU"
        },
        "flujo_datos": {
            "entrada": "1600D vector (from LOCAL expansion)",
            "procesamiento": "Sequential through layers 2-5",
            "salida": {
                "anomaly_prob": "1D (anomaly prediction)",
                "dendrite_adjustments": "16D (for LOCAL feedback)"
            }
        }
    }

@app.post("/diagnostico")
async def diagnostico():
    """Endpoint para diagnóstico del sistema"""
    try:
        # Test modelo con dummy input
        test_input = torch.randn(1, 1, 1600).to(device)
        with torch.no_grad():
            anomaly_pred, dendrite_pred = model_v2(test_input)
        
        return {
            "status": "diagnostico_completado",
            "modelo_funcional": True,
            "test_input_shape": list(test_input.shape),
            "test_output_anomaly": float(anomaly_pred.item()),
            "test_output_dendrites": dendrite_pred[0].detach().cpu().tolist(),
            "gpu_disponible": torch.cuda.is_available(),
            "gpu_info": {
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
                "memoria_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2 if torch.cuda.is_available() else 0
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "trace": traceback.format_exc()}


# ==========================================
# 4. Iniciar Servidor con ngrok
# ==========================================

if __name__ == "__main__":
    try:
        # ngrok.set_auth_token("TU_TOKEN_AQUI") # Descomentar y poner token si es necesario
        
        print("\n" + "="*80)
        print("🚀 INICIANDO OMEGA 21 - CORTEZA COGNITIVA DISTRIBUIDA")
        print("="*80)
        
        # Información del sistema
        print(f"\n📊 INFORMACIÓN DEL SISTEMA:")
        print(f"   • PyTorch version: {torch.__version__}")
        print(f"   • GPU disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   • GPU: {torch.cuda.get_device_name(0)}")
            print(f"   • CUDA version: {torch.version.cuda}")
        print(f"   • Dispositivo: {device}")
        
        # Información del modelo
        print(f"\n🧠 INFORMACIÓN DEL MODELO:")
        print(f"   • Arquitectura: CortezaCognitivaV2")
        print(f"   • Parámetros entrenables: {sum(p.numel() for p in model_v2.parameters() if p.requires_grad):,}")
        print(f"   • Input dimension: 1600")
        print(f"   • Output: [anomaly_prob (1D), dendrite_adjustments (16D)]")
        
        # Establecer túnel ngrok
        print(f"\n🌐 ESTABLECIENDO TÚNEL NGROK...")
        try:
            public_url = ngrok.connect(8000).public_url
            print(f"   ✅ Túnel establecido: {public_url}")
            print(f"\n   📝 COPIA ESTA URL EN: src/neural/configColab.ts")
            print(f"      export const COLAB_URL = '{public_url}';")
        except Exception as ngrok_error:
            print(f"   ⚠️  Error en ngrok: {ngrok_error}")
            public_url = "http://localhost:8000"
            print(f"   Usando URL local: {public_url}")
        
        print(f"\n📚 ENDPOINTS DISPONIBLES:")
        print(f"   • POST   /train_layer2 - Entrenar Capas 2-5")
        print(f"   • GET    /status - Estado del servidor")
        print(f"   • GET    /health - Health check")
        print(f"   • GET    /info - Información arquitectónica")
        print(f"   • POST   /diagnostico - Diagnóstico del sistema")
        print(f"\n📖 DOCUMENTACIÓN:")
        print(f"   • Swagger UI: {public_url}/docs")
        print(f"   • ReDoc: {public_url}/redoc")
        print(f"   • OpenAPI JSON: {public_url}/openapi.json")
        
        print(f"\n" + "="*80)
        print("✅ SERVIDOR LISTO - Esperando conexiones...")
        print("="*80 + "\n")
        
        # Ejecutar servidor
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Servidor detenido por el usuario")
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        print(f"Traceback: {traceback.format_exc()}")

