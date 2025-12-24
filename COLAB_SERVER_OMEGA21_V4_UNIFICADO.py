# @title 🧠 OMEGA 21 v4.0 - SERVIDOR UNIFICADO OPTIMIZADO
# Copia este CÓDIGO COMPLETO en una celda de Google Colab y ejecútalo.
# ESTE ES EL SERVIDOR FINAL UNIFICADO: Combina todas las capas (2-5) con feedback integrado

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import uvicorn
from pyngrok import ngrok
import json
import numpy as np
from datetime import datetime
import traceback
import asyncio
from collections import deque

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
        # Proyectar a dimensión común
        lstm_proj = self.proj_lstm(lstm_out) if self.proj_lstm else lstm_out
        trans_proj = self.proj_trans(trans_out) if self.proj_trans else trans_out
        
        # Gating
        gate_input = torch.cat([lstm_out, trans_out], dim=-1)
        gate = self.gate(gate_input)
        
        # Combinación ponderada
        return lstm_proj * gate + trans_proj * (1 - gate)

# ==========================================
# 2. DEFINICIÓN: Capa 2 - Espacio-Temporal
# ==========================================
class Capa2EspacioTemporal(nn.Module):
    """Capa 2: Procesamiento Temporal (LSTM) + Espacial (Transformer) + GMU"""
    def __init__(self, input_dim: int = 1600, hidden_dim: int = 512):
        super(Capa2EspacioTemporal, self).__init__()
        
        # 2A: Temporal - Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.2
        )
        
        # 2B: Espacial - Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # GMU: Fusionar LSTM (1024D) + Transformer (1600D)
        self.gmu = GMU(hidden_dim * 2, input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Input: [batch, seq_len, 1600]
        Output: [batch, 1600] (last timestep fused)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # [batch, seq, 1024]
        lstm_last = lstm_out[:, -1, :]  # [batch, 1024]
        
        # Transformer forward
        trans_out = self.transformer(x)  # [batch, seq, 1600]
        trans_last = trans_out[:, -1, :]  # [batch, 1600]
        
        # GMU fusion
        fused = self.gmu(lstm_last, trans_last)  # [batch, 1600]
        
        return fused, (h_n, c_n)

# ==========================================
# 3. DEFINICIÓN: Capa 3 - Asociativa Inferior
# ==========================================
class Capa3AsociativaInferior(nn.Module):
    """Capa 3: MLP Residual con Skip Connections para asociación inferior"""
    def __init__(self, input_dim: int = 1600, output_dim: int = 512):
        super(Capa3AsociativaInferior, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, output_dim)
        )
        
        # Proyección residual si dimensiones no coinciden
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch, 1600]
        Output: [batch, 512]
        """
        out = self.mlp(x)
        if self.skip:
            x_proj = self.skip(x)
            return out + x_proj * 0.1  # Residual light
        return out

# ==========================================
# 4. DEFINICIÓN: Capa 4 - Asociativa Superior
# ==========================================
class Capa4AsociativaSuper(nn.Module):
    """Capa 4: Self-Attention Multi-head para reasoning de alto nivel"""
    def __init__(self, hidden_dim: int = 512):
        super(Capa4AsociativaSuper, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [batch, 512] o [batch, seq_len, 512]
        Output: [batch, 512] o [batch, seq_len, 512]
        """
        # Asegurar que sea [batch, seq_len, dim] para attention
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, 512]
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)  # [batch, seq_len, 512]
        attn_out = self.norm(attn_out + x)  # Residual + norm
        
        # Feed-forward
        ffn_out = self.ffn(attn_out)
        out = self.norm(ffn_out + attn_out)  # Residual + norm
        
        # Retornar último timestep
        return out[:, -1, :] if out.size(1) > 1 else out.squeeze(1)

# ==========================================
# 5. DEFINICIÓN: Capa 5 - Ejecutiva (Decision Heads)
# ==========================================
class Capa5Ejecutiva(nn.Module):
    """Capa 5: 3 Decision Heads especializados para meta-cognición"""
    def __init__(self, hidden_dim: int = 512):
        super(Capa5Ejecutiva, self).__init__()
        
        # Head 1: Anomaly Detection (1D, 0-1)
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Dendrite Adjustments para feedback (16D, -1 a 1)
        self.dendrite_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 16),
            nn.Tanh()
        )
        
        # Head 3: Global Coherence para meta-cognición (64D, -1 a 1)
        self.coherence_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: [batch, 512]
        Outputs:
            - anomaly: [batch, 1]
            - dendrites: [batch, 16]
            - coherence: [batch, 64]
        """
        anomaly = self.anomaly_head(x)
        dendrites = self.dendrite_head(x)
        coherence = self.coherence_head(x)
        
        return anomaly, dendrites, coherence

# ==========================================
# 6. DEFINICIÓN: CortezaCognitivaV4 COMPLETA
# ==========================================
class CortezaCognitivaV4(nn.Module):
    """
    MODELO COMPLETO: Corteza Cognitiva con 5 Capas + GMU
    
    Flujo:
    Input (1600D) → Capa2 → Capa3 → Capa4 → Capa5 → 3 Outputs
    """
    def __init__(self, input_dim: int = 1600, hidden_dim: int = 512):
        super(CortezaCognitivaV4, self).__init__()
        
        self.capa2 = Capa2EspacioTemporal(input_dim, hidden_dim)
        self.capa3 = Capa3AsociativaInferior(input_dim, hidden_dim)
        self.capa4 = Capa4AsociativaSuper(hidden_dim)
        self.capa5 = Capa5Ejecutiva(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: [batch, seq_len, 1600] o [batch, 1600]
        
        Output dict:
        {
            'anomaly': [batch, 1],
            'dendrites': [batch, 16],
            'coherence': [batch, 64],
            'capa2_out': [batch, 1600],
            'capa3_out': [batch, 512],
            'capa4_out': [batch, 512]
        }
        """
        # Asegurar dimensión correcta
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, 1600]
        
        # Forward por cada capa con tracking
        c2_out, lstm_state = self.capa2(x)  # [batch, 1600]
        c3_out = self.capa3(c2_out)  # [batch, 512]
        c4_out = self.capa4(c3_out)  # [batch, 512]
        anomaly, dendrites, coherence = self.capa5(c4_out)  # [batch, 1], [batch, 16], [batch, 64]
        
        return {
            'anomaly': anomaly,
            'dendrites': dendrites,
            'coherence': coherence,
            'capa2_out': c2_out,
            'capa3_out': c3_out,
            'capa4_out': c4_out,
            'lstm_state': lstm_state
        }

# ==========================================
# 7. CONFIGURACIÓN DEL SERVIDOR
# ==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Dispositivo detectado: {device}")

model = CortezaCognitivaV4(input_dim=1600, hidden_dim=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion_anomaly = nn.BCELoss()

app = FastAPI(
    title="OMEGA 21 v4.0 - Corteza Cognitiva Distribuida Unificada",
    description="Servidor de entrenamiento con 5 capas + feedback bidireccional",
    version="4.0"
)

# ==========================================
# 8. MODELOS PYDANTIC
# ==========================================

class MuestraEntrenamiento(BaseModel):
    input_data: List[float]  # 1600D
    anomaly_label: int  # 0 o 1

class LoteEntrenamiento(BaseModel):
    samples: List[MuestraEntrenamiento]
    epochs: int = 1

class FeedbackDendritico(BaseModel):
    """Feedback del local para ajustes dendríticos"""
    ajustes_aplicados: List[float]  # 16D
    validacion: bool
    timestamp: str

# ==========================================
# 9. ESTADÍSTICAS Y MÉTRICAS
# ==========================================

class EstadisticasAvanzadas:
    def __init__(self):
        self.total_muestras = 0
        self.total_batches = 0
        self.total_loss = 0.0
        self.tiempo_inicio = datetime.now()
        
        # Historio de métricas
        self.historial_loss = deque(maxlen=1000)
        self.historial_anomalias = deque(maxlen=1000)
        self.historial_confianza = deque(maxlen=1000)
        
        # Feedback tracking
        self.feedback_recibido = 0
        self.feedback_exitoso = 0
        
    def registrar_entrenamiento(self, loss: float, batch_size: int, anomaly_mean: float):
        self.total_muestras += batch_size
        self.total_batches += 1
        self.total_loss += loss
        self.historial_loss.append(loss)
        self.historial_anomalias.append(float(anomaly_mean))
        
    def registrar_feedback(self, exitoso: bool):
        self.feedback_recibido += 1
        if exitoso:
            self.feedback_exitoso += 1
    
    def get_estadisticas(self) -> Dict:
        tiempo_transcurrido = (datetime.now() - self.tiempo_inicio).total_seconds()
        
        # Calcular promedios de últimos 100 batches
        ultimos_loss = list(self.historial_loss)[-100:] if self.historial_loss else [0]
        ultimas_anomalias = list(self.historial_anomalias)[-100:] if self.historial_anomalias else [0]
        
        return {
            "total_muestras": self.total_muestras,
            "total_batches": self.total_batches,
            "loss_promedio_global": self.total_loss / max(self.total_batches, 1),
            "loss_promedio_ultimos_100": np.mean(ultimos_loss) if ultimos_loss else 0,
            "anomalia_media": np.mean(ultimas_anomalias) if ultimas_anomalias else 0,
            "tiempo_transcurrido_seg": tiempo_transcurrido,
            "dispositivo": "CUDA" if torch.cuda.is_available() else "CPU",
            "gpu_memoria_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2) if torch.cuda.is_available() else 0,
            "feedback": {
                "recibido": self.feedback_recibido,
                "exitoso": self.feedback_exitoso,
                "tasa_exito": self.feedback_exitoso / max(self.feedback_recibido, 1)
            }
        }

estadisticas = EstadisticasAvanzadas()

# ==========================================
# 10. ENDPOINTS FUNCIONALES (7 total)
# ==========================================

@app.post("/train_layer2")
async def train_layer2(lote: LoteEntrenamiento):
    """
    ✅ ENDPOINT 1: Entrenar Corteza Cognitiva (Capas 2-5)
    
    Input: 1600D vector (expandido desde CapaSensorial LOCAL)
    Output: Loss + 3 decision heads + estado interno
    """
    try:
        if not lote.samples:
            return {"status": "error", "error": "No samples provided"}
        
        # Validar dimensión
        esperado = 1600
        actual = len(lote.samples[0].input_data)
        if actual != esperado:
            return {
                "status": "error",
                "error": f"Input mismatch: expected {esperado}D, got {actual}D",
                "help": "Verificar que CapaSensorial expande correctamente a 1600D"
            }
        
        # Preparar tensores
        inputs = torch.tensor(
            [s.input_data for s in lote.samples],
            dtype=torch.float
        ).to(device).unsqueeze(1)  # [batch, 1, 1600]
        
        labels = torch.tensor(
            [s.anomaly_label for s in lote.samples],
            dtype=torch.float
        ).view(-1, 1).to(device)
        
        # Forward
        model.train()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        anomaly_pred = outputs['anomaly']
        
        # Loss
        loss = criterion_anomaly(anomaly_pred, labels)
        loss.backward()
        optimizer.step()
        
        # Estadísticas
        estadisticas.registrar_entrenamiento(
            loss.item(),
            len(lote.samples),
            anomaly_pred.mean().item()
        )
        
        # Resultados
        return {
            "status": "trained",
            "loss": float(loss.item()),
            "batch_size": len(lote.samples),
            "outputs": {
                "anomaly_prob": float(anomaly_pred.mean().item()),
                "dendrite_adjustments": outputs['dendrites'][0].detach().cpu().tolist(),
                "coherence_state": outputs['coherence'][0].detach().cpu().tolist()
            },
            "capa_info": {
                "capa2_activations": float(outputs['capa2_out'].mean().item()),
                "capa3_activations": float(outputs['capa3_out'].mean().item()),
                "capa4_activations": float(outputs['capa4_out'].mean().item())
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc()
        }

@app.post("/feedback_dendritas")
async def feedback_dendritas(feedback: FeedbackDendritico):
    """
    ✅ ENDPOINT 2: Recibir feedback desde LOCAL
    
    Cuando LOCAL aplica ajustes dendríticos, reporta aquí
    """
    try:
        estadisticas.registrar_feedback(feedback.validacion)
        
        return {
            "status": "feedback_recibido",
            "validacion": feedback.validacion,
            "timestamp": datetime.now().isoformat(),
            "estadisticas": estadisticas.get_estadisticas()
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/status")
async def get_status():
    """✅ ENDPOINT 3: Estado completo del servidor"""
    try:
        return {
            "status": "online",
            "modelo": "CortezaCognitivaV4",
            "estadisticas": estadisticas.get_estadisticas(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/health")
async def health_check():
    """✅ ENDPOINT 4: Health check simple"""
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - estadisticas.tiempo_inicio).total_seconds()
    }

@app.get("/info")
async def get_info():
    """✅ ENDPOINT 5: Arquitectura detallada"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "nombre": "OMEGA 21 v4.0 - Corteza Cognitiva Distribuida",
        "version": "4.0",
        "arquitectura": {
            "capas": {
                "capa_2a": {"tipo": "Bi-LSTM", "hidden_dim": 512, "output_dim": 1024},
                "capa_2b": {"tipo": "Transformer Encoder", "nhead": 8, "num_layers": 2},
                "gmu": {"tipo": "Gated Multimodal Unit", "fusion": "LSTM+Transformer"},
                "capa_3": {"tipo": "MLP Residual", "hidden": [4096, 2048], "output_dim": 512},
                "capa_4": {"tipo": "Self-Attention Multi-head", "num_heads": 4, "output_dim": 512},
                "capa_5": {
                    "tipo": "3 Decision Heads",
                    "heads": {
                        "anomaly": {"output_dim": 1, "activation": "Sigmoid"},
                        "dendrites": {"output_dim": 16, "activation": "Tanh"},
                        "coherence": {"output_dim": 64, "activation": "Tanh"}
                    }
                }
            },
            "parametros_totales": total_params,
            "parametros_entrenables": trainable_params
        },
        "flujo": {
            "entrada": "1600D (CapaSensorial LOCAL)",
            "procesamiento": "Capas 2→3→4→5 secuencial",
            "salida": {
                "anomaly": "1D probabilidad",
                "dendrites": "16D feedback",
                "coherence": "64D meta-cognición"
            }
        }
    }

@app.post("/diagnostico")
async def diagnostico():
    """✅ ENDPOINT 6: Test del sistema"""
    try:
        test_input = torch.randn(2, 1, 1600).to(device)
        with torch.no_grad():
            outputs = model(test_input)
        
        return {
            "status": "diagnostico_ok",
            "test_input_shape": [2, 1, 1600],
            "outputs_shapes": {
                "anomaly": list(outputs['anomaly'].shape),
                "dendrites": list(outputs['dendrites'].shape),
                "coherence": list(outputs['coherence'].shape)
            },
            "gpu_info": {
                "cuda": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/metricas")
async def get_metricas():
    """✅ ENDPOINT 7: Métricas avanzadas de entrenamiento"""
    try:
        ultimos_loss = list(estadisticas.historial_loss)[-20:] if estadisticas.historial_loss else []
        
        return {
            "ultimos_20_losses": ultimos_loss,
            "tendencia": "mejorando" if len(ultimos_loss) > 2 and ultimos_loss[-1] < ultimos_loss[0] else "sin_cambios",
            "anomalias_detectadas": sum(1 for a in estadisticas.historial_anomalias if a > 0.5),
            "feedback_tasa_exito": estadisticas.get_estadisticas()['feedback']['tasa_exito']
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ==========================================
# 11. INICIAR SERVIDOR
# ==========================================

if __name__ == "__main__":
    try:
        print("\n" + "="*90)
        print("🚀 INICIANDO OMEGA 21 v4.0 - CORTEZA COGNITIVA DISTRIBUIDA UNIFICADA")
        print("="*90)
        
        print(f"\n📊 SISTEMA:")
        print(f"   • PyTorch: {torch.__version__}")
        print(f"   • GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"   • Dispositivo: {device}")
        
        print(f"\n🧠 MODELO CortezaCognitivaV4:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   • Parámetros totales: {total_params:,}")
        print(f"   • Capas: 2A (LSTM) → 2B (Transformer) → GMU → Capa3 (MLP) → Capa4 (Attention) → Capa5 (3 Heads)")
        
        print(f"\n📡 NGROK TUNNEL:")
        try:
            public_url = ngrok.connect(8000).public_url
            print(f"   ✅ {public_url}")
        except:
            public_url = "http://localhost:8000"
            print(f"   ⚠️ Local: {public_url}")
        
        print(f"\n📚 7 ENDPOINTS FUNCIONALES:")
        print(f"   1. POST   /train_layer2        - Entrenar modelo")
        print(f"   2. POST   /feedback_dendritas  - Recibir feedback LOCAL")
        print(f"   3. GET    /status              - Estado servidor")
        print(f"   4. GET    /health              - Health check")
        print(f"   5. GET    /info                - Arquitectura")
        print(f"   6. POST   /diagnostico         - Test sistema")
        print(f"   7. GET    /metricas            - Métricas avanzadas")
        
        print(f"\n📖 DOCUMENTACIÓN AUTOMÁTICA:")
        print(f"   • Swagger: {public_url}/docs")
        print(f"   • ReDoc: {public_url}/redoc")
        
        print(f"\n" + "="*90)
        print("✅ SERVIDOR LISTO - Esperando conexiones...")
        print("="*90 + "\n")
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except KeyboardInterrupt:
        print("\n⚠️ Servidor detenido")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(traceback.format_exc())

