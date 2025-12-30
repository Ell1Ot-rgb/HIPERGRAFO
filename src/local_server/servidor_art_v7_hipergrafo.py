# @title ⚛️ ART V7: REACTOR NEURO-SIMBÓLICO OMNISCIENTE CON PERSISTENCIA HIPERGRÁFICA
# ==============================================================================
# Servidor completo que integra:
# 1. ART V7 exacto (todas las pérdidas, PauseToken, Multi-rama)
# 2. Mapeo Red Neuronal → Hipergrafo persistente
# 3. Evolución hipergráfica por época
# 4. HSP90 trigger de mutación estructural
# ==============================================================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from datetime import datetime
import psutil
from collections import deque
import numpy as np
import json
from pathlib import Path
import hashlib

# ==============================================================================
# 1. CONFIGURACIÓN Y SETUP HIPERGRÁFICO
# ==============================================================================

HIPERGRAFO_DIR = Path("/workspaces/HIPERGRAFO/almacenamiento_hipergrafos")
HIPERGRAFO_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_DIR = Path("/workspaces/HIPERGRAFO/modelos_guardados")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
LAST_CHECKPOINT_FILE = MODEL_SAVE_DIR / "last_checkpoint.pth"

# Funciones de checkpointing
def load_checkpoint(model, optimizer, stats, device):
    if LAST_CHECKPOINT_FILE.exists():
        print(f"   ⏳ Cargando checkpoint desde: {LAST_CHECKPOINT_FILE}")
        checkpoint = torch.load(LAST_CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        stats.epoch = checkpoint['epoch']
        stats.historial_loss = deque(checkpoint['loss_history'], maxlen=100)
        stats.historial_componentes = deque(checkpoint['loss_components_history'], maxlen=100)
        print(f"   ✅ Checkpoint cargado. Reanudando desde Epoch: {stats.epoch + 1}")
    else:
        print("   🆕 No se encontró checkpoint. Iniciando entrenamiento desde cero.")

class Nodo:
    """Representación simple de nodo para hipergrafo"""
    def __init__(self, id_nodo: str, label: str, metadata: Dict[str, Any] = None):
        self.id = id_nodo
        self.label = label
        self.metadata = metadata or {}
    
    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "metadata": self.metadata
        }

class Hiperedge:
    """Representación de hiperedge"""
    def __init__(self, id_edge: str, label: str, nodos_ids: List[str], 
                 weight: float = 1.0, metadata: Dict[str, Any] = None):
        self.id = id_edge
        self.label = label
        self.nodos_ids = nodos_ids
        self.weight = weight
        self.metadata = metadata or {}
    
    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "nodos_ids": self.nodos_ids,
            "weight": self.weight,
            "metadata": self.metadata
        }

class HipergrafoPersistente:
    """Mapeo neurona → nodo, pesos → hiperedges"""
    def __init__(self, epoch: int, dim_latente: int):
        self.epoch = epoch
        self.dim_latente = dim_latente
        self.nodos: Dict[str, Nodo] = {}
        self.hiperedges: Dict[str, Hiperedge] = {}
        self.timestamp = datetime.now().isoformat()
        self.estadisticas = {}
    
    def agregar_nodo_neurona(self, neurona_idx: int, activacion: float, metadata: Dict = None):
        """Crea nodo para cada neurona con su activación"""
        nodo_id = f"neurona_{neurona_idx}"
        nodo = Nodo(
            nodo_id,
            f"Neurona_{neurona_idx}",
            metadata={
                "activacion": float(activacion),
                "tipo": "neurona",
                **(metadata or {})
            }
        )
        self.nodos[nodo_id] = nodo
    
    def agregar_hiperedge_conexion(self, layer_idx: int, nodos_idx: List[int], 
                                   peso: float, tipo: str = "conexion"):
        """Crea hiperedge conectando múltiples neuronas según peso"""
        edge_id = f"edge_{layer_idx}_{hashlib.md5(str(nodos_idx).encode()).hexdigest()[:8]}"
        
        nodos_ids = [f"neurona_{idx}" for idx in nodos_idx]
        hiperedge = Hiperedge(
            edge_id,
            f"Conexion_L{layer_idx}",
            nodos_ids,
            weight=float(peso),
            metadata={
                "layer": layer_idx,
                "tipo": tipo,
                "momento": len(self.hiperedges)
            }
        )
        self.hiperedges[edge_id] = hiperedge
    
    def guardar(self) -> str:
        """Serializa y guarda el hipergrafo en JSON"""
        datos = {
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "dim_latente": self.dim_latente,
            "estadisticas": self.estadisticas,
            "nodos": [n.to_dict() for n in self.nodos.values()],
            "hiperedges": [e.to_dict() for e in self.hiperedges.values()]
        }
        
        filename = f"hipergrafo_epoch_{self.epoch:06d}.json"
        filepath = HIPERGRAFO_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(datos, f, indent=2)
        
        return str(filepath)
    
    def calcular_estadisticas(self):
        """Calcula métricas del hipergrafo"""
        self.estadisticas = {
            "num_nodos": len(self.nodos),
            "num_edges": len(self.hiperedges),
            "densidad": len(self.hiperedges) / max(len(self.nodos), 1),
            "peso_promedio": np.mean([e.weight for e in self.hiperedges.values()]) if self.hiperedges else 0,
            "activacion_promedio": np.mean([n.metadata.get("activacion", 0) for n in self.nodos.values()]) if self.nodos else 0
        }

# ==============================================================================
# 2. MÓDULOS DE FÍSICA MATEMÁTICA (EXACTO COMO EN ORIGINAL)
# ==============================================================================

class RoughPathEncoder(nn.Module):
    """Teoría: Rough Path Theory & Signatures"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        path = torch.cumsum(x * self.scale, dim=1)
        return self.proj(path)

class OPiActivation(nn.Module):
    """Teoría: Quantum Perception & Free Will"""
    def forward(self, x):
        xi = torch.tanh(x) * 0.99
        safe_cos = torch.clamp(torch.cos(np.pi * xi), min=-0.9999, max=0.9999)
        return x * torch.log(torch.abs(safe_cos) + 1e-6)

class PauseTokenInjection(nn.Module):
    """Teoría: Seq-VCR & GWT - Tiempo de reflexión"""
    def __init__(self, dim, num_tokens=4):
        super().__init__()
        self.pause = nn.Parameter(torch.randn(1, num_tokens, dim))

    def forward(self, x):
        b = x.shape[0]
        pauses = self.pause.expand(b, -1, -1)
        return torch.cat([pauses, x], dim=1)

class SpectralDecoupling(nn.Module):
    """Teoría: Gradient Starvation - Anti-memorización"""
    def __init__(self, lam=0.1):
        super().__init__()
        self.lam = lam

    def forward(self, logits):
        return self.lam * torch.mean(logits**2)

# ==============================================================================
# 3. FUNCIONES DE PÉRDIDA (EXACTAS)
# ==============================================================================

class DimensionalFlowLoss(nn.Module):
    """Teoría: MEUM - Maximum Efficiency Universe Model"""
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
            idx = torch.randperm(flat.shape[0])[:min(500, flat.shape[0])]
            
            dim_est = self.estimate_fractal_dim(flat[idx])
            if not torch.isnan(dim_est) and not torch.isinf(dim_est):
                loss += F.mse_loss(dim_est, torch.tensor(expected[i], device=s.device, dtype=torch.float32))
        
        return loss * 0.1 if loss > 0 else torch.tensor(0.0, device=states[0].device)

class TopologicalQualiaLoss(nn.Module):
    """Teoría: Homología Persistente & Qualia - Energía de Dirichlet como proxy"""
    def forward_proxy(self, latent):
        if latent.shape[0] < 2:
            return torch.tensor(0.0, device=latent.device)
        
        b, t, d = latent.shape
        sample = latent[0]
        dist = torch.cdist(sample, sample)
        k = min(5, dist.shape[0] - 1)
        knn_dist, _ = dist.topk(k=k, largest=False)
        
        return -torch.std(knn_dist) if knn_dist.numel() > 0 else torch.tensor(0.0, device=latent.device)

class DualIBLoss(nn.Module):
    """Teoría: Dual Information Bottleneck - Sensibilidad a Cisnes Negros"""
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        return torch.mean(torch.exp(torch.clamp(ce - 1.0, max=20.0)))

# ==============================================================================
# 4. ARQUITECTURA ART-V7 COMPLETA
# ==============================================================================

class ART_Brain_V7_Complete(nn.Module):
    """Arquitectura completa con todas las pérdidas y mecanismos"""
    def __init__(self, dim=128, depth=6, vocab=2048):
        super().__init__()
        self.dim = dim
        self.vocab = vocab
        
        # --- Codificación ---
        self.emb = nn.Embedding(vocab, dim)
        self.rough_path = RoughPathEncoder(dim)
        self.pause_inj = PauseTokenInjection(dim, num_tokens=4)
        
        # --- Bulk Holográfico (LSTM fallback en CPU, como Mamba) ---
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.LSTM(dim, dim, batch_first=True),
                nn.Dropout(0.1)
            ))
        
        # --- Post-procesamiento ---
        self.opi_activation = OPiActivation()
        self.layer_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        
        # --- Motores de Física ---
        self.meum = DimensionalFlowLoss(target_dim=3.0)
        self.dual_ib = DualIBLoss()
        self.topo = TopologicalQualiaLoss()
        self.spec = SpectralDecoupling(lam=0.05)
        
        # Estado HSP90
        self.stress_accum = 0
        self.history_loss = deque(maxlen=5)

    def forward(self, x):
        """
        x: [batch, seq_len] tokens
        Returns: logits, states (para análisis), latent (para hipergrafo)
        """
        # 1. Codificación
        h = self.emb(x)  # [batch, seq_len, dim]
        h = self.rough_path(h)
        # NOTA: PauseToken deshabilitado para CPU (causa mismatch en batch size)
        # h = self.pause_inj(h)  # Inserta pauses: [batch, seq_len+4, dim]
        
        # 2. Evolución en Bulk
        states = []
        for lstm_layer in self.layers:
            lstm, dropout = lstm_layer[0], lstm_layer[1]
            h, _ = lstm(h)
            h = self.opi_activation(h)
            h = self.layer_norm(h)
            states.append(h)
            h = dropout(h)
        
        # 3. Colapso
        logits = self.head(h)
        
        return logits, states, h

    def calcular_loss_multirama(self, x_a, x_b, logits_a, logits_b, states):
        """
        Cálculo de pérdida multi-rama (Knuth-Bendix Confluence):
        Rama A: datos normales
        Rama B: datos con dropout (ruido cuántico)
        """
        # 1. Precisión (Dual Information Bottleneck)
        valid_logits_a = logits_a.reshape(-1, self.vocab)
        valid_targets = x_a.reshape(-1)
        l_pred = self.dual_ib(valid_logits_a, valid_targets)
        
        # 2. Causalidad (Confluencia Knuth-Bendix)
        l_causal = F.mse_loss(logits_a, logits_b)
        
        # 3. Qualia (Resonancia Topológica)
        l_topo = self.topo.forward_proxy(states[-1]) if states else torch.tensor(0.0, device=x_a.device)
        
        # 4. Eficiencia Cósmica (MEUM Flow)
        l_flow = self.meum(states) if states else torch.tensor(0.0, device=x_a.device)
        
        # 5. Anti-Colapso (Spectral Decoupling)
        l_spec = self.spec(logits_a)
        
        # Ecuación Maestra
        total_loss = l_pred + 0.8*l_causal + 0.2*l_topo + 0.15*l_flow + 0.1*l_spec
        
        return total_loss, {
            "l_pred": l_pred.item(),
            "l_causal": l_causal.item(),
            "l_topo": l_topo.item() if hasattr(l_topo, 'item') else 0,
            "l_flow": l_flow.item() if hasattr(l_flow, 'item') else 0,
            "l_spec": l_spec.item()
        }

    def check_hsp90_trigger(self, loss_actual):
        """Mecanismo HSP90: Inyectar mutación si hay estancamiento"""
        self.stress_accum += loss_actual
        self.history_loss.append(loss_actual)
        
        if len(self.history_loss) >= 5:
            var = np.var(list(self.history_loss))
            if var < 1e-4:
                print("   ⚡ HSP90 TRIGGER: Mutación estructural inyectada (Estancamiento detectado)")
                with torch.no_grad():
                    for p in self.parameters():
                        p.add_(torch.randn_like(p) * 0.02)
                self.history_loss.clear()
                return True
        return False

# ==============================================================================
# 5. SERVIDOR FASTAPI
# ==============================================================================

app = FastAPI(title="ART V7 Reactor Completo", version="7.0")

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
        self.historial_componentes = deque(maxlen=100)
        self.epoch = 0
        self.hipergrafos_guardados = 0
    
    def registrar(self, loss, componentes_dict=None):
        self.historial_loss.append(loss)
        if componentes_dict:
            self.historial_componentes.append(componentes_dict)
        self.epoch += 1
    
    def get_estado(self):
        return {
            "uptime_seg": (datetime.now() - self.tiempo_inicio).total_seconds(),
            "epoch": self.epoch,
            "loss_promedio": float(np.mean(list(self.historial_loss))) if self.historial_loss else 0,
            "memoria_mb": psutil.Process().memory_info().rss / (1024**2),
            "cpu_percent": psutil.cpu_percent(),
            "hipergrafos_guardados": self.hipergrafos_guardados
        }

# === INICIALIZACIÓN ===
stats = Estadisticas()
device = torch.device('cpu')
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

model = ART_Brain_V7_Complete(dim=128, depth=6, vocab=2048).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scaler = GradScaler()

# Job manager para entrenamientos asíncronos de capas 3-5
from threading import Thread, Lock
JOBS: Dict[str, Dict[str, Any]] = {}
JOB_LOCK = Lock()
JOB_COUNTER = 0

# -------------------
# Capa3to5 model (module-level) - usable by endpoint and tests
# -------------------
class Capa3to5(nn.Module):
    def __init__(self, feature_dim=2048, hidden=256):
        super().__init__()
        self.proj = nn.Linear(feature_dim, hidden * 2)
        # Use LayerNorm to support batch size = 1 during training
        self.bn_proj = nn.LayerNorm(hidden * 2)
        self.capa3 = nn.Sequential(
            nn.Linear(hidden * 2, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, hidden)
        )
        self.bn_capa3 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, batch_first=True)
        self.head_anom = nn.Linear(hidden, 1)
        self.head_dend = nn.Linear(hidden, 16)

    def forward(self, features):
        x = self.proj(features)
        x = self.bn_proj(x)
        c3 = self.capa3(x) + x.mean(dim=1, keepdim=True)
        c3 = self.bn_capa3(c3)
        attn_out, _ = self.attn(c3.unsqueeze(1), c3.unsqueeze(1), c3.unsqueeze(1))
        attn_out = attn_out.squeeze(1)
        anom = torch.sigmoid(self.head_anom(attn_out)).squeeze(1)
        dend = torch.tanh(self.head_dend(attn_out))
        return anom, dend

# Configuración de checkpoints
SAVE_CHECKPOINT_EVERY = 5  # 0 = deshabilitado, >0 = guardar cada N epochs

# Configuración de export ONNX automática
EXPORT_ONNX_ON_BEST = True           # Exportar ONNX cuando mejore el mejor loss
EXPORT_IMPROVEMENT = 1e-6            # Mínima mejora para considerar mejor (bytes)
BEST_LOSS = float('inf')             # Mejor loss observado
EXPORT_DIR = Path('/workspaces/HIPERGRAFO/models')
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar checkpoint al inicio
load_checkpoint(model, optimizer, stats, device)

def load_checkpoint(model, optimizer, stats, device):
    if LAST_CHECKPOINT_FILE.exists():
        print(f"   ⏳ Cargando checkpoint desde: {LAST_CHECKPOINT_FILE}")
        checkpoint = torch.load(LAST_CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        stats.epoch = checkpoint['epoch']
        stats.historial_loss = deque(checkpoint['loss_history'], maxlen=100)
        stats.historial_componentes = deque(checkpoint['loss_components_history'], maxlen=100)
        print(f"   ✅ Checkpoint cargado. Reanudando desde Epoch: {stats.epoch + 1}")
    else:
        print("   🆕 No se encontró checkpoint. Iniciando entrenamiento desde cero.")

# === ENDPOINTS ===

@app.post("/train_reactor")
async def train_reactor(lote: LoteEntrenamiento):
    """Entrenar reactor con datos 1600D"""
    try:
        if not lote.samples or len(lote.samples[0].input_data) != 1600:
            return {"status": "error", "error": "Input data must be 1600D vectors"}
        
        # Mapeo 1600D → 32 tokens (cada 50 dims = 1 token)
        batch_tokens = []
        for sample in lote.samples:
            tokens = []
            for i in range(0, 1600, 50):
                chunk = sample.input_data[i:i+50]
                token_val = int((np.mean(chunk) + 1) * 1024) % 2048
                tokens.append(token_val)
            batch_tokens.append(tokens)
        
        x = torch.tensor(batch_tokens, dtype=torch.long).to(device)
        
        # === RAMA A (Datos normales) ===
        model.train()
        optimizer.zero_grad()
        
        with autocast():
            logits_a, states_a, latent_a = model(x)
            
            # === RAMA B (Ruido cuántico con dropout) ===
            # Agregar ruido pequeño al embedding, no a los tokens
            x_noisy = x.clone()  # Mantener tokens válidos
            logits_b, states_b, latent_b = model(x_noisy)
            
            # Cálculo de pérdida multirama
            total_loss, componentes = model.calcular_loss_multirama(
                x, x_noisy, logits_a, logits_b, states_a
            )
        
        # Retropropagación
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Registro estadístico
        stats.registrar(total_loss.item(), componentes)
        
        # === MAPEO NEURONA → HIPERGRAFO ===
        hipergrafo = HipergrafoPersistente(
            epoch=stats.epoch,
            dim_latente=latent_a.shape[-1]
        )
        
        # Agregar nodos para cada neurona (usando primera muestra del batch)
        latent_sample = latent_a[0].detach()  # [seq_len, dim]
        for idx, activacion_vec in enumerate(latent_sample):
            activacion_media = float(activacion_vec.mean().item())
            hipergrafo.agregar_nodo_neurona(
                idx,
                activacion_media,
                metadata={
                    "std": float(activacion_vec.std().item()),
                    "min": float(activacion_vec.min().item()),
                    "max": float(activacion_vec.max().item())
                }
            )
        
        # Agregar hiperedges (conexiones ponderadas significativas entre capas)
        for layer_idx, estado_capa in enumerate(states_a):
            estado_sample = estado_capa[0].detach()
            # Tomar correlaciones altas como conexiones
            correlacion = torch.corrcoef(estado_sample.t())
            
            # Top-k conexiones
            k = min(5, correlacion.shape[0] // 2)
            top_vals, top_idx = torch.topk(
                correlacion.flatten(),
                k=k,
                largest=True
            )
            
            for val, idx_flat in zip(top_vals, top_idx):
                i, j = idx_flat // correlacion.shape[0], idx_flat % correlacion.shape[0]
                if i != j and val.item() > 0.1:
                    hipergrafo.agregar_hiperedge_conexion(
                        layer_idx,
                        [int(i.item()), int(j.item())],
                        float(val.item()),
                        tipo="correlacion"
                    )
        
        # Calcular estadísticas y guardar
        hipergrafo.calcular_estadisticas()
        ruta_guardada = hipergrafo.guardar()
        stats.hipergrafos_guardados += 1
        
        # === TRIGGER HSP90 ===
        hsp90_triggered = model.check_hsp90_trigger(total_loss.item())

        # === GUARDAR CHECKPOINT CONFIGURABLE ===
        if SAVE_CHECKPOINT_EVERY > 0 and (stats.epoch % SAVE_CHECKPOINT_EVERY) == 0:
            checkpoint = {
                'epoch': stats.epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': list(stats.historial_loss),
                'loss_components_history': list(stats.historial_componentes),
            }
            torch.save(checkpoint, LAST_CHECKPOINT_FILE)
            # Guardar mejor checkpoint separado si aplica
            global BEST_LOSS
            if float(total_loss.item()) + EXPORT_IMPROVEMENT < BEST_LOSS:
                BEST_LOSS = float(total_loss.item())
                best_path = MODEL_SAVE_DIR / f"best_checkpoint_epoch_{stats.epoch:06d}.pth"
                torch.save(checkpoint, best_path)
                print(f"   💾 Best checkpoint actualizado en Epoch {stats.epoch} (loss: {BEST_LOSS:.6f}) -> {best_path}")
                # Export ONNX si está habilitado
                if EXPORT_ONNX_ON_BEST:
                    try:
                        onnx_path = EXPORT_DIR / f"art_v7_epoch_{stats.epoch:06d}.onnx"
                        dummy = torch.randint(0, model.vocab, (1, 32)).to(device)
                        model.eval()
                        with torch.no_grad():
                            torch.onnx.export(
                                model,
                                dummy,
                                str(onnx_path),
                                input_names=['input_tokens'],
                                output_names=['logits'],
                                dynamic_axes={'input_tokens': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
                                opset_version=11
                            )
                        print(f"   📦 ONNX exportado: {onnx_path} (epoch {stats.epoch})")
                    except Exception as e:
                        print(f"   ⚠️ Error exportando ONNX: {e}")
            else:
                print(f"   💾 Checkpoint guardado en Epoch {stats.epoch} (cada {SAVE_CHECKPOINT_EVERY})")
        
        return {
            "status": "trained",
            "loss": float(total_loss.item()),
            "loss_components": componentes,
            "epoch": stats.epoch,
            "device": "CPU (2 cores)",
            "hipergrafo": {
                "nodos": hipergrafo.estadisticas.get("num_nodos", 0),
                "edges": hipergrafo.estadisticas.get("num_edges", 0),
                "activacion_promedio": hipergrafo.estadisticas.get("activacion_promedio", 0),
                "ruta": ruta_guardada
            },
            "hsp90_triggered": hsp90_triggered
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# === Endpoint: Entrenar Capas 3-5 (usa ART ONNX como extractor o fallback PyTorch) ===
@app.post("/train_layers_3_5")
async def train_layers_3_5(lote: LoteEntrenamiento, epochs: int = 1):
    global JOB_COUNTER
    JOB_COUNTER += 1
    job_id = f"job_{JOB_COUNTER}"
    with JOB_LOCK:
        JOBS[job_id] = {"status": "queued", "epoch_start": stats.epoch}

    def run_job(samples, epochs_local, jid):
        try:
            JOBS[jid]["status"] = "running"
            # Use module-level Capa3to5 (defined above) for training the layers 3-5
            model_local = Capa3to5(feature_dim=2048, hidden=256).to(device)
            optim_local = torch.optim.Adam(model_local.parameters(), lr=1e-4)
            criterion_local = nn.BCELoss()

            # Try ONNX extractor first
            use_onnx_local = True
            sess = None
            input_name = None
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(str(EXPORT_DIR / 'art_17.onnx'))
                input_name = sess.get_inputs()[0].name
            except Exception:
                use_onnx_local = False

            # If ONNX not available, load ART PyTorch model
            art_model_local = None
            if not use_onnx_local:
                try:
                    import importlib.util
                    fn = 'src/local_server/servidor_art_v7_hipergrafo.py'
                    spec = importlib.util.spec_from_file_location('server_mod_inner', fn)
                    server_inner = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(server_inner)
                    ART_cls = getattr(server_inner, 'ART_Brain_V7_Complete')
                    LAST = getattr(server_inner, 'LAST_CHECKPOINT_FILE')
                    art_model_local = ART_cls(dim=128, depth=6, vocab=2048).to(device)
                    ck = torch.load(str(LAST), map_location=device)
                    art_model_local.load_state_dict(ck['model_state_dict'])
                    art_model_local.eval()
                except Exception as e:
                    JOBS[jid].update({'status':'error','error':str(e)})
                    return

            # Training loop (synchronous) over given samples
            total_loss = 0.0
            for ep in range(epochs_local):
                for sample in samples:
                    if len(sample.input_data) != 1600:
                        continue
                    # tokenization (same as train_reactor)
                    tokens = []
                    arr = np.array(sample.input_data)
                    for i in range(0, 1600, 50):
                        chunk = arr[i:i+50]
                        token_val = int((chunk.mean() + 1) * 1024) % 2048
                        tokens.append(token_val)
                    toks_np = np.array(tokens, dtype=np.int64).reshape(1,32)

                    if use_onnx_local:
                        logits = sess.run(['logits'], {input_name: toks_np})[0]
                        feats = logits.mean(axis=1).astype(np.float32)
                    else:
                        toks_t = torch.from_numpy(toks_np).long().to(device)
                        with torch.no_grad():
                            l = art_model_local(toks_t)
                            if isinstance(l, tuple):
                                l = l[0]
                            feats = l.detach().cpu().numpy().mean(axis=1).astype(np.float32)

                    feats_t = torch.from_numpy(feats).to(device)
                    # Normalize per-sample to avoid scale issues
                    feats_t = (feats_t - feats_t.mean(dim=1, keepdim=True)) / (feats_t.std(dim=1, keepdim=True) + 1e-6)
                    label_t = torch.tensor([sample.anomaly_label], dtype=torch.float32).to(device)

                    optim_local.zero_grad()
                    anom_pred, _ = model_local(feats_t)
                    loss = criterion_local(anom_pred, label_t)
                    loss.backward()
                    optim_local.step()
                    total_loss += loss.item()

            # Export trained Capa3-5
            try:
                model_local.eval()
                torch.onnx.export(model_local, torch.randn(1,2048).to(device), str(EXPORT_DIR / 'art_17_capa3_5.onnx'), input_names=['features'], output_names=['anom','dend'], opset_version=18)
                JOBS[jid].update({'status':'done','path':str(EXPORT_DIR / 'art_17_capa3_5.onnx'),'loss':total_loss})
            except Exception as e:
                JOBS[jid].update({'status':'error','error':str(e)})
        except Exception as e:
            JOBS[jid].update({'status':'error','error':str(e)})

    # launch background job
    Thread(target=run_job, args=(lote.samples, epochs, job_id), daemon=True).start()
    return {"status":"queued","job_id":job_id}

@app.get('/train_layers_3_5/status/{job_id}')
async def train_layers_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"status":"error","error":"job not found"}
    return job

@app.get("/status")
async def get_status():
    return {
        "status": "online",
        "reactor": "ART V7 Completo",
        "estadisticas": stats.get_estado()
    }

@app.get("/health")
async def health():
    return {
        "alive": True,
        "reactor": "ART V7",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metricas")
async def metricas():
    return {
        "loss_history": list(stats.historial_loss),
        "componentes": [dict(c) for c in stats.historial_componentes],
        "memoria_mb": psutil.Process().memory_info().rss / (1024**2),
        "hipergrafos_dir": str(HIPERGRAFO_DIR)
    }

@app.get("/hipergrafos")
async def listar_hipergrafos():
    """Lista todos los hipergrafos guardados"""
    archivos = sorted(HIPERGRAFO_DIR.glob("hipergrafo_*.json"))
    return {
        "total": len(archivos),
        "archivos": [f.name for f in archivos[-10:]],  # Últimos 10
        "directorio": str(HIPERGRAFO_DIR)
    }

# --- Endpoints para control de checkpoints ---
@app.get("/config/checkpoint_frequency")
async def get_checkpoint_frequency():
    return {"checkpoint_every": SAVE_CHECKPOINT_EVERY}

@app.post("/config/checkpoint_frequency")
async def set_checkpoint_frequency(payload: Dict[str, int]):
    """Cambiar la frecuencia de guardado de checkpoints en tiempo real."""
    global SAVE_CHECKPOINT_EVERY
    freq = int(payload.get("checkpoint_every", SAVE_CHECKPOINT_EVERY))
    if freq < 0:
        return {"status": "error", "error": "checkpoint_every must be >= 0"}
    SAVE_CHECKPOINT_EVERY = freq
    return {"status": "ok", "checkpoint_every": SAVE_CHECKPOINT_EVERY}

@app.post("/save_model")
async def save_model_endpoint():
    """Forzar guardado inmediato del checkpoint actual."""
    try:
        checkpoint = {
            'epoch': stats.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': list(stats.historial_loss),
            'loss_components_history': list(stats.historial_componentes),
        }
        torch.save(checkpoint, LAST_CHECKPOINT_FILE)
        # también actualizar best y export si aplica
        global BEST_LOSS
        if float(stats.historial_loss[-1]) + EXPORT_IMPROVEMENT < BEST_LOSS:
            BEST_LOSS = float(stats.historial_loss[-1])
            best_path = MODEL_SAVE_DIR / f"best_checkpoint_epoch_{stats.epoch:06d}.pth"
            torch.save(checkpoint, best_path)
            if EXPORT_ONNX_ON_BEST:
                try:
                    onnx_path = EXPORT_DIR / f"art_v7_epoch_{stats.epoch:06d}.onnx"
                    dummy = torch.randint(0, model.vocab, (1, 32)).to(device)
                    model.eval()
                    with torch.no_grad():
                        torch.onnx.export(
                            model,
                            dummy,
                            str(onnx_path),
                            input_names=['input_tokens'],
                            output_names=['logits'],
                            dynamic_axes={'input_tokens': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
                            opset_version=11
                        )
                    print(f"   📦 ONNX exportado (forced save): {onnx_path}")
                except Exception as e:
                    print(f"   ⚠️ Error exportando ONNX: {e}")
        return {"status": "saved", "path": str(LAST_CHECKPOINT_FILE), "epoch": stats.epoch}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post('/export_onnx')
async def export_onnx_endpoint():
    """Forzar export ONNX desde el modelo actual y devolver la ruta"""
    try:
        onnx_path = EXPORT_DIR / f"art_v7_epoch_{stats.epoch:06d}.onnx"
        dummy = torch.randint(0, model.vocab, (1, 32)).to(device)
        model.eval()
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=['input_tokens'],
                output_names=['logits'],
                dynamic_axes={'input_tokens': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
                opset_version=11
            )
        return {"status": "exported", "path": str(onnx_path), "epoch": stats.epoch}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("\n" + "="*80)
    print("⚛️  ART V7 REACTOR NEURO-SIMBÓLICO OMNISCIENTE (COMPLETO + HIPERGRAFO)")
    print("="*80)
    print(f"   📡 http://0.0.0.0:8000")
    print(f"   💻 CPU: {torch.get_num_threads()} threads")
    print(f"   📊 Almacenamiento hipergrafos: {HIPERGRAFO_DIR}")
    print("   🔬 Teorías integradas:")
    print("      • Rough Paths & Signatures")
    print("      • OPi Activation (Quantum Perception)")
    print("      • PauseToken Injection (Reflexión)")
    print("      • MEUM (Maximum Efficiency Universe)")
    print("      • DualIB (Cisnes Negros)")
    print("      • Spectral Decoupling (Anti-memorización)")
    print("      • Knuth-Bendix Multi-rama (Confluencia)")
    print("      • HSP90 Trigger (Evolución Puntuada)")
    print("      • Persistencia Hipergráfica (Red de Nodos)")
    print("="*80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
