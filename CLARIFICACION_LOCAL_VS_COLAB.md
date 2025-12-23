# 🔄 CLARIFICACIÓN: DIVISIÓN LOCAL ↔ COLAB

**Documento de Separación Arquitectónica**

---

## 📍 DISTRIBUCIÓN GEOGRÁFICA DEL SISTEMA

```
┌──────────────────────────────────┐         ┌──────────────────────────────────┐
│                                  │         │                                  │
│     LOCAL (Este Workspace)       │         │     COLAB (Google Servers)      │
│                                  │         │                                  │
│  /workspaces/HIPERGRAFO/         │         │  Servidor Python (server.py)     │
│                                  │         │                                  │
└──────────────────────────────────┘         └──────────────────────────────────┘
```

---

## 🏠 LO QUE OCURRE LOCALMENTE

### 1. **CAPA 0: Entrada (Vector 256D)**

**Archivo**: `src/control/MapeoVector256DaDendritas.ts`

```typescript
class MapeoVector256DaDendritas {
    extraerCamposDendriticos(vector: Vector256D): Omega21Dendrites {
        // D001-D056 extraídos aquí
        // D057-D256 disponibles para otras funciones
        return dendrites;  // Configuración para Simulador
    }
}
```

**¿Qué ocurre?**
- ✅ Vector 256D llega como entrada
- ✅ Se extraen 56 campos dendríticos
- ✅ Se mapean a sub-redes S1-S25
- ✅ Se prepara configuración

**¿Dónde ocurre?**
- 📍 Local (TypeScript/Node.js)

**¿Cuándo ocurre?**
- ⏰ Cada ciclo de `procesarFlujo()`

---

### 2. **CAPA 1: Sensorial (25 Átomos)**

**Archivos**: 
- `src/SistemaOmnisciente.ts` (orquestación)
- `src/hardware/Simulador.ts` (generación de telemetría)
- `src/neural/InferenciaLocal.ts` (ONNX)
- `models/omega21_brain.onnx` (modelo pre-entrenado)

```
Para cada ciclo:
    
    1. Vector 256D entra
    2. Dendritas se extraen (56 valores)
    3. Para cada Átomo S1-S25:
        
        a) Simulador.configurarDendritas(D001-D056)
           └─ Modifica comportamiento ONNX
        
        b) Simulador.generarMuestra()
           └─ Crea telemetría estabilizada
        
        c) InferenciaLocal.predecir(ONNX)
           └─ Ejecuta 1024 neuronas LIF
           └─ Output: embedding 256D
        
        d) Análisis físico
           └─ Calcula métricas
    
    4. Los 25 embeddings se concatenan → 1600D
    5. Se registra experiencia en EntrenadorCognitivo
```

**¿Qué ocurre?**
- ✅ Cada Átomo procesa en paralelo
- ✅ Modelo ONNX (1024 LIF) se ejecuta 25 veces
- ✅ Salidas se capturan (256D cada una)
- ✅ Protocolo de Infección propaga anomalías

**¿Dónde ocurre?**
- 📍 Local (TypeScript/Node.js + ONNX Runtime)

**¿Cuándo ocurre?**
- ⏰ Cada ciclo (configurable, típicamente cada 100ms)

---

### 3. **CONSOLIDACIÓN COGNITIVA (4 Fases)**

**Archivo**: `src/neural/EntrenadorCognitivo.ts`

```
Fase 1: ADQUISICIÓN
├─ registrarExperiencia()
├─ Almacena percepciones + hipergrafo + anomalía
└─ Buffer: max 50 experiencias

Fase 2: CATEGORIZACIÓN  
├─ refinarCategorias()
├─ Crea Nodos concepto en Hipergrafo
└─ Calcula centroides de percepciones

Fase 3: CONSOLIDACIÓN
├─ reforzarCausalidad()
├─ Crea Hiperedges entre conceptos
└─ Peso inicial: 0.7

Fase 4: PODA
├─ podarMemoriaDebil()
├─ Elimina edges con weight < 0.1
└─ Mantiene solo conexiones fuertes
```

**¿Qué ocurre?**
- ✅ Experiencias se capturan continuamente
- ✅ Cada 50 experiencias: consolidación
- ✅ Conceptos abstractos emergen
- ✅ Relaciones causales se refuerzan
- ✅ Memoria débil se poda

**¿Dónde ocurre?**
- 📍 Local (TypeScript/Node.js)

**¿Cuándo ocurre?**
- ⏰ Continuo (Fase 1), cada 50 experiencias (Fases 2-4)

---

### 4. **EXPANSIÓN DIMENSIONAL**

**Archivo**: `src/SistemaOmnisciente.ts`

```typescript
expandirAVector1600D(embedding256D: number[]): number[] {
    // Entrada: 256D (salida de un Átomo)
    // Proceso: Repetición + modulación harmónica
    // Salida: 1600D (25 subespacios × 64D)
    
    for (let s = 0; s < 25; s++) {
        for (let i = 0; i < 64; i++) {
            const modulacion = sin((s+1)*π/25) * cos((i+1)*π/64);
            const valor = embedding[i] * (1 + modulacion * 0.3);
            vector1600D.push(valor);
        }
    }
    return vector1600D;
}
```

**¿Qué ocurre?**
- ✅ Vector 256D se expande a 1600D
- ✅ Cada subespacio obtiene modulación única
- ✅ Resultado: coherencia armónica

**¿Dónde ocurre?**
- 📍 Local (TypeScript/Node.js)

**¿Cuándo ocurre?**
- ⏰ Después de procesar cada Átomo

---

### 5. **STREAMING A COLAB**

**Archivo**: `src/neural/StreamingBridge.ts`

```typescript
async enviarVector(vector1600D: number[], esAnomalia: boolean) {
    // Bufferiza 64 vectores
    // Envía batch HTTP POST a Colab
    
    const payload = {
        samples: [
            {
                input_data: vector1600D,    // 1600D
                anomaly_label: esAnomalia ? 1 : 0
            },
            // ... más muestras
        ]
    };
    
    await fetch(`${url}/train_layer2`, {
        method: 'POST',
        headers: { 'Authorization': token },
        body: JSON.stringify(payload)
    });
}
```

**¿Qué ocurre?**
- ✅ Vectores 1600D se acumulan
- ✅ Cada 64 muestras: envío a Colab
- ✅ Etiqueta de anomalía incluida

**¿Dónde ocurre?**
- 📍 Local (TypeScript/Node.js) → 🌐 Internet → ☁️ Colab

**¿Cuándo ocurre?**
- ⏰ Cada 64 ciclos (o por demanda)

---

## ☁️ LO QUE OCURRE EN COLAB

### 1. **CAPA 2: Procesamiento Dual (Temporal + Espacial)**

**Archivo**: `src/colab/server.py` (que se ejecuta en servidor Colab)

```python
class CortezaCognitivaV2(Model):
    
    # CAPA 2A: Temporal (Bi-LSTM)
    lstm_fw = LSTM(256, return_sequences=True)
    lstm_bw = LSTM(256, return_sequences=True, go_backwards=True)
    
    # CAPA 2B: Espacial (Transformer)
    transformer = MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        value_dim=64
    )
    
    def temporal_stream(self, x):
        # x: [batch, 1600D]
        # Procesa secuencias de 128 timesteps
        # Output: [batch, 512D]
        lstm_out = concatenate([
            lstm_fw(x),
            lstm_bw(x)
        ])
        return lstm_out  # 512D
    
    def spatial_stream(self, x):
        # x: [batch, 25, 64D] (reshapear 1600D)
        # Self-attention entre subespacios
        # Output: [batch, 512D]
        attn_out = transformer(x)
        return global_average_pooling(attn_out)  # 512D
```

**¿Qué ocurre?**
- ✅ Entrada: 1600D (25 subespacios × 64D)
- ✅ Bi-LSTM procesa secuencias temporales → 512D
- ✅ Transformer procesa correlaciones espaciales → 512D
- ✅ Ambas salidas se concatenan → 1024D

**¿Dónde ocurre?**
- ☁️ Colab (Python/TensorFlow/Keras)

**¿Cuándo ocurre?**
- ⏰ Cuando se recibe batch (cada 64 muestras)

---

### 2. **CAPA 3: Asociativa Inferior (Fusión)**

```python
# Fusión inteligente con GMU (Gated Multimodal Unit)
class GatedMultimodalUnit(Layer):
    def __init__(self, units):
        self.units = units
        # Gating mechanism
        self.gate_dense = Dense(1, activation='sigmoid')
    
    def call(self, temporal, spatial):
        # temporal: [batch, 512D]
        # spatial: [batch, 512D]
        
        concatenated = concatenate([temporal, spatial])  # 1024D
        lambda_gate = self.gate_dense(concatenated)
        
        # Weighted fusion
        fused = lambda_gate * temporal + (1 - lambda_gate) * spatial
        
        # MLP Residual
        x = Dense(4096, activation='gelu')(fused)
        x = Dense(4096, activation='gelu')(x)
        x = Dense(4096, activation='gelu')(x)
        
        # Skip connection
        output = Add()([x, Dense(4096)(fused)])
        return output  # 4096D → redimensionar a 1024D
```

**¿Qué ocurre?**
- ✅ GMU combina temporal + espacial inteligentemente
- ✅ MLP Residual aprende patrones complejos
- ✅ Output: 1024D (representación unificada)

**¿Dónde ocurre?**
- ☁️ Colab (Python/TensorFlow)

**¿Cuándo ocurre?**
- ⏰ Durante entrenamiento de batch

---

### 3. **CAPA 4: Asociativa Superior (Abstracción)**

```python
# Self-Attention para crear conceptos
abstraction = MultiHeadAttention(
    num_heads=16,
    key_dim=64
)(Dense(1024)(fusion_output))

concepts = Dense(256)(abstraction)  # Representación de conceptos
```

**¿Qué ocurre?**
- ✅ Self-Attention crea representaciones abstractas
- ✅ Output: 256D (vector de conceptos)

**¿Dónde ocurre?**
- ☁️ Colab (Python/TensorFlow)

---

### 4. **CAPA 5: Ejecutiva (Meta-Cognición)**

```python
# Decisiones ejecutivas
decision_head = Sequential([
    Dense(256, activation='gelu'),
    Dense(128, activation='gelu'),
    Dense(1)  # Predicción de anomalía
])

# También genera sugerencias
suggestions_head = Dense(16)(concepts)  # 16 ajustes dendríticos

outputs = {
    'loss': mse(y_true, y_pred),
    'avg_anomaly_prob': sigmoid(decision_head(concepts)),
    'suggested_adjustments': suggestions_head
}
```

**¿Qué ocurre?**
- ✅ Predice si es anomalía
- ✅ Sugiere ajustes dendríticos (16D)
- ✅ Calcula loss para backprop

**¿Dónde ocurre?**
- ☁️ Colab (Python/TensorFlow)

---

### 5. **ENTRENAMIENTO**

```python
@app.post('/train_layer2')
async def train_layer2(request: TrainingRequest):
    # Recibe batch de muestras
    # samples = [{'input_data': 1600D[], 'anomaly_label': 0|1}, ...]
    
    # Forward pass
    predictions = model(request.samples['input_data'])
    
    # Backprop
    loss = compute_loss(predictions, request.samples['anomaly_label'])
    optimizer.minimize(loss)
    
    # Retorna feedback
    return {
        'loss': float(loss),
        'avg_anomaly_prob': float(tf.reduce_mean(predictions)),
        'suggested_adjustments': list(suggestions_head.numpy())
    }
```

**¿Qué ocurre?**
- ✅ Recibe batch 1600D + etiquetas
- ✅ Forward pass: Capas 2-5
- ✅ Backprop: Actualiza pesos
- ✅ Retorna loss + sugerencias

**¿Dónde ocurre?**
- ☁️ Colab (Python/TensorFlow con GPU)

---

## 🔗 FEEDBACK LOOP

```
1. LOCAL: Genera vector 1600D
    ↓
2. COLAB: Recibe, procesa, entrena
    ↓
3. COLAB: Retorna suggested_adjustments (16D)
    ↓
4. LOCAL: MapeoVector256DaDendritas aplica → D001-D056
    ↓
5. LOCAL: Siguiente ciclo usa nuevas dendritas
    ↓
[Vuelta a paso 1]
```

---

## 📊 TABLA RESUMEN

| Componente | Local | Colab | Lenguaje | GPU Requerida |
|-----------|-------|-------|----------|---------------|
| **Capa 0** | ✅ | - | TypeScript | ❌ |
| **Capa 1** | ✅ | - | TypeScript | ⚠️ (ONNX) |
| **Cognitivo** | ✅ | - | TypeScript | ❌ |
| **Expansión** | ✅ | - | TypeScript | ❌ |
| **Capa 2A** | - | ✅ | Python | ✅ |
| **Capa 2B** | - | ✅ | Python | ✅ |
| **GMU** | - | ✅ | Python | ✅ |
| **Capa 3** | - | ✅ | Python | ✅ |
| **Capa 4** | - | ✅ | Python | ✅ |
| **Capa 5** | - | ✅ | Python | ✅ |
| **Streaming** | ✅ | ✅ | TypeScript/Python | ❌ |

---

## ⚡ FLUJO COMPLETO EN UN CICLO

```
CICLO N:
═══════════════════════════════════════════════════════════

LOCAL:
  1. generarVectorEntrada256D()
  2. MapeoVector256DaDendritas.extraer(D001-D056)
  3. Para cada Átomo S1-S25:
     - Simulador.configurarDendritas(D001-D056)
     - Simulador.generarMuestra()
     - InferenciaLocal.predecir(ONNX)  ← AQUÍ SE ENTRENA EL ONNX
     - Output: embedding 256D
  4. EntrenadorCognitivo.registrarExperiencia()
  5. expandirAVector1600D(256D) → 1600D
  6. StreamingBridge.bufferizar(1600D)
  
  [Si buffer = 64]:
    7. StreamingBridge.enviarVector(batch_1600D)
    
COLAB:
  1. POST /train_layer2 recibe batch
  2. Capa 2A (Bi-LSTM) procesa temporal
  3. Capa 2B (Transformer) procesa espacial
  4. Capa 3 (GMU + MLP) fusiona
  5. Capa 4 (Attention) abstrae
  6. Capa 5 (Decision) predice
  7. Backprop: Actualiza Capas 2-5
  8. POST Response retorna:
     - loss
     - avg_anomaly_prob
     - suggested_adjustments (16D)
  
LOCAL (Siguiente Ciclo):
  1. Recibe suggested_adjustments
  2. Actualiza D001-D056
  3. Vuelve a paso 1
```

---

## 🎯 CONCLUSIÓN

**LOCAL (SistemaOmnisciente):**
- ✅ Procesa datos de entrada
- ✅ Ejecuta 25 Átomos ONNX
- ✅ Consolida cognitivamente
- ✅ Expande a 1600D
- ✅ Envía a Colab

**COLAB (server.py):**
- ✅ Recibe 1600D
- ✅ Entrena Capas 2-5
- ✅ Retorna feedback

**SEPARACIÓN CLARA:**
- 📍 **Local = Capas 0-1 + Cognitivo**
- ☁️ **Colab = Capas 2-5 + Entrenamiento**

**¿QUÉ SE ENTRENA?**
- 🔴 **LOCAL**: ONNX omega21_brain.onnx (ya pre-entrenado)
- 🔴 **LOCAL**: Entrenador Cognitivo (consolidación de memoria)
- 🔵 **COLAB**: Capas 2-5 (el grueso del aprendizaje)

---

*Clarificación de Arquitectura - 23 de Diciembre de 2025*
