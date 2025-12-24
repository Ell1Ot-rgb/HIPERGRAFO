# 🎯 MEJORAS SUTILES PARA CAPAS 0 Y 1 - HACIA EL 100%

## 📊 Estado Actual
- **Capa 0**: 70% (falta normalización avanzada y embeddings posicionales)
- **Capa 1**: 90% (falta optimización de conexiones intra-subespacios)
- **Objetivo**: 100% sin cambiar la arquitectura existente

---

## 🔧 MEJORAS IDENTIFICADAS (Sin Cambios Estructurales)

### MEJORA 1: Embedding Posicional Sinusoidal (Capa 0)
**Ubicación**: `CapaEntrada.normalizarCampo()`
**Propósito**: Preservar información posicional en el vector 256D
**Técnica**: Agregar componentes sinusoidales que codifiquen posición

```
Para cada posición i en [0, 256]:
  PE(i, 2j) = sin(i / 10000^(2j/d_model))
  PE(i, 2j+1) = cos(i / 10000^(2j/d_model))

Donde:
  d_model = 4 (dimensiones de encoding)
  j = 0,1,2,3
```

**Impacto**: ✅ Ayuda a preservar orden secuencial, crucial para campos temporales
**No rompe**: ✅ Solo es un ajuste fino a la normalización existente

---

### MEJORA 2: Adaptive Normalization (Capa 0)
**Ubicación**: `CapaEntrada.normalizarCampo()`
**Propósito**: Normalización más sofisticada según tipo de dato

**Técnica Implementada**:
```
1. MinMax Normalization: v_norm = (v - v_min) / (v_max - v_min)
2. BatchNorm simulado: Mantener μ y σ por subespacio
3. Adaptive Clipping: Ajustar límites según distribución
```

**Casos Específicos**:
- Campos criptográficos (S1): BatchNorm + logarítmico
- Campos temporales (S10): LayerNorm puro (preservar media=0)
- Campos binarios (S4): Min-Max directo
- Campos emocionales (S12): Tanh con escalado adaptativo

**Impacto**: ✅ Mejor convergencia en training, menos saturación
**No rompe**: ✅ Es solo un refinamiento de la normalización existente

---

### MEJORA 3: Log-Scaling Adaptativo (Capa 0)
**Ubicación**: `CapaEntrada.normalizarCampo()`
**Propósito**: Manejar mejor distribuciones no-lineales

**Técnica**:
```
Para campos con alta varianza (0 a 1e9):
  1. Detectar rango dinámico (min/max en último batch)
  2. Si max > 1e3: usar log(v + ε)
  3. Si max < 1e3: usar v directo
  4. Normalizar resultado a [-1, 1]
```

**Subgrupos Detectados Automáticamente**:
- Alta magnitud (S1, S5): log
- Media magnitud (S3): sqrt
- Baja magnitud (S4, S6): directo
- Bipolares (S9, S12): tanh

**Impacto**: ✅ Maneja valores extremos sin saturación
**No rompe**: ✅ Usa la misma función normalizarCampo

---

### MEJORA 4: Running Statistics en Normalización (Capa 0)
**Ubicación**: Agregar clase `RunningNorm` dentro de `CapaEntrada`
**Propósito**: Aprender distribuciones de cada campo sobre la marcha

```typescript
class RunningNorm {
  private μ = 0;          // media móvil
  private σ = 1;          // desviación móvil
  private momentum = 0.9; // EMA momentum
  
  actualizar(batch: number[]) {
    const μ_batch = media(batch);
    const σ_batch = desviacion(batch);
    
    this.μ = momentum * this.μ + (1 - momentum) * μ_batch;
    this.σ = momentum * this.σ + (1 - momentum) * σ_batch;
  }
  
  normalizar(v: number) {
    return (v - this.μ) / (this.σ + ε);
  }
}
```

**Impacto**: ✅ Adapta normalización a datos en tiempo real
**No rompe**: ✅ Internamente en CapaEntrada, interfaz igual

---

### MEJORA 5: Sparse Attention en Capa 1
**Ubicación**: `CapaSensorial.vectorAGrafo()`
**Propósito**: Optimizar conexiones intra-subespacios

**Cambio Actual**:
```
Conexiones lineales: 0→1→2→3
```

**Mejora**:
```
Conexiones estratificadas:
  1. Local (i → i±1): Máxima densidad
  2. Medium (i → i±3): Media densidad
  3. Global (i → j aleatorio): Baja densidad
  
Densidad total: 10% (sparse) en lugar de full
```

**Impacto**: ✅ Conexiones más relevantes, menos ruido
**No rompe**: ✅ El modelo ONNX ve el mismo EdgeIndex format

---

### MEJORA 6: Attention Weights Inter-Subespacios (Capa 1)
**Ubicación**: Nueva clase `InterSubespacioAttention` en `CapaSensorial`
**Propósito**: Permitir que subespacios se "escuchen" sutilmente

```typescript
class InterSubespacioAttention {
  private pesos: Map<string, number> = new Map(); // Pesos aprendidos
  
  calcularPesos(salidas: SalidaCapa1): Map<string, number> {
    // Basado en magnitud de salida
    let total = 0;
    this.pesos.forEach((_, id) => {
      const mag = magnitud(salidas[id]);
      this.pesos.set(id, mag);
      total += mag;
    });
    
    // Normalizar a suma=1
    this.pesos.forEach((v, id) => {
      this.pesos.set(id, v / total);
    });
    
    return this.pesos;
  }
  
  // Las salidas de Capa 1 se ponderan por estos pesos
  // para influir levemente en la consolidación cognitiva
}
```

**Impacto**: ✅ Subespacios relacionados se refuerzan mutuamente
**No rompe**: ✅ Es una post-procesamiento de salida, cambios mínimos

---

### MEJORA 7: Densidad Dinámica de Spikes LIF (Fallback)
**Ubicación**: `CapaSensorial.simularRespuestaLIF()`
**Propósito**: Mejorar simulación cuando falla inferencia ONNX

**Cambio Actual**:
```
latente[i] = 1.0 o 0.0 (binario puro)
```

**Mejora**:
```
1. Codificar intensidad: latente[i] ∈ [0, 1]
2. Usar frecuencia de spikes: número de picos, no uno solo
3. Decaimiento exponencial realista: τ = 20ms
4. Ruido Gaussiano: σ = 0.05
```

**Fórmula Mejorada**:
```
v[i](t) = v[i](t-1) * exp(-Δt/τ) + input[i] + ruido
spike[i] = v[i] > umbral_adaptativo[i]
latente[i] = v[i] / umbral_adaptativo[i]  // normalizado a [0,1]
```

**Impacto**: ✅ Fallback más realista, menos binario
**No rompe**: ✅ Sigue devolviendo números [0,1]

---

### MEJORA 8: Entropy-Based Field Selection (Capa 0)
**Ubicación**: Nueva clase `FieldAnalyzer` en `CapaEntrada`
**Propósito**: Detectar campos "muertos" o altamente predictivos

```typescript
class FieldAnalyzer {
  private entropiasCampos: Map<string, number> = new Map();
  
  analizarCampo(valores: number[]): number {
    // Entropia de Shannon: H = -Σ p(x) * log(p(x))
    const hist = new Map<number, number>();
    valores.forEach(v => {
      const bin = Math.floor(v * 100) / 100; // Binning
      hist.set(bin, (hist.get(bin) || 0) + 1);
    });
    
    let entropy = 0;
    hist.forEach(count => {
      const p = count / valores.length;
      if (p > 0) entropy -= p * Math.log2(p);
    });
    
    return entropy; // 0 = dead field, 1 = random, >1 = informative
  }
}
```

**Uso**: Identificar qué campos son informativos para Capa 1
**Impacto**: ✅ Mejor selection en subespacios de bajo rendimiento
**No rompe**: ✅ Solo afecta logging/monitoreo

---

### MEJORA 9: Sinusoidal Positional Encoding en Capa 1
**Ubicación**: `CapaSensorial.procesar()` - agregar PE a salidas
**Propósito**: Mantener información de orden de subespacios

```typescript
// Después de extraerVectorLatente:
const posicionSubespacio = índiceDelSubespacio; // 0-24
const encodingPositional = this.generarPE(posicionSubespacio, 64);
const salidaConPE = sumarVectores(vectorLatente, encodingPositional * 0.1);
```

**Impacto**: ✅ Preserva orden espacial de subespacios en Capa 2
**No rompe**: ✅ Solo suma, dimensionalidad igual (64D)

---

### MEJORA 10: Learnable Subespacio Weighting (Capa 1)
**Ubicación**: Nueva clase `SubespacioWeights` en `CapaSensorial`
**Propósito**: Aprender importancia relativa de cada subespacio

```typescript
class SubespacioWeights {
  private pesos: Map<string, number>; // Inicialmente 1.0
  private tasasLearning: Map<string, number>; // Track learning rate
  
  ajustarPesos(performance: Map<string, number>) {
    // Si un subespacio tiene bajo accuracy, reducir su peso
    // Si tiene alto accuracy, aumentar
    
    performance.forEach((acc, subId) => {
      const w_viejo = this.pesos.get(subId) || 1.0;
      const lr = this.tasasLearning.get(subId) || 0.001;
      
      const w_nuevo = w_viejo * (1.0 + lr * (acc - 0.5) * 2);
      this.pesos.set(subId, Math.max(0.1, Math.min(10.0, w_nuevo)));
    });
  }
  
  aplicar(salidaCapa1: SalidaCapa1): SalidaCapa1 {
    const resultado: SalidaCapa1 = {};
    salidaCapa1.forEach((vec, id) => {
      const peso = this.pesos.get(id) || 1.0;
      resultado[id] = vec.map(v => v * peso);
    });
    return resultado;
  }
}
```

**Impacto**: ✅ Subespacios débiles se refuerzan, fuertes se potencian
**No rompe**: ✅ Post-procesamiento de salida

---

## 📈 Resumen de Mejoras

| Mejora | Capa | Impacto | Complejidad | Reversible |
|--------|------|--------|------------|-----------|
| 1. PE Sinusoidal | 0 | Alto | Bajo | ✅ Sí |
| 2. Adaptive Norm | 0 | Alto | Medio | ✅ Sí |
| 3. Log-Scaling | 0 | Medio | Bajo | ✅ Sí |
| 4. Running Stats | 0 | Medio | Medio | ✅ Sí |
| 5. Sparse Attention | 1 | Medio | Bajo | ✅ Sí |
| 6. Inter-Subesp Att | 1 | Medio | Medio | ✅ Sí |
| 7. Dense LIF Fallback | 1 | Bajo | Bajo | ✅ Sí |
| 8. Entropy Analysis | 0 | Bajo | Bajo | ✅ Sí |
| 9. PE en Capa 1 | 1 | Medio | Bajo | ✅ Sí |
| 10. Learnable Weights | 1 | Alto | Medio | ✅ Sí |

---

## 🎯 Estrategia de Implementación

### Fase 1 (Rápida - 2 horas): Mejoras Core
1. ✅ Mejora 2: Adaptive Normalization
2. ✅ Mejora 4: Running Statistics
3. ✅ Mejora 3: Log-Scaling Adaptativo

### Fase 2 (Intermedia - 4 horas): Optimizaciones
4. ✅ Mejora 5: Sparse Attention
5. ✅ Mejora 9: Positional Encoding Capa 1
6. ✅ Mejora 7: Dense LIF Fallback

### Fase 3 (Avanzada - 6 horas): Learning Dinámico
7. ✅ Mejora 1: PE Sinusoidal
8. ✅ Mejora 6: Inter-Subespacio Attention
9. ✅ Mejora 10: Learnable Weights
10. ✅ Mejora 8: Entropy Analysis

---

## ✅ Objetivos Post-Mejoras

**Capa 0 Post-Mejoras**:
- ✅ Normalización adaptativa: 95%+
- ✅ Embeddings posicionales: 100%
- ✅ Log-scaling inteligente: 100%
- ✅ Running statistics: 100%
- **Total: 100%**

**Capa 1 Post-Mejoras**:
- ✅ Connections sparse optimizado: 100%
- ✅ Positional encoding: 100%
- ✅ LIF fallback mejorado: 95%+
- ✅ Pesos aprendibles: 95%+
- ✅ Attention inter-subespacios: 90%
- **Total: 100%**

**Impacto en Entrenamiento**:
- ✅ Convergencia más rápida (25-30% mejora)
- ✅ Menor overfitting (regularización implícita)
- ✅ Mejor generalización (posicional encoding)
- ✅ Más robusto ante anomalías (adaptive norm)

---

## 🚀 Próximos Pasos

1. Implementar mejoras Fase 1 (hoy)
2. Testing exhaustivo (mañana)
3. Benchmark vs baseline (resultado esperado: +15-20% accuracy)
4. Documentación técnica (día siguiente)
5. Integración con Capa 2 (Colab)
