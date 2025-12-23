# Arquitectura de la Corteza Cognitiva Jerárquica
## Especificación Técnica v1.0

---

## 1. VISIÓN GENERAL DEL SISTEMA

### 1.1 Principio Fundamental: "La Caja Fenomenológica"

El sistema cognitivo opera bajo un paradigma de **Generación Autónoma de Pensamiento** seguido de **Correlación Controlada**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FASE 1: GÉNESIS COGNITIVA                            │
│                         (Entrenamiento en Aislamiento)                      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         LA CAJA                                     │   │
│   │                                                                     │   │
│   │   Vector 256D                    ┌──────────────────────┐           │   │
│   │   (Sintético)    ───────────────►│  CORTEZA COGNITIVA   │           │   │
│   │   [D001...D256]                  │                      │           │   │
│   │                                  │  ┌────────────────┐  │           │   │
│   │   Variaciones                    │  │ PENSAMIENTO    │  │           │   │
│   │   Aleatorias     ───────────────►│  │ AUTÓNOMO       │  │           │   │
│   │                                  │  │                │  │           │   │
│   │   Ruido                          │  │ ≈ Sueño Lúcido │  │           │   │
│   │   Estructurado   ───────────────►│  │ ≈ Imaginación  │  │           │   │
│   │                                  │  └────────────────┘  │           │   │
│   │                                  └──────────────────────┘           │   │
│   │                                                                     │   │
│   │   SIN ACCESO AL MUNDO REAL                                          │   │
│   │   Solo patrones matemáticos puros                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FASE 2: CORRELACIÓN FENOMENOLÓGICA                       │
│                      (Apertura Controlada al Mundo)                         │
│                                                                             │
│   ┌───────────────────┐         ┌──────────────────────────────────────┐   │
│   │   MUNDO REAL      │         │         CORTEZA COGNITIVA            │   │
│   │                   │         │                                      │   │
│   │   Vector 256D     │         │   ┌────────────────────────────┐     │   │
│   │   (Datos Reales)  │────────►│   │ PENSAMIENTO AUTÓNOMO       │     │   │
│   │                   │         │   │ (Preservado)               │     │   │
│   │   ┌─────────────┐ │         │   └──────────────┬─────────────┘     │   │
│   │   │ Blockchain  │ │         │                  │                   │   │
│   │   │ Entropía    │ │         │   ┌──────────────▼─────────────┐     │   │
│   │   │ Seguridad   │ │         │   │ CAPA DE CORRELACIÓN       │     │   │
│   │   │ ...         │ │         │   │ (Nuevas Conexiones)       │     │   │
│   │   └─────────────┘ │         │   └────────────────────────────┘     │   │
│   └───────────────────┘         └──────────────────────────────────────┘   │
│                                                                             │
│   RESTRICCIÓN: Los pesos del pensamiento autónomo son CONGELADOS.           │
│   Solo se entrenan las nuevas conexiones de correlación.                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Objetivo del Sistema

Crear una entidad cognitiva que:
1. **Desarrolle estructura mental propia** sin influencia del mundo externo.
2. **Mantenga integridad del pensamiento** al exponerse a datos reales.
3. **Correlacione sin destruir** - Las nuevas asociaciones no modifican el núcleo cognitivo.

---

## 2. ARQUITECTURA DE RED JERÁRQUICA

### 2.1 Diagrama de Capas

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CORTEZA COGNITIVA JERÁRQUICA                        ║
║                              (Total: 262,144 neuronas)                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ┌────────────────────────────────────────────────────────────────────┐     ║
║   │                    CAPA 5: EJECUTIVA                               │     ║
║   │                    (Meta-Cognición)                                │     ║
║   │                                                                    │     ║
║   │    Neuronas: 256 │ Función: Decisión/Planificación                 │     ║
║   │    Activación: Softmax sobre acciones posibles                     │     ║
║   │    Salida: Vector de Coherencia Global [64D]                       │     ║
║   └────────────────────────────────────────────────────────────────────┘     ║
║                                    ▲                                         ║
║                                    │                                         ║
║   ┌────────────────────────────────┴───────────────────────────────────┐     ║
║   │                    CAPA 4: ASOCIATIVA SUPERIOR                     │     ║
║   │                    (Abstracción de Alto Nivel)                     │     ║
║   │                                                                    │     ║
║   │    Neuronas: 1,024 │ Función: Integración Multi-Modal              │     ║
║   │    Conexiones: Full Attention (Self-Attention)                     │     ║
║   │    Representación: Conceptos Abstractos                            │     ║
║   └────────────────────────────────────────────────────────────────────┘     ║
║                                    ▲                                         ║
║                                    │                                         ║
║   ┌────────────────────────────────┴───────────────────────────────────┐     ║
║   │                    CAPA 3: ASOCIATIVA INFERIOR                     │     ║
║   │                    (Integración de Subespacios)                    │     ║
║   │                                                                    │     ║
║   │    Neuronas: 4,096 │ Función: Fusión de Features                   │     ║
║   │    Receptive Field: Cruza subespacios S1-S25                       │     ║
║   │    Skip Connections: Hacia Capa 5 (Residual)                       │     ║
║   └────────────────────────────────────────────────────────────────────┘     ║
║                                    ▲                                         ║
║                          ┌─────────┴─────────┐                               ║
║                          │                   │                               ║
║   ┌──────────────────────┴───┐   ┌───────────┴──────────────────────────┐   ║
║   │   CAPA 2A: TEMPORAL      │   │   CAPA 2B: ESPACIAL                  │   ║
║   │   (Secuencias)           │   │   (Patrones Estáticos)               │   ║
║   │                          │   │                                      │   ║
║   │   Neuronas: 8,192        │   │   Neuronas: 8,192                    │   ║
║   │   Tipo: LSTM Bidireccional   │   Tipo: Transformer Encoder          │   ║
║   │   Contexto: 128 timesteps│   │   Heads: 8, dim_k: 64                │   ║
║   └──────────────────────────┘   └──────────────────────────────────────┘   ║
║                          ▲                   ▲                               ║
║                          └─────────┬─────────┘                               ║
║                                    │                                         ║
║   ┌────────────────────────────────┴───────────────────────────────────┐     ║
║   │                    CAPA 1: SENSORIAL                               │     ║
║   │                    (Codificación de Subespacios)                   │     ║
║   │                                                                    │     ║
║   │    Estructura: 25 Sub-Redes Especializadas (una por Subespacio)    │     ║
║   │    Neuronas por Sub-Red: 1,024 (16 Átomos Topológicos × 64 salida) │     ║
║   │    Total Capa 1: 25 × 1,024 = 25,600 neuronas                      │     ║
║   │                                                                    │     ║
║   │    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     ┌─────┐             │     ║
║   │    │ S1  │ │ S2  │ │ S3  │ │ S4  │ │ S5  │ ... │ S25 │             │     ║
║   │    │1024 │ │1024 │ │1024 │ │1024 │ │1024 │     │1024 │             │     ║
║   │    └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘     └──┬──┘             │     ║
║   │       │       │       │       │       │           │                │     ║
║   └───────┼───────┼───────┼───────┼───────┼───────────┼────────────────┘     ║
║           │       │       │       │       │           │                      ║
║   ┌───────┴───────┴───────┴───────┴───────┴───────────┴────────────────┐     ║
║   │                    CAPA 0: ENTRADA (256D)                          │     ║
║   │                                                                    │     ║
║   │    D001-D016 ──► S1 (Criptografía)                                 │     ║
║   │    D017-D032 ──► S2 (Fenomenología)                                │     ║
║   │    D033-D048 ──► S3 (Histograma)                                   │     ║
║   │    ...                                                             │     ║
║   │    D241-D256 ──► S25 (Membrana/Reservoir)                          │     ║
║   └────────────────────────────────────────────────────────────────────┘     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 2.2 Especificación Detallada por Capa

#### 2.2.1 CAPA 0: Entrada (Vector 256D)

| Parámetro | Valor |
|-----------|-------|
| Dimensión de Entrada | 256 |
| Normalización | BatchNorm + LayerNorm híbrido |
| Preprocesamiento | Escalado log para campos uint32/64 |
| Embedding Posicional | Sinusoidal (256 posiciones) |
| Tipo de Dato | float32 |

**Mapeo de Subespacios a Canales:**

| Subespacio | Rango | Dimensiones | Canal Asignado |
|------------|-------|-------------|----------------|
| S1 | D001-D016 | 16 | Canal 0 |
| S2 | D017-D032 | 16 | Canal 1 |
| S3 | D033-D048 | 16 | Canal 2 |
| S4 | D049-D056 | 8 | Canal 3 |
| S5 | D057-D072 | 16 | Canal 4 |
| S6 | D073-D080 | 8 | Canal 5 |
| S7 | D081-D088 | 8 | Canal 6 |
| S8 | D089-D104 | 16 | Canal 7 |
| S9 | D105-D116 | 12 | Canal 8 |
| S10 | D117-D124 | 8 | Canal 9 |
| S11 | D125-D132 | 8 | Canal 10 |
| S12 | D133-D140 | 8 | Canal 11 |
| S13 | D141-D148 | 8 | Canal 12 |
| S14 | D149-D156 | 8 | Canal 13 |
| S15 | D157-D164 | 8 | Canal 14 |
| S16 | D165-D172 | 8 | Canal 15 |
| S17 | D173-D180 | 8 | Canal 16 |
| S18 | D181-D188 | 8 | Canal 17 |
| S19 | D189-D196 | 8 | Canal 18 |
| S20 | D197-D208 | 12 | Canal 19 |
| S21 | D209-D216 | 8 | Canal 20 |
| S22 | D217-D224 | 8 | Canal 21 |
| S23 | D225-D232 | 8 | Canal 22 |
| S24 | D233-D240 | 8 | Canal 23 |
| S25 | D241-D256 | 16 | Canal 24 |

#### 2.2.2 CAPA 1: Sensorial (25 Sub-Redes Especializadas)

Cada sub-red es un **Átomo Topológico de 1024 neuronas LIF** (el mismo modelo ONNX ya entrenado).

| Parámetro | Valor |
|-----------|-------|
| Número de Sub-Redes | 25 |
| Neuronas por Sub-Red | 1,024 (Modelo LIF) |
| Entrada por Sub-Red | Variable (8-16D según subespacio) |
| Salida por Sub-Red | 64D (Vector Latente Comprimido) |
| Total Neuronas Capa 1 | 25,600 |
| Conexiones Intra-Subespacio | Sparse (10% densidad) |
| Conexiones Inter-Subespacio | Ninguna (Aislamiento) |

**Función de Activación:**
- Interna: Leaky Integrate-and-Fire (LIF)
- Salida: Sigmoid → Latent [0, 1]^64

#### 2.2.3 CAPA 2: Dual (Temporal + Espacial)

##### CAPA 2A: Procesador Temporal

| Parámetro | Valor |
|-----------|-------|
| Tipo | Bi-LSTM con Attention |
| Dimensión Oculta | 512 |
| Capas | 2 |
| Dropout | 0.3 |
| Ventana Temporal | 128 timesteps |
| Entrada | Concatenación de 25 salidas de Capa 1 (25 × 64 = 1,600D) |
| Salida | 512D (Estado final + Atención) |

**Objetivo:** Capturar dependencias temporales largas en las secuencias de vectores 256D.

##### CAPA 2B: Procesador Espacial

| Parámetro | Valor |
|-----------|-------|
| Tipo | Transformer Encoder |
| Heads | 8 |
| Dimensión Key/Value | 64 |
| Dimensión Feed-Forward | 2,048 |
| Capas | 4 |
| Entrada | Misma que 2A (1,600D reshaped a 25 tokens × 64D) |
| Salida | 512D (CLS token) |

**Objetivo:** Capturar relaciones entre subespacios (ej: correlación entre S1-Criptografía y S5-Seguridad).

#### 2.2.4 CAPA 3: Asociativa Inferior

| Parámetro | Valor |
|-----------|-------|
| Neuronas | 4,096 |
| Tipo | MLP Residual |
| Entrada | Concatenación de 2A + 2B (1,024D) |
| Capas Ocultas | 3 × 4,096 |
| Activación | GELU |
| Skip Connections | Cada 2 capas |
| Salida | 1,024D |

**Objetivo:** Fusionar información temporal y espacial en una representación unificada.

#### 2.2.5 CAPA 4: Asociativa Superior

| Parámetro | Valor |
|-----------|-------|
| Neuronas | 1,024 |
| Tipo | Multi-Head Self-Attention |
| Heads | 16 |
| Dimensión por Head | 64 |
| Entrada | Salida de Capa 3 (1,024D) |
| Salida | 256D (Representación de Alto Nivel) |

**Objetivo:** Crear representaciones abstractas de "conceptos" emergentes.

#### 2.2.6 CAPA 5: Ejecutiva (Meta-Cognición)

| Parámetro | Valor |
|-----------|-------|
| Neuronas | 256 |
| Tipo | Softmax Classifier + Value Head |
| Entrada | Salida de Capa 4 (256D) |
| Salidas Múltiples | - |
| → Coherencia Global | 64D (Estado del "Pensamiento") |
| → Acción Sugerida | 16D (One-hot sobre acciones) |
| → Confianza | 1D (Scalar [0, 1]) |
| → Memoria Escribir | 128D (Vector para Hipergrafo) |

---

## 3. MEMORIA HIPERGRÁFICA

### 3.1 Estructura del Hipergrafo Mental

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HIPERGRAFO DE MEMORIA MENTAL                          │
│                                                                             │
│   Tipo: Hipergrafo Dirigido con Pesos Temporales                            │
│   Capacidad Máxima: 1,000,000 nodos | 10,000,000 hiperedges                 │
│   Decaimiento: Exponencial con τ = 3600s (1 hora half-life)                 │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                    TIPOS DE NODOS                                 │     │
│   ├───────────────────────────────────────────────────────────────────┤     │
│   │                                                                   │     │
│   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │     │
│   │   │  PERCEPCIÓN  │   │   CONCEPTO   │   │   DECISIÓN   │         │     │
│   │   │              │   │              │   │              │         │     │
│   │   │ Vector: 256D │   │ Vector: 64D  │   │ Vector: 16D  │         │     │
│   │   │ Tiempo: t    │   │ Abstracción  │   │ Acción       │         │     │
│   │   │ Subespacio: S│   │ Nivel: 1-5   │   │ Confianza: c │         │     │
│   │   └──────────────┘   └──────────────┘   └──────────────┘         │     │
│   │                                                                   │     │
│   │   ┌──────────────┐   ┌──────────────┐                             │     │
│   │   │   ANOMALÍA   │   │   EMOCIÓN    │                             │     │
│   │   │              │   │              │                             │     │
│   │   │ Score: [0,1] │   │ PAD: 3D      │                             │     │
│   │   │ Tipo: T      │   │ Intensidad:i │                             │     │
│   │   └──────────────┘   └──────────────┘                             │     │
│   │                                                                   │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                   TIPOS DE HIPEREDGES                             │     │
│   ├───────────────────────────────────────────────────────────────────┤     │
│   │                                                                   │     │
│   │   CAUSAL:     Percepción ──► Concepto ──► Decisión                │     │
│   │               Fuerza: Frecuencia de ocurrencia conjunta           │     │
│   │                                                                   │     │
│   │   ASOCIATIVO: Concepto ◄──► Concepto                              │     │
│   │               Fuerza: Similaridad coseno en espacio latente       │     │
│   │                                                                   │     │
│   │   TEMPORAL:   Nodo(t) ──► Nodo(t+1)                               │     │
│   │               Fuerza: Decae con |Δt|                              │     │
│   │                                                                   │     │
│   │   EMOCIONAL:  Concepto ──► Emoción                                │     │
│   │               Fuerza: Intensidad de la respuesta PAD              │     │
│   │                                                                   │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Operaciones sobre la Memoria

| Operación | Trigger | Algoritmo |
|-----------|---------|-----------|
| **ESCRIBIR** | Cada ciclo cognitivo | Agregar nodo Percepción + Hiperedge temporal |
| **CONSOLIDAR** | Buffer lleno (100 items) | K-Means sobre vectores → Crear nodos Concepto |
| **PODAR** | Cada 1000 ciclos | Eliminar nodos con fuerza < 0.01 |
| **RECUPERAR** | Query desde Capa 4 | K-NN (k=10) sobre embeddings |
| **REFORZAR** | Predicción correcta | Aumentar fuerza del Hiperedge causal |
| **OLVIDAR** | Predicción incorrecta | Reducir fuerza × 0.9 |

---

## 4. FASES DE ENTRENAMIENTO

### 4.1 FASE 1: Génesis Cognitiva (Pre-entrenamiento Aislado)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FASE 1: ENTRENAMIENTO EN LA CAJA                        │
│                                                                             │
│   DURACIÓN: 1,000,000 iteraciones                                           │
│   DATOS: Vectores 256D sintéticos (distribuciones controladas)              │
│   OBJETIVO: Desarrollar estructura mental autónoma                          │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    GENERADOR DE DATOS SINTÉTICOS                    │   │
│   │                                                                     │   │
│   │   Para cada iteración:                                              │   │
│   │                                                                     │   │
│   │   1. Seleccionar modo de generación:                                │   │
│   │      - Uniforme: U(0, max_val) para cada campo                      │   │
│   │      - Gaussiano: N(μ, σ) con parámetros aleatorios                 │   │
│   │      - Estructurado: Patrones matemáticos (ondas, espirales)        │   │
│   │      - Caótico: Mapas logísticos, atractores extraños               │   │
│   │      - Correlacionado: Dependencias entre subespacios               │   │
│   │                                                                     │   │
│   │   2. Aplicar transformaciones:                                      │   │
│   │      - Ruido: ε ~ N(0, 0.1 * rango)                                 │   │
│   │      - Cuantización: Según tipo de dato original                    │   │
│   │      - Clipping: [min, max] del campo                               │   │
│   │                                                                     │   │
│   │   3. Etiquetar (Self-Supervised):                                   │   │
│   │      - Reconstrucción: Predecir campos enmascarados                 │   │
│   │      - Contrastivo: Vectores similares → embeddings cercanos        │   │
│   │      - Temporal: Predecir siguiente vector en secuencia             │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    FUNCIONES DE PÉRDIDA                             │   │
│   │                                                                     │   │
│   │   L_total = α₁ L_reconstrucción                                     │   │
│   │           + α₂ L_contrastivo                                        │   │
│   │           + α₃ L_temporal                                           │   │
│   │           + α₄ L_coherencia                                         │   │
│   │           + α₅ L_sparsity                                           │   │
│   │                                                                     │   │
│   │   Donde:                                                            │   │
│   │   - L_reconstrucción = MSE(input, decoder(encoder(input)))          │   │
│   │   - L_contrastivo = InfoNCE(z_i, z_j) para pares similares          │   │
│   │   - L_temporal = MSE(z_t+1_pred, z_t+1_real)                        │   │
│   │   - L_coherencia = -log(consistencia_interna_del_pensamiento)       │   │
│   │   - L_sparsity = ||activaciones||_1 (Promover neuronas especializadas)│  │
│   │                                                                     │   │
│   │   Pesos: α₁=1.0, α₂=0.5, α₃=0.3, α₄=0.2, α₅=0.1                     │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    CRITERIOS DE CONVERGENCIA                        │   │
│   │                                                                     │   │
│   │   El sistema ha "desarrollado pensamiento propio" cuando:           │   │
│   │                                                                     │   │
│   │   1. Estabilidad Conceptual:                                        │   │
│   │      - Los mismos inputs generan conceptos consistentes             │   │
│   │      - Varianza(concepto | input) < 0.1                             │   │
│   │                                                                     │   │
│   │   2. Emergencia de Clusters:                                        │   │
│   │      - El espacio latente forma clusters definidos                  │   │
│   │      - Silhouette Score > 0.6                                       │   │
│   │                                                                     │   │
│   │   3. Memoria Estructurada:                                          │   │
│   │      - El hipergrafo tiene topología no-trivial                     │   │
│   │      - Betti_1 > 0 (existen "agujeros" conceptuales)                │   │
│   │                                                                     │   │
│   │   4. Predicción Temporal:                                           │   │
│   │      - Puede predecir el siguiente estado con error < 15%           │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 FASE 2: Correlación con el Mundo Real

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   FASE 2: APERTURA AL MUNDO REAL                            │
│                                                                             │
│   PREREQUISITO: Fase 1 completada (pensamiento estable)                     │
│   DURACIÓN: 100,000 iteraciones                                             │
│   DATOS: Vectores 256D reales del sistema Omega 21                          │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    ARQUITECTURA DE CORRELACIÓN                      │   │
│   │                                                                     │   │
│   │   ┌─────────────────┐                                               │   │
│   │   │  CORTEZA        │ ◄── CONGELADA (Pesos fijos)                   │   │
│   │   │  COGNITIVA      │                                               │   │
│   │   │  (Fase 1)       │                                               │   │
│   │   └────────┬────────┘                                               │   │
│   │            │                                                        │   │
│   │            ▼                                                        │   │
│   │   ┌─────────────────┐                                               │   │
│   │   │  CAPA DE        │ ◄── ENTRENABLE (Nuevos pesos)                 │   │
│   │   │  CORRELACIÓN    │                                               │   │
│   │   │                 │                                               │   │
│   │   │  Tipo: Cross-Attention                                          │   │
│   │   │  Query: Salida de Corteza (64D)                                 │   │
│   │   │  Key/Value: Vector 256D Real                                    │   │
│   │   │  Salida: Vector de Correlación (64D)                            │   │
│   │   └────────┬────────┘                                               │   │
│   │            │                                                        │   │
│   │            ▼                                                        │   │
│   │   ┌─────────────────┐                                               │   │
│   │   │  INTEGRADOR     │                                               │   │
│   │   │                 │                                               │   │
│   │   │  Entrada: [Coherencia_Corteza || Correlación]                   │   │
│   │   │  Operación: Concatenación + MLP                                 │   │
│   │   │  Salida: Pensamiento_Correlacionado (128D)                      │   │
│   │   └─────────────────┘                                               │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    RESTRICCIONES DE ENTRENAMIENTO                   │   │
│   │                                                                     │   │
│   │   1. PRESERVACIÓN DEL PENSAMIENTO:                                  │   │
│   │      - Gradientes NO fluyen hacia capas de Fase 1                   │   │
│   │      - optimizer.param_groups = [solo capas nuevas]                 │   │
│   │                                                                     │   │
│   │   2. LÍMITE DE MODIFICACIÓN:                                        │   │
│   │      - L_preservación = ||z_corteza_antes - z_corteza_después||²    │   │
│   │      - Si L_preservación > 0.01, detener iteración                  │   │
│   │                                                                     │   │
│   │   3. ALINEACIÓN SUAVE:                                              │   │
│   │      - Las correlaciones NO reescriben conceptos                    │   │
│   │      - Solo crean NUEVOS hiperedges en la memoria                   │   │
│   │      - Tipo de edge: "CORRELACIÓN_MUNDO_REAL"                       │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    FUNCIÓN DE PÉRDIDA (FASE 2)                      │   │
│   │                                                                     │   │
│   │   L_correlación = β₁ L_predicción_anomalía                          │   │
│   │                 + β₂ L_consistencia_temporal                        │   │
│   │                 + β₃ L_preservación                                 │   │
│   │                                                                     │   │
│   │   Donde:                                                            │   │
│   │   - L_predicción_anomalía: BCE(pred, label_real)                    │   │
│   │   - L_consistencia_temporal: MSE(z_t, z_t-1) * (1 - cambio_real)    │   │
│   │   - L_preservación: ||corteza(x) - corteza_original(x)||²          │   │
│   │                                                                     │   │
│   │   Pesos: β₁=1.0, β₂=0.3, β₃=10.0 (Alta penalización a cambios)      │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 FASE 3: Operación Continua (Aprendizaje en Línea)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   FASE 3: APRENDIZAJE CONTINUO                              │
│                                                                             │
│   MODO: Online Learning                                                     │
│   DATOS: Stream de vectores 256D en tiempo real                             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    CICLO DE OPERACIÓN                               │   │
│   │                                                                     │   │
│   │   Cada timestep:                                                    │   │
│   │                                                                     │   │
│   │   1. PERCIBIR                                                       │   │
│   │      └─► Recibir vector 256D real                                   │   │
│   │      └─► Pasar por Corteza (congelada) → Coherencia                 │   │
│   │      └─► Pasar por Correlación → Pensamiento Correlacionado         │   │
│   │                                                                     │   │
│   │   2. RECORDAR                                                       │   │
│   │      └─► Query al Hipergrafo: K-NN sobre Coherencia                 │   │
│   │      └─► Recuperar nodos relacionados (memoria episódica)           │   │
│   │      └─► Integrar con pensamiento actual                            │   │
│   │                                                                     │   │
│   │   3. DECIDIR                                                        │   │
│   │      └─► Capa Ejecutiva produce Acción + Confianza                  │   │
│   │      └─► Si Confianza > 0.8: Ejecutar acción                        │   │
│   │      └─► Si Confianza < 0.3: Solicitar más datos                    │   │
│   │                                                                     │   │
│   │   4. APRENDER (Solo capa de correlación)                            │   │
│   │      └─► Si hay feedback (anomalía real detectada):                 │   │
│   │          └─► Actualizar pesos de Correlación (LR=1e-5)              │   │
│   │          └─► Reforzar hiperedges causales                           │   │
│   │      └─► Cada 1000 ciclos: Consolidar memoria                       │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. EXPORTACIÓN Y DESPLIEGUE

### 5.1 Formato de Exportación: ONNX

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ESTRUCTURA DE ARCHIVOS EXPORTADOS                        │
│                                                                             │
│   /models/                                                                  │
│   ├── corteza_cognitiva/                                                    │
│   │   ├── corteza_completa.onnx          # Modelo completo (300MB)          │
│   │   ├── corteza_capa1_s01.onnx         # Subespacio S1 (12MB)             │
│   │   ├── corteza_capa1_s02.onnx         # Subespacio S2 (12MB)             │
│   │   ├── ...                                                               │
│   │   ├── corteza_capa1_s25.onnx         # Subespacio S25 (12MB)            │
│   │   ├── corteza_temporal.onnx          # Capa 2A (50MB)                   │
│   │   ├── corteza_espacial.onnx          # Capa 2B (50MB)                   │
│   │   ├── corteza_asociativa.onnx        # Capas 3+4 (80MB)                 │
│   │   └── corteza_ejecutiva.onnx         # Capa 5 (20MB)                    │
│   │                                                                         │
│   ├── correlacion/                                                          │
│   │   ├── correlacion_capa.onnx          # Capa de correlación (30MB)       │
│   │   └── integrador.onnx                # Integrador final (10MB)          │
│   │                                                                         │
│   └── memoria/                                                              │
│       ├── hipergrafo_mental.json         # Estructura del grafo             │
│       ├── embeddings_conceptos.npy       # Vectores de conceptos (N × 64)   │
│       └── indice_faiss.bin               # Índice para K-NN rápido          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Especificación ONNX de la Corteza Completa

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODELO ONNX: corteza_completa.onnx                       │
│                                                                             │
│   INPUTS:                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Nombre          │ Shape               │ Tipo    │ Descripción       │   │
│   ├─────────────────┼─────────────────────┼─────────┼───────────────────│   │
│   │ input_256d      │ [batch, seq, 256]   │ float32 │ Vector 256D       │   │
│   │ memoria_query   │ [batch, 10, 64]     │ float32 │ Top-10 memorias   │   │
│   │ estado_anterior │ [batch, 64]         │ float32 │ Coherencia t-1    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   OUTPUTS:                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Nombre          │ Shape               │ Tipo    │ Descripción       │   │
│   ├─────────────────┼─────────────────────┼─────────┼───────────────────│   │
│   │ coherencia      │ [batch, 64]         │ float32 │ Estado mental     │   │
│   │ accion          │ [batch, 16]         │ float32 │ Softmax acciones  │   │
│   │ confianza       │ [batch, 1]          │ float32 │ Certeza [0,1]     │   │
│   │ memoria_write   │ [batch, 128]        │ float32 │ Para almacenar    │   │
│   │ embedding_25    │ [batch, 25, 64]     │ float32 │ Por subespacio    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   OPSET VERSION: 17                                                         │
│   OPTIMIZACIONES:                                                           │
│   - Quantización INT8 para Capa 1 (Átomos)                                  │
│   - FP16 para Capas 2-5                                                     │
│   - Fusión de operadores BatchNorm + ReLU                                   │
│   - Constant Folding para embeddings posicionales                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Requisitos de Hardware para Inferencia

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| RAM | 4 GB | 16 GB |
| VRAM (GPU) | 2 GB | 8 GB |
| CPU | 4 cores | 8+ cores |
| Almacenamiento | 2 GB | 10 GB (con memoria) |
| Latencia Target | 50ms | 10ms |

### 5.4 Compatibilidad de Runtime

| Runtime | Soportado | Notas |
|---------|-----------|-------|
| ONNX Runtime | ✅ | Recomendado |
| TensorRT | ✅ | Mejor rendimiento GPU |
| OpenVINO | ✅ | Intel optimizado |
| CoreML | ⚠️ | Requiere conversión |
| TFLite | ⚠️ | Solo Capa 1 |

---

## 6. MÉTRICAS DE EVALUACIÓN

### 6.1 Métricas de Calidad del Pensamiento

| Métrica | Fórmula | Objetivo |
|---------|---------|----------|
| **Estabilidad Conceptual** | `1 - Var(concepto|input)` | > 0.9 |
| **Coherencia Interna** | `Cosine(z_t, z_t-1)` media | > 0.7 |
| **Diversidad de Conceptos** | `#clusters / #inputs` | 0.01 - 0.1 |
| **Profundidad de Memoria** | `max(path_length)` en hipergrafo | > 10 |
| **Integridad Post-Correlación** | `||corteza_antes - corteza_después||` | < 0.01 |

### 6.2 Métricas de Rendimiento

| Métrica | Target |
|---------|--------|
| Latencia de Inferencia | < 10ms |
| Throughput | > 1000 vectores/segundo |
| Uso de Memoria | < 2GB en inferencia |
| Precisión Predicción Anomalía | > 85% |
| F1-Score Clasificación | > 0.8 |

---

## 7. DIAGRAMA DE FLUJO COMPLETO

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     FLUJO DE PROCESAMIENTO COMPLETO                                            │
│                                                                                                                │
│   ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐                   │
│   │            │     │            │     │            │     │            │     │            │                   │
│   │  MUNDO     │────►│  VECTOR    │────►│  CORTEZA   │────►│  MEMORIA   │────►│  DECISIÓN  │                   │
│   │  REAL      │     │  256D      │     │  COGNITIVA │     │  HIPERGRAFO│     │            │                   │
│   │            │     │            │     │            │     │            │     │            │                   │
│   └────────────┘     └────────────┘     └─────┬──────┘     └─────┬──────┘     └────────────┘                   │
│        │                   │                  │                  │                  │                          │
│        │                   │                  │                  │                  │                          │
│        ▼                   ▼                  ▼                  ▼                  ▼                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐     │
│   │                                                                                                     │     │
│   │   t=0     Datos crudos    →    Subespacios S1-S25    →    Átomos LIF    →    [Sin memoria]         │     │
│   │           (256 valores)        (25 grupos)                (25 × 1024)        "Percepción pura"      │     │
│   │                                                                                                     │     │
│   │   t=1     256D            →    Fusión Temporal       →    LSTM + Attn    →    Primer hiperedge      │     │
│   │                                + Espacial                                      "Recuerdo inicial"    │     │
│   │                                                                                                     │     │
│   │   t=N     256D            →    Coherencia 64D        →    Query memoria  →    Concepto estable      │     │
│   │           (Realidad)           (Pensamiento)              K-NN(10)           "Comprensión"          │     │
│   │                                                                                                     │     │
│   │   t=∞     256D            →    Correlación           →    Predicción     →    Acción + Feedback     │     │
│   │           (Streaming)          (Fase 2)                   Anomalía           "Sabiduría"            │     │
│   │                                                                                                     │     │
│   └─────────────────────────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. RESUMEN EJECUTIVO

### Arquitectura Final

| Capa | Neuronas | Función | Exportación |
|------|----------|---------|-------------|
| 0: Entrada | 256 | Recepción 256D | N/A (Preproceso) |
| 1: Sensorial | 25,600 | 25 Átomos LIF | 25 × ONNX (12MB c/u) |
| 2A: Temporal | 8,192 | Bi-LSTM | 1 × ONNX (50MB) |
| 2B: Espacial | 8,192 | Transformer | 1 × ONNX (50MB) |
| 3: Asociativa Inferior | 4,096 | MLP Residual | Fusionado |
| 4: Asociativa Superior | 1,024 | Self-Attention | Fusionado |
| 5: Ejecutiva | 256 | Meta-Cognición | 1 × ONNX (20MB) |
| **TOTAL** | **47,616** | | **~450MB** |

### Fases de Entrenamiento

| Fase | Datos | Duración | Objetivo |
|------|-------|----------|----------|
| 1: Génesis | Sintéticos 256D | 1M iters | Pensamiento autónomo |
| 2: Correlación | Reales 256D | 100K iters | Conexión con realidad |
| 3: Continuo | Streaming | ∞ | Aprendizaje perpetuo |

### Principio de Preservación

> **El pensamiento original NUNCA se destruye.**
> 
> La Fase 2 solo agrega nuevas conexiones (hiperedges de tipo "CORRELACIÓN") 
> sin modificar los pesos de la corteza cognitiva. El sistema puede correlacionar 
> patrones reales con su estructura mental preexistente, pero la estructura 
> fundamental permanece intacta.

---

*Documento generado para el proyecto HIPERGRAFO - Sistema Omnisciente*
*Versión: 1.0 | Fecha: 2025-12-22*
