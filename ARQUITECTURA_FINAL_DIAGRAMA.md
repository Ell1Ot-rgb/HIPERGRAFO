# ARQUITECTURA FINAL - DIAGRAMA TÉCNICO

## 1. Flujo de Datos Completo

```
┌───────────────────────────────────────────────────────────────────────┐
│                      ENTRADA: VECTOR 256D                             │
│                   (Sensores / Simulador)                              │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│            CAPA 0: EXTRACCIÓN DENDRÍTICA                              │
│        MapeoVector256DaDendritas.extraerCamposDendriticos()          │
│                   Extrae D001-D056 (16 señales)                       │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│            CAPA 1: PROCESAMIENTO CON 25 ÁTOMOS                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ S1  │ S2  │ S3  │ ... │ S25  (25 Átomos Topológicos)           │  │
│  │ ┌────────────────────────────────────────┐                    │  │
│  │ │ 1. Simulador.configurarDendritas()    │                    │  │
│  │ │    (Recibe D001-D056)                 │                    │  │
│  │ │ 2. Simulador.generarMuestra()         │                    │  │
│  │ │    (Produce telemetría estabilizada)  │                    │  │
│  │ │ 3. Cerebro.predecir()                 │                    │  │
│  │ │    (ONNX 1024 LIF neurons)            │                    │  │
│  │ │ 4. Output: ajustes_dendritas (256D)   │                    │  │
│  │ └────────────────────────────────────────┘                    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  PARALELO: Protocolo de Infección                                   │
│  - emitirSenal() si anomalía > 0.7                                  │
│  - recibirSenal() propaga a otros átomos                            │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│         ENTRENADOR COGNITIVO: 4 FASES                                 │
│                                                                        │
│ ┌────────────────────────────────────────────────────────────────┐   │
│ │ FASE 1: ADQUISICIÓN (registrarExperiencia)                    │   │
│ │ • Input: percepciones (72D), hipergrafo, fueFalla            │   │
│ │ • Almacena en bufferExperiencias (max 50)                    │   │
│ │ • Mapea concepto → experiencias                              │   │
│ └────────────────────────────────────────────────────────────────┘   │
│                         ↓ (buffer lleno)                             │
│ ┌────────────────────────────────────────────────────────────────┐   │
│ │ FASE 2: CATEGORIZACIÓN                                        │   │
│ │ refinarCategorias():                                          │   │
│ │ • Para cada concepto: Crea Nodo en Hipergrafo               │   │
│ │ • Calcula centroide de percepciones                          │   │
│ │ • Almacena frecuencia y centroide en metadata                │   │
│ └────────────────────────────────────────────────────────────────┘   │
│                         ↓                                             │
│ ┌────────────────────────────────────────────────────────────────┐   │
│ │ FASE 3: CONSOLIDACIÓN                                        │   │
│ │ reforzarCausalidad():                                         │   │
│ │ • Crea Hiperedges entre conceptos consecutivos               │   │
│ │ • Peso inicial: 0.7                                          │   │
│ │ • Representa relaciones temporales                            │   │
│ └────────────────────────────────────────────────────────────────┘   │
│                         ↓                                             │
│ ┌────────────────────────────────────────────────────────────────┐   │
│ │ FASE 4: PODA (podarMemoriaDebil)                             │   │
│ │ • Elimina Hiperedges con weight < 0.1                        │   │
│ │ • Mantiene solo conexiones fuertes                           │   │
│ │ • Log de edges podadas                                       │   │
│ └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│ OUTPUT: Hipergrafo con conceptos consolidados                        │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│        EXPANSIÓN DIMENSIONALIDAD: 256D → 1600D                        │
│                                                                        │
│  expandirAVector1600D(embedding256D):                                 │
│  • Itera 25 subespacios (S1-S25)                                     │
│  • Cada subespacio: 64 dimensiones (1600 / 25)                       │
│  • Transformación harmónica:                                         │
│    valor[i] = embedding[idxEmb] × (1 + modulación × 0.3)           │
│    donde modulación = sin(frecuencia) × cos(posición)                │
│                                                                        │
│  OUTPUT: vector1600D = [1600 valores]                                │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│              STREAMING BRIDGE → COLAB                                 │
│                                                                        │
│  StreamingBridge.enviarVector(vector1600D, esAnomalia):              │
│  • Batches de 64 muestras                                            │
│  • Endpoint: /train_layer2 en Colab                                  │
│  • Headers: Authorization, Content-Type: application/json            │
│  • Payload: {samples: [{input_data: 1600D[], anomaly_label: 0|1}]}  │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│              COLAB: CortezaCognitivaV2                                │
│                                                                        │
│  Input: 1600D (25 subespacios × 64D)                                 │
│                                                                        │
│  Capa 2: Bi-LSTM (Bidirectional)                                     │
│  • 256 hidden units                                                  │
│  • Procesa secuencia temporal                                        │
│                                                                        │
│  Capa 3: Transformer Encoder                                         │
│  • 8 attention heads                                                 │
│  • 512 hidden units                                                  │
│  • Self-attention sobre embeddings                                   │
│                                                                        │
│  Capa 4: Gated Multimodal Unit (GMU)                                 │
│  • Fusión inteligente LSTM + Transformer                             │
│  • Gating mechanism: λ = sigmoid(W_gate × [LSTM, Transformer])      │
│  • Output: λ × LSTM + (1-λ) × Transformer                            │
│                                                                        │
│  Capa 5: Associative Memory                                          │
│  • Matriz de asociación aprendida                                    │
│  • Pattern completion                                                │
│  • Anomaly detection: predicción_anomalía                            │
│                                                                        │
│  OUTPUT:                                                              │
│  • loss: MSE                                                         │
│  • avg_anomaly_prob: Probabilidad anomalía                           │
│  • suggested_adjustments: 16 ajustes dendríticos (D001-D056 subset) │
└───────────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────────┐
│              FEEDBACK LOOP (Para siguiente ciclo)                     │
│                                                                        │
│  Colab retorna: suggested_adjustments (16 valores)                   │
│  • Se mapean a D001-D056 (sistema de dendritas)                     │
│  • Se aplican en el siguiente ciclo → Capa 0                         │
│  • Afecta comportamiento de los 25 átomos                            │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. Estructura de Clases

```
SistemaOmnisciente (Orquestador)
├── atomos: Map<string, AtomoTopologico>
│   ├── S1: AtomoTopologico
│   │   ├── simulador: Omega21Simulador
│   │   │   ├── dendrites: Omega21Dendrites (D001-D056)
│   │   │   └── mezclar(): void (aplica configuración)
│   │   ├── cerebro: InferenciaLocal
│   │   │   └── predecir(): Promise<Prediccion>
│   │   ├── mapeador: MapeoOmegaAHipergrafo
│   │   └── hipergrafo: Hipergrafo
│   ├── S2-S25: ...
│   │
├── entrenador: EntrenadorCognitivo
│   ├── bufferExperiencias: Experiencia[]
│   ├── mapeoConceptos: Map<string, Experiencia[]>
│   ├── registrarExperiencia(percepciones, imagenMental, fueFalla): void
│   ├── ejecutarCicloConsolidacion(): void
│   │   ├── refinarCategorias(): void
│   │   ├── reforzarCausalidad(): void
│   │   └── podarMemoriaDebil(): void
│   └── obtenerEstadisticas(): object
│
├── corteza: CortezaCognitiva
│   ├── mapaMental: Hipergrafo
│   └── getMapaMental(): Hipergrafo
│
├── sensorial: ProcesadorSensorial
│   ├── 25 sub-redes LIF
│   └── procesar(vector256D): object
│
├── capa2: CapaEspacioTemporal
│   └── procesar(entrada): object
│
├── capa3: CapaCognitiva
│   └── procesar(entrada): object
│
├── bridge: StreamingBridge
│   └── enviarVector(vector1600D, esAnomalia): Promise<void>
│
├── procesarFlujo(id: string, telemetria: Omega21Telemetry): Promise
│   ├── 1. Alterar con dendritas
│   ├── 2. Procesar con átomo
│   ├── 3. Propagar anomalías (Infección)
│   ├── 4. Registrar en entrenador cognitivo
│   ├── 5. Expandir a 1600D
│   └── 6. Enviar a Colab
│
└── propagarInfeccion(): Promise<void>
    └── Emitir señales entre átomos
```

---

## 3. Flujo de Datos por Ciclo

```
CICLO DE EJECUCIÓN:
═══════════════════

1. ENTRADA (Vector 256D generado o recibido)
   ↓
2. EXTRACCIÓN DENDRÍTICA
   D001-D056 extraídos
   ↓
3. PARA CADA ÁTOMO (S1-S25 en paralelo):
   a) Configurar dendritas
   b) Generar muestra (simulador)
   c) Inferencia ONNX (predicción)
   d) Obtener salida (256D embedding)
   ↓
4. CONSOLIDACIÓN COGNITIVA:
   a) Registrar experiencia en buffer
   b) Si buffer lleno (50): Consolidar
      - Categorizar conceptos
      - Reforzar causalidad
      - Podar memoria débil
   ↓
5. EXPANSIÓN DIMENSIONALIDAD:
   256D × 25 átomos → 1600D
   ↓
6. ENVÍO A COLAB:
   StreamingBridge.enviarVector(1600D, anomaly)
   ↓
7. RECIBIR FEEDBACK:
   suggested_adjustments → D001-D056 para próx ciclo
   ↓
8. PROTOCOLO DE INFECCIÓN (cada 10 ciclos):
   Propagar anomalías entre átomos
   ↓
[Vuelta al paso 2 con nuevo vector]
```

---

## 4. Parámetros Clave

### Dendríticas (D001-D056)
- **Rango**: 0-100 (valores normalizados)
- **Propósito**: Estabilizar embeddings ONNX
- **Efecto**: Modular valores de salida en `mezclar()`

### Umbrales de Anomalía
- **Sensorial**: nov > 200 → anomalía
- **Sistema**: prediccion_anomalia > 0.7 → propagar
- **Colab**: avg_anomaly_prob → etiqueta 0|1

### Consolidación Cognitiva
- **Buffer**: max 50 experiencias
- **Trigger**: Buffer lleno → ejecutar ciclo
- **Poda**: weight < 0.1 → eliminar

### Expansión 1600D
- **Subespacios**: 25 (S1-S25)
- **Dimensiones/subespacio**: 64
- **Modulación**: sin(s+1)*cos(i+1) en [0,1]

---

## 5. Estados y Transiciones

```
ESTADOS DEL SISTEMA:
═══════════════════

[INICIALIZACIÓN]
  ↓
[OPERACIÓN NORMAL]
  ├→ Si anomalía detectada en átomo
  │   ↓
  │   [PROPAGACIÓN DE INFECCIÓN]
  │   └→ emitirSenal() → recibirSenal()
  │
  ├→ Si buffer entrenador lleno (50 exp)
  │   ↓
  │   [CONSOLIDACIÓN COGNITIVA]
  │   ├→ refinarCategorias()
  │   ├→ reforzarCausalidad()
  │   └→ podarMemoriaDebil()
  │
  └→ Cada ciclo
      ↓
      [ENVÍO A COLAB]
      └→ Recibir feedback
```

---

## 6. Verificación de Integridad

**✅ Compilación TypeScript**: SIN ERRORES  
**✅ Tests Unitarios**: 44/44 PASS  
**✅ Validación de Integración**: EXITOSA  
**✅ Protocolo de Infección**: FUNCIONAL  
**✅ Consolidación Cognitiva**: FUNCIONAL  
**✅ Expansión Dimensionalidad**: FUNCIONAL  
**✅ Envío a Colab**: LISTO  

---

*Documento de Arquitectura - Hipergrafo v3.0*  
*Última actualización: 23 de Diciembre de 2025*
