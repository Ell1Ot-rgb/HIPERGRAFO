# 🏗️ HIPERGRAFO + OMEGA 21: Arquitectura Completa del Sistema

## 📋 Índice
1. [Visión General](#visión-general)
2. [Componentes del Sistema](#componentes-del-sistema)
3. [Arquitectura de Red](#arquitectura-de-red)
4. [Flujos de Datos](#flujos-de-datos)
5. [Estructura de Directorios](#estructura-de-directorios)
6. [Plan de Implementación](#plan-de-implementación)
7. [Protocolos de Comunicación](#protocolos-de-comunicación)

---

## 1. Visión General

### El Sistema Completo
```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           ARQUITECTURA CIBERNÉTICA COMPLETA                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐ │
│   │  🖥️ PC LIMITADA      │      │  🐳 DOCKER/WSL      │      │  ☁️ GOOGLE COLAB    │ │
│   │  (Cliente Ligero)   │◄────►│  (Omega 21)         │      │  (Entrenamiento)    │ │
│   │                     │      │                     │      │                     │ │
│   │  • Hipergrafo.ts    │      │  • Renode           │      │  • PyTorch          │ │
│   │  • Análisis Local   │      │  • Zephyr RTOS      │      │  • Modelos IA       │ │
│   │  • Persistencia     │      │  • 16 Dendritas     │      │  • Optimización     │ │
│   │  • Control Feedback │      │  • 1024 LIF         │      │  • ONNX Export      │ │
│   │                     │      │  • Vector 256D      │      │                     │ │
│   └──────────┬──────────┘      └──────────┬──────────┘      └──────────┬──────────┘ │
│              │                            │                            │            │
│              │         RED LOCAL          │         INTERNET           │            │
│              │      (UDP/TCP/Telnet)      │      (HTTP/ngrok)          │            │
│              │                            │                            │            │
│              └────────────────────────────┴────────────────────────────┘            │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Objetivos del Sistema
1. **Percepción**: Omega 21 analiza datos con 256 sensores fenomenológicos
2. **Procesamiento**: 16 dendritas transforman datos a señales físicas
3. **Integración**: Red LIF 1024 integra y genera patrones
4. **Análisis**: Hipergrafo analiza la topología del estado
5. **Control**: Hipergrafo envía retroalimentación a las dendritas
6. **Persistencia**: Estados significativos se almacenan como hipergrafos
7. **Entrenamiento**: Colab optimiza los modelos de predicción

---

## 2. Componentes del Sistema

### 2.1 PC Limitada (Cliente TypeScript)
| Componente | Estado | Archivo | Función |
|------------|--------|---------|---------|
| Core Hipergrafo | ✅ Existe | `src/core/` | Nodos, Hiperedges, Estructura |
| Análisis | ✅ Existe | `src/analisis/` | Centralidad, Clustering, Espectral |
| Persistencia | ✅ Existe | `src/persistencia/` | JSON, Almacenamiento |
| ZX-Calculus | ✅ Existe | `src/zx/` | Reescritura, Simplificación |
| Puente Colab | ✅ Creado | `src/neural/ColabBridge.ts` | HTTP a Colab |
| **Omega21 Schema** | ⏳ Pendiente | `src/omega21/Schema.ts` | Tipos 256D |
| **Omega21 Client** | ⏳ Pendiente | `src/omega21/Client.ts` | TCP/UDP a Docker |
| **Dendrite Controller** | ⏳ Pendiente | `src/control/DendriteController.ts` | Retroalimentación |
| **Renode Bridge** | ⏳ Pendiente | `src/hardware/RenodeController.ts` | Telnet a Monitor |

### 2.2 Docker/WSL (Omega 21)
| Componente | Lenguaje | Función |
|------------|----------|---------|
| `main_omniscient.c` | C | Entry point firmware |
| `metrics_256.c` | C | Vector 256D fenomenológico |
| `dendrites.c` | C | Sistema 16 dendritas |
| `soma_integrator.c` | C | Modelo LIF |
| `neuro_interface.c` | C | Interfaz HW 1024 neuronas |
| `physics_loss.c` | C | Restricciones físicas |
| `monje_neuro.repl` | Renode | Mapa de memoria hardware |
| `neuro_peripheral.py` | Python | Periférico neuronal |

### 2.3 Google Colab (Entrenamiento)
| Componente | Estado | Función |
|------------|--------|---------|
| Servidor FastAPI | ✅ Creado | Recibe datos del Hipergrafo |
| Generador Sintético | ⏳ Pendiente | Simula datos Omega 21 |
| Modelo Traductor | ⏳ Pendiente | 256D → Topología Hipergrafo |
| Exportador ONNX | ⏳ Pendiente | Optimización para PC limitada |

---

## 3. Arquitectura de Red

### 3.1 Topología de Conexiones
```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              TOPOLOGÍA DE RED                                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                         PC LIMITADA (Windows/Linux)                         │  │
│  │                              IP: 192.168.x.x                                │  │
│  ├─────────────────────────────────────────────────────────────────────────────┤  │
│  │                                                                             │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │  │
│  │  │ Hipergrafo  │    │ Omega21     │    │ Renode      │    │ Colab       │   │  │
│  │  │ App         │    │ Client      │    │ Controller  │    │ Bridge      │   │  │
│  │  │ (Main)      │    │ (UDP/TCP)   │    │ (Telnet)    │    │ (HTTP)      │   │  │
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │  │
│  │         │                  │                  │                  │          │  │
│  └─────────┼──────────────────┼──────────────────┼──────────────────┼──────────┘  │
│            │                  │                  │                  │             │
│            │                  │                  │                  │             │
│  ┌─────────┼──────────────────┼──────────────────┼──────────────────┼──────────┐  │
│  │         │                  │                  │                  │          │  │
│  │  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐   │  │
│  │  │ Port 4561   │    │ Port 4561   │    │ Port 1234   │    │ ngrok URL   │   │  │
│  │  │ (UART JSON) │    │ (Telemetry) │    │ (Monitor)   │    │ (Internet)  │   │  │
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │  │
│  │         │                  │                  │                  │          │  │
│  │         └────────────┬─────┴────────────┬─────┘                  │          │  │
│  │                      │                  │                        │          │  │
│  │               ┌──────▼──────┐    ┌──────▼──────┐          ┌──────▼──────┐   │  │
│  │               │   DOCKER    │    │   DOCKER    │          │   GOOGLE    │   │  │
│  │               │   RENODE    │    │   RENODE    │          │   COLAB     │   │  │
│  │               │   (SoC)     │    │   (Monitor) │          │   (GPU)     │   │  │
│  │               └─────────────┘    └─────────────┘          └─────────────┘   │  │
│  │                                                                             │  │
│  │                         WSL2 / Docker Desktop                               │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Puertos y Protocolos
| Puerto | Protocolo | Dirección | Contenido | Latencia |
|--------|-----------|-----------|-----------|----------|
| 4561 | TCP/Socket | Omega→Hipergrafo | JSON Telemetría | ~1ms |
| 1234 | Telnet | Hipergrafo→Omega | Comandos sysbus | ~5ms |
| 8000 | HTTP | Hipergrafo↔Colab | JSON Análisis | ~100ms |
| 5000 | UDP | Hipergrafo→Omega | Control rápido | <1ms |

---

## 4. Flujos de Datos

### 4.1 Flujo de Telemetría (Omega → Hipergrafo)
```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           FLUJO DE TELEMETRÍA                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   OMEGA 21 (C/Zephyr)                                                            │
│   ┌────────────────────────────────────────────────────────────────────────┐     │
│   │                                                                        │     │
│   │  1. Entrada: 256 bytes de datos                                        │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  2. calculate_metrics_256d() → Vector[256] floats                      │     │
│   │     │                                                                  │     │
│   │     ├──► S1-S25: 25 subespacios fenomenológicos                        │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  3. project_to_72d() → Vector[72] floats                               │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  4. dendrites_process() → 16 corrientes dendríticas                    │     │
│   │     │                                                                  │     │
│   │     ├──► D1-D4:   Eléctricas (Ohm, Power, Capacitor)                   │     │
│   │     ├──► D3-D4:   Térmicas (Altitude, Dew, Entropy)                    │     │
│   │     ├──► D5-D7:   Espaciales (Distance, Velocity, Accel)               │     │
│   │     ├──► D8-D11:  Temporales (Phase, Freq, Delay, Memory)              │     │
│   │     └──► D12-D14: Químicas (Decay, Michaelis, Hill)                    │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  5. soma_integrate() → Potencial de membrana + Spike                   │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  6. neuro_infer() → Patrón ID, Similitud, Novedad, Categoría           │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  7. emit_omniscient_json() → JSON por UART                             │     │
│   │                                                                        │     │
│   └────────────────────────────────────────────────────────────────────────┘     │
│                                        │                                         │
│                                        │ TCP :4561                               │
│                                        ▼                                         │
│   HIPERGRAFO (TypeScript)                                                        │
│   ┌────────────────────────────────────────────────────────────────────────┐     │
│   │                                                                        │     │
│   │  8. Omega21Client.recibirTelemetria() → JSON parseado                  │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  9. Omega21Schema.decodificar() → Objeto tipado {meta, logic, neuro,   │     │
│   │     │                              dendrites, sig}                     │     │
│   │     ▼                                                                  │     │
│   │ 10. MapeoOmegaAHipergrafo.mapear() → Hipergrafo con nodos por          │     │
│   │     │                                subespacio                        │     │
│   │     ▼                                                                  │     │
│   │ 11. AnálisisTopológico.analizar() → Centralidad, Clustering,          │     │
│   │     │                               Anomalías                          │     │
│   │     ▼                                                                  │     │
│   │ 12. ServicioPersistencia.guardar() → JSON/SQLite                       │     │
│   │                                                                        │     │
│   └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Flujo de Control (Hipergrafo → Omega)
```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           FLUJO DE CONTROL (FEEDBACK)                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   HIPERGRAFO (TypeScript)                                                        │
│   ┌────────────────────────────────────────────────────────────────────────┐     │
│   │                                                                        │     │
│   │  1. AnalisisTopologico detecta condición:                              │     │
│   │     │                                                                  │     │
│   │     ├──► Alta Entropía Nodo S2 (D017) → Posible ataque/ruido           │     │
│   │     ├──► Baja Novedad Neuro (nov=0) → Sistema estancado                │     │
│   │     ├──► Spike Burst D4 (Dew) → Condensación inminente                 │     │
│   │     └──► Anomalía Clustering S11 → Patrón desconocido                  │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  2. DendriteController.evaluarAccion(estado) → Acción                  │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  3. DendriteController.generarComando(accion) → {dendrita, param, val} │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  4. RenodeController.enviarComando(cmd) → Telnet string                │     │
│   │                                                                        │     │
│   └────────────────────────────────────────────────────────────────────────┘     │
│                                        │                                         │
│                                        │ Telnet :1234                            │
│                                        ▼                                         │
│   RENODE MONITOR                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────┐     │
│   │                                                                        │     │
│   │  5. Recibe comando:                                                    │     │
│   │     "sysbus WriteDoubleWord 0x53000014 0x7F"                           │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  6. Escribe en memoria mapeada del periférico neuronal                 │     │
│   │                                                                        │     │
│   └────────────────────────────────────────────────────────────────────────┘     │
│                                        │                                         │
│                                        │ Memory Write                            │
│                                        ▼                                         │
│   OMEGA 21 FIRMWARE                                                              │
│   ┌────────────────────────────────────────────────────────────────────────┐     │
│   │                                                                        │     │
│   │  7. Registro REG_REWARD (0x53000014) actualizado                       │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  8. physics_stdp.c lee el reward y ajusta pesos sinápticos             │     │
│   │     │                                                                  │     │
│   │     ▼                                                                  │     │
│   │  9. Próxima inferencia usa nuevos pesos                                │     │
│   │                                                                        │     │
│   └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Estructura de Directorios

### 5.1 Proyecto HIPERGRAFO (TypeScript - Este Repo)
```
/workspaces/HIPERGRAFO/
├── 📁 src/
│   ├── 📁 core/                      # ✅ EXISTENTE - Núcleo del Hipergrafo
│   │   ├── Hipergrafo.ts             # Clase principal H = (V, E)
│   │   ├── Nodo.ts                   # Vértices
│   │   ├── Hiperedge.ts              # Aristas generalizadas
│   │   └── index.ts                  # Exports
│   │
│   ├── 📁 analisis/                  # ✅ EXISTENTE - Métricas matemáticas
│   │   ├── CentralidadHipergrafo.ts  # Grado, Betweenness, Eigenvector
│   │   ├── ClusteringHipergrafo.ts   # Coeficientes, Modularidad
│   │   ├── DualidadHipergrafo.ts     # Transformación dual H*
│   │   ├── PropiedadesEspectrales.ts # Laplaciana, Eigenvalores
│   │   └── index.ts
│   │
│   ├── 📁 persistencia/              # ✅ EXISTENTE - Almacenamiento
│   │   ├── ServicioPersistencia.ts   # Serialización JSON
│   │   ├── GestorAlmacenamiento.ts   # Disco/DB
│   │   └── index.ts
│   │
│   ├── 📁 zx/                        # ✅ EXISTENTE - ZX-Calculus
│   │   ├── ZXDiagram.ts              # Diagrama ZX
│   │   ├── ZXSpider.ts               # Arañas Z/X
│   │   ├── Reglas.ts                 # Reglas de reescritura
│   │   ├── MotorZX.ts                # Motor de inferencia
│   │   └── index.ts
│   │
│   ├── 📁 neural/                    # 🔄 MIXTO - Integración IA
│   │   ├── MapeoRedNeuronalAHipergrafo.ts  # ✅ Genérico
│   │   ├── tipos.ts                        # ✅ Tipos base
│   │   ├── ColabBridge.ts                  # ✅ Cliente HTTP
│   │   ├── IntegradorHipergrafoColo.ts     # ✅ Orquestador
│   │   ├── configColab.ts                  # ✅ URL ngrok
│   │   ├── ConfiguracionDistribuida.ts     # ✅ Config cliente/servidor
│   │   └── index.ts
│   │
│   ├── 📁 omega21/                   # ⏳ NUEVO - Integración Omega 21
│   │   ├── Schema.ts                 # Tipos para 256D + 16 dendritas
│   │   ├── Decodificador.ts          # JSON → Objeto tipado
│   │   ├── MapeoOmegaAHipergrafo.ts  # Estado Omega → Hipergrafo
│   │   ├── SubespaciosFenomenologicos.ts # S1-S25 definiciones
│   │   └── index.ts
│   │
│   ├── 📁 hardware/                  # ⏳ NUEVO - Comunicación con Docker/Renode
│   │   ├── SensoresInterface.ts      # ✅ Interfaz abstracta (Mock)
│   │   ├── Omega21Client.ts          # Cliente TCP/UDP para telemetría
│   │   ├── RenodeController.ts       # Cliente Telnet para control
│   │   ├── ProtocoloComandos.ts      # Definición de comandos sysbus
│   │   └── index.ts
│   │
│   ├── 📁 control/                   # ⏳ NUEVO - Bucle de retroalimentación
│   │   ├── DendriteController.ts     # Traductor Estado → Ajustes
│   │   ├── ReglasControl.ts          # Lógica de decisión (PID/Fuzzy)
│   │   ├── ParametrosDendrita.ts     # R, τ, Km, n, etc.
│   │   └── index.ts
│   │
│   ├── 📁 pruebas/                   # Archivos de prueba
│   │   └── prueba_colab.ts           # ✅ Test conexión Colab
│   │
│   └── index.ts                      # Export principal
│
├── 📁 docs/                          # Documentación
│   ├── ARQUITECTURA_DISTRIBUIDA.md   # ✅ Creado
│   ├── FASE3_MATEMATICA.md           # ✅ Existente
│   └── TECNICA.md                    # ✅ Existente
│
├── 📁 scripts/                       # Scripts de utilidad
│   ├── conectar_omega21.sh           # ⏳ Pendiente
│   └── entrenar_colab.sh             # ⏳ Pendiente
│
├── PROYECTO_COMPLETO.md              # 📍 ESTE ARCHIVO
├── PUENTE_COLAB.md                   # ✅ Documentación del puente
├── ARQUITECTURA_PUENTE.md            # ✅ Diagrama visual
├── verificar_puente.sh               # ✅ Script verificación
├── EJEMPLO_SERVIDOR_COLAB.py         # ✅ Código para Colab
├── package.json                      # Dependencias
├── tsconfig.json                     # Config TypeScript
└── jest.config.js                    # Config tests
```

### 5.2 Proyecto Omega 21 (C/Renode - Otro Repo)
```
/path/to/omega21/
├── 📁 src/
│   ├── main_omniscient.c             # Entry point
│   ├── metrics_256.c                 # Vector 256D
│   ├── sha256.c                      # Hash
│   ├── neuro_interface.c             # HW 1024 LIF
│   └── 📁 neural/
│       ├── dendrites.c               # Sistema unificado
│       ├── dendrite_ohm.c            # D1
│       ├── dendrite_power.c          # D2
│       ├── ...                       # D3-D15
│       ├── dendrite_entropy.c        # D16
│       ├── soma_integrator.c         # SOMA
│       ├── physics_loss.c            # Pérdidas
│       └── physics_stdp.c            # STDP
│
├── 📁 include/
│   ├── metrics_256.h
│   ├── dendrites.h
│   ├── neuro_interface.h
│   └── ...
│
├── 📁 renode/
│   ├── monje_neuro.repl              # Mapa hardware
│   ├── monje_omniscient.resc         # Script arranque
│   ├── neuro_peripheral.py           # Periférico Python
│   └── internal_watcher_clean.py     # Watcher archivos
│
├── 📁 zephyr/
│   ├── prj.conf                      # Config Zephyr
│   ├── CMakeLists.txt                # Build system
│   ├── app.overlay                   # Device Tree
│   └── mmu_regions.c                 # Regiones MMU
│
└── start_omniscient.sh               # Script Docker
```

---

## 6. Plan de Implementación

### Fase 1: Esquema de Datos (Prioridad Alta) 🔴
| Tarea | Archivo | Descripción | Dependencia |
|-------|---------|-------------|-------------|
| 1.1 | `src/omega21/Schema.ts` | Interfaces TypeScript para Vector 256D | - |
| 1.2 | `src/omega21/SubespaciosFenomenologicos.ts` | Constantes S1-S25 | 1.1 |
| 1.3 | `src/omega21/Decodificador.ts` | Parser JSON → Objeto tipado | 1.1, 1.2 |
| 1.4 | Test unitarios Schema | Verificar parsing correcto | 1.3 |

### Fase 2: Comunicación con Docker (Prioridad Alta) 🔴
| Tarea | Archivo | Descripción | Dependencia |
|-------|---------|-------------|-------------|
| 2.1 | `src/hardware/Omega21Client.ts` | Socket TCP puerto 4561 | 1.3 |
| 2.2 | `src/hardware/RenodeController.ts` | Telnet puerto 1234 | - |
| 2.3 | `src/hardware/ProtocoloComandos.ts` | Comandos sysbus tipados | - |
| 2.4 | Test integración | Conectar a Renode real | 2.1, 2.2 |

### Fase 3: Mapeo a Hipergrafo (Prioridad Media) 🟡
| Tarea | Archivo | Descripción | Dependencia |
|-------|---------|-------------|-------------|
| 3.1 | `src/omega21/MapeoOmegaAHipergrafo.ts` | Estado 256D → Nodos/Edges | 1.3 |
| 3.2 | Nodos Subespacio | Un nodo por cada S1-S25 | 3.1 |
| 3.3 | Hiperedges Correlación | Conexiones por correlación | 3.1, 3.2 |
| 3.4 | Test topología | Verificar estructura generada | 3.3 |

### Fase 4: Control de Retroalimentación (Prioridad Media) 🟡
| Tarea | Archivo | Descripción | Dependencia |
|-------|---------|-------------|-------------|
| 4.1 | `src/control/ParametrosDendrita.ts` | Enum de parámetros ajustables | - |
| 4.2 | `src/control/ReglasControl.ts` | Lógica if/then o PID | 3.1 |
| 4.3 | `src/control/DendriteController.ts` | Orquestador de control | 4.1, 4.2, 2.2 |
| 4.4 | Test bucle cerrado | Enviar comando y verificar | 4.3 |

### Fase 5: Entrenamiento en Colab (Prioridad Baja) 🟢
| Tarea | Archivo | Descripción | Dependencia |
|-------|---------|-------------|-------------|
| 5.1 | `colab/GeneradorSintetico.py` | Genera datos 256D falsos | 1.1 |
| 5.2 | `colab/ModeloTraductor.py` | Red 256D → Topología | 5.1 |
| 5.3 | `colab/ExportadorONNX.py` | Cuantización + Export | 5.2 |
| 5.4 | Integrar en PC limitada | Cargar modelo ONNX | 5.3 |

---

## 7. Protocolos de Comunicación

### 7.1 Telemetría JSON (Omega → Hipergrafo)
```json
{
  "meta": {
    "ts": 154100,
    "blk": 19312,
    "sz": 256
  },
  "logic": {
    "h": 0,
    "lz": 14,
    "chi": 65280,
    "pad": [-1000, -29172, -496]
  },
  "neuro": {
    "id": 0,
    "sim": 0,
    "nov": 0,
    "cat": 0
  },
  "sig": {
    "fp": "5d98aeb4af636e93",
    "lsh": 170,
    "eq": 0,
    "sc": 0
  },
  "dendrites": {
    "voltage": 0,
    "current": 0,
    "power": 0,
    "altitude": 8627,
    "dew_temp": -1726,
    "velocity": -6590,
    "phase": 75,
    "freq": 0,
    "soma_v": -7500,
    "spike": 1,
    "loss": 1893849984
  },
  "metrics_256": [/* Array de 256 valores uint16/32 */]
}
```

### 7.2 Comandos de Control (Hipergrafo → Omega)
```
# Formato: sysbus WriteDoubleWord <ADDRESS> <VALUE>

# Registros del periférico neuronal (0x53000000)
REG_CTRL      = 0x53000000  # Control: START(1), LEARN(2), RESET(4)
REG_STATUS    = 0x53000004  # Estado: BUSY(1), DONE(2), READY(4), ERROR(8)
REG_PATRON_ID = 0x53000008  # ID del patrón (0-1023)
REG_SIMILITUD = 0x5300000C  # Similitud (0-255)
REG_NOVEDAD   = 0x53000010  # Novedad (0-255)
REG_REWARD    = 0x53000014  # Recompensa R-STDP (-128 a +127)
REG_STATE     = 0x53000018  # Estado FSM

# Ejemplos de comandos:
"sysbus WriteDoubleWord 0x53000014 0x7F"   # Reward +127 (máximo positivo)
"sysbus WriteDoubleWord 0x53000014 0x80"   # Reward -128 (máximo negativo)
"sysbus WriteDoubleWord 0x53000000 0x02"   # Activar aprendizaje
"sysbus WriteDoubleWord 0x53000000 0x04"   # Reset
```

### 7.3 Control de Parámetros Dendríticos
```typescript
// Enum de parámetros ajustables por dendrita
enum DendriteParam {
  // D1: Ohm
  D1_RESISTANCE = 0x100,    // R (Ω)
  D1_WEIGHT     = 0x104,    // Peso sináptico

  // D3: Altitude
  D3_P_REFERENCE = 0x200,   // Presión referencia

  // D4: Dew
  D4_MARGIN_CRITICAL = 0x300, // Margen crítico (°C)

  // D13: Michaelis
  D13_VMAX = 0x400,         // Velocidad máxima
  D13_KM   = 0x404,         // Constante Michaelis

  // D14: Hill
  D14_KD = 0x500,           // Constante disociación
  D14_N  = 0x504,           // Coeficiente cooperatividad

  // D15: Capacitor
  D15_TAU = 0x600,          // τ = RC

  // SOMA
  SOMA_THRESHOLD = 0x700,   // V_thresh
  SOMA_TAU_M     = 0x704,   // τ membrana
}
```

---

## 📊 Estado Actual del Proyecto

| Componente | Progreso | Notas |
|------------|----------|-------|
| Core Hipergrafo | █████████░ 90% | Funcional |
| Análisis Matemático | █████████░ 90% | Funcional |
| Persistencia | ████████░░ 80% | Funcional |
| ZX-Calculus | ████████░░ 80% | Funcional |
| Puente Colab | ██████████ 100% | ✅ Probado |
| Esquema Omega21 | ░░░░░░░░░░ 0% | ⏳ Siguiente |
| Cliente Omega21 | ░░░░░░░░░░ 0% | ⏳ Siguiente |
| Control Dendritas | ░░░░░░░░░░ 0% | ⏳ Pendiente |
| Entrenamiento ONNX | ░░░░░░░░░░ 0% | Última fase |

---

## 🚀 Siguiente Paso Recomendado

**Crear `src/omega21/Schema.ts`** con las interfaces TypeScript para:
1. El JSON de telemetría completo
2. Los 25 subespacios (S1-S25)
3. Las 16 dendritas (D1-D16)
4. Los tipos de cada dimensión (uint8, uint16, uint32, int16)

¿Procedo con la implementación de la **Fase 1**?
