# 🔷 ARQUITECTURA HEXAGONAL DEL SISTEMA ORGANISMO VIVO

**Confirmación**: El sistema está estructurado en **Arquitectura Hexagonal** (Ports and Adapters / Onion Architecture)

---

## 📐 ESTRUCTURA HEXAGONAL

### Vista General

```
                    ┌─────────────────────────────────────────┐
                    │            🌐 MUNDO EXTERNO              │
                    │  PC2 · Redis · Neo4j · n8n · LightRAG   │
                    └────────────────────┬────────────────────┘
                                         │
                    ╔════════════════════╧════════════════════╗
                    ║         📡 CAPA ADAPTADORES             ║
                    ║  ┌──────────────┐  ┌──────────────┐    ║
                    ║  │   INBOUND    │  │   OUTBOUND   │    ║
                    ║  │ tcp_neuro.py │  │ neo4j_repo.py│    ║
                    ║  │ redis_in.py  │  │ redis_pub.py │    ║
                    ║  │ webhook.py   │  │ lightrag.py  │    ║
                    ║  └──────┬───────┘  └───────┬──────┘    ║
                    ╚═════════╧══════════════════╧════════════╝
                              │                  ▲
                    ╔═════════╧══════════════════╧════════════╗
                    ║         🔌 CAPA INTERFACES              ║
                    ║  neural_ports.py · system_facade.py    ║
                    ║              health_monitor.py          ║
                    ╚═════════════════╤════════════════════════╝
                                      │
            ╔═════════════════════════╧═════════════════════════╗
            ║                  🧠 NÚCLEO (CORE)                  ║
            ║  ┌─────────────────────────────────────────────┐  ║
            ║  │            📊 ENGINES (Motores)              │  ║
            ║  │  ┌──────────┐ ┌──────────┐ ┌──────────┐    │  ║
            ║  │  │    S1    │ │    S2    │ │    S3    │    │  ║
            ║  │  │ Fenomeno │→│Emergencia│→│  Lógica  │    │  ║
            ║  │  │ logía    │ │          │ │  Pura    │    │  ║
            ║  │  └──────────┘ └──────────┘ └──────────┘    │  ║
            ║  └─────────────────────────────────────────────┘  ║
            ║  ┌─────────────────────────────────────────────┐  ║
            ║  │               🌀 CHAOS                       │  ║
            ║  │     Autómatas 1D/2D · Regulador · Lyapunov  │  ║
            ║  └─────────────────────────────────────────────┘  ║
            ║  ┌─────────────────────────────────────────────┐  ║
            ║  │              📦 DOMAIN                       │  ║
            ║  │    Concepto · Axioma · Grundzug · Instancia │  ║
            ║  └─────────────────────────────────────────────┘  ║
            ╚═══════════════════════════════════════════════════╝
```

---

## 🗂️ ESTRUCTURA DE CARPETAS HEXAGONAL

```
sistema_terminado/
│
├── 📁 core/                          # 🟢 SIN DEPENDENCIAS EXTERNAS
│   │
│   ├── 📁 domain/                    # Entidades del dominio
│   │   ├── concepto.py
│   │   ├── axioma.py
│   │   ├── grundzug.py
│   │   ├── instancia.py
│   │   └── configuracion.py
│   │
│   ├── 📁 engines/                   # Motores de procesamiento
│   │   ├── 📁 s1_fenomenologia/      # CAPA EMPÍRICA
│   │   │   ├── tokenizer.py
│   │   │   ├── embedder.py
│   │   │   ├── clasificador.py
│   │   │   ├── grundzug_tracker.py
│   │   │   └── esn.py
│   │   │
│   │   ├── 📁 s2_emergencia/         # CAPA EMERGENCIA
│   │   │   ├── motor_emergencia.py
│   │   │   ├── fca_processor.py
│   │   │   ├── grafo_conceptual.py
│   │   │   └── apoptosis.py
│   │   │
│   │   └── 📁 s3_logica/             # CAPA LÓGICA
│   │       ├── motor_axiomas.py
│   │       ├── mundo_hipotetico.py
│   │       └── logica_pura.py
│   │
│   └── 📁 chaos/                     # BORDE DEL CAOS
│       ├── automata_1d.py
│       ├── regulador.py
│       └── metricas.py
│
├── 📁 adapters/                      # 🔴 CONEXIONES EXTERNAS
│   │
│   ├── 📁 inbound/                   # Entrada al sistema
│   │   ├── tcp_neuromorphic.py       # PC2 FPGA
│   │   ├── redis_listener.py         # Capa 1
│   │   └── webhook_handler.py        # n8n
│   │
│   └── 📁 outbound/                  # Salida del sistema
│       ├── neo4j_repository.py       # Persistencia
│       ├── redis_publisher.py        # Eventos
│       ├── lightrag_client.py        # RAG
│       └── n8n_integrator.py         # Webhooks
│
├── 📁 interfaces/                    # 🔵 CONTRATOS PÚBLICOS
│   ├── neural_ports.py               # Conexiones #1-#4
│   ├── system_facade.py              # Orquestador
│   └── health_monitor.py             # Salud
│
├── 📁 config/
├── 📁 tests/
└── 📁 docs/
```

---

## 🔷 CONCEPTO: VOHEXISTENCIA

El término **Vohexistencia** en el sistema NO es solo un nombre - tiene significado arquitectónico:

> **Vo-hex-istencia** = Co-existencia en red hexagonal (6 dimensiones relacionales)

### Las 6 Dimensiones Relacionales

```
                    Dimensión 1
                        ▲
                       /│\
           Dim 6 ─────●─────── Dim 2
                     /│\│
                    / │ \│
           Dim 5 ──/──│──\── Dim 3
                      │
                      ▼
                  Dimensión 4
```

Cada **Vohexistencia** (patrón emergente) puede relacionarse en 6 direcciones/dimensiones:

1. **Temporal** - Antes/Después
2. **Causal** - Causa/Efecto
3. **Semántica** - Similar/Diferente
4. **Lógica** - Implica/Contradice
5. **Afectiva** - Positivo/Negativo
6. **Estructural** - Parte/Todo

---

## 🔌 PUERTOS Y CONEXIONES

### Puertos de Entrada (Inbound)

| Puerto | Adaptador | Protocolo | Origen |
|--------|-----------|-----------|--------|
| **T1** | `tcp_neuromorphic.py` | TCP | PC2 FPGA |
| **R1** | `redis_listener.py` | Redis Sub | Capa 1 Monje |
| **W1** | `webhook_handler.py` | HTTP POST | n8n |

### Puertos de Salida (Outbound)

| Puerto | Adaptador | Protocolo | Destino |
|--------|-----------|-----------|---------|
| **N1** | `neo4j_repository.py` | Bolt | Neo4j DB |
| **R2** | `redis_publisher.py` | Redis Pub | Eventos |
| **L1** | `lightrag_client.py` | HTTP | LightRAG API |

### Conexiones Neuronales (#1-#4)

| # | Nombre | Dirección | Formato | Descripción |
|---|--------|-----------|---------|-------------|
| **#1** | Embedding Out | S1 → Ext | `float32[64]` | Estado semántico |
| **#2** | Concept Inject | Ext → S2 | `(str, float)` | Inyección de conceptos |
| **#3** | Temporal Pred | ESN → Ext | `float32[64]` | Predicción temporal |
| **#4** | Axioma Bridge | S2 ↔ S3 | `Axioma` | Puente lógico |

---

## 🎯 PRINCIPIOS HEXAGONALES APLICADOS

### 1. Independencia del Core
```
core/ → SIN dependencias externas
       → Testeable sin mocks
       → Portable a cualquier entorno
```

### 2. Inversión de Dependencias
```
NÚCLEO ← define interfaces
ADAPTERS → implementan interfaces
```

### 3. Separación de Responsabilidades
```
🟢 VERDE (core/)    = Lógica pura, testeable
🔴 ROSA (adapters/) = Requiere mocks para tests
🔵 AZUL (interfaces/) = Puente entre capas
```

---

## 📊 ARCHIVOS EN `sistema_terminado/`

| Carpeta | Archivos | Contenido |
|---------|:--------:|-----------|
| `core_new/domain/` | 4 | Entidades |
| `core_new/engines/s1_fenomenologia/` | 8 | Fenomenología |
| `core_new/engines/s2_emergencia/` | 9 | Emergencia |
| `core_new/engines/s3_logica/` | 8 | Lógica |
| `core_new/engines/chaos/` | 3 | Autómatas |
| `core_new/engines/bio/` | 34 | Bio-subsistemas |
| `adapters/inbound/` | 2 | Entrada |
| `adapters/outbound/` | 7 | Salida |
| `interfaces/` | 6 | Puertos |
| `tests/` | 5 | Validación |
| **TOTAL** | **99** | - |

---

## 📍 PUNTOS DE ENTRADA

1. **Sistema Principal**: `interfaces/neural_ports.py`
2. **Health Monitor**: `interfaces/health_monitor.py`
3. **Benchmark**: `interfaces/benchmark.py`

---

## ✅ CONCLUSIÓN

El sistema **Organismo Vivo v100** implementa una **Arquitectura Hexagonal** que:

1. **Aísla el núcleo** (core/) de dependencias externas
2. **Usa puertos/adaptadores** para comunicación
3. **Permite testing** sin infraestructura real
4. **Facilita cambios** en bases de datos o servicios externos
5. **Mantiene la Vohexistencia** como patrón de 6 dimensiones relacionales

---

*Documento generado el 31 de Diciembre de 2025*  
*Arquitectura Hexagonal del Sistema Organismo Vivo*
