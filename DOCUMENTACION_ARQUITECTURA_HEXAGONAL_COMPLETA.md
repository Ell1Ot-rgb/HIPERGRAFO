# 🔷 DOCUMENTACIÓN COMPLETA: ARQUITECTURA HEXAGONAL DEL ORGANISMO VIVO

**Sistema**: Organismo Vivo v100  
**Patrón Arquitectónico**: Hexagonal (Ports & Adapters) / Onion Architecture  
**Fecha de Documentación**: 31 de Diciembre de 2025  

---

## 📚 ÍNDICE

1. [Introducción a la Arquitectura Hexagonal](#1-introducción)
2. [Principios Fundamentales](#2-principios-fundamentales)
3. [Estructura del Sistema](#3-estructura-del-sistema)
4. [El Concepto Vohexistencia](#4-el-concepto-vohexistencia)
5. [Capas del Sistema](#5-capas-del-sistema)
6. [Puertos y Adaptadores](#6-puertos-y-adaptadores)
7. [Motores Cognitivos (S1, S2, S3)](#7-motores-cognitivos)
8. [Flujo de Datos](#8-flujo-de-datos)
9. [Implementación en Código](#9-implementación-en-código)
10. [Guía de Extensión](#10-guía-de-extensión)

---

## 1. INTRODUCCIÓN

### ¿Qué es la Arquitectura Hexagonal?

La **Arquitectura Hexagonal** (también conocida como "Ports and Adapters") fue propuesta por Alistair Cockburn. Su objetivo principal es crear sistemas que sean:

- **Independientes de frameworks**
- **Testables** sin infraestructura
- **Independientes de la UI**
- **Independientes de la base de datos**
- **Independientes de agentes externos**

### ¿Por qué Hexagonal para el Organismo Vivo?

El Organismo Vivo necesita:
1. Conectarse a múltiples fuentes (Redis, Neo4j, TCP, Webhooks)
2. Ser testeable sin hardware real
3. Permitir cambios en almacenamiento sin afectar la lógica
4. Soportar múltiples interfaces (API, CLI, Neuromorfo)

---

## 2. PRINCIPIOS FUNDAMENTALES

### 2.1 Inversión de Dependencias

```
❌ INCORRECTO:
   Lógica de Negocio → Base de Datos
   
✅ CORRECTO:
   Lógica de Negocio ← Interface ← Base de Datos
```

El núcleo define las interfaces, los adaptadores las implementan.

### 2.2 Separación de Responsabilidades

```
🟢 CORE (Verde)     = Lógica pura, sin dependencias externas
🔴 ADAPTERS (Rojo)  = Conexiones al mundo exterior
🔵 INTERFACES (Azul) = Contratos entre capas
```

### 2.3 Regla de Dependencia

Las dependencias solo pueden apuntar hacia adentro:

```
EXTERIOR → ADAPTERS → INTERFACES → CORE
    ↑          ↑           ↑         ↓
    └──────────┴───────────┴─────────┘
         Las dependencias apuntan al centro
```

---

## 3. ESTRUCTURA DEL SISTEMA

### 3.1 Vista Hexagonal

```
                              ┌─────────────────┐
                              │   🌐 EXTERNO    │
                              │ Neo4j · Redis   │
                              │ n8n · LightRAG  │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
          ┌─────────▼─────────┐              ┌────────────▼────────────┐
          │  📥 INBOUND       │              │  📤 OUTBOUND            │
          │  tcp_neuromorphic │              │  neo4j_repository       │
          │  redis_listener   │              │  redis_publisher        │
          │  webhook_handler  │              │  lightrag_client        │
          └─────────┬─────────┘              └────────────┬────────────┘
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │         🔌 INTERFACES               │
                    │     neural_ports.py (#1-#4)         │
                    │     system_facade.py                │
                    │     health_monitor.py               │
                    └──────────────────┬──────────────────┘
                                       │
            ╔══════════════════════════▼══════════════════════════╗
            ║                   🧠 NÚCLEO                          ║
            ║  ┌────────────────────────────────────────────────┐  ║
            ║  │              📊 ENGINES                        │  ║
            ║  │  ┌────────┐  ┌────────┐  ┌────────┐           │  ║
            ║  │  │   S1   │→ │   S2   │→ │   S3   │           │  ║
            ║  │  │ Fenome │  │Emergen │  │ Lógica │           │  ║
            ║  │  │ nología│  │  cia   │  │  Pura  │           │  ║
            ║  │  └────────┘  └────────┘  └────────┘           │  ║
            ║  └────────────────────────────────────────────────┘  ║
            ║  ┌────────────────────────────────────────────────┐  ║
            ║  │              🌀 CHAOS                          │  ║
            ║  │    Autómatas · Regulador · Lyapunov            │  ║
            ║  └────────────────────────────────────────────────┘  ║
            ║  ┌────────────────────────────────────────────────┐  ║
            ║  │              📦 DOMAIN                         │  ║
            ║  │  Concepto · Axioma · Grundzug · Instancia      │  ║
            ║  └────────────────────────────────────────────────┘  ║
            ╚═════════════════════════════════════════════════════╝
```

### 3.2 Estructura de Carpetas

```
sistema_terminado/
│
├── 📁 core/                          # 🟢 SIN DEPENDENCIAS EXTERNAS
│   │
│   ├── 📁 domain/                    # Entidades del dominio
│   │   ├── concepto.py               # Concepto, ConceptoEmergente
│   │   ├── axioma.py                 # Axioma, Proposición
│   │   ├── grundzug.py               # Grundzug, TipoYO
│   │   ├── instancia.py              # Instancia, InstanciaAbstracta
│   │   └── configuracion.py          # ConfiguracionSistema
│   │
│   ├── 📁 engines/                   # Motores de procesamiento
│   │   ├── 📁 s1_fenomenologia/      # Tokenización, Embeddings
│   │   ├── 📁 s2_emergencia/         # FCA, Grafos, Apoptosis
│   │   ├── 📁 s3_logica/             # Axiomas, Mundos
│   │   └── 📁 bio/                   # 17 subsistemas biológicos
│   │
│   └── 📁 chaos/                     # Borde del caos
│       ├── automata_1d.py
│       ├── automata_2d.py
│       └── regulador.py
│
├── 📁 adapters/                      # 🔴 CONEXIONES EXTERNAS
│   │
│   ├── 📁 inbound/                   # Entrada
│   │   ├── tcp_neuromorphic.py       # FPGA/PC2
│   │   ├── redis_listener.py         # Capa 1
│   │   └── webhook_handler.py        # n8n
│   │
│   └── 📁 outbound/                  # Salida
│       ├── neo4j_repository.py       # Persistencia
│       ├── redis_publisher.py        # Eventos
│       ├── lightrag_client.py        # RAG
│       └── n8n_integrator.py         # Webhooks
│
├── 📁 interfaces/                    # 🔵 CONTRATOS
│   ├── neural_ports.py               # Puertos #1-#4
│   ├── system_facade.py              # Orquestador
│   └── health_monitor.py             # Salud
│
└── 📁 config/
    └── settings.py
```

---

## 4. EL CONCEPTO VOHEXISTENCIA

### 4.1 Etimología

> **Vo-hex-istencia** = **Co-existencia** en red **hexagonal** (6 dimensiones relacionales)

El nombre NO es arbitrario. Representa la estructura topológica del sistema.

### 4.2 Las 6 Dimensiones Relacionales

```
                    Dim 1: TEMPORAL
                         ▲
                        /│\
                       / │ \
        Dim 6: ───────●──│──●─────── Dim 2:
        ESTRUCTURAL  /   │   \      CAUSAL
                    /    │    \
        Dim 5: ────●─────│─────●──── Dim 3:
        AFECTIVA         │          SEMÁNTICA
                         ▼
                    Dim 4: LÓGICA
```

| Dim | Nombre | Relación | Ejemplo |
|-----|--------|----------|---------|
| 1 | **Temporal** | Antes/Después | "almuerzo → cena" |
| 2 | **Causal** | Causa/Efecto | "lluvia → mojado" |
| 3 | **Semántica** | Similar/Diferente | "perro ~ lobo" |
| 4 | **Lógica** | Implica/Contradice | "mortal → finito" |
| 5 | **Afectiva** | Positivo/Negativo | "alegría ↔ tristeza" |
| 6 | **Estructural** | Parte/Todo | "rueda ⊂ carro" |

### 4.3 Código de Vohexistencia

```python
@dataclass
class Vohexistencia:
    """Nivel 1: Agrupación de instancias con patrón compartido"""
    
    id: str                        # vohex_xxxxxxxx
    nombre: str
    descripcion: str
    instancias: List[Dict]         # IDs participantes
    constante_emergente: str       # Patrón compartido
    peso_coexistencial: float      # 0.0 - 1.0
    ejes_relacionales: List[str]   # Las 6 dimensiones activas
    timestamp: str
```

---

## 5. CAPAS DEL SISTEMA

### 5.1 Capa 1: Física (Monje Gemelo)

```
┌─────────────────────────────────────────────────────────────┐
│                   CAPA 1: FÍSICA                            │
├─────────────────────────────────────────────────────────────┤
│  Origen: Renode + Zephyr (Simulación de hardware)           │
│                                                              │
│  Vector Físico:                                              │
│  {                                                           │
│    "tiempo": 1250000,        // Ciclos CPU                  │
│    "instrucciones": 45823,   // Instrucciones ejecutadas    │
│    "energia": 3420,          // Microjoules                 │
│    "entropia": 2847563921,   // Shannon (uint32)            │
│    "concepto": "TÉCNICO",    // Clasificación ML            │
│    "confianza": 0.87,        // Certeza [0-1]               │
│    "hash": "8a3f2e91c4..."   // Blake3                      │
│  }                                                           │
│                                                              │
│  Transporte: Redis Pub/Sub                                   │
│  Canales: monje/fenomenologia/{texto|imagen|audio|video}    │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Capa 2: Cognitiva (YO Estructural)

```
┌─────────────────────────────────────────────────────────────┐
│                   CAPA 2: COGNITIVA                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  S1: FENOMENOLOGÍA (El Cuerpo)                      │    │
│  │  • TokenizerLite (MD5 % vocab_size)                 │    │
│  │  • EmbedderCompact (64-dim Int8)                    │    │
│  │  • ClasificadorYO (Dasein/Vorhandene/Zuhandene)    │    │
│  │  • GrundzugTracker (Count-Min Sketch 5×2718)       │    │
│  │  • EchoStateNetwork (100 neuronas reservoir)        │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼ (Grundzugs frecuentes)           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  S2: EMERGENCIA (La Mente)                          │    │
│  │  • FCA Proxy (MinHash + LSH)                        │    │
│  │  • Grafo Conceptual                                 │    │
│  │  • Curvatura de Forman                              │    │
│  │  • Apoptosis (muerte celular de conceptos)          │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼ (Conceptos estables)             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  S3: LÓGICA PURA (La Razón)                         │    │
│  │  • Generador de Axiomas                             │    │
│  │  • Mundo Lógico (consistencia)                      │    │
│  │  • Lógica Modal de 3 valores                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Capa 3: Neuromorfa (PC2)

```
┌─────────────────────────────────────────────────────────────┐
│                   CAPA 3: NEUROMORFA                        │
├─────────────────────────────────────────────────────────────┤
│  Hardware: FPGA / Procesadores Neuromorfos                  │
│                                                              │
│  Conexiones:                                                 │
│  • #1: Recibe embeddings de S1 (float32[64])               │
│  • #2: Inyecta conceptos a S2 (str, float)                 │
│  • #3: Recibe predicciones temporales de ESN               │
│                                                              │
│  Protocolo: TCP (neuro_result_t)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. PUERTOS Y ADAPTADORES

### 6.1 Puertos de Entrada (Inbound)

| ID | Adaptador | Protocolo | Origen | Archivo |
|----|-----------|-----------|--------|---------|
| T1 | tcp_neuromorphic | TCP | PC2 FPGA | `adapters/inbound/tcp_neuromorphic.py` |
| R1 | redis_listener | Redis Sub | Capa 1 | `adapters/inbound/redis_listener.py` |
| W1 | webhook_handler | HTTP POST | n8n | `adapters/inbound/webhook_handler.py` |

### 6.2 Puertos de Salida (Outbound)

| ID | Adaptador | Protocolo | Destino | Archivo |
|----|-----------|-----------|---------|---------|
| N1 | neo4j_repository | Bolt | Neo4j | `adapters/outbound/neo4j_repository.py` |
| R2 | redis_publisher | Redis Pub | Eventos | `adapters/outbound/redis_publisher.py` |
| L1 | lightrag_client | HTTP | LightRAG | `adapters/outbound/lightrag_client.py` |
| W2 | n8n_integrator | Webhook | n8n | `adapters/outbound/n8n_integrator.py` |

### 6.3 Conexiones Neuronales (#1-#4)

```python
# interfaces/neural_ports.py

class NeuralPorts:
    """Puertos de conexión neuronal"""
    
    # #1: Embedding Output (S1 → Externo)
    def get_embedding(self) -> np.ndarray:
        """Retorna float32[64] - Estado semántico actual"""
        return self.s1.embedder.last_embedding
    
    # #2: Concept Injection (Externo → S2)
    def inject_concept(self, name: str, certainty: float):
        """Inyecta concepto con certeza dada"""
        self.s2.motor.inyectar_concepto(name, certainty)
    
    # #3: Temporal Prediction (ESN → Externo)
    def get_prediction(self) -> np.ndarray:
        """Retorna float32[64] - Predicción del siguiente estado"""
        return self.s1.esn.predict()
    
    # #4: Axiom Bridge (S2 ↔ S3)
    def transfer_axiom(self, axiom: Axioma):
        """Transfiere axioma de S2 a S3"""
        self.s3.logica.agregar_axioma(axiom)
```

---

## 7. MOTORES COGNITIVOS

### 7.1 S1: Fenomenología

**Propósito**: Procesamiento inmediato, generación de "qualia" matemático.

```python
class S1Fenomenologia:
    """Motor fenomenológico - El Cuerpo"""
    
    def __init__(self, config: ConfiguracionSistema):
        self.tokenizer = TokenizerLite(config)      # MD5 hash
        self.embedder = EmbedderCompact(config)     # 64-dim
        self.clasificador = ClasificadorYO(config)  # 3 clases
        self.tracker = GrundzugTracker(config)      # Count-Min
        self.emociones = MotorEmociones(config)     # PAD
        self.esn = EchoStateNetwork(config)         # Reservoir
    
    def procesar(self, texto: str) -> Dict:
        tokens = self.tokenizer.tokenize(texto)
        embedding = self.embedder.embed(tokens)
        tipo_yo, probs = self.clasificador.predict(embedding)
        
        # Actualizar trackers
        for t in tokens:
            self.tracker.actualizar(t)
        
        grundzugs = [t for t in tokens if self.tracker.es_grundzug(t)]
        
        return {
            "tokens": tokens,
            "embedding": embedding,
            "tipo_yo": tipo_yo,
            "grundzugs": grundzugs
        }
```

### 7.2 S2: Emergencia

**Propósito**: Abstracción, emergencia de conceptos desde patrones.

```python
class S2Emergencia:
    """Motor de emergencia - La Mente"""
    
    def __init__(self, config: ConfiguracionSistema):
        self.fca = FCAProxy(config)           # MinHash + LSH
        self.grafo = GrafoConceptual(config)  # Topología
        self.conceptos = {}                    # Conceptos activos
    
    def actualizar(self, grundzugs: List[int], timestamp: float):
        # Agregar al FCA
        self.fca.agregar_objeto(len(self.fca.objetos), set(grundzugs))
        
        # Detectar patrones frecuentes
        conceptos_nuevos = self._detectar_conceptos()
        
        # Aplicar apoptosis (muerte de conceptos débiles)
        self._aplicar_apoptosis()
        
        return conceptos_nuevos
    
    def inyectar_concepto(self, nombre: str, certeza: float):
        """Conexión #2: Inyección externa"""
        c = Concepto(nombre=nombre, certeza=certeza, origen="inyectado")
        self.conceptos[c.id] = c
```

### 7.3 S3: Lógica Pura

**Propósito**: Validación formal, construcción de verdad.

```python
class S3LogicaPura:
    """Motor lógico - La Razón"""
    
    def __init__(self, config: ConfiguracionSistema):
        self.axiomas = {}              # Libro de axiomas
        self.mundo = set()             # Objetos existentes
    
    def procesar_conceptos(self, conceptos: Dict, timestamp: float):
        nuevos = 0
        
        for c in conceptos.values():
            if c.certeza > 0.7:
                # Crear axioma de existencia
                ax = Axioma(
                    proposicion=f"exists({c.nombre})",
                    tipo="existencia",
                    certeza=c.certeza
                )
                self.axiomas[ax.id] = ax
                self.mundo.add(c.nombre)
                nuevos += 1
        
        return {"axiomas_totales": len(self.axiomas), "nuevos": nuevos}
```

---

## 8. FLUJO DE DATOS

### 8.1 Diagrama de Secuencia

```
┌─────────┐    ┌─────────┐    ┌────┐    ┌────┐    ┌────┐    ┌───────┐
│ Entrada │    │ Webhook │    │ S1 │    │ S2 │    │ S3 │    │ Neo4j │
└────┬────┘    └────┬────┘    └──┬─┘    └──┬─┘    └──┬─┘    └───┬───┘
     │              │            │          │          │          │
     │ POST /yo     │            │          │          │          │
     │─────────────>│            │          │          │          │
     │              │ procesar() │          │          │          │
     │              │───────────>│          │          │          │
     │              │            │ tokenize │          │          │
     │              │            │ embed    │          │          │
     │              │            │ classify │          │          │
     │              │            │──────────│          │          │
     │              │            │grundzugs │          │          │
     │              │            │─────────>│          │          │
     │              │            │          │ emergir  │          │
     │              │            │          │──────────│          │
     │              │            │          │ conceptos│          │
     │              │            │          │─────────>│          │
     │              │            │          │          │ axiomas  │
     │              │            │          │          │──────────│
     │              │            │          │          │ MERGE    │
     │              │            │          │          │─────────>│
     │              │ respuesta  │          │          │          │
     │<─────────────│            │          │          │          │
     │              │            │          │          │          │
```

### 8.2 Ciclo Completo (5ms)

```
1. ESTÍMULO (0ms)     → Llega texto al webhook
2. S1 (1ms)           → Tokeniza, embed, clasifica, detecta grundzugs
3. S2 (3ms)           → FCA, grafo, curvatura, emergencia
4. S3 (1ms)           → Valida, genera axiomas
5. RESPUESTA (0ms)    → JSON con estado completo
```

---

## 9. IMPLEMENTACIÓN EN CÓDIGO

### 9.1 Archivo Principal: sistema_vivo_v100_completo.py

**Ubicación**: `sistema_terminado/core/optimized/sistema_vivo_v100_completo.py`  
**Líneas**: 661  
**Tamaño**: 26 KB

```python
# Componentes principales
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from enum import Enum
import numpy as np

@dataclass
class ConfiguracionSistema:
    vocab_size: int = 8192
    embed_dim: int = 64
    num_clases: int = 3
    cm_width: int = 2718
    cm_depth: int = 5
    minhash_funciones: int = 100
    esn_reservoir_size: int = 100
    # ... más configuraciones

class TipoYO(Enum):
    DASEIN = 0      # Ser-ahí
    VORHANDENE = 1  # Presente-a-la-mano
    ZUHANDENE = 2   # A-la-mano

def main():
    # Inicializar
    config = ConfiguracionSistema()
    clasificador = ClasificadorYO(config)
    tracker = GrundzugTracker(config)
    emergencia = MotorEmergencia(config)
    logica = S3LogicaPura(config)
    esn = EchoStateNetwork(config)
    
    # Procesar
    for texto in textos:
        resultado = procesar_texto(texto, config, ...)
```

### 9.2 Verificación de Instalación

```python
# tests/test_validacion.py
def test_sistema_completo():
    config = ConfiguracionSistema()
    
    # S1
    cls = ClasificadorYO(config)
    assert cls.W.shape == (3, 64)
    
    # S2
    motor = MotorEmergencia(config)
    assert len(motor.conceptos) == 0
    
    # S3
    logica = S3LogicaPura(config)
    assert len(logica.axiomas) == 0
    
    print("✅ Sistema v100 validado")
```

---

## 10. GUÍA DE EXTENSIÓN

### 10.1 Agregar Nuevo Adaptador de Entrada

```python
# adapters/inbound/mqtt_listener.py
class MQTTListener:
    """Nuevo adaptador para MQTT"""
    
    def __init__(self, facade: SystemFacade):
        self.facade = facade
        self.client = mqtt.Client()
    
    def on_message(self, client, userdata, msg):
        # Traducir mensaje MQTT a formato interno
        data = self._parse_mqtt(msg)
        
        # Usar la fachada (no acceder al core directamente)
        resultado = self.facade.procesar(data)
        
        return resultado
```

### 10.2 Agregar Nuevo Motor

```python
# core/engines/s4_prediccion/motor_prediccion.py
class S4Prediccion:
    """Nuevo motor de predicción"""
    
    def __init__(self, config: ConfiguracionSistema):
        # Solo dependencias internas del core
        self.esn = EchoStateNetwork(config)
    
    def predecir(self, embedding: np.ndarray) -> np.ndarray:
        return self.esn.predict_train(embedding)
```

### 10.3 Testear sin Infraestructura

```python
# tests/test_core_aislado.py
def test_s1_sin_redis():
    """Test S1 sin Redis real"""
    config = ConfiguracionSistema()
    s1 = S1Fenomenologia(config)
    
    resultado = s1.procesar("El ser es tiempo")
    
    assert len(resultado["tokens"]) > 0
    assert resultado["embedding"].shape == (64,)
    assert resultado["tipo_yo"] in TipoYO
```

---

## CONCLUSIÓN

La **Arquitectura Hexagonal** del Organismo Vivo permite:

1. ✅ **Testabilidad**: Core testeable sin infraestructura
2. ✅ **Flexibilidad**: Cambiar Neo4j por PostgreSQL sin tocar lógica
3. ✅ **Claridad**: Separación clara de responsabilidades
4. ✅ **Extensibilidad**: Agregar adaptadores sin modificar core
5. ✅ **Mantenibilidad**: Código organizado y predecible

El concepto de **Vohexistencia** (6 dimensiones relacionales) refleja esta arquitectura hexagonal a nivel conceptual, creando un sistema coherente desde el código hasta la ontología.

---

*Documentación generada el 31 de Diciembre de 2025*  
*Sistema: Organismo Vivo v100*  
*Arquitectura: Hexagonal (Ports & Adapters)*
