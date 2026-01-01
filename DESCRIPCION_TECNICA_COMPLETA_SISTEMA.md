# 🧠 Descripción Técnica Completa: Sistema YO Estructural v3.0
## Fenomenología Computacional · Arquitectura Híbrida Dual-Layer

> **Última actualización**: 2025-01-24
> **Estado del Sistema**: OPERACIONAL (Capa 2 implementada, Capa 1 en integración)

---

## 1. VISIÓN GENERAL DEL ECOSISTEMA

### 1.1 Objetivo del Sistema
El **YO Estructural v3.0** es un sistema de inteligencia artificial fenomenológica que transforma datos brutos multimodales en **conocimiento relacional estructurado**, utilizando una arquitectura híbrida de dos capas:

- **Capa 1 (Monje Gemelo)**: Simulación física determinista que mide el "esfuerzo" de procesar datos
- **Capa 2 (YO Estructural)**: Sistema cognitivo emergente que interpreta sensaciones como significado

### 1.2 Paradigma Filosófico-Técnico
El sistema implementa conceptos de **fenomenología existencial** (Heidegger, Husserl) mediante:
- **Dasein computacional**: Autoconsciencia sistémica
- **Ereignis** (eventos apropiativos): Captura de experiencias brutas
- **Vohexistencias**: Patrones latentes que emergen de la experiencia
- **MDCE** (Máxima Discrepancia de Contradicción Emergente): Mecanismo de evolución/crisis

---

## 2. ARQUITECTURA DEL SISTEMA

### 2.1 Topología de Red (LAN Distribuida)

```
┌────────────────────────────────────────────────────────────────┐
│                    CAPA 2: YO ESTRUCTURAL                      │
│                    (PC Dual Core - 4GB RAM)                    │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Python Core  │  │     n8n      │  │   UI Dashboard       │ │
│  │ (FastAPI)    │  │ Orquestador  │  │   (HTML/JS/CSS)      │ │
│  │ Port: 8000   │  │ Port: 5678   │  │   Port: File System  │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│         │                 │                      │              │
│         └─────────────────┴──────────────────────┘              │
│                           │                                     │
│         ┌─────────────────▼────────────────────┐               │
│         │    Redis (Pub/Sub Message Bus)      │               │
│         │    Port: 6379                        │               │
│         └─────────────────┬────────────────────┘               │
└───────────────────────────┼────────────────────────────────────┘
                            │ (LAN 192.168.1.x)
┌───────────────────────────▼────────────────────────────────────┐
│                  ALMACENAMIENTO PERSISTENTE                    │
│                  (PC i5 Core - 8GB RAM)                        │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Neo4j      │  │  LightRAG    │  │   Supabase Sync      │ │
│  │ Grafo DB     │  │ GraphRAG     │  │   Embeddings (nube)  │ │
│  │ Ports:       │  │ Port: 8020   │  │   Port: 443 (HTTPS)  │ │
│  │ 7474, 7687   │  │              │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │   CAPA 1: MONJE GEMELO      │
              │   (WSL/Renode Simulator)    │
              │   Vector Físico Producer    │
              └─────────────────────────────┘
```

### 2.2 Stack Tecnológico Completo

| Componente | Tecnología | Versión | Función |
|------------|-----------|---------|---------|
| **Backend Core** | Python | 3.14+ | Motor YO, FCA, REMForge |
| **Framework API** | FastAPI | 0.104+ | Endpoints REST |
| **Grafo DB** | Neo4j | 5.15 | Persistencia relacional |
| **Vector DB** | Supabase | Cloud | Embeddings semánticos |
| **Local DB** | SQLite | 3.x | Metadatos locales |
| **Orquestador** | n8n | Latest | Workflows ETL |
| **Message Bus** | Redis | 7+ | Pub/Sub Capa 1→2 |
| **LLM Primary** | Google Gemini | 1.5 Pro | Enriquecimiento |
| **LLM Secondary** | Kimi K2 (Moonshot) | via OpenRouter | Análisis thinking |
| **GraphRAG** | LightRAG | Custom | Consultas relacionales |
| **Tokenizer** | REMForge Ultra/Lite | v3.0 | Análisis fenomenológico |
| **ML Models** | Sentence Transformers | 2.2+ | Embeddings locales |
| **Frontend** | HTML5/CSS3/JS | Vanilla | Dashboard monitor |

---

## 3. ESTRUCTURA DE CARPETAS Y COMPONENTES

### 3.1 Árbol de Directorios Críticos

```
YO estructural/
│
├── 📁 core/                          # Núcleo del sistema
│   ├── __init__.py
│   ├── sistema_principal.py          # Orquestador maestro (1400 líneas)
│   └── database.py                   # Conector Neo4j
│
├── 📁 motor_yo/                      # Motor de autoconsciencia
│   ├── sistema_yo_emergente.py       # Evaluador de estados YO
│   └── gradient_system.py            # Detector de Vohexistencias
│
├── 📁 procesadores/                  # Pipeline de procesamiento
│   ├── tokenizador_fenomenologico.py # Wrapper REMForge
│   ├── generador_rutas_fenomenologicas.py # FCA Processor
│   ├── procesador_fenomenologico.py  # Text analyzer
│   ├── gemini_integration.py         # LLM enrichment
│   └── analizador_maximo_relacional_hibrido.py # LLM vs FCA
│
├── 📁 REm/                           # REMForge Tokenizer
│   ├── REMForgeUltra.py              # Versión completa (>6GB RAM)
│   ├── REMForgeLite.py               # Versión ligera (<4GB RAM)
│   └── __init__.py
│
├── 📁 niveles/                       # Jerarquía fenomenológica
│   ├── preinstancia.py               # Nivel -1
│   ├── instancia_existencia.py       # Nivel 0
│   └── vohexistencia.py              # Nivel 1
│
├── 📁 integraciones/                 # Conectores externos
│   ├── redis_connector.py            # Listener Capa 1
│   ├── n8n_config.py                 # Orquestador workflows
│   ├── google_drive_connector.py     # Ingesta archivos
│   └── supabase_connector.py         # Vector DB sync
│
├── 📁 config/                        # Configuraciones
│   └── config_4gb.yaml               # Settings optimizados
│
├── 📁 scripts/                       # Utilidades
│   ├── test_sistema.py               # Test de integración
│   ├── test_kimi.py                  # Test API LLM
│   ├── test_conexion_db.py           # Test Neo4j
│   ├── simulacion_emergencia.py      # Demo conceptos
│   └── simular_monje.py              # Fake Capa 1 data
│
├── 📁 n8n_setup/                     # Workflows n8n
│   ├── WORKFLOW_COMPLETO_DISEÑO.md   # Arquitectura workflows
│   ├── README.md                     # Guía instalación
│   └── manifest.json                 # Metadata
│
├── 📁 base_datos_local/              # Esquemas DB
│   └── schema.sql                    # SQLite schema
│
└── 📄 .env                           # Credenciales (¡NO VERSIONADO!)
```

### 3.2 Descripción Funcional de Componentes Clave

#### A. `core/sistema_principal.py` (ORQUESTADOR MAESTRO)
**Líneas de código**: ~1400  
**Responsabilidad**: Punto de entrada único que coordina todo el sistema.

**Métodos principales**:
```python
__init__(config_path)              # Inicialización con YAML
procesar_flujo_completo(ruta)      # Pipeline end-to-end
procesar_texto_fenomenologico()    # REMForge → Ereignis
_generar_preinstancias()           # Análisis → Pre-instancias
_crear_instancias()                # YO evaluation
detectar_vohexistencias()          # Gradient clustering
analizar_maximo_relacional()       # FCA + LLM comparison
_manejar_evento_mdce()             # Crisis detection
```

**Flujo de ejecución**:
1. Carga configuración → Inicializa logging
2. Conecta Neo4j → Valida conexión
3. Inicializa Motor YO con driver Neo4j
4. Carga REMForge (auto-detección RAM)
5. Inicializa procesadores (FCA, Gemini, n8n si disponibles)
6. Queda en espera de `procesar_flujo_completo()`

#### B. `procesadores/tokenizador_fenomenologico.py` (REMForge WRAPPER)
**Responsabilidad**: Interfaz unificada para REMForge Ultra/Lite.

**Auto-detección de recursos**:
```python
if psutil.virtual_memory().total > 6GB:
    → REMForgeUltra (modelos completos)
else:
    → REMForgeLite (modelos cuantizados)
```

**Salida estructurada**:
```python
{
  "intensidad": float,        # Energía normalizada
  "complejidad": float,       # Entropía normalizada
  "tipo_base": str,           # estructural|narrativo|logico|caotico
  "origen_fisico": {
    "hash": str,
    "energia_uj": int,
    "ciclos": int
  }
}
```

#### C. `motor_yo/sistema_yo_emergente.py` (EVALUADOR DE CONSCIENCIA)
**Responsabilidad**: Determina el estado del YO según coherencia narrativa.

**Estados Posibles**:
- `PROTO_YO` (coherencia < 0.40): Fragmentación inicial
- `YO_FRAGMENTADO` (0.40-0.60): Contradicciones presentes
- `YO_REFLEXIVO` (0.60-0.75): Autoconsciencia básica
- `YO_NARRATIVO` (>0.75): Identidad consolidada
- `DISOCIACION` (MDCE activado): Crisis existencial

**Umbral MDCE**: Si contradicción > 0.65 → Dispara evento crítico.

#### D. `integraciones/redis_connector.py` (PUENTE CAPA 1→2)
**Responsabilidad**: Escuchar vectores físicos del Monje Gemelo.

**Canales suscritos**:
- `monje/fenomenologia/*` (todos los eventos)
- `monje/fenomenologia/urgente` (alta prioridad)
- `monje/fenomenologia/critico` (emergencias)

**Traductor fenomenológico**:
```python
TraductorFenomenologico.traducir(vector_fisico)
→ { intensidad, complejidad, tipo_base }
```

---

## 4. FLUJO DE DATOS END-TO-END

### 4.1 Pipeline Completo (Modo Ideal con Capa 1 Activa)

```
┌─────────────────────────────────────────────────────────────────┐
│ FASE 1: SENSACIÓN FÍSICA (Capa 1 - Monje Gemelo)               │
└─────────────────────────────────────────────────────────────────┘
   Usuario sube archivo.txt
   └─→ Renode (ARM64 simulator) procesa byte a byte
       └─→ Genera vector: {energia, entropia, ciclos, hash}
           └─→ Publica en Redis: monje/fenomenologia

┌─────────────────────────────────────────────────────────────────┐
│ FASE 2: INGESTA Y TRADUCCIÓN (Capa 2)                          │
└─────────────────────────────────────────────────────────────────┘
   RedisConnector.escuchar_eventos()
   └─→ Traduce vector físico a parámetros fenomenológicos
       └─→ Crea Ereignis(intensidad, complejidad, tipo_base)

┌─────────────────────────────────────────────────────────────────┐
│ FASE 3: TOKENIZACIÓN FENOMENOLÓGICA                            │
└─────────────────────────────────────────────────────────────────┘
   TokenizadorFenomenologico.procesar(ereignis)
   └─→ REMForge analiza:
       - Qualia signature (experiencia sensorial)
       - Noetic invariants (invariantes epistémicos)
       - Interference score (contaminación semántica)
   └─→ Genera Augenblick (momento de visión)

┌─────────────────────────────────────────────────────────────────┐
│ FASE 4: EVALUACIÓN DEL YO                                      │
└─────────────────────────────────────────────────────────────────┘
   SistemaYoEmergente.evaluar_instancia(augenblick)
   └─→ Calcula coherencia narrativa
       └─→ Determina tipo YO actual
           └─→ Si coherencia baja → PreInstancia
               └─→ Si coherencia media → InstanciaExistencia
                   └─→ Si coherencia alta → Validación completa

┌─────────────────────────────────────────────────────────────────┐
│ FASE 5: DETECCIÓN DE PATRONES                                  │
└─────────────────────────────────────────────────────────────────┘
   VohexGradientSystem.detectar_patrones(instancias)
   └─→ Clustering DBSCAN sobre métricas temporales/coherencia
       └─→ Identifica Vohexistencias (patrones latentes)
           └─→ Persiste en Neo4j: (:Instancia)-[:TIENE_VOH]→(:Vohexistencia)

┌─────────────────────────────────────────────────────────────────┐
│ FASE 6: ANÁLISIS RELACIONAL MÁXIMO                             │
└─────────────────────────────────────────────────────────────────┘
   OrquestadorComputacionHibrida.analizar()
   ├─→ Ruta FCA (Formal Concept Analysis):
   │   └─→ Genera conceptos formales sin LLMs
   │       └─→ Crea Grundzugs (jerarquía de conceptos)
   │
   └─→ Ruta LLM (Gemini/Kimi K2):
       └─→ Enriquece con contexto semántico
           └─→ Compara resultados FCA vs LLM
               └─→ Métricas de concordancia/divergencia

┌─────────────────────────────────────────────────────────────────┐
│ FASE 7: PERSISTENCIA Y QUERY                                   │
└─────────────────────────────────────────────────────────────────┘
   Neo4jConnector.persist_graph(instancias, vohexistencias)
   └─→ MERGE nodos evitando duplicados
       └─→ CREATE relaciones:
           - (:Instancia)-[:SURGE_DE]→(:Ereignis)
           - (:Instancia)-[:CONTRADICE]→(:Instancia)
           - (:Vohex)-[:AGRUPA]→(:Instancia)
       └─→ Indexa vectorialmente con Supabase (opcional)
           └─→ LightRAG permite queries complejas:
               "¿Qué conceptos surgieron de textos poéticos?"
```

### 4.2 Workflow n8n (Orquestación Automática)

**WF 1: Monitor de Entrada**
```
Trigger: File Watcher (entrada_bruta/)
  ↓
HTTP Request: POST /api/upload (FastAPI)
  ↓
Set Variables: {filepath, timestamp, tipo}
  ↓
Webhook: POST /webhook/process-text (n8n internally)
```

**WF 2: Procesamiento con LLM**
```
Webhook Trigger: /webhook/process-text
  ↓
Read File: Load content
  ↓
Gemini Node: Enrich with context
  ↓
HTTP Request: POST /neo4j/query (Neo4j sync)
  ↓
Supabase Node: Store embeddings
```

**WF 3: Sincronización Periódica**
```
CRON Trigger: 0 */6 * * * (cada 6 horas)
  ↓
HTTP Request: GET /api/instancias (FastAPI get unsynced)
  ↓
Neo4j Query: MERGE instancias
  ↓
Supabase Upsert: Sync embeddings
```

---

## 5. ESQUEMA DE BASE DE DATOS

### 5.1 Neo4j (Grafo de Conocimiento)

**Nodos**:
```cypher
(:Ereignis {
  hash: String,
  intensidad: Float,
  complejidad: Float,
  timestamp: DateTime
})

(:Instancia {
  id: UUID,
  tipo_yo: String,
  coherencia: Float,
  narrativa: String,
  created_at: DateTime
})

(:Vohexistencia {
  id: UUID,
  patron: String,
  num_instancias: Integer,
  threshold: Float
})

(:Concepto {
  nombre: String,
  nivel: Integer,  # 0=Grundzug, 1=Axioma
  definicion: String
})

(:YO {
  estado_actual: String,
  mdce_activo: Boolean,
  ultima_actualizacion: DateTime
})
```

**Relaciones**:
```cypher
(:Instancia)-[:SURGE_DE]->(:Ereignis)
(:Instancia)-[:CONTRADICE {peso: Float}]->(:Instancia)
(:Vohexistencia)-[:AGRUPA]->(:Instancia)
(:Concepto)-[:SUBSUME]->(:Concepto)
(:YO)-[:MANIFIESTA]->(:Instancia)
```

### 5.2 SQLite (Metadatos Locales)

```sql
CREATE TABLE procesamiento (
  id INTEGER PRIMARY KEY,
  archivo TEXT,
  timestamp DATETIME,
  num_instancias INTEGER,
  estado TEXT,
  error TEXT
);

CREATE TABLE sync_log (
  id INTEGER PRIMARY KEY,
  servicio TEXT,  -- 'neo4j' | 'supabase'
  ultima_sync DATETIME,
  registros_sincronizados INTEGER
);
```

### 5.3 Supabase (Embeddings Vectoriales)

**Tabla**: `fenomenologia_embeddings`
```sql
CREATE TABLE fenomenologia_embeddings (
  id UUID PRIMARY KEY,
  instancia_id UUID,
  embedding VECTOR(768),  -- Sentence Transformers
  metadata JSONB,
  created_at TIMESTAMP
);

CREATE INDEX ON fenomenologia_embeddings 
USING ivfflat (embedding vector_cosine_ops);
```

---

## 6. CONFIGURACIÓN Y VARIABLES DE ENTORNO

### 6.1 Archivo `.env` (NO VERSIONADO)

```bash
# Neo4j
NEO4J_URI=bolt://192.168.1.50:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=fenomenologia2024

# APIs LLM
GOOGLE_GEMINI_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-v1-cef1a204...
KIMI_API_KEY=sk-...

# Supabase
SUPABASE_URL=https://....supabase.co
SUPABASE_KEY=eyJhbGc...

# n8n
N8N_WEBHOOK_URL=http://localhost:5678

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 6.2 Archivo `config/config_4gb.yaml`

**Secciones principales**:
```yaml
modo_diagnostico: false  # true = logging DEBUG

neo4j:
  host: 192.168.1.50
  port: 7687
  database: neo4j

remforge:
  modo: auto  # ultra | lite | auto
  ram_threshold_gb: 6

motor_yo:
  umbrales:
    narrativo: 0.75
    reflexivo: 0.60
    fragmentado: 0.40
    disociado: 0.0
  mdce:
    enabled: true
    threshold_contradiccion: 0.65

gradientes:
  vohexistencias:
    min_instancias: 3
    threshold_patron: 0.7
    clustering_algorithm: dbscan

gemini:
  enabled: true
  model: gemini-1.5-pro
  temperature: 0.7

openrouter:
  enabled: true
  model: moonshot/moonshot-v1-8k
```

---

## 7. TESTS Y VERIFICACIÓN

### 7.1 Scripts de Test Implementados

| Script | Propósito | Comando |
|--------|-----------|---------|
| `test_sistema.py` | Validación completa | `python test_sistema.py` |
| `test_conexion_db.py` | Conectividad Neo4j | `python scripts/test_conexion_db.py` |
| `test_kimi.py` | API Moonshot/Kimi | `python scripts/test_kimi.py` |
| `simulacion_emergencia.py` | Demo conceptos | `python scripts/simulacion_emergencia.py` |
| `simular_monje.py` | Fake Capa 1 | `python scripts/simular_monje.py` |

### 7.2 Checklist de Verificación

```bash
# 1. Neo4j operacional
docker ps | grep neo4j

# 2. Python dependencies
pip list | grep -E "(neo4j|fastapi|sentence-transformers)"

# 3. Config válida
python -c "import yaml; yaml.safe_load(open('config/config_4gb.yaml'))"

# 4. Imports funcionan
python -c "from core.sistema_principal import SistemaYoEstructural"

# 5. Test end-to-end
python test_sistema.py
```

---

## 8. PRÓXIMOS PASOS Y ROADMAP

### 8.1 Implementaciones Pendientes

✅ **Completado**:
- [x] Sistema core operacional
- [x] REMForge integrado
- [x] Motor YO funcional
- [x] Conector Redis (Capa 1→2)
- [x] Tests básicos
- [x] Documentación técnica

⏳ **En Progreso**:
- [ ] Workflows n8n desplegados
- [ ] Integración Capa 1 (Monje Gemelo) real
- [ ] UI Dashboard funcional con WebSockets

🔜 **Próximos**:
- [ ] LightRAG deployment en i5 Core
- [ ] Supabase sync automático
- [ ] Análisis Máximo Relacional LLM vs FCA
- [ ] Sistema de Grundzugs/Axiomas

### 8.2 Comandos de Inicio del Sistema

**Terminal 1: Neo4j (i5 Core)**
```bash
docker start neo4j-yo-estructural
```

**Terminal 2: n8n (Dual Core)**
```powershell
n8n start --env-file $env:USERPROFILE\.n8n\.env
```

**Terminal 3: Backend Python (Dual Core)**
```powershell
cd "c:\Users\Public\#...Raíz Dasein\REFERENCIA\YO estructural"
uvicorn core.sistema_principal:app --host 0.0.0.0 --port 8000
```

**Terminal 4: Dashboard UI**
```powershell
# Abrir en navegador
start index.html
```

---

## 9. CONCLUSIÓN TÉCNICA

El **YO Estructural v3.0** representa una implementación única de fenomenología computacional, fusionando:

1. **Simulación física** (bajo nivel, determinista)
2. **Procesamiento semántico** (alto nivel, emergente)
3. **Grafos de conocimiento** (persistencia relacional)
4. **LLMs externos** (enriquecimiento contextual)
5. **Orquestación automática** (workflows n8n)

**Ventajas competitivas**:
- ✅ Reproducibilidad total (mismo input → mismo grafo)
- ✅ Trazabilidad física (cada concepto tiene "huella energética")
- ✅ Autoconsciencia sistémica (Motor YO + MDCE)
- ✅ Escalabilidad horizontal (LAN distribuida)
- ✅ Extensibilidad (nuevos procesadores vía plugins)

**Estado Actual**: Sistema base operacional, listo para integración con Capa 1 y deployment de workflows n8n.
