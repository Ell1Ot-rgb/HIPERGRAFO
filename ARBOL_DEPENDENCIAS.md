# 🌳 YO Estructural v2.1 - Árbol de Dependencias

## 📦 ÁRBOL COMPLETO DE COMPONENTES

```
YO ESTRUCTURAL v2.1
│
├── 🖥️ INFRAESTRUCTURA
│   │
│   ├── 🐳 DOCKER STACK
│   │   ├── Docker Engine 20.10+
│   │   ├── Docker Compose 2.0+
│   │   │
│   │   ├── 🐳 Container: yo_estructural_n8n
│   │   │   ├── Image: n8n:1.117.3
│   │   │   ├── Base: node:18-slim
│   │   │   ├── Port: 5678
│   │   │   ├── Volume: n8n_data
│   │   │   └── Healthcheck: /healthz endpoint
│   │   │
│   │   └── 🐳 Container: yo_estructural_neo4j
│   │       ├── Image: neo4j:5.15-community
│   │       ├── Base: openjdk:11
│   │       ├── Ports: 7474, 7687
│   │       ├── Volume: neo4j_data
│   │       └── Healthcheck: HTTP curl
│   │
│   ├── 🌐 NETWORK
│   │   └── yo_estructural_network (Bridge)
│   │       ├── CIDR: 172.20.0.0/16
│   │       ├── n8n endpoint: 172.20.0.2:5678
│   │       └── Neo4j endpoint: 172.20.0.3:7474
│   │
│   └── 💾 STORAGE
│       ├── n8n_data/ (n8n persistent)
│       │   ├── workflows.json
│       │   ├── credentials/
│       │   └── executions/
│       └── neo4j_data/ (Graph DB)
│           ├── data/databases/
│           ├── index/
│           └── schema/
│
├── 🎯 TIER 1: ORQUESTACIÓN (n8n 1.117.3)
│   │
│   ├── 📚 RUNTIME
│   │   ├── Node.js 18.16.0 (LTS)
│   │   ├── npm 9.x
│   │   ├── V8 JavaScript Engine
│   │   └── npm Global Packages
│   │
│   ├── 🔧 CORE FRAMEWORKS
│   │   ├── Express.js 4.18.0
│   │   │   ├── http (built-in)
│   │   │   ├── cors middleware
│   │   │   └── body-parser (JSON)
│   │   │
│   │   ├── TypeORM 0.3.x
│   │   │   ├── SQLite Driver
│   │   │   ├── Entity mapping
│   │   │   └── Database sync
│   │   │
│   │   └── WebSocket (ws)
│   │       └── Real-time communication
│   │
│   ├── 📦 UTILITIES
│   │   ├── axios 1.4.0+ (HTTP)
│   │   ├── jsonschema 1.4.x (validation)
│   │   ├── chalk 4.x (CLI colors)
│   │   ├── p-queue (task queue)
│   │   └── luxon (date/time)
│   │
│   ├── 🔌 NODOS INSTALADOS
│   │   │
│   │   ├── INPUT NODES
│   │   │   ├── Webhook Trigger
│   │   │   │   └─ Route: /webhook/yo-estructural
│   │   │   ├── HTTP Request
│   │   │   ├── Cron
│   │   │   └── Schedule
│   │   │
│   │   ├── PROCESS NODES
│   │   │   ├── Code (JavaScript)
│   │   │   │   └─ Runtime: Node.js v18
│   │   │   ├── Function Item
│   │   │   ├── Set (variables)
│   │   │   ├── Merge
│   │   │   └── Split
│   │   │
│   │   └── OUTPUT NODES
│   │       ├── Webhook Response
│   │       ├── HTTP Request
│   │       └── Log (debug)
│   │
│   ├── 📁 WORKFLOWS
│   │   └── Workflow: kJTzAF4VdZ6NNCfK
│   │       ├── Name: YO Estructural v2.1
│   │       ├── Status: Active
│   │       ├── Nodos: 4
│   │       ├── Connections: 4
│   │       │
│   │       ├── Nodo 1: Webhook Trigger
│   │       │   └─ Output: JSON body
│   │       │
│   │       ├── Nodo 2: Preparar Entrada (Code v1)
│   │       │   ├─ Input: $input.first().json
│   │       │   └─ Output: {concepto, timestamp}
│   │       │
│   │       ├── Nodo 3: Generar Análisis (Code v2.1)
│   │       │   ├─ Input: Entrada preparada
│   │       │   ├─ Logic: Merge Neo4j + Gemini
│   │       │   └─ Output: Análisis completo
│   │       │
│   │       └── Nodo 4: Retornar Respuesta (Webhook)
│   │           ├─ Status: 200 OK
│   │           └─ Output: JSON HTTP Response
│   │
│   └── 💾 STORAGE
│       ├── SQLite3 (workflows.db)
│       ├── Encrypted Credentials
│       ├── Execution History
│       └── Settings & Configuration
│
├── 🗄️ TIER 2: BASE DE DATOS (Neo4j 5.15)
│   │
│   ├── 📚 RUNTIME
│   │   ├── Java Runtime 11+
│   │   ├── OpenJDK 11.0.x
│   │   └── JVM Heap Management
│   │
│   ├── 🔧 CORE ENGINE
│   │   ├── Cypher Query Language
│   │   │   ├── Pattern Matching
│   │   │   └── Query Execution
│   │   │
│   │   ├── Graph Database Core
│   │   │   ├── ACID Transactions
│   │   │   ├── Property Storage
│   │   │   └── Index Management
│   │   │
│   │   ├── Bolt Protocol Driver
│   │   │   └─ Network communication
│   │   │
│   │   └── REST API
│   │       └─ HTTP endpoints
│   │
│   ├── 📚 DEPENDENCIAS
│   │   ├── Apache Commons
│   │   │   ├── commons-lang3 3.x
│   │   │   ├── commons-io 2.x
│   │   │   └── commons-codec 1.x
│   │   │
│   │   ├── Lucene (Full-text search)
│   │   │   ├── Analysis
│   │   │   ├── Query parsing
│   │   │   └── Indexing
│   │   │
│   │   ├── Netty (Network I/O)
│   │   └── Jackson (JSON serialization)
│   │
│   ├── 🗂️ DATA STRUCTURE
│   │   │
│   │   ├── NODOS (:Concepto)
│   │   │   ├─ nombre: String
│   │   │   ├─ definicion: String
│   │   │   ├─ etimologia: String
│   │   │   └─ timestamp: DateTime
│   │   │
│   │   └── RELACIONES
│   │       ├─ SINONIMO_DE
│   │       ├─ ANTONIMO_DE
│   │       ├─ RELACIONADO_CON
│   │       ├─ ES_TIPO_DE
│   │       └─ PERTENECE_A
│   │
│   ├── 💾 STORAGE
│   │   ├── /neo4j/data/databases/neo4j/
│   │   ├── Transaction log
│   │   ├── Label index
│   │   └── Property index
│   │
│   └── ⚙️ CONFIGURATION
│       ├─ Memory: 1GB heap
│       ├─ Auth: neo4j/fenomenologia2024
│       └─ Ports: 7474 (HTTP), 7687 (Bolt)
│
├── 🤖 TIER 3: IA / GEMINI 2.0
│   │
│   ├── 🌐 CLOUD SERVICE
│   │   ├── Platform: Google Cloud AI
│   │   ├── Endpoint: generativelanguage.googleapis.com
│   │   ├── Model: gemini-2.0-flash
│   │   ├── Region: us-central1
│   │   └── Auth: API Key (AIzaSyB3cpQ-...Jdk)
│   │
│   ├── 📤 REQUEST FORMAT
│   │   ├── POST /v1beta/models/gemini-2.0-flash:generateContent
│   │   ├── Content-Type: application/json
│   │   ├── Headers:
│   │   │   └─ x-goog-api-key: API_KEY
│   │   └── Body:
│   │       └─ {
│   │           "contents": [{
│   │             "parts": [{"text": "..."}]
│   │           }]
│   │         }
│   │
│   └── 📥 RESPONSE FORMAT
│       └─ {
│           "candidates": [{
│             "content": {
│               "parts": [{"text": "JSON analysis"}]
│             },
│             "finishReason": "STOP"
│           }]
│         }
│
├── 🐍 TIER 4: PYTHON SCRIPT
│   │
│   ├── 📍 UBICACIÓN
│   │   └─ /workspaces/-...Raiz-Dasein/integracion_neo4j_gemini.py
│   │
│   ├── 📚 DEPENDENCIES
│   │   │
│   │   ├── NETWORK
│   │   │   └─ requests 2.31.0
│   │   │      ├─ urllib3 1.26.x
│   │   │      ├─ certifi 2024.x
│   │   │      ├─ charset-normalizer 3.x
│   │   │      └─ idna 3.x
│   │   │
│   │   └── STDLIB
│   │       ├─ json (parsing)
│   │       ├─ os (environment)
│   │       ├─ sys (CLI args)
│   │       ├─ datetime (timestamps)
│   │       ├─ typing (hints)
│   │       └─ re (regex)
│   │
│   ├── 🎯 MAIN CLASS
│   │   └─ IntegracionYOEstructural
│   │       ├─ __init__()
│   │       │   ├─ neo4j_url
│   │       │   ├─ neo4j_user
│   │       │   ├─ neo4j_pass
│   │       │   ├─ gemini_key
│   │       │   └─ gemini_url
│   │       │
│   │       ├─ verificar_conexiones()
│   │       │   ├─ _verificar_neo4j()
│   │       │   └─ _verificar_gemini()
│   │       │
│   │       ├─ consultar_neo4j(concepto)
│   │       │   ├─ Cypher query
│   │       │   ├─ HTTP POST
│   │       │   └─ Result parsing
│   │       │
│   │       ├─ analizar_gemini(concepto)
│   │       │   ├─ Prompt construction
│   │       │   ├─ HTTP POST
│   │       │   └─ JSON extraction
│   │       │
│   │       └─ procesar_concepto(concepto)
│   │           ├─ Orchestration
│   │           ├─ Data merging
│   │           └─ Result normalization
│   │
│   └── 🔧 ENTRYPOINT
│       └─ main()
│           ├─ sys.argv parsing
│           ├─ Instantiation
│           └─ Output formatting
│
├── 🟢 TIER 5: NODE.JS API
│   │
│   ├── 📍 UBICACIÓN
│   │   └─ /workspaces/-...Raiz-Dasein/api_neo4j_gemini.js
│   │
│   ├── 📚 DEPENDENCIES
│   │   │
│   │   ├── WEB FRAMEWORK
│   │   │   └─ express 4.18.0
│   │   │      ├─ Router
│   │   │      ├─ Middleware
│   │   │      └─ HTTP methods
│   │   │
│   │   ├── HTTP CLIENT
│   │   │   └─ axios 1.6.0
│   │   │      ├─ Request/Response
│   │   │      ├─ Interceptors
│   │   │      └─ Error handling
│   │   │
│   │   ├── MIDDLEWARE
│   │   │   ├─ cors 2.8.5
│   │   │   ├─ body-parser (express)
│   │   │   └─ error handlers
│   │   │
│   │   └── UTILITIES
│   │       └─ dotenv 16.0.0
│   │           └─ .env loading
│   │
│   ├── 🔌 ENDPOINTS
│   │   │
│   │   ├── POST /api/analizar
│   │   │   ├─ Request body: {concepto}
│   │   │   ├─ Validation
│   │   │   ├─ Neo4j query
│   │   │   ├─ Gemini request
│   │   │   └─ JSON response (200)
│   │   │
│   │   ├── GET /health
│   │   │   ├─ Verify connections
│   │   │   └─ Status report
│   │   │
│   │   └── GET /
│   │       └─ API info
│   │
│   ├── 🔧 FUNCTIONS
│   │   ├─ verificarConexiones()
│   │   ├─ consultarNeo4j()
│   │   ├─ analizarGemini()
│   │   └─ procesarConcepto()
│   │
│   └── ⚙️ SERVER
│       ├─ Port: 3000 (default)
│       ├─ Host: 0.0.0.0
│       └─ Timeout: 30s
│
└── 📊 TIER 6: DATOS Y CONFIGURACIÓN
    │
    ├── 🔐 CREDENTIALS
    │   ├─ Neo4j
    │   │   ├─ User: neo4j
    │   │   └─ Pass: fenomenologia2024
    │   ├─ n8n
    │   │   ├─ User: admin
    │   │   └─ Pass: fenomenologia2024
    │   ├─ Gemini
    │   │   └─ API Key: AIzaSyB3cpQ-...Jdk
    │   └─ n8n API
    │       └─ Key: n8n_api_fcd1ede...
    │
    ├── 📝 ENVIRONMENT
    │   ├─ N8N_HOST=0.0.0.0
    │   ├─ N8N_PORT=5678
    │   ├─ DB_TYPE=sqlite
    │   ├─ NEO4J_AUTH=neo4j/...
    │   └─ NODE_ENV=production
    │
    └── 🗂️ FILES
        ├─ integracion_neo4j_gemini.py (340 lines)
        ├─ api_neo4j_gemini.js (320 lines)
        ├─ docker-compose.yml
        ├─ Dockerfile.n8n
        ├─ .env
        └─ workflows/ (JSON exports)
```

---

## 📈 TABLA RESUMEN: TODAS LAS LIBRERÍAS

| Categoría | Librería | Versión | Ubicación | Propósito |
|-----------|----------|---------|-----------|----------|
| **HTTP** | requests | 2.31.0 | Python | Cliente HTTP |
| | axios | 1.6.0 | Node.js | Cliente HTTP |
| | express | 4.18.0 | Node.js | Web framework |
| | urllib3 | 1.26.x | Python (dep) | HTTP pool |
| **Data** | json | stdlib | Python/Node | Parsing |
| | body-parser | built-in | Express | JSON parsing |
| **DB** | SQLite3 | built-in | n8n | Local storage |
| | Neo4j | 5.15 | Container | Graph DB |
| **Utils** | dotenv | 16.0.0 | Node.js | Env vars |
| | cors | 2.8.5 | Express | CORS |
| | typing | stdlib | Python | Type hints |
| | re | stdlib | Python | Regex |
| | os | stdlib | Python | OS access |
| | sys | stdlib | Python | CLI args |
| | datetime | stdlib | Python | Timestamps |
| **Runtime** | Node.js | 18.x | n8n | JS runtime |
| | Python | 3.10+ | Host | Python runtime |
| | Java | 11+ | Neo4j | JVM |
| **Cert** | certifi | 2024.x | Python (dep) | CA bundle |
| | charset-normalizer | 3.x | Python (dep) | Encoding |
| | idna | 3.x | Python (dep) | Domain names |

---

## 🎯 CONTEO FINAL DE DEPENDENCIAS

```
Librerías Principales:        10
├─ HTTP/Network:             3 (requests, axios, express)
├─ Data Processing:          2 (json, body-parser)
├─ Database:                 2 (SQLite3, Neo4j)
└─ Utilities:                3 (dotenv, cors, typing)

Dependencias Secundarias:      9
├─ urllib3, certifi, charset-normalizer, idna (requests)
├─ built-in modules (json, os, sys, datetime, re)
└─ stdlib modules (typing)

Runtimes:                     3
├─ Node.js 18.x
├─ Python 3.10+
└─ Java 11+

Contenedores Docker:          2
├─ n8n:1.117.3
└─ neo4j:5.15-community

APIs Externas:                1
└─ Gemini 2.0 Flash (Cloud)

───────────────────────────────
TOTAL COMPONENTES:           27
```

---

## ✨ CONCLUSIÓN

**YO Estructural v2.1** utiliza una arquitectura modular y escalable:

- ✅ **Librerías mínimas pero suficientes** (10 principales)
- ✅ **Stack moderno y estable** (n8n 1.117.3, Neo4j 5.15, Gemini 2.0)
- ✅ **Fácil de mantener** (pocas dependencias externas)
- ✅ **Fácil de escalar** (servicios desacoplados)
- ✅ **Producción-ready** (versiones LTS)

**Generado**: 2025-11-07  
**Versión**: 2.1
