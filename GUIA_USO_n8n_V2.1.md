# 📚 GUÍA COMPLETA DE USO - n8n v1.117.3 + Neo4j + Gemini

**Versión:** 2.1  
**Fecha:** 7 de Noviembre de 2025  
**Nivel:** Avanzado / Desarrolladores

---

## 📖 TABLA DE CONTENIDOS

1. [Introducción](#introducción)
2. [Arquitectura](#arquitectura)
3. [Instalación](#instalación)
4. [Configuración](#configuración)
5. [Uso del Webhook](#uso-del-webhook)
6. [Integraciones Avanzadas](#integraciones-avanzadas)
7. [Troubleshooting](#troubleshooting)
8. [Ejemplos Prácticos](#ejemplos-prácticos)

---

## 🎯 Introducción

**YO Estructural v2.1** es un sistema de análisis fenomenológico que integra:

- **n8n 1.117.3**: Orquestador de flujos (versión estable)
- **Neo4j 5.15**: Base de datos de grafos (almacenamiento conceptual)
- **Gemini 2.0 Flash**: API de IA de Google (análisis de lenguaje)

### Características Principales

✅ **Análisis fenomenológico de conceptos**  
✅ **5 rutas de análisis** (etimológica, sinonímica, antonímica, metafórica, contextual)  
✅ **Cálculo de certeza combinada** (Neo4j + Gemini)  
✅ **Webhook HTTP público** (acceso remoto)  
✅ **Respuesta en JSON** (fácil de integrar)  
✅ **Respuesta rápida** (<100ms)  

---

## 🏗️ Arquitectura

### Componentes del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENTE EXTERNO                         │
│  (cURL, Python, JavaScript, Postman, App Web, etc.)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                    HTTP POST
                   (JSON Body)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    n8n 1.117.3                              │
│                  (Puerto 5678)                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Webhook Trigger (/webhook/yo-estructural)        │  │
│  │    ├─ Recibe: {"concepto": "XXXX"}                  │  │
│  │    └─ Output: payload sin procesar                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────┴──────────────────────────────┐   │
│  │ 2. Preparar Entrada (Code Node v3.0)               │   │
│  │    ├─ Extrae: concepto del JSON                    │   │
│  │    ├─ Default: "SOPORTE"                           │   │
│  │    └─ Output: {concepto, timestamp, estado}       │   │
│  └──────────────────────────────────────────────────────┘  │
│           │ (copia flujo)          │ (copia flujo)         │
│      ┌────▼─────┐            ┌─────▼──────┐                │
│      │           │            │            │                │
│  ┌───▼────┐  ┌──▼─────┐                                   │
│  │Neo4j   │  │ Gemini │                                   │
│  └────────┘  └────────┘                                   │
│      │           │                                          │
│  ┌───▼───────────▼──────────────────────────────────────┐  │
│  │ 3. Combinar Resultados (Code Node v2.1)             │  │
│  │    ├─ Merge Neo4j + Gemini outputs                  │  │
│  │    ├─ Calcula certeza_combinada                     │  │
│  │    └─ Output: JSON completo                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │ 4. Webhook Response (Respondent)                    │   │
│  │    ├─ Status: 200 OK                                │   │
│  │    └─ Body: JSON completo                           │   │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬─────────────────────────────────┘
                         │
                    HTTP 200
                  (JSON Response)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     CLIENTE EXTERNO                         │
│  (Recibe JSON con análisis fenomenológico)                  │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Datos Detallado

```
Entrada: {"concepto": "FENOMENOLOGIA"}
   │
   ├─→ Node 1: Webhook Trigger
   │   Output: raw payload
   │
   ├─→ Node 2: Preparar Entrada
   │   ├─ Lee: body.concepto
   │   ├─ Valida: concepto !== null
   │   ├─ Default: "SOPORTE"
   │   Output: {concepto: "FENOMENOLOGIA", timestamp: "...", estado: "procesando"}
   │
   ├─→ Node 3: Query Neo4j (paralelo)
   │   ├─ Conecta a: http://neo4j:7474/db/neo4j/tx/commit
   │   ├─ Auth: neo4j / fenomenologia2024
   │   ├─ Cypher: MATCH (c:Concepto {nombre: $concepto}) RETURN c
   │   Output: {encontrado: true/false, relacionados: [...], etimologia: "..."}
   │
   ├─→ Node 4: Gemini Análisis (paralelo)
   │   ├─ Conecta a: generativelanguage.googleapis.com
   │   ├─ Prompt: "Analiza fenomenológicamente..."
   │   ├─ Modelo: gemini-2.0-flash
   │   Output: {rutas: {etimologica, sinonímica, ...}, sintesis: "..."}
   │
   ├─→ Node 5: Combinar Resultados
   │   ├─ Merge Neo4j + Gemini
   │   ├─ Calcula: certeza_combinada = 0.92
   │   ├─ Fuentes: neo4j, gemini, ambas
   │   Output: {concepto, rutas, certeza, integraciones, timestamp, sistema}
   │
   └─→ Node 6: Webhook Response
       └─ Retorna: JSON completo (200 OK)

Salida: {
  "concepto": "FENOMENOLOGIA",
  "es_maximo_relacional": true,
  "certeza_combinada": 0.92,
  "rutas_fenomenologicas": [...],
  ...
}
```

---

## 💾 Instalación

### Prerequisitos

- Docker y Docker Compose instalados
- Puerto 5678 disponible (n8n)
- Conexión a internet (Gemini API)
- Cuenta de GitHub (Codespaces)

### Pasos de Instalación

#### 1. Clonar el repositorio

```bash
git clone https://github.com/Ell1Ot-rgb/-...Raiz-Dasein.git
cd "-...Raiz-Dasein"
cd "YO estructural"
```

#### 2. Configurar variables de entorno

```bash
cat > .env << EOF
# n8n
N8N_USER_MANAGEMENT_DISABLED=false
N8N_PROTOCOL=http
N8N_HOST=0.0.0.0
N8N_PORT=5678
N8N_EXECUTION_MODE=regular
N8N_SECURITY_BASIC_AUTH_ACTIVE=true
N8N_SECURITY_BASIC_AUTH_USER=admin
N8N_SECURITY_BASIC_AUTH_PASSWORD=fenomenologia2024

# Neo4j
NEO4J_AUTH=neo4j/fenomenologia2024
NEO4J_INITIAL_SERVER_MODE_CONSTRAINT_VERIFICATION=WARN

# Gemini
GEMINI_API_KEY=AIzaSyB3cpQ-nVNn8qeC6fUhwozpgYxEFoB_Jdk
EOF
```

#### 3. Levantar los servicios

```bash
docker compose up -d
```

#### 4. Verificar estado

```bash
# Todos los contenedores corriendo
docker compose ps

# Logs de n8n
docker compose logs n8n | tail -20

# Logs de Neo4j
docker compose logs neo4j | tail -20
```

---

## ⚙️ Configuración

### n8n - Credenciales

#### 1. Acceder a n8n

- URL: `http://localhost:5678`
- Usuario: `admin`
- Contraseña: `fenomenologia2024`

#### 2. Crear Credencial de Neo4j (Opcional - ya existe)

```
Nombre: Neo4j Credentials
Tipo: Database - Neo4j
URL: http://neo4j:7474
Usuario: neo4j
Contraseña: fenomenologia2024
```

#### 3. Crear Credencial de Gemini (Opcional - ya existe)

```
Nombre: Gemini API Key
Tipo: Generic Credential
API Key: AIzaSyB3cpQ-nVNn8qeC6fUhwozpgYxEFoB_Jdk
```

### Neo4j - Inicializar Base de Datos

#### 1. Acceder a Neo4j Browser

```bash
# Desde dentro del Codespace:
curl -u neo4j:fenomenologia2024 http://neo4j:7474/browser/
```

#### 2. Crear Índices (Opcional)

```cypher
CREATE INDEX idx_concepto_nombre FOR (c:Concepto) ON (c.nombre);
```

#### 3. Crear Nodos de Ejemplo (Opcional)

```cypher
CREATE (n1:Concepto {nombre: 'FENOMENOLOGIA', definicion: 'Estudio de fenómenos', etimologia: 'Del griego phainomenon'})
CREATE (n2:Concepto {nombre: 'DASEIN', definicion: 'Ser-ahí en alemán', etimologia: 'Da (ahí) + Sein (ser)'})
CREATE (n1)-[:RELACIONADO_CON]->(n2)
```

---

## 🔌 Uso del Webhook

### Sintaxis Básica

```http
POST /webhook/yo-estructural HTTP/1.1
Host: localhost:5678
Content-Type: application/json

{"concepto": "CONCEPTO_A_ANALIZAR"}
```

### Ejemplos de Uso

#### Ejemplo 1: cURL - Análisis Simple

```bash
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"FENOMENOLOGIA"}' | jq '.'
```

#### Ejemplo 2: cURL - Análisis por Defecto

```bash
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{}'
```

#### Ejemplo 3: Python - Script de Análisis

```python
#!/usr/bin/env python3
import requests
import json
from typing import Dict, Any

def analizar_concepto(concepto: str) -> Dict[str, Any]:
    """Analiza un concepto usando YO Estructural"""
    url = "http://localhost:5678/webhook/yo-estructural"
    
    try:
        resp = requests.post(
            url,
            json={"concepto": concepto},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

# Uso
resultado = analizar_concepto("FENOMENOLOGIA")

if resultado:
    print(f"Concepto: {resultado['concepto']}")
    print(f"Certeza: {resultado['certeza_combinada']:.0%}")
    print(f"Rutas: {len(resultado['rutas_fenomenologicas'])}/5")
    
    # Mostrar todas las rutas
    for ruta in resultado['rutas_fenomenologicas']:
        print(f"  • {ruta['tipo']}: {ruta['certeza']:.0%}")
```

#### Ejemplo 4: JavaScript - Análisis con Async/Await

```javascript
// analizar.js
async function analizarConcepto(concepto) {
  const url = 'http://localhost:5678/webhook/yo-estructural';
  
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({concepto})
    });
    
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    
    const data = await resp.json();
    return data;
  } catch (err) {
    console.error('Error:', err);
    return null;
  }
}

// Uso
analizarConcepto('FENOMENOLOGIA').then(resultado => {
  console.log(`Concepto: ${resultado.concepto}`);
  console.log(`Certeza: ${(resultado.certeza_combinada * 100).toFixed(0)}%`);
  console.log(`Rutas:`);
  resultado.rutas_fenomenologicas.forEach(r => {
    console.log(`  • ${r.tipo}: ${(r.certeza * 100).toFixed(0)}%`);
  });
});
```

#### Ejemplo 5: Node.js - Script de Batch

```javascript
// batch_analysis.js
const axios = require('axios');

const conceptos = [
  'FENOMENOLOGIA',
  'DASEIN',
  'EXISTENCIA',
  'LIBERTAD',
  'ESENCIA'
];

async function analizarBatch() {
  const url = 'http://localhost:5678/webhook/yo-estructural';
  const resultados = [];
  
  for (const concepto of conceptos) {
    try {
      const resp = await axios.post(url, {concepto});
      resultados.push({
        concepto,
        certeza: resp.data.certeza_combinada,
        estado: resp.data.estado_integracion
      });
      console.log(`✅ ${concepto}: ${resp.data.certeza_combinada.toFixed(2)}`);
    } catch (err) {
      console.error(`❌ ${concepto}: ${err.message}`);
    }
  }
  
  return resultados;
}

analizarBatch().then(r => console.table(r));
```

---

## 🔗 Integraciones Avanzadas

### Integración con Express.js

```javascript
// api_server.js
const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

const WEBHOOK_URL = 'http://localhost:5678/webhook/yo-estructular';

// Proxy del webhook
app.post('/analizar', async (req, res) => {
  try {
    const {concepto} = req.body;
    const resp = await axios.post(WEBHOOK_URL, {concepto});
    res.json(resp.data);
  } catch (err) {
    res.status(500).json({error: err.message});
  }
});

// Con cache
const cache = new Map();

app.post('/analizar-cached', async (req, res) => {
  const {concepto} = req.body;
  
  if (cache.has(concepto)) {
    return res.json({...cache.get(concepto), cached: true});
  }
  
  try {
    const resp = await axios.post(WEBHOOK_URL, {concepto});
    cache.set(concepto, resp.data);
    res.json({...resp.data, cached: false});
  } catch (err) {
    res.status(500).json({error: err.message});
  }
});

app.listen(3000, () => console.log('🚀 API en puerto 3000'));
```

### Integración con FastAPI

```python
# api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import asyncio

app = FastAPI()

WEBHOOK_URL = "http://localhost:5678/webhook/yo-estructural"

class ConceptoRequest(BaseModel):
    concepto: str

@app.post("/analizar")
async def analizar(req: ConceptoRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            WEBHOOK_URL,
            json={"concepto": req.concepto}
        )
        return resp.json()

# Cache
cache = {}

@app.post("/analizar-cached")
async def analizar_cached(req: ConceptoRequest):
    if req.concepto in cache:
        return {**cache[req.concepto], "cached": True}
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            WEBHOOK_URL,
            json={"concepto": req.concepto}
        )
        data = resp.json()
        cache[req.concepto] = data
        return {**data, "cached": False}

# Ejecución: uvicorn api_server:app --reload
```

---

## 🐛 Troubleshooting

### ❌ Problema: "Connection refused"

**Síntomas:** 
```
Error: connection refused at 127.0.0.1:5678
```

**Solución:**
```bash
# 1. Verificar que Docker está corriendo
docker compose ps

# 2. Si no está, levantarlo
docker compose up -d

# 3. Esperar a que n8n inicie (30-60 segundos)
docker compose logs n8n | grep "n8n started"

# 4. Verificar healthz
curl http://localhost:5678/healthz
```

---

### ❌ Problema: "Webhook not found"

**Síntomas:**
```json
{"code":"WEBHOOK_ERROR","level":"warning","message":"Webhook not found"}
```

**Solución:**
```bash
# 1. Verificar que el workflow está activo
curl http://localhost:5678/api/v1/workflows/kJTzAF4VdZ6NNCfK \
  -H "X-N8N-API-KEY: YOUR_API_KEY" | jq '.active'

# 2. Si retorna false, activarlo desde el Dashboard
# Dashboard → Workflows → Click en workflow → Botón "Active"

# 3. Recrear el webhook
# En el workflow, edita el nodo "Webhook Trigger"
# Guarda y activa nuevamente
```

---

### ❌ Problema: "Neo4j connection failed"

**Síntomas:**
```
Cannot connect to http://neo4j:7474
```

**Solución:**
```bash
# 1. Verificar que Neo4j está corriendo
docker compose logs neo4j | tail -20

# 2. Verificar desde dentro del contenedor
docker exec -it yo_estructural_neo4j \
  curl -u neo4j:fenomenologia2024 http://localhost:7474/db/neo4j/tx/commit \
  -X POST -d '{"statements":[{"statement":"RETURN 1"}]}'

# 3. Reiniciar Neo4j
docker compose restart neo4j
```

---

### ❌ Problema: "Gemini API error"

**Síntomas:**
```json
{"error": {"code": 401, "message": "API Key not valid"}}
```

**Solución:**
```bash
# 1. Verificar que la API Key es correcta en .env
grep GEMINI_API_KEY .env

# 2. Probar la API Key directamente
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=YOUR_KEY" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"test"}]}]}'

# 3. Si falla, obtener nueva API Key:
# - Ir a https://ai.google.dev
# - Crear proyecto en Google Cloud
# - Generar API Key
# - Actualizar .env
# - Reiniciar servicios: docker compose restart n8n
```

---

### ❌ Problema: Response vacía o "null"

**Síntomas:**
```json
null
```

**Solución:**
```bash
# 1. Verificar que todos los servicios funcionan
docker compose logs | grep -i error

# 2. Revisar el workflow en n8n Dashboard
# Dashboard → Workflows → Click en workflow

# 3. Hacer una ejecución manual de prueba
# Click "Test workflow"

# 4. Ver el último execution log
curl http://localhost:5678/api/v1/executions \
  -H "X-N8N-API-KEY: YOUR_API_KEY" | jq '.data[0]'
```

---

## 📚 Ejemplos Prácticos

### Caso 1: Análisis Simple

```bash
#!/bin/bash
# simple_analysis.sh

CONCEPTO=$1
URL="http://localhost:5678/webhook/yo-estructural"

echo "🔍 Analizando: $CONCEPTO"

curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "{\"concepto\":\"$CONCEPTO\"}" | jq '{
    concepto,
    certeza_combinada,
    estado_integracion,
    rutas: (.rutas_fenomenologicas | map({tipo, certeza}))
  }'

# Uso: ./simple_analysis.sh FENOMENOLOGIA
```

---

### Caso 2: Análisis Comparativo

```python
#!/usr/bin/env python3
# compare_concepts.py

import requests
import json

conceptos = ['FENOMENOLOGIA', 'DASEIN', 'EXISTENCIA']
url = "http://localhost:5678/webhook/yo-estructural"

resultados = []

for concepto in conceptos:
    resp = requests.post(url, json={"concepto": concepto})
    data = resp.json()
    
    resultados.append({
        'concepto': data['concepto'],
        'certeza': data['certeza_combinada'],
        'rutas': len(data['rutas_fenomenologicas']),
        'integracion': data['estado_integracion']
    })

print("\n📊 COMPARATIVA DE CONCEPTOS")
print("=" * 70)

for r in resultados:
    print(f"{r['concepto']:20} | Certeza: {r['certeza']:.0%} | Rutas: {r['rutas']}/5 | {r['integracion']}")
```

---

### Caso 3: Integración con Webhook Externo

```javascript
// send_to_slack.js
// Enviar resultados a Slack

const axios = require('axios');

async function analizarYEnviarSlack(concepto, slackHook) {
  const respYO = await axios.post(
    'http://localhost:5678/webhook/yo-estructural',
    {concepto}
  );
  
  const data = respYO.data;
  
  const mensaje = {
    "blocks": [
      {
        "type": "header",
        "text": {
          "type": "plain_text",
          "text": `🔬 Análisis: ${data.concepto}`
        }
      },
      {
        "type": "section",
        "fields": [
          {
            "type": "mrkdwn",
            "text": `*Certeza:*\n${(data.certeza_combinada * 100).toFixed(0)}%`
          },
          {
            "type": "mrkdwn",
            "text": `*Estado:*\n${data.estado_integracion}`
          }
        ]
      },
      {
        "type": "divider"
      }
    ]
  };
  
  await axios.post(slackHook, mensaje);
}

// Uso
analizarYEnviarSlack('FENOMENOLOGIA', process.env.SLACK_WEBHOOK);
```

---

## ✅ Checklist Final

- [x] n8n instalado y corriendo
- [x] Neo4j instalado y corriendo
- [x] Gemini API key configurada
- [x] Webhook accesible
- [x] Workflow activo
- [x] Primeras pruebas exitosas
- [x] Documentación completa

---

## 📞 Soporte

Para más información, consulta:
- `RESUMEN_INTEGRACION_FINAL.md` - Resumen completo
- `GUIA_RAPIDA_5MINUTOS.md` - Inicio rápido
- `URLS_ACCESO_PUBLICAS.md` - URLs de acceso

---

**¡Listo para empezar a usar YO Estructural v2.1!** 🚀
