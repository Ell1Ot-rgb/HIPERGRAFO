# 🌐 URLS DE ACCESO PÚBLICO - YO Estructural v2.1

**Última Actualización:** 7 de Noviembre de 2025  
**Versión del Sistema:** 2.1  
**Ambiente:** GitHub Codespaces (Ubuntu 24.04 LTS)

---

## 📡 ENDPOINTS PÚBLICOS

### 🔴 WEBHOOK PRINCIPAL (Análisis de Conceptos)

**Nombre:** YO Estructural Webhook v2.1

**URL Pública:**
```
POST https://sinister-wand-5vqjp756r4xcvpvw-5678.app.github.dev/webhook/yo-estructural
```

**URL Local (desde dentro del Codespace):**
```
POST http://localhost:5678/webhook/yo-estructural
```

**Headers Requeridos:**
```
Content-Type: application/json
```

**Body (Ejemplo):**
```json
{
  "concepto": "FENOMENOLOGIA"
}
```

**Respuesta (200 OK):**
```json
{
  "concepto": "FENOMENOLOGIA",
  "es_maximo_relacional": true,
  "certeza_combinada": 0.92,
  "estado_integracion": "completo",
  "rutas_fenomenologicas": [
    {"tipo": "etimologica", "certeza": 0.95},
    {"tipo": "sinonímica", "certeza": 0.88},
    {"tipo": "antonímica", "certeza": 0.82},
    {"tipo": "metafórica", "certeza": 0.90},
    {"tipo": "contextual", "certeza": 0.85}
  ],
  "timestamp": "2025-11-07T06:15:00.000Z",
  "sistema": "YO Estructural v2.1 - Neo4j + Gemini Ready"
}
```

---

## 🖥️ INTERFACES DE ADMINISTRACIÓN

### n8n Dashboard

**URL Pública:**
```
https://sinister-wand-5vqjp756r4xcvpvw-5678.app.github.dev/
```

**URL Local:**
```
http://localhost:5678/
```

**Credenciales:**
- Usuario: `admin`
- Contraseña: `fenomenologia2024`

**Acceso a Workflow:**
```
Dashboard → Workflows → 🚀 YO Estructural - Demostración Funcional
```

**Workflow ID:** `kJTzAF4VdZ6NNCfK`

---

### Neo4j Browser

**URL Pública:** ❌ NO disponible públicamente (por seguridad)

**URL Local (desde Codespace):**
```
http://neo4j:7474/browser/
```

**Credenciales:**
- Usuario: `neo4j`
- Contraseña: `fenomenologia2024`

**Acceso directo a HTTP API:**
```
http://neo4j:7474/db/neo4j/tx/commit
```

---

## 🔌 APIs DE INTEGRACIÓN

### HTTP Request Nodes (Usadas por n8n internamente)

#### Neo4j Database API
```
Protocolo: HTTP
Host: neo4j
Puerto: 7474
Ruta: /db/neo4j/tx/commit
Método: POST
Auth: Basic (neo4j / fenomenologia2024)
```

**Body Ejemplo:**
```json
{
  "statements": [
    {
      "statement": "MATCH (n:Concepto {nombre: $concepto}) RETURN n LIMIT 1",
      "parameters": {
        "concepto": "FENOMENOLOGIA"
      }
    }
  ]
}
```

---

#### Gemini API
```
Protocolo: HTTPS
Host: generativelanguage.googleapis.com
Puerto: 443
Ruta: /v1beta/models/gemini-2.0-flash:generateContent
Método: POST
Auth: API Key (Query Parameter)
```

**URL Completa:**
```
https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyB3cpQ-nVNn8qeC6fUhwozpgYxEFoB_Jdk
```

**Body Ejemplo:**
```json
{
  "contents": [
    {
      "parts": [
        {
          "text": "Analiza fenomenológicamente el concepto FENOMENOLOGIA"
        }
      ]
    }
  ]
}
```

---

## 🧪 PRUEBAS RÁPIDAS

### Test 1: Con cURL desde Terminal

```bash
# Concepto 1
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"FENOMENOLOGIA"}'

# Concepto 2
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"DASEIN"}'

# Sin parámetro (default = SOPORTE)
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

### Test 2: Desde Python

```python
import requests
import json

url = "http://localhost:5678/webhook/yo-estructural"

# Test 1
resp = requests.post(url, json={"concepto": "FENOMENOLOGIA"})
print(json.dumps(resp.json(), indent=2))

# Test 2
resp = requests.post(url, json={"concepto": "DASEIN"})
print(json.dumps(resp.json(), indent=2))
```

---

### Test 3: Desde JavaScript/Node.js

```javascript
// test_webhook.js
const axios = require('axios');

const testWebhook = async (concepto) => {
  const url = 'http://localhost:5678/webhook/yo-estructural';
  const resp = await axios.post(url, { concepto });
  console.log(JSON.stringify(resp.data, null, 2));
};

testWebhook('FENOMENOLOGIA');
testWebhook('DASEIN');
```

---

### Test 4: Desde Navegador (DevTools Console)

```javascript
const analizar = (concepto) => {
  fetch('http://localhost:5678/webhook/yo-estructural', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({concepto})
  })
  .then(r => r.json())
  .then(d => console.log(JSON.stringify(d, null, 2)));
};

// Ejecutar pruebas
analizar('FENOMENOLOGIA');
analizar('DASEIN');
analizar('MAXIMOS_RELACIONALES');
```

---

## 🔐 SEGURIDAD Y ACCESO

### Restricciones de Acceso

| Componente | Público | Autenticación |
|-----------|---------|---------------|
| Webhook Análisis | ✅ Sí | ❌ No requerida |
| n8n Dashboard | ✅ Sí | ✅ Usuario/Contraseña |
| Neo4j Browser | ❌ No | ✅ Usuario/Contraseña |
| Neo4j HTTP API | ❌ No | ✅ Basic Auth |
| Gemini API | ✅ Sí | ✅ API Key |

### CORS / Access-Control

```
Webhook:
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

---

## 🔄 ENDPOINT DE MANTENIMIENTO

### Health Check (Estado del Sistema)

**URL Local:**
```
GET http://localhost:5678/healthz
```

**Respuesta:**
```json
{
  "status": "ok"
}
```

---

## 📊 ESTADÍSTICAS DE USO

### Monitoreo de Workflows

**URL Local (requiere API Key):**
```
GET http://localhost:5678/api/v1/workflows
Headers: X-N8N-API-KEY: [API_KEY]
```

**Workflow Actual:**
```
ID: kJTzAF4VdZ6NNCfK
Nombre: 🚀 YO Estructural - Demostración Funcional
Estado: ACTIVO ✅
Nodos: 6 (Webhook, Code, HTTP, HTTP, Code, Response)
Conexiones: 5
Creado: 2025-11-07T03:20:42.021Z
Actualizado: 2025-11-07T06:01:07.126Z
```

---

## 🎯 CASOS DE USO RECOMENDADOS

### 1. Integración con Aplicación Web
```
Tu Frontend → POST /webhook/yo-estructural → Recibe JSON
```

### 2. Integración con Pipeline de Datos
```
Sistema ETL → POST /webhook/yo-estructural → Enriquece datos
```

### 3. Análisis Batch
```
Script Python → Itera conceptos → POST a webhook → Compila resultados
```

### 4. Chatbot / Asistente Virtual
```
Chatbot → Detecta concepto → POST /webhook → Retorna análisis
```

---

## 🆘 SOPORTE

### Verificar Conectividad

```bash
# ¿El webhook está accesible?
curl -I -X POST "http://localhost:5678/webhook/yo-estructural"

# ¿Gemini funciona?
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=YOUR_KEY" \
  -X POST -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"test"}]}]}'

# ¿Neo4j funciona? (desde Codespace)
docker exec -it yo_estructural_neo4j \
  curl -u neo4j:fenomenologia2024 http://localhost:7474/db/neo4j/tx/commit \
  -X POST -d '{"statements":[{"statement":"RETURN 1"}]}'
```

---

## 📝 LIMITES Y CONSIDERACIONES

### Rate Limiting
- Sin limite por ahora (usar responsablemente)
- Recomendado: máximo 100 requests/minuto

### Timeouts
- Webhook: 30 segundos
- Neo4j: 10 segundos
- Gemini: 30 segundos

### Tamaño de Payload
- Request: máximo 1MB
- Response: típicamente 5-10KB

---

## 🔗 REFERENCIAS RÁPIDAS

### Documentación
- Resumen Final: `RESUMEN_INTEGRACION_FINAL.md`
- Guía Rápida: `GUIA_RAPIDA_5MINUTOS.md`
- Guía Completa: `GUIA_USO_n8n_V2.1.md`

### Archivos de Código
- Script Python: `integracion_neo4j_gemini.py`
- API Express: `api_neo4j_gemini.js`
- Test Script: `test_webhook.sh`

---

## ✨ CONCLUSIÓN

**YO Estructural v2.1** es completamente accesible desde internet a través de su webhook público. Puedes integrar esta API en tus proyectos sin necesidad de configuración adicional.

**Endpoint Principal:**
```
POST https://sinister-wand-5vqjp756r4xcvpvw-5678.app.github.dev/webhook/yo-estructural
```

**¡Empieza a analizar conceptos ahora mismo!** 🚀

---

**Última actualización:** 7 de Noviembre de 2025  
**Sistema:** YO Estructural v2.1  
**Estado:** ✅ Operativo
