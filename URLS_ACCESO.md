# 🌐 YO Estructural v2.1 - URLs de Acceso Público

## 🔗 Acceso Remoto (GitHub Codespaces)

### n8n Dashboard
```
URL: https://sinister-wand-5vqjp756r4xcvpvw-5678.app.github.dev
Usuario: admin
Contraseña: fenomenologia2024
```

### Webhook Principal
```
URL: https://sinister-wand-5vqjp756r4xcvpvw-5678.app.github.dev/webhook/yo-estructural
Método: POST
Content-Type: application/json
Body: {"concepto":"CONCEPTO_A_ANALIZAR"}
```

### Ejemplo cURL Público
```bash
curl -X POST "https://sinister-wand-5vqjp756r4xcvpvw-5678.app.github.dev/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"FENOMENOLOGIA"}'
```

---

## 💻 Acceso Local (Dev Container)

### Webhook Local
```
URL: http://localhost:5678/webhook/yo-estructural
Método: POST
Content-Type: application/json
```

### n8n Local
```
URL: http://localhost:5678
Usuario: admin
Contraseña: fenomenologia2024
```

### Neo4j Local
```
URL: http://neo4j:7474
Usuario: neo4j
Contraseña: fenomenologia2024
```

---

## 📊 Ejemplo de Uso Completo (Local)

```bash
# Concepto simple
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"SER"}'

# Concepto filosófico
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"DASEIN"}'

# Guardar resultado
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"VERDAD"}' | jq '.' > resultado_verdad.json

# Ver solo certeza
curl -s -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"RELACION"}' | jq '.certeza_combinada'
```

---

## 🔐 APIs y Credenciales

### Gemini API Key
```
AIzaSyB3cpQ-nVNn8qeC6fUhwozpgYxEFoB_Jdk
(Configurado en integracion_neo4j_gemini.py)
```

### Neo4j HTTP API
```
Endpoint: http://neo4j:7474/db/neo4j/tx/commit
Usuario: neo4j
Contraseña: fenomenologia2024
Autenticación: Basic Auth
```

### n8n API
```
Endpoint: http://localhost:5678/api/v1
API Key: n8n_api_fcd1ede386b72b3cb67f2f7e46d0882f2a000eeeb48214741ec32910330024a57e60d6fc97bb3c7a
```

---

## ⚡ Test Rápido (Copiar y Pegar)

### Test 1: Webhook Responsivo
```bash
curl -s "http://localhost:5678/webhook/yo-estructural" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"concepto":"TEST"}' | jq '.sistema'
```

**Resultado esperado:**
```json
"YO Estructural v2.1 - Neo4j + Gemini Ready"
```

### Test 2: Neo4j Conectado
```bash
curl -s -u neo4j:fenomenologia2024 \
  -X POST http://neo4j:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"RETURN 1"}]}' | jq '.results[0]'
```

**Resultado esperado:**
```json
{"data": [{"row": [1], "meta": [null]}]}
```

### Test 3: Gemini Verificado
```bash
python3 integracion_neo4j_gemini.py "TEST" json 2>&1 | jq '.estado_conexiones.gemini'
```

**Resultado esperado:**
```json
true
```

---

## 📈 Estadísticas en Tiempo Real

```bash
# Webhook - Última hora
curl -s "http://localhost:5678/api/v1/workflows/kJTzAF4VdZ6NNCfK" \
  -H "X-N8N-API-KEY: n8n_api_fcd1ede386b72b3cb67f2f7e46d0882f2a000eeeb48214741ec32910330024a57e60d6fc97bb3c7a" | \
  jq '.active'

# Estado Neo4j
curl -s -u neo4j:fenomenologia2024 \
  http://neo4j:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"MATCH (n) RETURN COUNT(n)"}]}' | \
  jq '.results[0].data[0].row[0]'
```

---

## 🛠️ Debugging

### Ver logs de n8n
```bash
docker logs yo_estructural_n8n -f --tail 50
```

### Ver logs de Neo4j
```bash
docker logs yo_estructural_neo4j -f --tail 50
```

### Ver workflow activos
```bash
curl -s http://localhost:5678/api/v1/workflows \
  -H "X-N8N-API-KEY: n8n_api_fcd1ede386b72b3cb67f2f7e46d0882f2a000eeeb48214741ec32910330024a57e60d6fc97bb3c7a" | \
  jq '.data[] | select(.active == true) | {id, name}'
```

### Healthcheck completo
```bash
echo "=== n8n ===" && \
curl -s http://localhost:5678/healthz && \
echo -e "\n=== Neo4j ===" && \
curl -s -u neo4j:fenomenologia2024 -X POST \
  http://neo4j:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"RETURN 1"}]}' | jq '.results[0].data' && \
echo -e "\n=== Gemini ===" && \
python3 integracion_neo4j_gemini.py "HEALTH" json 2>&1 | jq '.estado_conexiones'
```

---

## 📱 Integración en Otras Aplicaciones

### cURL Básico
```bash
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"CONCEPTO"}'
```

### Python Requests
```python
import requests

response = requests.post(
    "http://localhost:5678/webhook/yo-estructural",
    json={"concepto": "FENOMENOLOGIA"}
)
print(response.json())
```

### JavaScript Fetch
```javascript
fetch("http://localhost:5678/webhook/yo-estructural", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({concepto: "DASEIN"})
})
.then(r => r.json())
.then(data => console.log(data))
```

### PowerShell
```powershell
$body = @{concepto = "SOPORTE"} | ConvertTo-Json
$response = Invoke-WebRequest -Uri "http://localhost:5678/webhook/yo-estructural" `
  -Method POST -ContentType "application/json" -Body $body
$response.Content | ConvertFrom-Json
```

---

## 🔄 Automación (Bash Script)

```bash
#!/bin/bash
# Script para procesar múltiples conceptos

WEBHOOK_URL="http://localhost:5678/webhook/yo-estructural"
CONCEPTOS=("DASEIN" "FENOMENOLOGIA" "VERDAD" "RELACION" "MAXIMO")

for concepto in "${CONCEPTOS[@]}"; do
  echo "Analizando: $concepto"
  curl -s -X POST "$WEBHOOK_URL" \
    -H "Content-Type: application/json" \
    -d "{\"concepto\":\"$concepto\"}" | \
    jq '{concepto: .concepto, certeza: .certeza_combinada, estado: .estado_integracion}'
  echo "---"
done
```

---

## 📡 Monitoreo Continuo

```bash
# Healthcheck cada 30 segundos
watch -n 30 'curl -s http://localhost:5678/healthz | jq "."'

# Monitoreo de respuestas
while true; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') - Webhook:"
  curl -s -X POST "http://localhost:5678/webhook/yo-estructural" \
    -H "Content-Type: application/json" \
    -d '{"concepto":"MONITOR"}' | jq '.timestamp, .certeza_combinada'
  sleep 60
done
```

---

## 📞 Información de Contacto Técnico

- **Versión**: 2.1
- **Última Actualización**: 2025-11-07T06:15:00Z
- **Estado**: ✅ OPERATIVO
- **Soporte**: Ver GUIA_INTEGRACION_COMPLETA.md

---

**¡Sistema listo para producción!** 🚀
