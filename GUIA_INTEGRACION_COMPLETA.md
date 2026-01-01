# 🚀 YO Estructural v2.1 - Integración Neo4j + Gemini en n8n

## 📋 Resumen del Sistema

Sistema completo de análisis fenomenológico que integra:
- **n8n 1.117.3** (versión estable): Orquestación de workflows
- **Neo4j 5.15**: Base de datos de conceptos relacionados
- **Gemini 2.0 Flash API**: Análisis de lenguaje natural
- **Python/Node.js**: Scripts de procesamiento

## ✅ Estado Actual

```
┌─────────────────────────────────────────────┐
│ 🔬 YO Estructural v2.1 - OPERATIVO         │
├─────────────────────────────────────────────┤
│ ✅ n8n 1.117.3        [HEALTHY]            │
│ ✅ Neo4j 5.15         [HEALTHY]            │
│ ✅ Gemini API         [VERIFIED]           │
│ ✅ Webhook Funcional  [TESTING]            │
└─────────────────────────────────────────────┘
```

## 🎯 Endpoints Disponibles

### 1. **Webhook n8n (Principal)**

```bash
# Solicitud
curl -X POST "https://sinister-wand-5vqjp756r4xcvpvw-5678.app.github.dev/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"DASEIN"}'

# Respuesta
{
  "concepto": "DASEIN",
  "es_maximo_relacional": true,
  "integracion_neo4j": {
    "encontrado": true,
    "nodos": ["concepto_relacionado_1", "concepto_relacionado_2"],
    "relaciones": ["sinonimia", "antonimia"]
  },
  "integracion_gemini": {
    "analisis_completado": true,
    "modelos_analizados": ["etimologico", "sinonimico", ...]
  },
  "certeza_combinada": 0.92,
  "similitud_promedio": 0.88,
  "rutas_fenomenologicas": [
    {"tipo": "etimologica", "certeza": 0.95, "fuente": "neo4j + gemini"},
    {"tipo": "sinonímica", "certeza": 0.88, "fuente": "neo4j"},
    ...
  ],
  "estado_integracion": "completo",
  "sistema": "YO Estructural v2.1 - Neo4j + Gemini Ready",
  "timestamp": "2025-11-07T06:02:42.459Z"
}
```

### 2. **Script Python (CLI)**

```bash
# Con output JSON
python3 integracion_neo4j_gemini.py "FENOMENOLOGIA" json

# Con output formateado
python3 integracion_neo4j_gemini.py "DASEIN"
```

### 3. **API Express (Futuro)**

```bash
# Disponible cuando se ejecute: node api_neo4j_gemini.js

# Health check
curl http://localhost:3000/health

# Analizar concepto
curl -X POST http://localhost:3000/api/analizar \
  -H "Content-Type: application/json" \
  -d '{"concepto":"SOPORTE"}'
```

## 🔧 Arquitectura del Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Webhook Input (POST /webhook/yo-estructural)               │
│  Body: {"concepto": "DASEIN"}                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Nodo: Preparar Entrada (Code Node v2.1)                   │
│  • Extrae concepto del body                                 │
│  • Valida formato de entrada                               │
│  • Genera timestamp                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────────────┐  ┌──────────────────────┐
│ Query Neo4j          │  │ Gemini Análisis      │
│ • Busca concepto     │  │ • Análisis 5 rutas   │
│ • Obtiene relaciones │  │ • Extrae JSON        │
│ • Extrae definición  │  │ • Calcula certeza    │
└──────────────────────┘  └──────────────────────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Nodo: Combinar Resultados (Code Node v2.1)                │
│  • Merge Neo4j + Gemini                                    │
│  • Calcula certeza combinada                              │
│  • Estructura rutas fenomenológicas                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Webhook Response (JSON completo)                          │
│  ✅ 200 OK con análisis fenomenológico                      │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Estructura de Respuesta

```json
{
  "concepto": "string",
  "timestamp": "ISO-8601",
  "estado_conexiones": {
    "neo4j": boolean,
    "gemini": boolean
  },
  "es_maximo_relacional": boolean,
  "integracion_neo4j": {
    "encontrado": boolean,
    "concepto": "string",
    "relacionados": array,
    "definicion": "string | null",
    "etimologia": "string | null"
  },
  "integracion_gemini": {
    "analisis_completado": boolean,
    "rutas": {
      "ruta_etimologica": {
        "analisis": "string",
        "certeza": number
      },
      ...
    }
  },
  "certeza_combinada": number,
  "similitud_promedio": number,
  "estado_integracion": "completo|parcial|degradado",
  "rutas_fenomenologicas": [
    {
      "tipo": "string",
      "certeza": number,
      "fuente": "string"
    }
  ],
  "sistema": "YO Estructural v2.1 - Neo4j + Gemini Integrado"
}
```

## 🚀 Cómo Usar

### Opción 1: Webhook n8n (Recomendado)

```bash
# URL pública en Codespaces
WEBHOOK_URL="https://sinister-wand-5vqjp756r4xcvpvw-5678.app.github.dev/webhook/yo-estructural"

# Solicitud
curl -X POST "$WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "concepto": "FENOMENOLOGIA",
    "usuario": "usuario@example.com",
    "timestamp_cliente": "2025-11-07T06:00:00Z"
  }'
```

### Opción 2: Script Python (Local)

```bash
# Instalar dependencias (una sola vez)
pip install requests

# Ejecutar análisis
python3 integracion_neo4j_gemini.py "DASEIN" json

# Output JSON para procesar
python3 integracion_neo4j_gemini.py "SOPORTE" json | jq '.rutas_fenomenologicas'
```

### Opción 3: API Express (Cuando esté disponible)

```bash
# Iniciar servidor (en terminal separado)
node api_neo4j_gemini.js

# Usar desde otra aplicación
curl -X POST http://localhost:3000/api/analizar \
  -H "Content-Type: application/json" \
  -d '{"concepto":"VERDAD"}'
```

## 🔐 Credenciales

| Servicio | Usuario | Password | URL |
|----------|---------|----------|-----|
| Neo4j | `neo4j` | `fenomenologia2024` | `http://neo4j:7474` |
| n8n | `admin` | `fenomenologia2024` | `http://localhost:5678` |
| Gemini API | `API Key` | `AIzaSyB3cpQ-nVNn8qeC6fUhwozpgYxEFoB_Jdk` | Pública |

## 📈 Validación de Estado

```bash
# Verificar n8n
curl -s http://localhost:5678/healthz | jq '.'

# Verificar Neo4j
curl -s http://neo4j:7474/db/neo4j/tx/commit \
  -u neo4j:fenomenologia2024 \
  -d '{"statements":[{"statement":"RETURN 1"}]}' | jq '.'

# Verificar Gemini (desde el script Python)
python3 integracion_neo4j_gemini.py "TEST" json | jq '.estado_conexiones'
```

## 🔍 Pruebas Sugeridas

```bash
# 1. Concepto simple
curl -X POST "$WEBHOOK_URL" -H "Content-Type: application/json" \
  -d '{"concepto":"SER"}'

# 2. Concepto complejo
curl -X POST "$WEBHOOK_URL" -H "Content-Type: application/json" \
  -d '{"concepto":"FENOMENOLOGIA"}'

# 3. Concepto de dominio
curl -X POST "$WEBHOOK_URL" -H "Content-Type: application/json" \
  -d '{"concepto":"HEIDEGGER"}'

# 4. Batch de conceptos (desde script)
for concepto in "DASEIN" "VERDA" "SOPORTE" "RELACION"; do
  echo "Analizando: $concepto"
  curl -s -X POST "$WEBHOOK_URL" \
    -H "Content-Type: application/json" \
    -d "{\"concepto\":\"$concepto\"}" | jq '.certeza_combinada'
done
```

## 🛠️ Resolución de Problemas

### Neo4j no se conecta

```bash
# Verificar contenedor
docker ps | grep neo4j

# Ver logs
docker logs yo_estructural_neo4j

# Reconectar manualmente
curl -X POST http://neo4j:7474/db/neo4j/tx/commit \
  -u neo4j:fenomenologia2024 \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"RETURN 1"}]}'
```

### Gemini API no responde

```bash
# Verificar API key
echo $GEMINI_API_KEY

# Probar directamente
curl -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"test"}]}]}'
```

### n8n webhook no responde

```bash
# Verificar workflow activo
curl -s http://localhost:5678/api/v1/workflows \
  -H "X-N8N-API-KEY: n8n_api_fcd1ede386b72b3cb67f2f7e46d0882f2a000eeeb48214741ec32910330024a57e60d6fc97bb3c7a" | jq '.data[] | select(.active == true)'

# Ver workflow específico
curl -s http://localhost:5678/api/v1/workflows/kJTzAF4VdZ6NNCfK \
  -H "X-N8N-API-KEY: n8n_api_fcd1ede386b72b3cb67f2f7e46d0882f2a000eeeb48214741ec32910330024a57e60d6fc97bb3c7a" | jq '.active'
```

## 📚 Estructura de Archivos

```
/workspaces/-...Raiz-Dasein/
├── integracion_neo4j_gemini.py      # Script principal Python
├── api_neo4j_gemini.js               # API Express (futuro)
├── docker-compose.yml                # Configuración servicios
├── YO estructural/
│   ├── Dockerfile                    # n8n custom
│   ├── main.py                       # Scripts adicionales
│   └── ...
└── GUIA_INTEGRACION_COMPLETA.md      # Esta documentación
```

## 🔄 Próximos Pasos

### Fase 1: Optimización Actual ✅
- [x] Integración Neo4j + Gemini en workflow
- [x] Verificación de conectividad
- [x] Script Python operativo
- [x] Webhook respondiendo correctamente

### Fase 2: Expansión (En Progreso)
- [ ] Agregar caching de resultados
- [ ] Persistencia de análisis en Neo4j
- [ ] Webhook de múltiples conceptos
- [ ] Rate limiting

### Fase 3: Producción (Futuro)
- [ ] Despliegue en servidor real
- [ ] Base de datos centralizada
- [ ] Métricas y logging
- [ ] API pública

## 📞 Soporte

Para consultas, verificar:
1. Estado de conexiones: `python3 integracion_neo4j_gemini.py TEST json`
2. Logs de n8n: `docker logs yo_estructural_n8n -f`
3. Logs de Neo4j: `docker logs yo_estructural_neo4j -f`
4. Estado del webhook: Verificar workflow activo en n8n UI

---

**Última actualización**: 2025-11-07  
**Versión**: 2.1  
**Estado**: ✅ OPERATIVO
