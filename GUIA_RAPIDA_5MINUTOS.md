# ⚡ QUICK START - YO Estructural v2.1 (5 Minutos)

## 🎯 Lo que necesitas saber

**YO Estructural** es un sistema de análisis fenomenológico que:
- Consulta conceptos en **Neo4j** (base de datos de grafos)
- Analiza con **Gemini 2.0 Flash** (IA)
- Orquesta todo con **n8n** (sin-código)

**Resultado:** Análisis profundo de cualquier concepto en JSON

---

## 🚀 INICIO RÁPIDO (30 segundos)

### Opción 1: cURL (Más simple)

```bash
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"FENOMENOLOGIA"}'
```

**Respuesta (ejemplo):**
```json
{
  "concepto": "FENOMENOLOGIA",
  "certeza_combinada": 0.92,
  "estado_integracion": "completo",
  "rutas_fenomenologicas": [
    {"tipo": "etimologica", "certeza": 0.95},
    {"tipo": "sinonímica", "certeza": 0.88},
    ...
  ]
}
```

### Opción 2: Desde Navegador

Abre esta URL en tu navegador y usa la consola DevTools:

```javascript
fetch('http://localhost:5678/webhook/yo-estructural', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({concepto: 'DASEIN'})
})
.then(r => r.json())
.then(d => console.log(d))
```

---

## 📊 CONCEPTOS QUE PUEDES ANALIZAR

```bash
# Prueba estos:
FENOMENOLOGIA
DASEIN
MAXIMOS_RELACIONALES
SOPORTE
EXISTENCIA
ESENCIA
RELACION
```

---

## 🔍 INTERPRETAR LA RESPUESTA

```json
{
  "concepto": "FENOMENOLOGIA",                    // Concepto analizado
  
  "es_maximo_relacional": true,                   // ¿Se encontró en Neo4j?
  
  "certeza_combinada": 0.92,                      // Nivel de confianza (0-1)
  
  "similitud_promedio": 0.88,                     // Promedio de similitudes
  
  "estado_integracion": "completo",               // completo | parcial | degradado
  
  "rutas_fenomenologicas": [                      // 5 análisis diferentes
    {
      "tipo": "etimologica",                      // Origen del término
      "certeza": 0.95,                            // Confianza en este análisis
      "fuente": "neo4j + gemini"                  // Dónde vino el dato
    },
    // ... 4 rutas más
  ],
  
  "integracion_neo4j": {                          // Datos de la BD
    "encontrado": true,
    "relacionados": ["concepto1", "concepto2"]
  },
  
  "integracion_gemini": {                         // Datos de IA
    "analisis_completado": true,
    "modelos_analizados": [...]
  },
  
  "timestamp": "2025-11-07T...",                  // Cuándo se hizo
  
  "sistema": "YO Estructural v2.1 - Neo4j + Gemini Ready"
}
```

---

## 🎮 CASOS DE USO

### 1️⃣ Búsqueda Simple de un Concepto
```bash
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"FILOSOFIA"}'
```

### 2️⃣ Análisis por Defecto (si no especificas concepto)
```bash
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{}'  # Usa "SOPORTE" por defecto
```

### 3️⃣ Desde Python
```python
import requests

resp = requests.post(
    "http://localhost:5678/webhook/yo-estructural",
    json={"concepto": "ONTOLOGIA"}
)

resultado = resp.json()
print(f"Certeza: {resultado['certeza_combinada']:.0%}")
print(f"Rutas: {len(resultado['rutas_fenomenologicas'])}/5")
```

### 4️⃣ Desde JavaScript
```javascript
const analizar = async (concepto) => {
  const resp = await fetch(
    'http://localhost:5678/webhook/yo-estructural',
    {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({concepto})
    }
  );
  return resp.json();
};

analizar('FENOMENOLOGIA').then(r => {
  console.log(`Estado: ${r.estado_integracion}`);
  console.log(`Certeza: ${r.certeza_combinada}`);
});
```

---

## 🔧 ARQUITECTURA EN 60 SEGUNDOS

```
         Tu Cliente
              ↓
    [POST con concepto]
              ↓
        n8n Webhook
              ↓
         [Workflow]
         /         \
     Neo4j       Gemini
    (BD local)  (API Cloud)
         \         /
          Combina
              ↓
        JSON Respuesta
              ↓
         Tu Cliente
```

---

## 🆘 TROUBLESHOOTING

### ❌ "Cannot POST /webhook/yo-estructural"
- n8n no está corriendo
- Solución: `docker compose up -d`

### ❌ "Connection refused"
- URL incorrecta o n8n no accesible
- Solución: Verifica que `http://localhost:5678` funcione

### ❌ "No such host: neo4j"
- Neo4j no está en la red Docker correcta
- Solución: Verifica `docker network ls` y `docker compose config`

### ❌ Respuesta vacía o "null"
- Neo4j o Gemini no conectan
- Solución: Ejecuta el health check (ver abajo)

---

## 🏥 HEALTH CHECK

### Verificar que todo funciona:

```bash
# 1. ¿n8n está vivo?
curl -s http://localhost:5678/healthz

# 2. ¿Neo4j está vivo? (requiere acceso a Docker)
curl -s -u neo4j:fenomenologia2024 http://neo4j:7474/db/neo4j/tx/commit \
  -X POST -d '{"statements":[{"statement":"RETURN 1"}]}'

# 3. ¿Workflows existen?
curl -s http://localhost:5678/api/v1/workflows \
  -H "X-N8N-API-KEY: [TU_API_KEY]" | jq '.data | length'
```

---

## 📚 RUTAS FENOMENOLÓGICAS EXPLICADAS

La respuesta siempre incluye **5 rutas de análisis**:

| Ruta | Qué es | Ejemplo |
|------|--------|---------|
| **Etimológica** | Origen del término | "Fenomenología = feno (aparecer) + logía (estudio)" |
| **Sinonímica** | Palabras similares | "Fenomenología ≈ Filosofía de la experiencia" |
| **Antonímica** | Opuestos | "Fenomenología ≠ Objetividad pura" |
| **Metafórica** | Comparaciones | "Fenomenología es como observar el amanecer" |
| **Contextual** | Usos reales | "En filosofía, en psicología, en ciencia..." |

---

## 🎓 EJEMPLO COMPLETO

```bash
# Comando
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"LIBERTAD"}'

# Respuesta simplificada
{
  "concepto": "LIBERTAD",
  "certeza_combinada": 0.92,
  "estado_integracion": "completo",
  "rutas_fenomenologicas": [
    {"tipo": "etimologica", "certeza": 0.95, "fuente": "neo4j + gemini"},
    {"tipo": "sinonímica", "certeza": 0.88, "fuente": "neo4j"},
    {"tipo": "antonímica", "certeza": 0.82, "fuente": "gemini"},
    {"tipo": "metafórica", "certeza": 0.90, "fuente": "gemini"},
    {"tipo": "contextual", "certeza": 0.85, "fuente": "neo4j + gemini"}
  ],
  "timestamp": "2025-11-07T06:15:00.000Z",
  "sistema": "YO Estructural v2.1 - Neo4j + Gemini Ready"
}
```

---

## 🚀 PRÓXIMO NIVEL

Quieres más? Consulta:
- `RESUMEN_INTEGRACION_FINAL.md` - Documentación técnica completa
- `GUIA_USO_n8n_V2.1.md` - Guía de uso avanzada
- `URLS_ACCESO_PUBLICAS.md` - URLs públicas accesibles

---

## ✨ TL;DR (Muy corto)

```bash
# Esto es todo lo que necesitas:
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"FENOMENOLOGIA"}'

# Obtienes JSON con análisis, certeza y 5 rutas fenomenológicas
```

---

**¡Listo! Ahora ya puedes analizar conceptos con YO Estructural v2.1** 🎉
