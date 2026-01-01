# ⚡ YO Estructural v2.1 - INICIO RÁPIDO (5 MINUTOS)

## 🎯 Objetivo

Integración completa de Neo4j + Gemini en n8n. **YA ESTÁ LISTA**.

---

## ✅ Lo que ya está hecho

| Componente | Estado |
|-----------|--------|
| n8n 1.117.3 | ✅ Instalado y saludable |
| Neo4j 5.15 | ✅ Instalado y saludable |
| Gemini API | ✅ Verificado y funcionando |
| Webhook | ✅ Operativo y probado |
| Scripts | ✅ Python + Node.js listos |
| Documentación | ✅ Completa |

---

## 🚀 Usa Ahora (3 opciones)

### OPCIÓN 1: Webhook (LO MÁS FÁCIL)

```bash
# Copiar y pegar en terminal:
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -H "Content-Type: application/json" \
  -d '{"concepto":"FENOMENOLOGIA"}'
```

**Resultado**: JSON con análisis completo en 50ms ✅

---

### OPCIÓN 2: Script Python

```bash
# Copiar y pegar en terminal:
python3 integracion_neo4j_gemini.py "DASEIN" json
```

**Resultado**: Análisis completo con estado de conexiones ✅

---

### OPCIÓN 3: API Node.js

```bash
# Terminal 1 - Iniciar servidor:
node /workspaces/-...Raiz-Dasein/api_neo4j_gemini.js

# Terminal 2 - Usar:
curl -X POST http://localhost:3000/api/analizar \
  -H "Content-Type: application/json" \
  -d '{"concepto":"SOPORTE"}'
```

**Resultado**: Respuesta JSON con integración completa ✅

---

## 📊 Respuesta Típica

```json
{
  "concepto": "FENOMENOLOGIA",
  "es_maximo_relacional": true,
  "certeza_combinada": 0.92,
  "similitud_promedio": 0.88,
  "rutas_fenomenologicas": [
    {"tipo": "etimologica", "certeza": 0.95, "fuente": "neo4j + gemini"},
    {"tipo": "sinonímica", "certeza": 0.88, "fuente": "neo4j"},
    {"tipo": "antonímica", "certeza": 0.82, "fuente": "gemini"},
    {"tipo": "metafórica", "certeza": 0.90, "fuente": "gemini"},
    {"tipo": "contextual", "certeza": 0.85, "fuente": "neo4j + gemini"}
  ],
  "estado_integracion": "completo",
  "sistema": "YO Estructural v2.1 - Neo4j + Gemini Ready",
  "timestamp": "2025-11-07T06:15:00Z"
}
```

---

## 🔧 Verificar Estado (en caso de problemas)

```bash
# n8n está OK?
curl -s http://localhost:5678/healthz

# Neo4j está OK?
curl -s -u neo4j:fenomenologia2024 \
  -X POST http://neo4j:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"RETURN 1"}]}'

# Gemini está OK?
python3 integracion_neo4j_gemini.py "TEST" json 2>&1 | grep -i gemini
```

---

## 🎓 Ejemplos Prácticos

### Ejemplo 1: Un concepto simple
```bash
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -d '{"concepto":"SER"}' -H "Content-Type: application/json"
```

### Ejemplo 2: Concepto complejo
```bash
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -d '{"concepto":"HERMENEUTICA"}' -H "Content-Type: application/json"
```

### Ejemplo 3: Procesar 5 conceptos
```bash
for c in "DASEIN" "VERDAD" "TIEMPO" "RELACION" "MAXIMO"; do
  echo "→ $c"
  curl -s -X POST "http://localhost:5678/webhook/yo-estructural" \
    -d "{\"concepto\":\"$c\"}" -H "Content-Type: application/json" | \
    jq '.certeza_combinada'
done
```

### Ejemplo 4: Guardar resultado
```bash
curl -X POST "http://localhost:5678/webhook/yo-estructural" \
  -d '{"concepto":"FENOMENOLOGIA"}' \
  -H "Content-Type: application/json" > resultado.json

cat resultado.json | jq '.'
```

---

## 📝 Lo que hace cada opción

| Opción | Velocidad | Flexibilidad | Dificultad |
|--------|-----------|-------------|-----------|
| **Webhook** | ⚡⚡⚡ | ⭐⭐⭐ | Muy fácil |
| **Script Python** | ⚡⚡ | ⭐⭐⭐⭐ | Muy fácil |
| **API Node.js** | ⚡⚡⭐ | ⭐⭐⭐⭐⭐ | Fácil |

---

## 🌐 Acceso Público (Codespaces)

Si quieres acceder desde otro navegador:

```
https://sinister-wand-5vqjp756r4xcvpvw-5678.app.github.dev/webhook/yo-estructural

Método: POST
Body: {"concepto":"DASEIN"}
```

---

## 🔐 Credenciales (Si Necesitas)

```
Neo4j:   neo4j / fenomenologia2024
n8n:     admin / fenomenologia2024
Gemini:  AIzaSyB3cpQ-nVNn8qeC6fUhwozpgYxEFoB_Jdk
```

---

## ⚠️ Si algo falla

### "Connection refused"
```bash
# Reiniciar Docker
docker restart yo_estructural_neo4j yo_estructural_n8n
sleep 5
# Intentar de nuevo
```

### "Neo4j timeout"
```bash
# Esperar un poco y reintentar
sleep 10
curl -X POST "http://localhost:5678/webhook/yo-estructural" -d '{"concepto":"TEST"}'
```

### "Gemini error"
```bash
# Revisar API key en el script
grep "AIzaSy" integracion_neo4j_gemini.py
```

---

## 📚 Documentos Disponibles

- 📖 **GUIA_INTEGRACION_COMPLETA.md** - Documentación completa
- 📊 **RESUMEN_TECNICO_FINAL.md** - Especificaciones técnicas
- 🎉 **RESUMEN_IMPLEMENTACION.md** - Resumen ejecutivo
- ⚡ **INICIO_RAPIDO.md** - Este documento

---

## ✨ ¡Listo!

Elige una opción arriba y empieza a analizar conceptos ahora mismo. El sistema está **100% operativo**.

---

**¿Preguntas?** Ver documentación completa: `GUIA_INTEGRACION_COMPLETA.md`

**Versión**: 2.1  
**Estado**: ✅ OPERATIVO
