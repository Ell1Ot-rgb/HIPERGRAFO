# 🔄 GUÍA CICLO PROMPT MÁXIMO RELACIONAL

## 📌 Descripción General

El **Ciclo Prompt Máximo Relacional** es un sistema AISLADO del YO Estructural v2.1 que descubre dinámicamente NUEVAS rutas fenomenológicas para cualquier concepto.

**Objetivo Principal**: Encontrar las "Máximas Rutas Fenomenológicas Posibles"

**Independencia**: ✅ Completamente independiente del workflow n8n actual

---

## 🎯 Factor Clave: "Máximas Rutas Fenomenológicas"

```
Factor = Total de rutas fenomenológicas disponibles para un concepto

Para DESTRUCCION:
- Rutas Canónicas: 10 (etimológica, sinonímica, antonímica, metafórica, contextual, 
                       histórica, fenomenológica, dialéctica, semiótica, axiológica)
- Rutas Descubiertas (Ciclo): X rutas nuevas
- TOTAL = 10 + X = "Máximas Rutas Fenomenológicas Posibles"

El objetivo del ciclo es MAXIMIZAR X
```

---

## 🚀 CÓMO EJECUTAR

### Opción 1: Ejecución Directa (Standalone)

```bash
cd /workspaces/-...Raiz-Dasein

# Ejecución completa (3 iteraciones de descubrimiento)
python3 ciclo_prompt_maximo_relacional.py

# Salida: 
# - RESULTADO_CICLO_MAXIMO_RELACIONAL.json
# - REPORTE_CICLO_MAXIMO_RELACIONAL.md
```

### Opción 2: Ejecución desde Python

```python
from ciclo_prompt_maximo_relacional import ejecutar_ciclo_completo

concepto = "DESTRUCCION"
gemini_key = "AIzaSyB3cpQ-nVNn8qeC6fUhwozpgYxEFoB_Jdk"
iteraciones = 3

resultado, reporte = ejecutar_ciclo_completo(concepto, gemini_key, iteraciones)

# resultado = dict con todas las rutas
# reporte = markdown formateado
```

### Opción 3: Integración en n8n (Aislada)

```javascript
// En un Code node de n8n:

const payload = {
  concepto: "DESTRUCCION",
  gemini_key: "AIzaSyB3cpQ-nVNn8qeC6fUhwozpgYxEFoB_Jdk",
  iteraciones: 2
};

// Llamar Python script que retorna JSON
const resultado = await ejecutarCicloDesdeNode(payload);
```

---

## 🔄 FLUJO DEL CICLO

```
INICIO
  ↓
[Iteración 1] Descubrir nuevas rutas no canónicas
  ↓
  └─ Prompt a Gemini 2.0: "¿Nuevas dimensiones de análisis?"
  ↓
  └─ Extraer rutas nuevas
  ↓
[Iteración 2] Profundizar en rutas descubiertas
  ↓
  └─ Prompt a Gemini: "Análisis profundo de cada ruta nueva"
  ↓
  └─ Calcular certeza para cada ruta
  ↓
[Iteración 3] Validar y expandir (opcional)
  ↓
  └─ Consolidar todas las rutas
  ↓
SALIDA: {
  rutas_canonicas: 10,
  rutas_nuevas: X,
  total: 10+X,
  factor_maximo: "Máximas Rutas Fenomenológicas Posibles"
}
  ↓
FIN
```

---

## 📊 ESTRUCTURA DE SALIDA

```json
{
  "ciclo_info": {
    "timestamp": "2025-11-07T12:34:56.789Z",
    "concepto": "DESTRUCCION",
    "iteraciones_ejecutadas": 3,
    "estado": "✅ COMPLETADO"
  },
  "estadisticas": {
    "rutas_canonicas": 10,
    "rutas_nuevas_descubiertas": 5,
    "total_rutas": 15,
    "certeza_promedio_nuevas": 0.87
  },
  "rutas_canonicas": [
    "etimologica",
    "sinonímica",
    "antonímica",
    "metafórica",
    "contextual",
    "histórica",
    "fenomenológica",
    "dialéctica",
    "semiótica",
    "axiológica"
  ],
  "rutas_nuevas": {
    "ruta_nueva_1": {
      "iteracion_descubrimiento": 1,
      "descripcion": "...",
      "analisis": {...},
      "certeza": 0.92
    },
    "ruta_nueva_2": {...}
  },
  "factor_maximo": {
    "nombre": "Máximas Rutas Fenomenológicas Posibles",
    "valor": 15,
    "descriptor": "El concepto 'DESTRUCCION' alcanza 15 dimensiones de análisis fenomenológico"
  }
}
```

---

## 🔧 CONFIGURACIÓN

### Parámetros Principales

```python
# ciclo_prompt_maximo_relacional.py

concepto = "DESTRUCCION"          # Concepto a analizar
gemini_key = "AIzaSyB3..."        # API Key de Gemini
iteraciones = 3                    # Número de iteraciones (1-5)
```

### Ajuste de Iteraciones

| Iteraciones | Tiempo Est. | Rutas Esperadas | Profundidad |
|-------------|-------------|-----------------|-------------|
| 1 | ~30s | 2-3 rutas | Superficial |
| 2 | ~60s | 4-6 rutas | Media |
| **3** | ~90s | **6-8 rutas** | **Profunda** |
| 4 | ~120s | 8-10 rutas | Muy Profunda |
| 5 | ~150s | 10-12 rutas | Exhaustiva |

---

## 📍 ARCHIVOS GENERADOS

### 1. `RESULTADO_CICLO_MAXIMO_RELACIONAL.json`
- JSON completo con todas las rutas
- Incluye certeza, análisis profundo, ejemplos
- Compatible con integración n8n

### 2. `REPORTE_CICLO_MAXIMO_RELACIONAL.md`
- Reporte markdown legible
- Tablas comparativas
- Estadísticas consolidadas

### 3. Terminal Output
- Log en tiempo real
- Iteraciones completadas
- Rutas descubiertas

---

## 🎨 INDEPENDENCIA DEL SISTEMA ACTUAL

El Ciclo Prompt Máximo Relacional es **COMPLETAMENTE INDEPENDIENTE**:

```
┌─────────────────────────────────────┐
│   YO ESTRUCTURAL v2.1 (ACTUAL)      │
│  ├─ n8n Workflow                    │
│  ├─ Neo4j Integration               │
│  └─ Gemini Basic Analysis           │
└─────────────────────────────────────┘

        SISTEMA AISLADO ↓

┌─────────────────────────────────────┐
│ CICLO PROMPT MÁXIMO RELACIONAL      │
│ ├─ Descubrimiento Dinámico          │
│ ├─ 10 → N Rutas                    │
│ └─ Gemini Iterativo (3+ loops)     │
└─────────────────────────────────────┘

NO COMPARTE:
❌ Neo4j connections
❌ n8n workflow state
❌ Credenciales del sistema
✅ Solo usa: Gemini API + Lógica Python
```

---

## 💡 CASOS DE USO

### Caso 1: Análisis Profundo Aislado
```bash
python3 ciclo_prompt_maximo_relacional.py

# Genera análisis completo sin tocar n8n
# Ideal para: Investigación, validación, testing
```

### Caso 2: Integración en Workflow n8n
```javascript
// En un HTTP POST node de n8n
// URL: Ejecutar script Python remoto
// Output: JSON con 15+ rutas

POST /api/ciclo-maximo-relacional
{
  "concepto": "DESTRUCCION",
  "iteraciones": 3
}
```

### Caso 3: Comparación de Conceptos
```python
# Ejecutar ciclo para múltiples conceptos
conceptos = ["DESTRUCCION", "CREACION", "TRANSFORMACION"]

for concepto in conceptos:
    resultado, reporte = ejecutar_ciclo_completo(concepto, gemini_key, 2)
    # Guardar y comparar resultados
```

---

## 📈 MEJORAS ESPERADAS

| Métrica | Antes | Después (Ciclo) | Mejora |
|---------|-------|-----------------|--------|
| Rutas Fenomenológicas | 5 | 10 | +100% |
| Rutas Totales (Con ciclo) | - | 15+ | +200% |
| Profundidad Media | 0.85 | 0.90 | +5% |
| Descubrimiento Dinámico | No | Sí | ✅ |

---

## 🚨 NOTAS IMPORTANTES

1. **Independencia**: El ciclo NO afecta el workflow n8n actual
2. **Gemini API**: Requiere clave válida y cuota disponible
3. **Tiempo**: Cada iteración toma ~30s. 3 iteraciones = ~90s
4. **Rutas Nuevas**: Cada ejecución puede descubrir rutas diferentes
5. **Certeza**: Varía según Gemini; rango 0.70-0.95

---

## 🔗 INTEGRACIÓN FUTURA

### Opción A: Agregar Nodo a Workflow Actual
```
Webhook Input
  ↓
YO Estructural v2.1 (5 rutas)
  ↓
CICLO PROMPT MÁXIMO RELACIONAL (5-10 rutas nuevas)
  ↓
Merge & Consolidate
  ↓
Output: 15+ Rutas Totales
```

### Opción B: Mantener Completamente Aislado
```
CICLO PROMPT MÁXIMO RELACIONAL (Standalone)
  ↓
Resultados independientes
  ↓
Comparar con YO Estructural v2.1
  ↓
Insights & Validación
```

---

## 📞 SOPORTE

**Archivos relacionados:**
- `ciclo_prompt_maximo_relacional.py` - Sistema completo
- `ciclo_maximo_relacional_n8n.py` - Versión n8n
- Este archivo: `GUIA_CICLO_MAXIMO_RELACIONAL.md`

**Comando de ayuda:**
```bash
python3 ciclo_prompt_maximo_relacional.py --help
```

---

**Versión**: 1.0  
**Creado**: 2025-11-07  
**Estado**: ✅ ACTIVO  
**Factor Máximo**: Máximas Rutas Fenomenológicas Posibles
