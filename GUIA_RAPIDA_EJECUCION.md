# 🚀 GUÍA RÁPIDA DE EJECUCIÓN

**Sistema**: YO Estructural v2.1 + Ciclo Máximo Relacional Optimizado  
**Fecha**: 2025-11-08

---

## ⚡ INICIO RÁPIDO (5 MINUTOS)

### 1. Ejecutar Ciclo Original (v1.0)

```bash
cd /workspaces/-...Raiz-Dasein
python3 ciclo_prompt_maximo_relacional.py
```

**Resultado esperado**:
- 15 rutas nuevas descubiertas
- Certeza promedio: 0.85
- Archivos generados:
  - `RESULTADO_CICLO_MAXIMO_RELACIONAL.json`
  - `REPORTE_CICLO_MAXIMO_RELACIONAL.md`

---

### 2. Ejecutar Ciclo Optimizado (v2.0)

```bash
cd /workspaces/-...Raiz-Dasein
python3 ciclo_maximo_relacional_optimizado.py
```

**Resultado esperado**:
- 8-12 rutas nuevas (depende de rate limits)
- Profundidad promedio: 4.3-4.5/5.0
- Tokens usados: 15K-20K
- Archivos generados:
  - `RESULTADO_CICLO_OPTIMIZADO.json`
  - `REPORTE_CICLO_OPTIMIZADO.md`

---

### 3. Ver Resultados

```bash
# Ver reporte optimizado
cat REPORTE_CICLO_OPTIMIZADO.md | head -100

# Ver JSON completo
cat RESULTADO_CICLO_OPTIMIZADO.json | jq '.'

# Ver métricas clave
cat RESULTADO_CICLO_OPTIMIZADO.json | jq '.metricas_optimizacion'
```

---

## 📊 COMPARAR VERSIONES

### Ver Tabla Comparativa

```bash
cat ANALISIS_COMPARATIVO_CICLOS.md | grep -A 20 "TABLA COMPARATIVA"
```

**Output esperado**:
```
| Métrica              | v1.0       | v2.0      | Mejora   |
|----------------------|------------|-----------|----------|
| Tokens Usados        | ~50,000    | 17,190    | -65.6%   |
| Llamadas API         | ~50        | 13        | -74%     |
| Rutas Nuevas         | 15         | 8         | -47%*    |
| Profundidad Promedio | N/A        | 4.38/5.0  | +100%    |
```

---

## 🔧 CONFIGURACIÓN

### API Key de Gemini

```bash
export GEMINI_API_KEY="AIzaSyAKWPJb7uG84PwQLMCFlxbJNuWZGpdMzNg"
```

**Verificar configuración**:
```bash
grep "GEMINI_KEY" ciclo_maximo_relacional_optimizado.py
```

---

### Cambiar Concepto a Analizar

**Editar archivo**:
```bash
nano ciclo_maximo_relacional_optimizado.py
```

**Buscar línea**:
```python
CONCEPTO = "DESTRUCCION"  # ← Cambiar aquí
```

**Conceptos sugeridos**:
- SER
- VERDAD
- RELACION
- FENOMENOLOGIA
- TIEMPO
- ESPACIO

---

### Ajustar Número de Iteraciones

**Editar**:
```python
ITERACIONES = 3  # ← Cambiar (1-5 recomendado)
```

**Recomendaciones**:
- `1-2`: Pruebas rápidas (5-10 min)
- `3`: Producción estándar (15-20 min)
- `4-5`: Análisis profundo (30-40 min)

---

## 📚 DOCUMENTACIÓN

### Ver Guía Completa de LangChain

```bash
cat GUIA_OPTIMIZACION_LANGCHAIN_GRAFOS.md | less
```

**Secciones clave**:
- Fundamentos teóricos
- Implementación paso a paso
- Problemas comunes (429, schema errors)
- Mejores prácticas

---

### Ver Resumen Ejecutivo

```bash
cat RESUMEN_EJECUTIVO_OPTIMIZACION.md | less
```

**Contenido**:
- Resultados clave
- Optimizaciones implementadas
- Desafíos identificados
- Roadmap v2.1

---

### Navegar Índice Completo

```bash
cat INDICE_COMPLETO_OPTIMIZACION.md | less
```

**Uso**: Mapa de navegación de toda la documentación

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### Error 429: Rate Limit

**Síntoma**:
```
❌ Error API 429: Resource exhausted
```

**Soluciones**:

1. **Esperar y reintentar**:
```bash
sleep 60  # Esperar 1 minuto
python3 ciclo_maximo_relacional_optimizado.py
```

2. **Reducir iteraciones**:
```python
ITERACIONES = 2  # En lugar de 3
```

3. **Aumentar delays** (editar archivo):
```python
time.sleep(10)  # Entre iteraciones
```

---

### Error 400: Schema Error

**Síntoma**:
```
❌ Error API 400: properties should be non-empty
```

**Causa**: Schema de grafos con propiedades vacías

**Estado**: Conocido, fix pendiente v2.1

**Workaround**: No afecta descubrimiento de rutas, solo extracción de grafos

---

### Sin Resultados

**Si no se generan archivos**:

```bash
# Verificar permisos
ls -l ciclo_maximo_relacional_optimizado.py

# Dar permisos de ejecución
chmod +x ciclo_maximo_relacional_optimizado.py

# Ejecutar con logs
python3 ciclo_maximo_relacional_optimizado.py 2>&1 | tee ciclo_output.log
```

---

### Verificar Instalación

```bash
# Verificar Python
python3 --version  # Debe ser 3.8+

# Verificar dependencias
pip list | grep -E "requests|json"
```

---

## 📈 MÉTRICAS EN TIEMPO REAL

### Durante Ejecución

**Monitorear tokens**:
```bash
tail -f /tmp/ciclo_optimizado_output.txt | grep "Tokens usados"
```

**Ver rutas descubiertas**:
```bash
tail -f /tmp/ciclo_optimizado_output.txt | grep "🆕"
```

---

### Después de Ejecución

**Contar rutas totales**:
```bash
cat RESULTADO_CICLO_OPTIMIZADO.json | jq '.estadisticas.total_rutas'
```

**Ver tokens usados**:
```bash
cat RESULTADO_CICLO_OPTIMIZADO.json | jq '.metricas_optimizacion.tokens_totales_usados'
```

**Ver profundidad promedio**:
```bash
cat RESULTADO_CICLO_OPTIMIZADO.json | jq '.estadisticas.nivel_profundidad_promedio'
```

---

## 🔄 INTEGRACIÓN CON N8N

### Exportar a n8n

**Crear workflow n8n**:
```bash
python3 ciclo_maximo_relacional_n8n.py
```

**Importar en n8n**:
1. Abrir n8n: http://localhost:5678
2. Ir a "Workflows" → "Import from URL or File"
3. Seleccionar archivo generado
4. Activar workflow

---

### Webhook Endpoint

**URL**:
```
http://localhost:5678/webhook/ciclo-maximo-relacional
```

**Payload**:
```json
{
  "concepto": "DESTRUCCION",
  "iteraciones": 3
}
```

**Test con curl**:
```bash
curl -X POST http://localhost:5678/webhook/ciclo-maximo-relacional \
  -H "Content-Type: application/json" \
  -d '{"concepto":"DESTRUCCION","iteraciones":3}'
```

---

## 📊 VISUALIZACIÓN

### Gráficos de Métricas

**Crear gráfico comparativo** (requiere matplotlib):
```python
import matplotlib.pyplot as plt
import json

# Cargar datos
with open('RESULTADO_CICLO_OPTIMIZADO.json') as f:
    data = json.load(f)

# Crear gráfico
rutas = data['estadisticas']['rutas_nuevas_descubiertas']
tokens = data['metricas_optimizacion']['tokens_totales_usados']

plt.bar(['v1.0', 'v2.0'], [15, rutas])
plt.title('Rutas Descubiertas')
plt.savefig('comparativa_rutas.png')
```

---

### Ver Grafo (cuando esté funcional)

```python
import networkx as nx
import matplotlib.pyplot as plt

# Cargar grafo
with open('RESULTADO_CICLO_OPTIMIZADO.json') as f:
    data = json.load(f)

grafo = data['grafo_conocimiento']

# Crear networkx graph
G = nx.DiGraph()
for nodo in grafo['nodos']:
    G.add_node(nodo['id'])
for rel in grafo['relaciones']:
    G.add_edge(rel['origen'], rel['destino'])

# Visualizar
nx.draw(G, with_labels=True)
plt.savefig('grafo_conocimiento.png')
```

---

## 🎯 CASOS DE USO

### Caso 1: Análisis Rápido (1 Concepto)

```bash
# Editar concepto
sed -i 's/DESTRUCCION/SER/g' ciclo_maximo_relacional_optimizado.py

# Ejecutar
python3 ciclo_maximo_relacional_optimizado.py

# Ver resumen
cat REPORTE_CICLO_OPTIMIZADO.md | head -50
```

---

### Caso 2: Comparativa Múltiples Conceptos

```bash
#!/bin/bash
CONCEPTOS=("SER" "VERDAD" "RELACION" "FENOMENOLOGIA")

for concepto in "${CONCEPTOS[@]}"; do
  echo "Analizando: $concepto"
  sed -i "s/CONCEPTO = .*/CONCEPTO = \"$concepto\"/g" ciclo_maximo_relacional_optimizado.py
  python3 ciclo_maximo_relacional_optimizado.py
  mv RESULTADO_CICLO_OPTIMIZADO.json "resultado_${concepto}.json"
  sleep 120  # Esperar 2 min entre conceptos
done
```

---

### Caso 3: Integración en Pipeline

```yaml
# .github/workflows/analisis_conceptual.yml
name: Análisis Conceptual Automatizado

on:
  push:
    branches: [main]

jobs:
  analizar:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install requests
      - name: Run analysis
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: python3 ciclo_maximo_relacional_optimizado.py
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: resultados
          path: RESULTADO_CICLO_OPTIMIZADO.json
```

---

## 🔒 SEGURIDAD

### Proteger API Key

**No commitear en git**:
```bash
echo "*.env" >> .gitignore
echo "GEMINI_API_KEY=tu_key" > .env
```

**Usar variable de entorno**:
```python
import os
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
```

---

### Rate Limits

**Verificar cuota actual**:
```bash
curl https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY
```

**Monitorear uso**:
- Ver dashboard: https://aistudio.google.com/

---

## 📞 SOPORTE

### Logs

**Ver logs de ejecución**:
```bash
tail -100 /tmp/ciclo_optimizado_output.txt
```

**Guardar logs**:
```bash
python3 ciclo_maximo_relacional_optimizado.py 2>&1 | tee logs_$(date +%Y%m%d_%H%M%S).txt
```

---

### Debugging

**Modo verbose**:
```python
# En ciclo_maximo_relacional_optimizado.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Verificar respuestas API**:
```python
# Añadir antes de response.json()
print(f"Status: {response.status_code}")
print(f"Body: {response.text[:500]}")
```

---

## 🎉 ÉXITO

**Verificar que todo funcionó**:

```bash
# Archivos generados
ls -lh RESULTADO_CICLO_OPTIMIZADO.json REPORTE_CICLO_OPTIMIZADO.md

# Rutas descubiertas
cat RESULTADO_CICLO_OPTIMIZADO.json | jq '.estadisticas'

# Tokens usados
cat RESULTADO_CICLO_OPTIMIZADO.json | jq '.metricas_optimizacion.tokens_totales_usados'
```

**Output esperado**:
```json
{
  "rutas_canonicas": 10,
  "rutas_nuevas_descubiertas": 8,
  "total_rutas": 18,
  "certeza_promedio_nuevas": 0.719,
  "nivel_profundidad_promedio": 4.38
}

17190
```

---

## 📚 RECURSOS ADICIONALES

### Documentación
- `GUIA_OPTIMIZACION_LANGCHAIN_GRAFOS.md` - Guía completa
- `RESUMEN_EJECUTIVO_OPTIMIZACION.md` - Resumen ejecutivo
- `ANALISIS_COMPARATIVO_CICLOS.md` - Comparativa v1 vs v2
- `INDICE_COMPLETO_OPTIMIZACION.md` - Índice navegable

### Código
- `ciclo_maximo_relacional_optimizado.py` - Ejecutable principal
- `ciclo_prompt_maximo_relacional.py` - Versión original

### Resultados
- `RESULTADO_CICLO_OPTIMIZADO.json` - Resultado JSON
- `REPORTE_CICLO_OPTIMIZADO.md` - Reporte legible

---

**Última Actualización**: 2025-11-08  
**Versión**: 1.0  
**Estado**: ✅ LISTO PARA USAR
