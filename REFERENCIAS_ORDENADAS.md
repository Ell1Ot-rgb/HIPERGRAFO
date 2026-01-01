# REFERENCIAS ORDENADAS - Sistema YO Estructural v3.0

Este documento contiene las 40 referencias citadas en el informe técnico, organizadas secuencialmente.

## 1. Modelos de Lenguaje Externos (Kimi, OpenRouter)
Los archivos `ciclo_relacional_kimi_openrouter.py`, `ciclo_openrouter_runner.py`, `ciclo_kimi_free.py`, `chat_openrouter.py` y `GUIA_OPENROUTER_KIMI_PYTHON.md` indican una interacción directa con Grandes Modelos de Lenguaje (LLMs) externos.

**Funcionalidad**: Integración con APIs de modelos de lenguaje para el descubrimiento de rutas fenomenológicas y análisis conceptual.

## 2. Integración Neo4j-Gemini
Los archivos `integracion_neo4j_gemini.py` y `api_neo4j_gemini.js` contienen scripts específicos para la integración de Neo4j con Google Gemini.

**Funcionalidad**: Puente entre la base de datos de grafos y el modelo generativo para enriquecimiento semántico.

## 3. Documentación y Reportes
Archivos `.md` y `.txt` que contienen resúmenes, reportes y guías con detalles sobre arquitectura y funcionamiento.

**Funcionalidad**: Documentación técnica exhaustiva del sistema.

## 4. Ciclo Relacional Kimi OpenRouter
```python
# ciclo_relacional_kimi_openrouter.py
# Ejecutable que implementa un "Ciclo Relacional" utilizando la API de OpenRouter
# con el modelo gratuito Kimi K2
```

**Funcionalidad**: Descubrimiento de nuevas rutas conceptuales y análisis profundo.

## 5. Ciclo Kimi Free
```python
# ciclo_kimi_free.py
# Implementa un "Ciclo Relacional" con Kimi K2 gratuito a través de OpenRouter
# Genera 8-12 rutas conceptuales y analiza las 3 principales
```

**Funcionalidad**: Generación optimizada de rutas fenomenológicas.

## 6. Ciclo Relacional
```python
# ciclo_relacional.py
# Versión adaptada del "ciclo optimizado v2.0"
# Utiliza la API de OpenRouter con GPT-3.5-Turbo como modelo predeterminado
```

**Funcionalidad**: Descubrimiento del "Máximo Relacional".

## 7. Integración Neo4j Gemini
```python
# integracion_neo4j_gemini.py
# Clase IntegracionYOEstructural
# Verifica conexiones a Neo4j y Gemini
# Realiza consultas simples a Neo4j
# Genera rutas base con Gemini
```

**Funcionalidad**: Verificación y coordinación de servicios críticos.

## 8. API Neo4j Gemini
```javascript
// api_neo4j_gemini.js
// API REST en Node/Express
// Endpoints:
// - GET /health: verifica Neo4j y Gemini
// - POST /neo4j/query: consulta conceptos y relaciones
// - POST /gemini: invoca Gemini para enriquecement
```

**Funcionalidad**: Capa de servicio HTTP para interacciones externas.

## 9. PowerShellDeployment Script
```powershell
# n8n_setup/deploy-n8n-complete.ps1
$NEO4J_HOST_IP = "192.168.1.50" 
$NEO4J_PORT_BOLT = 7687
$PROJECT_ROOT = "C:\ruta\a\YO estructural"
$LOCAL_DATA_PATH = "$PROJECT_ROOT\entrada_ruta_del_proyecto"
```

**Funcionalidad**: Despliegue automatizado de n8n en red local.

## 10. Configuración de Variables de Entorno n8n
```powershell
@'
N8N_HOST=localhost
N8N_PORT=5678
NEO4J_HOST=$NEO4J_HOST_IP
NEO4J_PORT=$NEO4J_PORT_BOLT
NEO4J_USER=neo4j
NEO4J_PASSWORD=TuPasswordSegura
LOCAL_DATA_PATH=$LOCAL_DATA_PATH
YAML_OUTPUT_PATH=$YAML_OUTPUT_PATH
'@ | Out-File -FilePath "$env:USERPROFILE\.n8n\.env"
```

**Funcionalidad**: Configuración de credenciales y rutas para n8n.

## 11. Inicio del Servicio n8n
```powershell
n8n start --env-file $env:USERPROFILE\.n8n\.env
```

**Funcionalidad**: Lanzamiento del motor de orquestación.

## 12-18. Workflows de n8n (workflow_1, workflow_2, workflow_3)
Definiciones JSON de los flujos de trabajo de n8n que orquestan:
- Workflow 1: Monitoreo de archivos
- Workflow 2: Sincronización con Neo4j
- Workflow 3: Procesamiento de texto y generación de embeddings

**Funcionalidad**: Orquestación automatizada del pipeline de datos.

## 19. Nodo Analizador Fenomenológico (Workflow 3)
```javascript
// Workflow 3 - Nodo: "2. Analizador Fenomenológico y Extracción"
const payload = items.json;
const extracted_text = // ... lógica de extracción multimodal (OCR/API Gemini);

// Creación de Ereignis (Acontecimiento)
const ereignis = {
 id: `ereignis_${payload.document_id.substring(0, 8)}_${Date.now()}`,
 timestamp_extraccion: new Date().toISOString(),
 texto_original: extracted_text.substring(0, 500),
 tipo: 'evento'
};
```

**Funcionalidad**: Extracción y estructuración fenomenológica de datos.

## 20-23. Generación de Augenblick
```javascript
// Generación de Augenblick (Instante-de-Visión)
const augenblick = {
 id: `augenblick_${ereignis.id.substring(8)}_${Math.random().toString(36).substring(7)}`,
 timestamp_inicio: new Date().toISOString(),
 ereignisse_constituyentes: [ereignis.id],
 estado_fenomenologico: 'perceptivo',
 propiedades_emergentes: {
  coherencia_interna: 0.85,
  complejidad_semantica: 0.72,
  intencionalidad: 'directa'
 }
};
```

**Funcionalidad**: Creación de unidades mínimas de experiencia interpretada.

## 24. Creación de Texto Fenomenológico
```javascript
const texto_fenomenologico = `[FENOMENOLOGICO] ${extracted_text} | Coherencia Interna: 0.85 | Nivel YO: Narrativo`;
```

**Funcionalidad**: Enriquecimiento semántico del contenido.

## 25-27. Sincronización Neo4j (Workflow 2)
```cypher
// Workflow 2 - Consulta Cypher en Nodo Neo4j
MERGE (i:Instancia {doc_id: $json.docID})
SET i.texto_fenomenologico = $json.texto_fenomenologico, 
 i.metrica_yo = toFloat($json.metrica_yo),
 i.fecha = datetime(),
 i.embeddings = split($json.embeddings, ", ")

// Crear nodo Fenomeno y relacionarlo
MERGE (f:Fenomeno {tipo: 'Narrativo'})
MERGE (i)-[:SURGE_DE {peso_existencial: toFloat($json.metrica_yo)}]->(f)
```

**Funcionalidad**: Persistencia idempotente en grafo de conocimiento.

## 28-30. GraphRAG - Persistencia de Vectores
La persistencia de embeddings en Neo4j permite:
- Búsqueda vectorial dentro del grafo
- Recuperación aumentada combinando similitud semántica y proximidad relacional
- GraphRAG (Recuperación Aumentada por Grafos)

**Funcionalidad**: Arquitectura híbrida de recuperación vectorial y grafo.

## 31-32. Integrador n8n Python
```python
# integraciones/n8n_config.py
import requests
import json

class N8nIntegrator:
    """Clase para interactuar con los Webhooks de n8n (Dual Core)."""
    
    def __init__(self, base_url="http://localhost:5678", process_webhook="/process-text"):
        self.base_url = base_url 
        self.process_webhook = process_webhook

    def enviar_datos_a_procesamiento(self, doc_data: dict):
        """Envía datos al Workflow 3 de n8n para su análisis."""
        url = self.base_url + self.process_webhook
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(doc_data))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error al llamar al webhook de n8n: {e}") 
            return None
```

**Funcionalidad**: Bridge Python → n8n para disparar workflows.

## 33-37. Sistema Yo Emergente - Clasificación
```python
# motor_yo/sistema_yo_emergente.py - función _actualizar_tipo_yo()
def _actualizar_tipo_yo(self, coherencia: float) -> str:
    """
    Actualiza el tipo de YO basándose en métricas de coherencia.
    """
    if coherencia >= 0.75:
        return "Narrativo"
    elif coherencia >= 0.60:
        return "Reflexivo"
    elif coherencia >= 0.40:
        return "Fragmentado"
    else:
        return "Disociado"
```

**Funcionalidad**: Clasificación del estado del YO basada en umbrales.

## 38-40. Gemini CLI - Enriquecimiento Conceptual
Comandos conceptuales CLI para el Agente IA:

| Fase | Comando | Finalidad |
|------|---------|-----------|
| 1. Enriquecimiento Estructurado | `gemini --prompt "Extrae 5 rutas fenomenológicas del texto:" --output-format json` | Determinar el Máximo Relacional y crear Grundzugs |
| 2. Transformación a Grafo | `gemini --prompt "Transforma el siguiente texto en nodos y relaciones Cypher:" --schema-file graph_schema.json` | LLM Graph Transformer (LangChain) |
| 3. Análisis de Contradicción | `gemini --prompt "Analiza la coherencia narrativa del texto e identifica tensiones estructurales." --output-format json` | Métricas cualitativas para Tipo YO y Binarización Adaptativa |

**Funcionalidad**: Interfaz CLI para enriquecimiento fenomenológico con LLMs.

---

## Resumen de Arquitectura

### Componentes Principales:
1. **Neo4j** (i5 Core, Docker/WSL) - Base de Datos de Grafos
2. **n8n** (Dual Core, PowerShell) - Motor de Orquestación Headless
3. **Python Core** (Dual Core) - Analizador Fenomenológico

### Flujo de Datos:
```
Datos Brutos → n8n (Workflow 1) → Procesamiento (Workflow 3) → Neo4j (Workflow 2) → Python (Análisis FCA)
                     ↓                                                                     ↓
                  LLMs (Gemini/Kimi)                                              Grundzugs/Axiomas
```

### Patrones de Integración:
- **Patrón 1**: Python invoca n8n (via webhooks)
- **Patrón 2**: n8n invoca servicios externos (Gemini, OCR)
- **Patrón 3**: Todos escriben a Neo4j (única fuente de verdad)
