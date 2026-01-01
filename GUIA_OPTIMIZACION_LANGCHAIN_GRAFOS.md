# 🔧 GUÍA DE OPTIMIZACIÓN: LANGCHAIN + GEMINI PARA GRAFOS DE CONOCIMIENTO

**Fecha**: 2025-11-08  
**Objetivo**: Integrar LLM Graph Transformer para extraer grafos de conceptos fenomenológicos  
**Stack**: Gemini 2.0 Flash + LangChain + Neo4j

---

## 📚 FUNDAMENTOS TEÓRICOS

### ¿Qué es LLM Graph Transformer?

**LLM Graph Transformer** de LangChain es una herramienta que convierte texto no estructurado en **grafos de conocimiento** usando LLMs.

**Características Clave**:
- ✅ **Compatibilidad multi-LLM**: OpenAI GPT-4, Gemini, Claude, modelos open-source
- ✅ **Structured Output Preferred**: Usa salida estructurada cuando disponible
- ✅ **Fallback Prompt-based**: Recurre a prompts si no hay structured output
- ✅ **Estandarización automática**: Maneja diferencias entre modelos

### Ventajas de Usar Gemini con Graph Transformer

1. **Salida Estructurada Nativa**:
   ```python
   # Gemini 2.0 Flash soporta JSON Schema
   generationConfig = {
       "responseMimeType": "application/json",
       "responseSchema": {...}
   }
   ```

2. **Tokens Eficientes**:
   - Gemini 2.0 Flash: 1M tokens contexto
   - Gemini 2.0 Flash Experimental: Gratis durante beta

3. **Multimodal** (futuro):
   - Puede extraer grafos de imágenes/videos

---

## 🛠️ IMPLEMENTACIÓN COMPLETA

### PASO 1: Instalación de Dependencias

```bash
# Instalar LangChain ecosystem
pip install langchain langchain-google-genai langchain-community langchain-experimental

# Instalar Neo4j driver
pip install neo4j

# Instalar Google AI SDK
pip install google-generativeai
```

### PASO 2: Configuración Básica

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document

# Inicializar LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key="TU_API_KEY",
    temperature=0.7,
    max_tokens=8192
)

# Inicializar Graph Transformer
graph_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Concepto", "Ruta", "Dimension", "Ejemplo"],
    allowed_relationships=["PERTENECE_A", "RELACIONADO_CON", "EJEMPLIFICA"],
    node_properties=["nombre", "descripcion", "certeza"]
)
```

### PASO 3: Extracción de Grafo desde Texto

```python
# Crear documento
texto = """
El concepto de DESTRUCCION tiene múltiples dimensiones:

1. Ruta Etimológica: Del latín destruere, significa deshacer lo construido.
2. Ruta Fenomenológica: La experiencia vivida de la destrucción implica una ruptura.
3. Ruta Neurocientífica: La poda sináptica es destrucción necesaria para el aprendizaje.
"""

documento = Document(page_content=texto)

# Transformar a grafo
grafos = graph_transformer.convert_to_graph_documents([documento])

# Acceder al grafo
grafo = grafos[0]
print(f"Nodos: {len(grafo.nodes)}")
print(f"Relaciones: {len(grafo.relationships)}")

# Ver nodos
for nodo in grafo.nodes:
    print(f"Nodo: {nodo.id} ({nodo.type})")
    print(f"Propiedades: {nodo.properties}")

# Ver relaciones
for rel in grafo.relationships:
    print(f"{rel.source.id} --[{rel.type}]--> {rel.target.id}")
```

**Salida Esperada**:
```
Nodos: 5
Relaciones: 4

Nodo: DESTRUCCION (Concepto)
Propiedades: {'nombre': 'DESTRUCCION'}

Nodo: Ruta Etimológica (Ruta)
Propiedades: {'descripcion': 'Del latín destruere...'}

DESTRUCCION --[TIENE_RUTA]--> Ruta Etimológica
Ruta Etimológica --[RELACIONADO_CON]--> Ruta Fenomenológica
```

### PASO 4: Persistencia en Neo4j

```python
# Conectar a Neo4j
neo4j_graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password123"
)

# Almacenar grafo
neo4j_graph.add_graph_documents(grafos)

# Consultar
resultado = neo4j_graph.query("""
MATCH (c:Concepto {nombre: 'DESTRUCCION'})-[r]->(ruta:Ruta)
RETURN c.nombre, type(r), ruta.descripcion
""")

print(resultado)
```

### PASO 5: Uso con Structured Output (Gemini)

```python
# Schema personalizado para Gemini
SCHEMA_GRAFO = {
    "type": "object",
    "properties": {
        "nodos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "tipo": {"type": "string"},
                    "propiedades": {
                        "type": "object",
                        "additionalProperties": True  # ⭐ IMPORTANTE
                    }
                },
                "required": ["id", "tipo"]
            }
        },
        "relaciones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "origen": {"type": "string"},
                    "tipo": {"type": "string"},
                    "destino": {"type": "string"}
                },
                "required": ["origen", "tipo", "destino"]
            }
        }
    },
    "required": ["nodos", "relaciones"]
}

# Llamar con structured output
import requests

response = requests.post(
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={API_KEY}",
    json={
        "contents": [{
            "parts": [{"text": prompt_extraccion_grafo}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA_GRAFO
        }
    }
)

grafo_json = response.json()
```

---

## 🔥 CASO DE USO: CICLO MÁXIMO RELACIONAL

### Integración Completa

```python
class CicloConGrafos:
    def __init__(self, concepto, gemini_key):
        self.concepto = concepto
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_key
        )
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["Concepto", "Ruta", "Dimension", "Ejemplo", "Aplicacion"],
            allowed_relationships=[
                "TIENE_RUTA", "RELACIONADO_CON", "EJEMPLIFICA",
                "APLICA_EN", "CONTRASTA_CON", "DERIVA_DE"
            ]
        )
        self.neo4j = Neo4jGraph(
            url="bolt://neo4j:7687",
            username="neo4j",
            password="password123"
        )
    
    def descubrir_y_graficar(self, iteraciones=3):
        for i in range(iteraciones):
            # 1. Descubrir nuevas rutas (con structured output)
            rutas = self._descubrir_rutas()
            
            # 2. Crear documento con rutas
            texto = self._crear_documento_rutas(rutas)
            documento = Document(page_content=texto)
            
            # 3. Extraer grafo
            grafos = self.graph_transformer.convert_to_graph_documents([documento])
            
            # 4. Persistir en Neo4j
            self.neo4j.add_graph_documents(grafos)
            
            # 5. Analizar conexiones emergentes
            conexiones = self._analizar_conexiones()
            
            print(f"Iteración {i+1}: {len(grafos[0].nodes)} nodos, {len(grafos[0].relationships)} relaciones")
        
        return self._generar_reporte_final()
```

### Visualización del Grafo

```python
# Consulta Cypher para visualizar
query = """
MATCH (c:Concepto {nombre: 'DESTRUCCION'})
OPTIONAL MATCH (c)-[r1]->(ruta:Ruta)
OPTIONAL MATCH (ruta)-[r2]->(dim:Dimension)
RETURN c, r1, ruta, r2, dim
"""

resultado = neo4j_graph.query(query)

# Exportar a formato visualizable
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
for record in resultado:
    G.add_node(record['c']['nombre'])
    if record['ruta']:
        G.add_edge(record['c']['nombre'], record['ruta']['nombre'])

nx.draw(G, with_labels=True)
plt.show()
```

---

## ⚠️ PROBLEMAS COMUNES Y SOLUCIONES

### Problema 1: Schema Error "properties should be non-empty"

**Error**:
```
400: properties["propiedades"].properties: should be non-empty
```

**Causa**: Gemini requiere que objetos tengan `properties` definidas o `additionalProperties: true`

**Solución**:
```python
# ❌ MAL
"propiedades": {"type": "object"}

# ✅ BIEN - Opción 1: additionalProperties
"propiedades": {
    "type": "object",
    "additionalProperties": True
}

# ✅ BIEN - Opción 2: properties definidas
"propiedades": {
    "type": "object",
    "properties": {
        "nombre": {"type": "string"},
        "valor": {"type": "string"}
    }
}
```

### Problema 2: Rate Limit 429

**Error**:
```
429: Resource exhausted. Please try again later.
```

**Soluciones**:

1. **Retry con Exponential Backoff**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60)
)
def llamar_gemini_con_retry(...):
    response = requests.post(...)
    if response.status_code == 429:
        raise Exception("Rate limit")
    return response
```

2. **Delays Entre Llamadas**:
```python
import time

for i in range(iteraciones):
    resultado = llamar_gemini(...)
    time.sleep(5)  # 5 segundos entre llamadas
```

3. **Circuit Breaker**:
```python
class CircuitBreaker:
    def __init__(self, max_failures=3, timeout=60):
        self.failures = 0
        self.max_failures = max_failures
        self.timeout = timeout
        self.opened_at = None
    
    def call(self, func, *args, **kwargs):
        if self.failures >= self.max_failures:
            if time.time() - self.opened_at < self.timeout:
                raise Exception("Circuit open")
            else:
                self.failures = 0  # Reset
        
        try:
            result = func(*args, **kwargs)
            self.failures = 0
            return result
        except RateLimitError:
            self.failures += 1
            self.opened_at = time.time()
            raise
```

### Problema 3: Nodos Duplicados

**Problema**: LLM crea múltiples nodos para la misma entidad con nombres ligeramente diferentes.

**Solución**: Normalización y deduplicación

```python
def normalizar_nodo(nombre):
    # Minúsculas, sin acentos, sin espacios extra
    import unidecode
    return unidecode.unidecode(nombre.lower().strip())

def deduplicar_nodos(grafo):
    nodos_normalizados = {}
    for nodo in grafo.nodes:
        key = normalizar_nodo(nodo.id)
        if key not in nodos_normalizados:
            nodos_normalizados[key] = nodo
    
    grafo.nodes = list(nodos_normalizados.values())
    return grafo
```

---

## 📈 MÉTRICAS Y EVALUACIÓN

### Métricas de Calidad del Grafo

```python
def evaluar_grafo(grafo):
    metricas = {
        "nodos_totales": len(grafo.nodes),
        "relaciones_totales": len(grafo.relationships),
        "densidad": len(grafo.relationships) / (len(grafo.nodes) ** 2),
        "nodos_por_tipo": {},
        "relaciones_por_tipo": {}
    }
    
    for nodo in grafo.nodes:
        tipo = nodo.type
        metricas["nodos_por_tipo"][tipo] = metricas["nodos_por_tipo"].get(tipo, 0) + 1
    
    for rel in grafo.relationships:
        tipo = rel.type
        metricas["relaciones_por_tipo"][tipo] = metricas["relaciones_por_tipo"].get(tipo, 0) + 1
    
    return metricas
```

### Métricas de Uso de Tokens

```python
class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.llamadas = 0
    
    def registrar(self, tokens):
        self.total_tokens += tokens
        self.llamadas += 1
    
    def promedios(self):
        return {
            "total": self.total_tokens,
            "llamadas": self.llamadas,
            "promedio_por_llamada": self.total_tokens / self.llamadas
        }
```

---

## 🎯 MEJORES PRÁCTICAS

### 1. Definir Ontología Clara

```python
# Antes de extraer grafos, define tu ontología
ONTOLOGIA = {
    "nodos_permitidos": [
        "Concepto",      # Concepto principal
        "Ruta",          # Ruta fenomenológica
        "Dimension",     # Dimensión de análisis
        "Ejemplo",       # Caso concreto
        "Aplicacion",    # Aplicación práctica
        "Paradoja"       # Tensión o contradicción
    ],
    "relaciones_permitidas": [
        "TIENE_RUTA",          # Concepto → Ruta
        "PERTENECE_A",         # Ruta → Concepto
        "RELACIONADO_CON",     # Ruta ↔ Ruta
        "EJEMPLIFICA",         # Ejemplo → Ruta
        "APLICA_EN",           # Aplicacion → Ruta
        "CONTRASTA_CON",       # Paradoja → Ruta
        "DERIVA_DE"            # Ruta → Ruta (jerarquía)
    ]
}
```

### 2. Prompt Engineering para Grafos

```python
PROMPT_EXTRACCION_GRAFO = """
Eres un experto en extracción de grafos de conocimiento.

TEXTO A ANALIZAR:
{texto}

ONTOLOGÍA:
- Nodos: {nodos_permitidos}
- Relaciones: {relaciones_permitidas}

INSTRUCCIONES:
1. Identifica TODAS las entidades mencionadas
2. Clasifícalas según la ontología
3. Determina relaciones explícitas e implícitas
4. Usa nombres concisos y consistentes
5. Añade propiedades relevantes

IMPORTANTE:
- Un concepto puede tener múltiples rutas
- Las rutas se relacionan entre sí
- Los ejemplos ejemplifican rutas específicas
- Las paradojas contrastan o tensan rutas

Responde en JSON con nodos y relaciones.
"""
```

### 3. Validación Post-Extracción

```python
def validar_grafo(grafo, ontologia):
    errores = []
    
    # Validar tipos de nodos
    for nodo in grafo.nodes:
        if nodo.type not in ontologia["nodos_permitidos"]:
            errores.append(f"Tipo de nodo inválido: {nodo.type}")
    
    # Validar tipos de relaciones
    for rel in grafo.relationships:
        if rel.type not in ontologia["relaciones_permitidas"]:
            errores.append(f"Tipo de relación inválida: {rel.type}")
    
    # Validar conectividad
    if len(grafo.relationships) == 0:
        errores.append("Grafo sin relaciones (desconectado)")
    
    return errores
```

### 4. Enriquecimiento Iterativo

```python
def enriquecer_grafo_iterativamente(grafo_base, iteraciones=3):
    for i in range(iteraciones):
        # 1. Identificar nodos con pocas relaciones
        nodos_debiles = [n for n in grafo_base.nodes 
                        if contar_relaciones(n, grafo_base) < 2]
        
        # 2. Pedir al LLM más información sobre estos nodos
        for nodo in nodos_debiles:
            prompt = f"Expande información sobre {nodo.id}"
            texto_adicional = llm.invoke(prompt)
            
            # 3. Extraer nuevo subgrafo
            doc = Document(page_content=texto_adicional)
            subgrafos = graph_transformer.convert_to_graph_documents([doc])
            
            # 4. Fusionar con grafo base
            grafo_base = fusionar_grafos(grafo_base, subgrafos[0])
    
    return grafo_base
```

---

## 📚 RECURSOS Y REFERENCIAS

### Documentación Oficial

1. **LangChain Graph Transformers**:
   - https://python.langchain.com/docs/use_cases/graph/constructing

2. **Gemini Structured Output**:
   - https://ai.google.dev/gemini-api/docs/structured-output

3. **Neo4j Python Driver**:
   - https://neo4j.com/docs/api/python-driver/current/

### Ejemplos de Código

1. **LangChain + Gemini + Neo4j**:
   ```bash
   git clone https://github.com/langchain-ai/langchain
   cd langchain/cookbook/
   # Ver: knowledge_graph_*.ipynb
   ```

2. **Nuestra Implementación**:
   - `ciclo_maximo_relacional_optimizado.py` (este proyecto)

### Papers Relevantes

1. **Knowledge Graphs from LLMs**: 
   - "From Text to Knowledge Graphs with LLMs" (2024)
   
2. **Structured Output for LLMs**:
   - "Constrained Decoding for LLMs" (2023)

---

## ✅ CHECKLIST DE IMPLEMENTACIÓN

- [ ] Instalar dependencias (langchain, langchain-google-genai, neo4j)
- [ ] Configurar API key de Gemini
- [ ] Configurar conexión a Neo4j
- [ ] Definir ontología (nodos y relaciones permitidas)
- [ ] Crear schemas JSON para structured output
- [ ] Implementar extracción de grafos con LLMGraphTransformer
- [ ] Añadir retry logic para rate limits
- [ ] Implementar validación de grafos
- [ ] Configurar persistencia en Neo4j
- [ ] Crear visualizaciones
- [ ] Medir métricas de calidad
- [ ] Documentar resultados

---

**Última Actualización**: 2025-11-08  
**Autor**: Sistema de Optimización YO Estructural  
**Versión**: 1.0
