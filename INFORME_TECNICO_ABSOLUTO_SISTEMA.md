# INFORME TÉCNICO ABSOLUTO DEL SISTEMA "YO ESTRUCTURAL"

Fecha: 2025-11-08
Ámbito: Código, arquitectura, dependencias, servicios, seguridad, flujos, ejecución, métricas y recomendaciones.

---

## 1) Descripción general y objetivos

"YO Estructural" es un sistema para análisis fenomenológico y relacional de conceptos mediante LLMs y grafos de conocimiento. Combina extracción generativa (Gemini / OpenRouter), análisis de grafos (Neo4j, NetworkX) e integración operativa (n8n, FastAPI) con documentación extensa y tooling Docker para un despliegue reproducible.

Objetivos clave:
- Descubrir rutas fenomenológicas de un concepto (p. ej., DESTRUCCIÓN).
- Analizar y evaluar profundidad, certeza, aplicaciones y paradojas.
- Extraer un grafo de conocimiento y calcular métricas.
- Integrar con Neo4j y n8n para persistencia/automatización.
- Optimizar uso de tokens y robustecer la calidad de salida.

---

## 2) Arquitectura lógica y de servicios

Componentes principales:
- Capa de aplicación:
  - `YO estructural/api_generador_maximo.py` (FastAPI REST)
  - `ciclo_maximo_relacional_optimizado.py` (Ciclo v2.0 optimizado, aislado)
  - `integracion_neo4j_gemini.py` (complemento más simple)
  - `api_neo4j_gemini.js` (API Node/Express opcional)
- Procesadores internos (YO estructural/procesadores):
  - `gemini_integration.py` (SDK google-generativeai)
  - `analizador_maximo_relacional_hibrido.py` (NetworkX + Neo4j GDS)
- Persistencia y cómputo:
  - Neo4j 5.15 (DB Grafos)
  - Redis (caché opcional)
- Orquestación/Integración: n8n (webhooks y flujos)
- Monitoreo/Proxy opcional: Prometheus, Grafana, Nginx

Topología (docker-compose):
- `neo4j` (puertos 7474, 7687)
- `n8n` (puerto 5678)
- `yo_estructural_api` (FastAPI en 8000)
- `yo_estructural_automation` (procesos batch)
- `redis` (6379), `nginx` (80/443), `prometheus` (9090), `grafana` (3000)

---

## 3) Flujos de alto nivel

1. Ciclo aislado (v2.0) — `ciclo_maximo_relacional_optimizado.py`:
   - Entrada: concepto y clave Gemini.
   - Fase 1: Descubrimiento de rutas (JSON Schema → structured output).
   - Fase 2: Extracción de grafo (JSON Schema específico).
   - Fase 3: Análisis profundo por ruta (estructura validada).
   - Salidas: JSON con métricas y reporte Markdown.

2. API REST — `YO estructural/api_generador_maximo.py`:
   - Endpoints: `/`, `/health`, `/api/generador/rutas`.
   - Usa embeddings locales y (opcional) Gemini para enriquecer.
   - Opcional: guarda en Neo4j y envía a n8n.

3. Integración Gemini SDK — `gemini_integration.py`:
   - Análisis de convergencia de rutas (estructura JSON estricta).

4. Análisis de grafos híbrido — `analizador_maximo_relacional_hibrido.py`:
   - Para grandes grafos: combina NetworkX (local) con Neo4j GDS (remoto).

5. OpenRouter — `test_openrouter.py` y `chat_openrouter.py`:
   - Consumo directo de OpenRouter API para pruebas y chat interactivo.

---

## 4) Módulos clave y responsabilidades

### 4.1 `ciclo_maximo_relacional_optimizado.py` (696 líneas)
- Clase: `CicloMaximoRelacionalOptimizado`.
- Esquemas JSON (structured output):
  - `SCHEMA_RUTAS_DESCUBIERTAS`: nombre, descripción, justificación, ejemplo, nivel_profundidad; resumen + total_encontradas.
  - `SCHEMA_ANALISIS_PROFUNDO`: analisis_profundo (≥500 chars), ejemplos (5–8), certeza (0–1), aplicaciones (3–5), paradojas (2–4), dimensiones_relacionadas.
  - `SCHEMA_GRAFO_CONOCIMIENTO`: nodos (id, tipo, propiedades) y relaciones (origen, tipo, destino, propiedades). Nota: requiere `additionalProperties: true` en `propiedades`.
- Métodos principales:
  - `_llamar_gemini_structured(prompt, schema, temperature)`: invoca Gemini con `responseMimeType=application/json` y `responseSchema`, acumula `tokens_usados` y conteo de llamadas.
  - `descubrir_nuevas_rutas_optimizado(iteraciones=3)`: orquesta Fase 1, 2 y 3; imprime métricas por iteración; compila resultados finales.
  - `_descubrir_rutas_structured()`: genera nuevas rutas evitando canónicas; retorna lista de rutas válidas.
  - `_extraer_grafo_structured(rutas)`: estructura texto y solicita un grafo; agrega nodos/relaciones a estado.
  - `_analizar_rutas_profundo()`: para rutas recientes, produce análisis completo según `SCHEMA_ANALISIS_PROFUNDO`.
  - Compilación de métricas: tokens totales, llamadas, promedios; genera reporte Markdown.
- Entradas: concepto, GEMINI key, modelo (por defecto `gemini-2.0-flash-exp`).
- Salidas: JSON de resultados y reporte detallado.
- Estado interno: `rutas_descubiertas`, `grafo_conocimiento`, `tokens_usados`, `llamadas_api`.

### 4.2 `YO estructural/api_generador_maximo.py` (FastAPI)
- Inicializa:
  - `GeneradorRutasFenomenologicas` (embeddings y Neo4j).
  - `GeminiEnriquecedor` (opcional; SDK oficial).
  - `N8nIntegrator` (opcional; webhook).
- Endpoints:
  - `/`: estado de componentes y rutas disponibles.
  - `/health`: health general con timestamps.
  - `POST /api/generador/rutas`: entrada `ConceptoRequest {concepto, usar_neo4j, usar_gemini, enviar_a_n8n}` → salida `MaximoRelacionalResponse` con rutas, certeza combinada, flags de guardado y envío.
- Flujo interno:
  - Genera rutas con el generador (local/embeddings).
  - Enriquecer con Gemini (si habilitado) y ajustar convergencia.
  - Guardar en Neo4j si es máximo relacional.
  - Enviar a n8n en tarea background (si se solicita).

### 4.3 `YO estructural/procesadores/gemini_integration.py`
- Clase `GeminiEnriquecedor` con `google-generativeai`:
  - Requiere `GOOGLE_GEMINI_API_KEY`.
  - Configuración: temperature 0.3, max_output_tokens 2048; `safety_settings` en `BLOCK_NONE`.
  - `analizar_convergencia(concepto, rutas)`: retorna JSON con `convergen`, `razon`, `definicion_unificada`, `confianza` y `recomendaciones`.
  - `enriquecer_ruta(...)`: análisis semántico por ruta (estructura JSON).

### 4.4 `YO estructural/procesadores/analizador_maximo_relacional_hibrido.py`
- Arquitectura híbrida para grandes grafos:
  - `AnalizadorNetworkX`: cargar grafo, PageRank, betweenness, densidad, métricas y tiempos.
  - Soporte para lotes de nodos/arcos, logging y control de memoria.
  - Plantilla para combinar resultados con Neo4j GDS (cálculo remoto cuando el grafo escala).

### 4.5 `integracion_neo4j_gemini.py`
- Clase `IntegracionYOEstructural`:
  - Verificación de conexiones a Neo4j y Gemini.
  - Consultas simples a Neo4j (Cypher) para un concepto.
  - Generación con Gemini de rutas base (sin structured output avanzado).

### 4.6 `api_neo4j_gemini.js` (Node/Express)
- Rutas:
  - `GET /health`: verifica Neo4j y Gemini (status HTTP 200 check).
  - `POST /neo4j/query`: consulta conceptos y relaciones.
  - `POST /gemini`: invoca Gemini para enriquecer (JSON simple).
- Config: `NEO4J_URL/USER/PASS`, `GEMINI_API_KEY` vía env.

### 4.7 OpenRouter utilities
- `test_openrouter.py`: prueba simple (2+2, consulta filosófica). Usa cabeceras `HTTP-Referer` y `X-Title`.
- `chat_openrouter.py`: chat interactivo desde terminal con selección de modelo, historial, conteo de tokens y guardado a JSON.

---

## 5) Datos, esquemas y persistencia

- Esquemas JSON (strict) en ciclo v2.0 controlan formato de salida del LLM.
- Grafo de conocimiento: nodos con `id`, `tipo` y `propiedades`; relaciones con `origen`, `tipo`, `destino` y `propiedades`.
- Neo4j: modelos `Concepto`, relaciones `RELACIONADO_CON` y otras según generador.
- Persistencia opcional: `GeneradorRutasFenomenologicas.guardar_maximo_en_neo4j`.

---

## 6) Integraciones externas

- Gemini (Google Generative AI): REST directo (v2.0) y SDK oficial (gemini_integration.py).
- OpenRouter: endpoint `/api/v1/chat/completions` con Bearer token. Headers recomendados: `HTTP-Referer`, `X-Title`.
- n8n: webhook configurable, envío en background desde API.
- Neo4j: HTTP (tx/commit) y Bolt (para servicios en docker-compose).

---

## 7) Configuración, variables y seguridad

Archivos y variables relevantes:
- `YO estructural/.env` (actualmente incluye secretos reales — riesgo):
  - `NEO4J_URI/USER/PASSWORD`
  - `GOOGLE_GEMINI_API_KEY`
  - `N8N_WEBHOOK_URL`, `N8N_API_KEY`, credenciales de usuario
  - `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_TOKEN_FILE`
- `docker-compose.yml`: credenciales (p. ej., `NEO4J_AUTH=neo4j/fenomenologia2024`).

Observaciones de seguridad:
- Claves Gemini expuestas en varios archivos (`ciclo_*`, `integracion_neo4j_gemini.py`, docs y scripts n8n).
- Token JWT de n8n y contraseñas de Neo4j en texto plano.
- Directorio `YO estructural/venv/` versionado en git (ruido y posible fuga).

Recomendaciones inmediatas:
- Rotar TODAS las claves expuestas y eliminar hardcodes. Usar solo variables de entorno.
- Añadir `.gitignore` para `.env`, `venv/`, artefactos, y limpiar historial si claves reales fueron commitadas.
- Cambiar contraseñas por valores fuertes y almacenarlas en secret manager.

---

## 8) Dependencias y packaging

`YO estructural/requirements.txt` (~60+ libs): FastAPI, Pydantic v2, Neo4j, Supabase, scikit-learn, transformers, torch, google-generativeai, requests/httpx, pandas/numpy, logging, pytest, etc.

Notas:
- Mezcla de dependencias de desarrollo (Jupyter/plotly/matplotlib) y producción.
- Recomendado separar `requirements.txt` (runtime) y `requirements-dev.txt` (dev/test/docs).
- Considerar `pipreqs`/`uv`/`pip-tools` para reducir a lo usado realmente.

---

## 9) Ejecución y endpoints

- API (FastAPI):
  - `uvicorn YO estructural/api_generador_maximo:app --host 0.0.0.0 --port 8000`
  - Endpoints: `/`, `/health`, `POST /api/generador/rutas`
- Neo4j: http://localhost:7474 (Auth inicial definido en compose)
- n8n: http://localhost:5678
- Grafana: http://localhost:3000 (credencial inicial en compose)

Ciclo aislado v2.0:
- Requiere `GEMINI_KEY` y concepto; produce JSON/MD de resultados.

OpenRouter:
- `test_openrouter.py` y `chat_openrouter.py` usan `OPENROUTER_API_KEY`.

---

## 10) Métricas conocidas (v2.0)

- Rutas nuevas descubiertas: 8 (limitado por 429 en pruebas).
- Profundidad promedio: 4.38/5.0
- Certeza promedio: 0.719
- Tokens usados: 17,190 (vs ~50,000 v1.0) → -65.6%
- Llamadas API: 13 (vs ~50 v1.0) → -74%

---

## 11) Calidad: gates y salud

- Build: PASS (no se detectaron errores de sintaxis con tooling estándar en lectura; APIs arrancan si se configuran envs).
- Lint/Typecheck: NO EVALUADO (no hay pipeline activo).
- Tests: BAJO (tests no presentes; `pytest` instalado pero sin suite). 

Acciones sugeridas:
- Añadir CI (GitHub Actions) con `pytest`, `ruff/flake8`, `black` y `bandit`.
- Crear suite mínima de tests sobre:
  - Formato de salida de structured output (validación de schemas).
  - Manejo de errores 4xx/5xx y reintentos para Gemini/OpenRouter.
  - Serialización/guardado a Neo4j y consistencia básica del grafo.

---

## 12) Riesgos y problemas detectados

1. Seguridad (crítico): claves API y contraseñas hardcodeadas en múltiples archivos y docs.
2. Schema de grafo (bloqueante): falta `additionalProperties: true` → 400 en extracción.
3. Rate limiting (operacional): errores 429; falta retry/backoff/circuit-breaker.
4. Duplicación de código (mantenibilidad): múltiples versiones del mismo ciclo.
5. Requisitos pesados (operación): requirements con librerías de notebook/plotting en entorno productivo.
6. venv versionado (higiene): innecesario y ruidoso.

---

## 13) Roadmap y recomendaciones concretas

Prioridad inmediata (Día 1):
- Rotar y remover claves hardcodeadas; `.gitignore` y limpieza de `venv/`.
- Cambiar contraseñas default en `docker-compose` y `.env`.

Alta (Semana 1):
- Arreglar `SCHEMA_GRAFO_CONOCIMIENTO` (agregar `additionalProperties: true`).
- Añadir retry/backoff con `tenacity` en `_llamar_gemini_structured`.
- Consolidar a v2.0; marcar v1.0/n8n como legacy o delegar a API.

Media (Mes 1):
- Separar `requirements` en runtime/dev y minimizar librerías.
- CI con tests básicos y linters.
- Logging estructurado (`loguru`) y métricas (Prometheus) en API.

Evolución funcional:
- Análisis Existencial-Experiencial (nuevo schema y función).
- Matriz conceptual entre conceptos (script y reporte).
- Sincronización masiva a Neo4j y dashboards.

---

## 14) Inventario de archivos relevantes (no exhaustivo)

- Raíz:
  - `ciclo_maximo_relacional_optimizado.py` (núcleo v2.0)
  - `ciclo_prompt_maximo_relacional.py` (v1.0)
  - `ciclo_maximo_relacional_n8n.py` (adaptador n8n)
  - `integracion_neo4j_gemini.py` (integración simple)
  - `api_neo4j_gemini.js` (API Node alternativa)
  - `test_openrouter.py`, `chat_openrouter.py` (utilidades OpenRouter)
  - Documentación: `README.md`, `RESUMEN_*.md`, `GUIA_*.md`, `ANALISIS_*.md`, `VISUALIZACION_*.md`, etc.
- `YO estructural/`:
  - `api_generador_maximo.py` (FastAPI)
  - `procesadores/gemini_integration.py`, `procesadores/analizador_maximo_relacional_hibrido.py`
  - `docker-compose.yml`, `.env`, `requirements.txt`

---

## 15) Conexiones externas y puertos

- Gemini REST: `https://generativelanguage.googleapis.com/v1beta/models/...:generateContent`
- OpenRouter: `https://openrouter.ai/api/v1/chat/completions`
- Neo4j: 7474/7687 (HTTP/Bolt)
- n8n: 5678
- FastAPI: 8000
- Nginx: 80/443
- Redis: 6379
- Prometheus: 9090
- Grafana: 3000

---

## 16) Conclusión

El sistema está bien diseñado y documentado, con una versión v2.0 que mejora significativamente la eficiencia (structured output + reducción de tokens/llamadas). Para producción, urge resolver los puntos críticos de seguridad y robustez (claves expuestas, schema de grafo y rate limits) y establecer una base de calidad (tests y CI). Con esos ajustes, el sistema es viable para escalar a múltiples conceptos y análisis cruzados con persistencia en grafos y pipelines automatizados.

---

Fin del informe.
