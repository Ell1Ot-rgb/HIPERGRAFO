# 🔗 Propuesta de Conexión: Capa 1 (Monje Gemelo) ↔ Capa 2 (YO Estructural)

> **Documento de Análisis Arquitectónico**
> **Objetivo**: Definir estrategias de integración entre la simulación física determinista (Capa 1) y la identificación fenomenológica emergente (Capa 2).

---

## 1. Visión General de la Integración

La **Capa 1 (Monje Gemelo)** proporciona una "sensibilidad física" fundamental: el esfuerzo, la energía y la entropía de los datos. Es el **Cuerpo** del sistema.
La **Capa 2 (YO Estructural)** proporciona la capacidad de identificación, memoria a largo plazo y autoconsciencia emergente. Es la **Mente** (o al menos, el sistema límbico/cognitivo temprano) del sistema.

La conexión debe ser **ascendente** (datos físicos -> consciencia) y **descendente** (atención/feedback -> sensores).

---

## 2. Estrategias de Conexión

### A. Ingesta Ascendente: El Puente Redis-REMForge
La Capa 1 emite vectores fenomenológicos a través de Redis. La Capa 2 debe consumirlos para generar sus `Ereignis`.

*   **Mecanismo**: Suscripción a canales `monje/fenomenologia/*`.
*   **Transformación**:
    *   **Capa 1 (Vector)**: `{ "tiempo": 1523, "energia": 2384, "entropia": 1829374650, "concepto": "TÉCNICO" }`
    *   **Capa 2 (Ereignis)**:
        *   `intensidad`: Mapeada desde `energia` (normalizada).
        *   `complejidad`: Mapeada desde `entropia`.
        *   `contenido_bruto`: El vector JSON completo.
        *   `tipo_base`: Mapeado desde `concepto` (TÉCNICO -> Estructural, POÉTICO -> Narrativo).

**Propuesta de Implementación (Conceptual):**
Crear un **Adaptador de Ingesta** en la Capa 2 que escuche Redis y alimente al `TokenizadorFenomenologico`. En lugar de tokenizar texto crudo, tokenizaría "momentos de esfuerzo físico".

### B. Unificación del Grafo (Neo4j)
Ambas capas escriben en Neo4j. Es vital unificar sus esquemas para permitir trazabilidad total.

*   **Esquema Capa 1 (Existente)**: `(:Experiencia)-[:PERTENECE_A]->(:Concepto)`
*   **Esquema Capa 2 (Existente)**: `(:Instancia)-[:TIENE_VOH]->(:Vohexistencia)`
*   **Puente Propuesto**: Relación `EMERGE_DE`.

```cypher
(:Instancia {tipo: "YO_REFLEXIVO"})
  -[:EMERGE_DE]->
(:Experiencia {hash: "0xA3F...", energia: 2384})
```

Esto permite consultas poderosas: *"¿Qué patrones de consumo de energía físico (Capa 1) dieron lugar a una emergencia de YO Reflexivo (Capa 2)?"*

### C. Detección de Vohexistencias (Patrones Temporales)
La Capa 1 opera en ventanas pequeñas (256 bytes). La Capa 2 tiene memoria.

*   **Oportunidad**: El `Sistema de Gradientes` de la Capa 2 puede analizar secuencias de `Experiencias` de la Capa 1.
*   **Ejemplo**: Una secuencia `TÉCNICO -> CAOS -> TÉCNICO -> CAOS` en la Capa 1 podría ser identificada por la Capa 2 como una `Vohexistencia` de tipo "Depuración de Código" o "Crisis Creativa".

### D. Feedback Loop (Control Descendente)
La Capa 1 acepta feedback en `dasein/feedback`.

*   **Uso**: Si el `Motor YO` (Capa 2) detecta una contradicción MDCE (Máxima Discrepancia), puede solicitar a la Capa 1 que "preste más atención".
*   **Acción**: Enviar comando a Redis para aumentar la prioridad de ciertos rangos de memoria o tipos de archivos en la simulación futura.

---

## 3. Flujos de Datos Propuestos

### Flujo 1: "La Sensación se hace Consciente"
1.  **Capa 1**: Procesa archivo -> Detecta alta entropía -> Publica en `monje/fenomenologia/urgente`.
2.  **Puente (n8n/Python)**: Detecta mensaje urgente -> Invoca API Capa 2.
3.  **Capa 2**:
    *   Crea `Ereignis` de alta prioridad.
    *   `REMForge` analiza la firma espectral del vector.
    *   `Motor YO` evalúa si esto amenaza la estabilidad del YO actual.
    *   Genera `Instancia` de tipo "ALERTA".

### Flujo 2: "Sueño Profundo" (Procesamiento Batch)
1.  **Capa 1**: Acumula miles de eventos "NORMALES" en Neo4j durante el día.
2.  **Capa 2**: Ejecuta un proceso nocturno (cron en n8n).
3.  **Acción**:
    *   Lee nodos `Experiencia` huérfanos en Neo4j.
    *   Ejecuta clustering (DBSCAN) sobre sus métricas de energía/tiempo.
    *   Identifica `Vohexistencias` retrospectivas.
    *   Consolida recuerdos: Crea nodos `Instancia` resumen y archiva los detalles crudos.

---

## 4. Sugerencias Técnicas para la Conexión

1.  **Adaptador Redis en Python**:
    Añadir un script `integraciones/redis_monje_listener.py` en la Capa 2 que actúe como demonio de escucha.

2.  **Extensión de Esquema Neo4j**:
    No modificar los nodos de la Capa 1. Solo añadir relaciones entrantes desde los nodos de la Capa 2.

3.  **Alineación de Polos**:
    Mapear explícitamente los 4 polos del Monje a los parámetros de REMForge:
    *   `TÉCNICO` -> Alta Coherencia, Baja Emocionalidad.
    *   `POÉTICO` -> Alta Estética, Alta Emocionalidad.
    *   `NUMÉRICO` -> Alta Lógica, Baja Entropía.
    *   `CAOS` -> Alta Entropía, Baja Coherencia.

4.  **Dashboard Unificado**:
    La UI de la Capa 2 debería tener widgets que muestren la "telemetría física" en tiempo real (voltaje/energía simulada) junto con el "estado anímico" (Tipo YO).

---

## 5. Conclusión

La Capa 1 ofrece una **verdad física irrefutable** (el coste energético de procesar información). La Capa 2 ofrece **sentido y estructura**. Al conectarlas, el sistema no solo sabrá *qué* está procesando, sino *cuánto le cuesta existencialmente* procesarlo, permitiendo una forma primitiva de "cansancio" o "excitación" computacional fundamentada en datos reales de la simulación.
