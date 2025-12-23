# 🌌 Documentación del Entorno Virtual de Simulación Omega 21

Esta documentación detalla la arquitectura, acceso y funcionamiento del entorno virtual que simula la **Red de Nodos (Hipergrafo)** y su interacción con la **Corteza Cognitiva** distribuida.

---

## 1. 🚀 Acceso y Ejecución

El entorno dispone de dos modos de operación principales:

### A. Modo Entrenamiento (Headless)
Este modo está optimizado para velocidad y rendimiento. Ejecuta el ciclo completo de estabilización dendrítica y envío de datos a Colab sin interfaz gráfica.

**Comando:**
```bash
npx ts-node src/run_entrenamiento_completo.ts
```

**Flujo:**
1. Inicializa los **25 Átomos Topológicos** (S1-S25).
2. Genera/Recibe el Vector 256D.
3. Aplica estabilización por dendritas.
4. Envía vectores de 1600D a la nube (Colab).

### B. Modo Omnisciente (Visualización en Tiempo Real)
Este modo levanta un servidor web local para visualizar la actividad neuronal, la topología del hipergrafo y las métricas físicas en tiempo real.

**Comando:**
```bash
npm run simular_cognicion
```
*(Alternativa directa: `npx ts-node src/run_omnisciente.ts`)*

**Acceso Web:**
Abra su navegador en: **[http://localhost:3000](http://localhost:3000)**

**Características Visuales:**
- **Cerebro Wolfram**: Actividad de las 1024 neuronas del átomo activo.
- **Jerarquía Cognitiva**: Estado de las Capas 1, 2 y 3.
- **Gráficos de Física**: Entropía, Energía y Tensión del sistema.

---

## 2. 🏗️ Estructura del Código

El entorno virtual está modularizado para separar la simulación física, el control neuronal y la comunicación.

### 📂 Núcleo del Sistema (`src/`)
*   **`SistemaOmnisciente.ts`**: **El Orquestador**. Clase principal que gestiona el ciclo de vida de los 25 átomos. Contiene el mapa `atomos: Map<string, AtomoTopologico>`.
*   **`core/AtomoTopologico.ts`**: La unidad fundamental. Cada instancia (S1...S25) contiene:
    *   `InferenciaLocal`: Motor ONNX (cerebro local).
    *   `Omega21Simulador`: Generador de realidad y telemetría.
    *   `Hipergrafo`: Estructura de memoria a corto plazo.

### 🎛️ Control y Estabilización (`src/control/`, `src/hardware/`)
*   **`control/MapeoVector256DaDendritas.ts`**: **Componente Crítico**. Extrae los subespacios D001-D056 del vector de entrada y los transforma en señales de control físico.
*   **`hardware/Simulador.ts`**: Implementa la física de los átomos.
    *   Método `configurarDendritas()`: Recibe las señales de control.
    *   Método `generarMuestra()`: Produce telemetría "estabilizada" (no aleatoria) basada en las dendritas.

### 🧠 Conexión Neuronal Distribuida (`src/colab/`, `src/neural/`)
*   **`colab/server.py`**: El "Cerebro Remoto". Script Python que corre en Google Colab con la arquitectura **CortezaCognitivaV2** (5 Capas: LSTM + Transformer + Asociativa + Ejecutiva).
*   **`neural/StreamingBridge.ts`**: Puente de datos. Gestiona el buffer y el envío eficiente (batching) de vectores 1600D a Colab.
*   **`neural/configColab.ts`**: Archivo de configuración donde se define la URL del túnel `ngrok`.

---

## 3. ⚙️ Funcionalidad y Flujo de Datos

El sistema opera en un bucle continuo de **Percepción-Estabilización-Aprendizaje**:

### Paso 1: Entrada (Vector 256D)
El sistema recibe o genera un vector de 256 dimensiones que representa el estado actual del mundo (datos sensoriales, criptográficos, ambientales, etc.).

### Paso 2: Estabilización Dendrítica
Antes de que los átomos "piensen", son estabilizados físicamente:
1.  `MapeoVector256DaDendritas` extrae los campos **D001-D056**.
2.  Estos valores se inyectan en los 25 simuladores (`atom.simulador.configurarDendritas`).
3.  Los átomos ajustan sus parámetros internos (voltaje, memoria, entropía) para alinearse con la entrada. **Esto evita que el sistema alucine sobre ruido aleatorio.**

### Paso 3: Procesamiento Distribuido (Capa 1)
Los 25 átomos (S1-S25) procesan su realidad local en paralelo:
1.  Generan telemetría estabilizada.
2.  Ejecutan inferencia ONNX local (`omega21_brain.onnx`).
3.  Producen un **Embedding Latente** (representación comprimida de su subespacio).

### Paso 4: Entrenamiento Cortical (Capas 2-5)
1.  Los embeddings de los 25 átomos se agregan en un **Vector Global (1600D)**.
2.  `StreamingBridge` envía este vector a la **Corteza Cognitiva** en Colab.
3.  La red en Colab procesa la información temporal y espacial, detecta anomalías complejas y devuelve **Ajustes de Dendritas** para el siguiente ciclo.

---

## 4. 🧬 Arquitectura de Átomos (Capa 1)

El entorno virtual despliega automáticamente 25 átomos especializados, mapeados a los subespacios del protocolo Omega 21:

| ID | Subespacio | Función Principal |
|----|------------|-------------------|
| **S1** | Criptografía | Seguridad base, Blockchain |
| **S2** | Fenomenología | Sensores físicos directos |
| **S3** | Histograma | Análisis estadístico rápido |
| **S4** | Streaming | Flujo de datos en tiempo real |
| ... | ... | ... |
| **S12** | Emocional | Modelo PAD (Placer-Activación-Dominancia) |
| **S25** | Membrana | Interfaz límite y Reservoir Computing |

Cada átomo mantiene su propia memoria y estado, pero comparte "firmas de anomalía" con otros átomos a través del **Protocolo de Infección** (LSH), permitiendo una inteligencia de enjambre.
