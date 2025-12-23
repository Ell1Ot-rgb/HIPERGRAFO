# HIPERGRAFO

**Sistema riguroso de mapeo de redes hipergráficas para generar hipergrafos persistentes a partir de redes neuronales**

[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Visión del Proyecto

HIPERGRAFO es un sistema que implementa con **rigor teórico** el mapeo de redes neuronales (1024 neuronas) a estructuras de **hipergrafos persistentes**. El proyecto busca capturar la complejidad de las redes neuronales mediante la teoría de hipergrafos, donde las relaciones pueden ser de orden superior (no solo conexiones binarias).

### Concepto Fundamental

Un **hipergrafo** $H = (V, E)$ es una generalización de un grafo donde:
- $V$ es el conjunto de **nodos**
- $E$ es el conjunto de **hiperedges** (cada hiperedge es un subconjunto de $V$)
- Una hiperedge puede conectar **más de dos nodos** simultáneamente

## 🏗️ Arquitectura del Proyecto

```
src/
├── core/                  # Abstracciones fundamentales
│   ├── Nodo.ts           # Clase que representa nodos
│   ├── Hiperedge.ts      # Clase que representa hiperedges
│   └── Hipergrafo.ts     # Clase principal del hipergrafo
│
├── neural/               # Mapeo de redes neuronales
│   ├── tipos.ts          # Definiciones de tipos y configuración
│   └── MapeoRedNeuronalAHipergrafo.ts
│
├── persistencia/         # Almacenamiento y recuperación
│   ├── ServicioPersistencia.ts
│   └── GestorAlmacenamiento.ts
│
└── index.ts             # Punto de entrada
```

## 🚀 Instalación y Configuración

### Requisitos
- Node.js 18+
- npm o yarn

### Pasos de Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd HIPERGRAFO

# Instalar dependencias
npm install

# Compilar TypeScript
npm run build

# Ejecutar pruebas
npm test

# Modo desarrollo con watch
npm run dev
```

## 📚 Uso Básico

### 1. Crear un Hipergrafo Manual

```typescript
import { Hipergrafo, Nodo, Hiperedge } from './src';

// Crear hipergrafo
const hg = new Hipergrafo('Mi Hipergrafo');

// Crear nodos
const n1 = new Nodo('Neurona_1', { activacion: 0.8 });
const n2 = new Nodo('Neurona_2', { activacion: 0.6 });
const n3 = new Nodo('Neurona_3', { activacion: 0.9 });

// Agregar nodos
hg.agregarNodos([n1, n2, n3]);

// Crear hiperedge (conecta múltiples nodos)
const edge = new Hiperedge('Activacion_Alta', [n1, n3], 1.0);

// Agregar hiperedge
hg.agregarHiperedge(edge);

// Consultar propiedades
console.log(`Nodos: ${hg.cardinalV()}`);       // 3
console.log(`Hiperedges: ${hg.cardinalE()}`);   // 1
console.log(`Grado promedio: ${hg.gradoPromedio()}`);
```

### 2. Mapear una Red Neuronal a Hipergrafo

```typescript
import { MapeoRedNeuronalAHipergrafo } from './src';

// Definir red neuronal con 1024 neuronas
const redNeuronal = {
  neuronas: [
    { id: 0, activacion: 0.8, sesgo: 0.1 },
    { id: 1, activacion: 0.6, sesgo: 0.2 },
    // ... 1022 neuronas más
  ],
  pesos: [/* matriz 3D de pesos */],
  capas: [1024, 512, 256],  // Arquitectura de la red
};

// Crear mapeador con configuración personalizada
const mapeador = new MapeoRedNeuronalAHipergrafo({
  umbralPeso: 0.1,
  umbralActivacion: 0.5,
  agruparPorCapas: true,
  detectarPatrones: true,
  tamanoMinimoPatron: 5
});

// Mapear red neuronal a hipergrafo
const hipergrafo = mapeador.mapear(redNeuronal);

console.log(`Hipergrafo creado con ${hipergrafo.cardinalV()} nodos`);
console.log(`Densidad: ${hipergrafo.densidad()}`);
```

### 3. Persistencia de Hipergrafos

```typescript
import { GestorAlmacenamiento } from './src';

// Crear gestor de almacenamiento
const gestor = new GestorAlmacenamiento('./hipergrafos');

// Guardar hipergrafo
gestor.guardarHipergrafo(hipergrafo, 'red_neuronal_1024');

// Cargar hipergrafo
const hipergrafoCargado = gestor.cargarHipergrafo('red_neuronal_1024');

// Listar hipergrafos guardados
const lista = gestor.listarHipergrafos();
console.log('Hipergrafos disponibles:', lista);

// Exportar a CSV para análisis
gestor.exportarACSV(hipergrafo, 'analisis_red_neuronal');
```

## 🔬 Operaciones Matemáticas

El proyecto implementa operaciones rigurosas sobre hipergrafos:

### Operaciones de Nodos

```typescript
// Grado de un nodo (cantidad de hiperedges que lo contienen)
const grado = hipergrafo.calcularGradoNodo(nodo.id);

// Vecinos de un nodo
const vecinos = hipergrafo.obtenerVecinos(nodo.id);

// Hiperedges incidentes
const hiperedges = hipergrafo.obtenerHiperedgesDelNodo(nodo.id);
```

### Operaciones de Hipergrafo

```typescript
// Cardinalidad
const V = hipergrafo.cardinalV();  // Número de nodos
const E = hipergrafo.cardinalE();  // Número de hiperedges

// Densidad del hipergrafo
const densidad = hipergrafo.densidad();

// Grado promedio
const gradoPromedio = hipergrafo.gradoPromedio();

// Matriz de incidencia M[i,j] = 1 si nodo i está en hiperedge j
const matrizIncidencia = hipergrafo.calcularMatrizIncidencia();

// Verificar conectividad
const conectados = hipergrafo.estaConectados(nodo1.id, nodo2.id);
```

### Análisis Avanzado (Fase 3)

#### Dualidad del Hipergrafo

```typescript
import { DualidadHipergrafo } from 'hipergrafo/analisis';

// Calcular el hipergrafo dual H* = (V*, E*)
const dual = DualidadHipergrafo.calcularDual(hipergrafo);

// Verificar si es autodual (H ≅ H*)
const esAutodual = DualidadHipergrafo.esAutodual(hipergrafo);

// Calcular período hasta convergencia
const periodo = DualidadHipergrafo.calcularPeriodoDualidad(hipergrafo);
```

#### Centralidad de Nodos

```typescript
import { CentralidadHipergrafo } from 'hipergrafo/analisis';

// Centralidad de grado (normalizada)
const cent = CentralidadHipergrafo.centralidadGrado(hipergrafo, nodoId);

// Centralidad ponderada (suma de pesos)
const centPond = CentralidadHipergrafo.centralidadPonderada(hipergrafo, nodoId);

// Betweenness centrality
const centBetween = CentralidadHipergrafo.centralidadBetweenness(hipergrafo, nodoId);

// Eigenvector centrality
const eigencents = CentralidadHipergrafo.centralidadEigenvector(hipergrafo);

// Ranking por tipo de centralidad
const ranking = CentralidadHipergrafo.rankingPorCentralidad(hipergrafo, 'grado');
```

#### Coeficiente de Clustering

```typescript
import { ClusteringHipergrafo } from 'hipergrafo/analisis';

// Clustering local (para un nodo)
const clust = ClusteringHipergrafo.coeficienteClusteringLocal(hipergrafo, nodoId);

// Clustering global (transitivity)
const clustGlobal = ClusteringHipergrafo.coeficienteClusteringGlobal(hipergrafo);

// Clustering promedio
const clustPromedio = ClusteringHipergrafo.coeficienteClusteringPromedio(hipergrafo);

// Índice de homofilia
const homofilia = ClusteringHipergrafo.indiceHomofilia(hipergrafo, 'atributo');

// Modularidad (para particiones)
const mod = ClusteringHipergrafo.calcularModularidad(hipergrafo, particion);
```

#### Propiedades Espectrales

```typescript
import { PropiedadesEspectrales } from 'hipergrafo/analisis';

// Matriz de adyacencia
const A = PropiedadesEspectrales.calcularMatrizAdyacencia(hipergrafo);

// Matriz de grados
const D = PropiedadesEspectrales.calcularMatrizGrados(hipergrafo);

// Matriz Laplaciana normalizada
const L = PropiedadesEspectrales.calcularMatrizLaplacianaNormalizada(hipergrafo);

// Energía espectral
const energia = PropiedadesEspectrales.calcularEnergiaEspectral(hipergrafo);

// Spectral gap (algebraic connectivity)
const gap = PropiedadesEspectrales.calcularGapEspectral(hipergrafo);

// Índice de Wiener espectral
const wiener = PropiedadesEspectrales.indiceWienerEspectral(hipergrafo);
```

## 🧪 Pruebas

El proyecto incluye suite completa de pruebas:

```bash
# Ejecutar todas las pruebas
npm test

# Modo watch
npm test -- --watch

# Con cobertura
npm test -- --coverage
```

### Áreas de Cobertura

- ✅ Operaciones básicas de Nodo y Hiperedge
- ✅ Construcción y manipulación de Hipergrafos
- ✅ Mapeo de redes neuronales
- ✅ Persistencia y serialización
- ✅ Análisis de propiedades matemáticas
- ✅ **Fase 3**: Dualidad, Centralidad, Clustering, Propiedades Espectrales

### Estado de Desarrollo

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | Fundamentos y Core | ✅ Completada |
| 2 | Mapeo y Persistencia | ✅ Completada |
| 3 | Rigor Teórico Avanzado | ✅ Completada |
| 4 | Herramientas y Visualización | ⏳ En cola |
| 5 | Integración y Escala | ⏳ En cola |

## 🔧 Configuración Avanzada

### Parámetros de Mapeo

```typescript
interface ConfiguracionMapeo {
  umbralPeso: number;           // [0, 1] - Umbral para conexiones significativas
  umbralActivacion: number;     // [0, 1] - Umbral para neurona "activa"
  agruparPorCapas: boolean;     // Crear hiperedges por capa
  detectarPatrones: boolean;    // Detectar patrones de activación
  tamanoMinimoPatron: number;   // Mínimo de neuronas para un patrón
  incluirPesos: boolean;        // Incluir pesos en metadata
}
```

## 📊 Análisis y Reportesarchivo

```typescript
const servicio = new ServicioPersistencia();

// Generar reporte de estadísticas
const reporte = servicio.generarReporte(hipergrafo);

console.log(reporte);
// {
//   label: 'Hipergrafo de Red Neuronal (1024 neuronas)',
//   cardinalV: 1024,
//   cardinalE: 5432,
//   gradoPromedio: 3.45,
//   densidad: 0.0023,
//   nodos: [...],
//   hiperedges: [...]
// }

// Calcular hash para validación
const hash = servicio.calcularHash(hipergrafo);
```

## 🎓 Fundamentos Teóricos

### Teoría de Hipergrafos

Un hipergrafo generaliza los grafos permitiendo aristas que conecten más de dos vértices.

**Definición formal:**
$$H = (V, E) \text{ donde } E = \{E_1, E_2, ..., E_m\}, E_i \subseteq V$$

**Grado de un vértice:**
$$\deg(v) = |\{E \in E : v \in E\}|$$

**Densidad:**
$$\rho(H) = \frac{|E|}{2^{|V|}}$$

### Mapeo de Redes Neuronales

La estrategia de mapeo es:

1. **Cada neurona → Nodo** con metadata (activación, sesgo)
2. **Conexiones significativas → Hiperedges** con peso = magnitud de conexión
3. **Capas → Hiperedges especiales** que agrupan neuronas por nivel
4. **Patrones de activación → Hiperedges dinámicas** detectadas por análisis

## 📈 Ejemplo Completo: Red de 1024 Neuronas

```typescript
import { MapeoRedNeuronalAHipergrafo, GestorAlmacenamiento } from './src';

// Crear red neuronal de 1024 neuronas
const redNeuronal = {
  neuronas: Array.from({ length: 1024 }, (_, i) => ({
    id: i,
    activacion: Math.random(),
    sesgo: Math.random() * 0.5
  })),
  pesos: crearMatrizPesos(1024),
  capas: [1024, 512, 256],
  metadata: {
    arquitectura: 'MLP',
    epocas_entrenamiento: 100,
    dataset: 'ImageNet'
  }
};

// Mapear a hipergrafo
const mapeador = new MapeoRedNeuronalAHipergrafo({
  umbralPeso: 0.15,
  detectarPatrones: true,
  agruparPorCapas: true
});

const hipergrafo = mapeador.mapear(redNeuronal);

// Persistir
const gestor = new GestorAlmacenamiento('./resultados');
gestor.guardarHipergrafo(hipergrafo, 'red_1024_entrenada');

// Análisis
const stats = gestor.obtenerInfoArchivo('red_1024_entrenada');
console.log(`Hipergrafo persistido: ${stats.tamanio} bytes`);
```

## 🤝 Contribuciones

Este proyecto está en fase temprana y aceptamos contribuciones en:

- Optimizaciones del mapeo neuronal
- Nuevas operaciones matemáticas
- Formatos de persistencia adicionales (GraphML, GExf)
- Visualización de hipergrafos
- Análisis de propiedades espectrales

## 📝 Licencia

MIT - Ver archivo [LICENSE](LICENSE)

## 👨‍💻 Autor

Ell1Ot-rgb

## 🔗 Referencias

- [Hypergraph Theory - Wikipedia](https://en.wikipedia.org/wiki/Hypergraph)
- [Neural Networks and Graph Theory](https://arxiv.org/search/)
- Documentación completa en `docs/`

---

**Última actualización:** Diciembre 2025

