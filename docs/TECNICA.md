# Documentación Técnica - HIPERGRAFO

## 📖 Índice

1. [Conceptos Fundamentales](#conceptos-fundamentales)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Módulo Core](#módulo-core)
4. [Mapeo de Redes Neuronales](#mapeo-de-redes-neuronales)
5. [Persistencia](#persistencia)
6. [API Reference](#api-reference)

---

## Conceptos Fundamentales

### ¿Qué es un Hipergrafo?

Un **hipergrafo** es una generalización matemática de un grafo donde las aristas (ahora llamadas **hiperedges**) pueden conectar más de dos vértices simultáneamente.

**Definición formal:**
$$H = (V, E)$$

Donde:
- $V = \{v_1, v_2, ..., v_n\}$ es el conjunto de **vértices/nodos**
- $E = \{E_1, E_2, ..., E_m\}$ es el conjunto de **hiperedges**
- Cada hiperedge $E_i \subseteq V$ es un subconjunto no vacío de vértices

### Propiedades Matemáticas

#### 1. Grado de un Nodo
El grado de un nodo $v$ es la cantidad de hiperedges que lo contienen:
$$\deg(v) = |\{E \in E : v \in E\}|$$

#### 2. Tamaño de una Hiperedge
El tamaño (o rango) de una hiperedge $E$ es la cantidad de nodos que conecta:
$$|E_i| = \text{cardinalidad de } E_i$$

#### 3. Densidad del Hipergrafo
$$\rho(H) = \frac{|E|}{2^{|V|}}$$

Esta mide qué fracción de todos los posibles subconjuntos de $V$ forman hiperedges.

#### 4. Matriz de Incidencia
Matriz $M$ de dimensiones $|V| \times |E|$ donde:
$$M[i,j] = \begin{cases} 1 & \text{si } v_i \in E_j \\ 0 & \text{otherwise} \end{cases}$$

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────┐
│         APLICACIÓN DEL USUARIO          │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
    ┌───▼────────┐    ┌──────▼─────┐
    │ Construcción│    │  Mapeo de  │
    │   Manual    │    │   Redes    │
    │ de Hipergr. │    │ Neuronales │
    └───┬────────┘    └──────┬─────┘
        │                    │
        └──────────┬─────────┘
                   │
        ┌──────────▼──────────┐
        │  CORE HIPERGRAFO   │
        │  ┌────────────────┐│
        │  │  Hipergrafo    ││
        │  │  Nodo          ││
        │  │  Hiperedge     ││
        │  └────────────────┘│
        └──────────┬─────────┘
                   │
        ┌──────────▼──────────┐
        │   PERSISTENCIA     │
        │  ┌────────────────┐│
        │  │ Serialización  ││
        │  │ Almacenamiento ││
        │  │ Reportes       ││
        │  └────────────────┘│
        └────────────────────┘
```

---

## Módulo Core

### Clase Nodo

Representa un vértice en el hipergrafo.

```typescript
class Nodo {
  readonly id: string;              // UUID único
  readonly label: string;           // Etiqueta legible
  metadata: Record<string, any>;    // Datos adicionales
  
  constructor(label: string, metadata?: Record<string, any>)
  clone(): Nodo
}
```

**Propiedades Clave:**
- `id`: Identificador único (UUID v4)
- `label`: Nombre/etiqueta del nodo
- `metadata`: Diccionario flexible para datos personalizados

**Ejemplo:**
```typescript
const neurona = new Nodo('Neurona_42', {
  activacion: 0.85,
  sesgo: 0.12,
  tipo: 'oculta',
  capa: 3
});
```

### Clase Hiperedge

Representa una arista generalizada que conecta múltiples nodos.

```typescript
class Hiperedge {
  readonly id: string;
  readonly nodos: Set<string>;      // IDs de nodos
  readonly label: string;
  weight: number;                   // Peso de la hiperedge
  metadata: Record<string, any>;
  
  constructor(label: string, nodos?: Nodo[], weight?: number, metadata?: Record<string, any>)
  agregarNodo(nodo: Nodo): void
  removerNodo(nodoId: string): boolean
  contiene(nodoId: string): boolean
  grado(): number
}
```

**Propiedades Clave:**
- `nodos`: Conjunto de IDs de nodos que conecta
- `weight`: Peso asociado (ej: magnitud de conexión neuronal)
- `grado()`: Retorna cantidad de nodos conectados

**Ejemplo:**
```typescript
const conexion = new Hiperedge(
  'Conexion_Capa_1_a_2',
  [neurona1, neurona2, neurona3],
  0.45,
  { tipo: 'conexion', capaOrigen: 1 }
);
```

### Clase Hipergrafo

Estructura principal que contiene nodos e hiperedges.

```typescript
class Hipergrafo {
  label: string;
  
  // Gestión de nodos
  agregarNodo(nodo: Nodo): void
  agregarNodos(nodos: Nodo[]): void
  obtenerNodo(nodoId: string): Nodo | undefined
  obtenerNodos(): Nodo[]
  
  // Gestión de hiperedges
  agregarHiperedge(hiperedge: Hiperedge): void
  obtenerHiperedge(edgeId: string): Hiperedge | undefined
  obtenerHiperedges(): Hiperedge[]
  
  // Análisis
  cardinalV(): number                                    // |V|
  cardinalE(): number                                    // |E|
  calcularGradoNodo(nodoId: string): number
  obtenerHiperedgesDelNodo(nodoId: string): Hiperedge[]
  obtenerVecinos(nodoId: string): Nodo[]
  estaConectados(nodoId1: string, nodoId2: string): boolean
  
  // Operaciones matemáticas
  calcularMatrizIncidencia(): number[][]
  gradoPromedio(): number
  densidad(): number
  
  // Utilidades
  clone(): Hipergrafo
  limpiar(): void
}
```

---

## Mapeo de Redes Neuronales

### Estrategia de Mapeo

El mapeo convierte una red neuronal a un hipergrafo siguiendo estas reglas:

#### Regla 1: Neuronas → Nodos
Cada neurona se convierte en un nodo con metadata sobre su estado:
```
Neurona i → Nodo con label "Neurona_i"
           metadata: {
             idNeurona: i,
             activacion: valor ∈ [0, 1],
             sesgo: valor,
             activa: activacion ≥ umbralActivacion
           }
```

#### Regla 2: Conexiones Significativas → Hiperedges
Las conexiones de peso superior a un umbral forman hiperedges:
```
Si |peso[de][a]| ≥ umbralPeso:
  Crear hiperedge conectando neurona_de y neurona_a
  weight = |peso[de][a]|
```

#### Regla 3: Capas → Hiperedges Especiales
Opcionalmente, agrupamos neuronas por capa:
```
Para cada capa en la red:
  Crear hiperedge que conecta todas las neuronas de esa capa
  label: "Capa_i"
  metadata: { tipoHiperedge: 'capa', capaIdx: i }
```

#### Regla 4: Patrones de Activación → Hiperedges Dinámicas
Detectamos grupos de neuronas con activación similar:
```
Agrupar neuronas por banda de activación:
  - inactiva: activacion < 0.2
  - baja: 0.2 ≤ activacion < 0.4
  - media: 0.4 ≤ activacion < 0.6
  - alta: 0.6 ≤ activacion < 0.8
  - muy_alta: activacion ≥ 0.8

Si grupo.size ≥ tamanoMinimoPatron:
  Crear hiperedge con label "Patron_Activacion_[banda]"
```

### Configuración de Mapeo

```typescript
interface ConfiguracionMapeo {
  umbralPeso: number;              // [0, 1] - Conexión significativa
  umbralActivacion: number;        // [0, 1] - Neurona "activa"
  agruparPorCapas: boolean;        // Crear hiperedges de capas
  detectarPatrones: boolean;       // Detectar patrones
  tamanoMinimoPatron: number;      // Min neuronas en patrón
  incluirPesos: boolean;           // Incluir pesos en metadata
}
```

**Valores por Defecto:**
```typescript
{
  umbralPeso: 0.1,
  umbralActivacion: 0.5,
  agruparPorCapas: true,
  detectarPatrones: true,
  tamanoMinimoPatron: 3,
  incluirPesos: true
}
```

### Ejemplo de Mapeo

```typescript
import { MapeoRedNeuronalAHipergrafo } from 'hipergrafo';

const redNeuronal = {
  neuronas: [
    { id: 0, activacion: 0.9, sesgo: 0.1 },
    { id: 1, activacion: 0.3, sesgo: 0.2 },
    // ...
  ],
  pesos: [/* matriz 3D */],
  capas: [1024, 512, 256]
};

const mapeador = new MapeoRedNeuronalAHipergrafo({
  umbralPeso: 0.15,
  detectarPatrones: true
});

const hipergrafo = mapeador.mapear(redNeuronal);
```

---

## Persistencia

### Servicio de Persistencia

Maneja serialización y deserialización.

```typescript
class ServicioPersistencia {
  serializarAJSON(hipergrafo: Hipergrafo): string
  deserializarDesdeJSON(jsonString: string): Hipergrafo
  calcularHash(hipergrafo: Hipergrafo): string
  generarReporte(hipergrafo: Hipergrafo): Record<string, any>
}
```

**Formato JSON:**
```json
{
  "label": "Mi Hipergrafo",
  "nodos": [
    {
      "id": "uuid-1",
      "label": "Neurona_1",
      "metadata": { "activacion": 0.8 }
    }
  ],
  "hiperedges": [
    {
      "id": "edge-1",
      "label": "Conexion_1",
      "nodosIds": ["uuid-1", "uuid-2"],
      "weight": 0.45,
      "metadata": {}
    }
  ],
  "version": "1.0.0",
  "fechaCreacion": "2025-12-20T..."
}
```

### Gestor de Almacenamiento

Maneja archivos en disco.

```typescript
class GestorAlmacenamiento {
  constructor(directorio: string = './hipergrafos')
  
  guardarHipergrafo(hipergrafo: Hipergrafo, nombre: string): string
  cargarHipergrafo(nombre: string): Hipergrafo
  listarHipergrafos(): string[]
  eliminarHipergrafo(nombre: string): void
  exportarACSV(hipergrafo: Hipergrafo, nombre: string): string
  obtenerInfoArchivo(nombre: string): Record<string, any>
}
```

---

## API Reference

### Métodos Principales

#### Hipergrafo

| Método | Descripción | Retorna |
|--------|-------------|---------|
| `cardinalV()` | Cantidad de nodos | `number` |
| `cardinalE()` | Cantidad de hiperedges | `number` |
| `gradoPromedio()` | Grado promedio de nodos | `number` |
| `densidad()` | Densidad del hipergrafo | `number` |
| `calcularMatrizIncidencia()` | Matriz incidencia M[i,j] | `number[][]` |
| `obtenerVecinos(nodoId)` | Nodos conectados a uno | `Nodo[]` |

#### MapeoRedNeuronalAHipergrafo

| Método | Descripción | Retorna |
|--------|-------------|---------|
| `mapear(redNeuronal)` | Mapea red neuronal | `Hipergrafo` |
| `actualizarConfiguracion(config)` | Actualiza parámetros | `void` |
| `obtenerConfiguracion()` | Obtiene configuración actual | `ConfiguracionMapeo` |

#### ServicioPersistencia

| Método | Descripción | Retorna |
|--------|-------------|---------|
| `serializarAJSON(hg)` | Convierte a JSON | `string` |
| `deserializarDesdeJSON(json)` | Convierte desde JSON | `Hipergrafo` |
| `calcularHash(hg)` | Hash de verificación | `string` |
| `generarReporte(hg)` | Estadísticas | `Record<string, any>` |

---

**Última actualización:** Diciembre 2025
