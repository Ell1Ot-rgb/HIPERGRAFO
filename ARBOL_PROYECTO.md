# Árbol del Proyecto HIPERGRAFO

```
HIPERGRAFO/
│
├── 📄 Archivos de Configuración
│   ├── package.json              # Dependencias y scripts
│   ├── tsconfig.json             # Configuración TypeScript
│   ├── jest.config.js            # Configuración de pruebas
│   ├── .eslintrc.json            # Linter configuration
│   └── .gitignore                # Git ignore
│
├── 📚 Documentación Principal
│   ├── README.md                 # Guía completa de usuario
│   ├── ESTADO_PROYECTO.md        # Estado actual del proyecto
│   ├── FASE3_RESUMEN.md          # Resumen de Fase 3
│   └── ARBOL_PROYECTO.md         # Este archivo
│
├── 📖 Documentación Técnica
│   └── docs/
│       ├── TECNICA.md            # API Reference detallada
│       └── FASE3_MATEMATICA.md   # Derivaciones matemáticas
│
├── 🧪 Ejemplos Ejecutables
│   ├── ejemplo.ts                # Ejemplo básico (100 neuronas)
│   └── ejemplo_fase3.ts          # Ejemplo avanzado con análisis
│
├── 💻 Código Fuente (src/)
│   │
│   ├── 🔧 CORE - Abstracciones Fundamentales
│   │   ├── Nodo.ts               # Vértices con metadata
│   │   ├── Hiperedge.ts          # Aristas generalizadas
│   │   ├── Hipergrafo.ts         # Estructura principal
│   │   └── index.ts              # Exportaciones
│   │   
│   │   📊 Operaciones:
│   │   • Crear/obtener nodos
│   │   • Crear/obtener hiperedges
│   │   • Calcular grado
│   │   • Matriz de incidencia
│   │   • Vecinos y conectividad
│   │
│   ├── 🧠 NEURAL - Mapeo de Redes Neuronales
│   │   ├── tipos.ts              # Tipos e interfaces
│   │   ├── MapeoRedNeuronalAHipergrafo.ts
│   │   └── index.ts              # Exportaciones
│   │   
│   │   📊 Funcionalidades:
│   │   • Mapeo de 1024 neuronas → Hipergrafo
│   │   • Detección de patrones
│   │   • Agrupación por capas
│   │   • Configuración flexible
│   │
│   ├── 💾 PERSISTENCIA - Almacenamiento y E/S
│   │   ├── ServicioPersistencia.ts
│   │   ├── GestorAlmacenamiento.ts
│   │   └── index.ts              # Exportaciones
│   │   
│   │   📊 Operaciones:
│   │   • Serializar a JSON
│   │   • Deserializar desde JSON
│   │   • Guardar/Cargar archivos
│   │   • Exportar a CSV
│   │   • Reportes estadísticos
│   │
│   ├── 📐 ANALISIS - Rigor Teórico Avanzado
│   │   ├── DualidadHipergrafo.ts
│   │   ├── CentralidadHipergrafo.ts
│   │   ├── ClusteringHipergrafo.ts
│   │   ├── PropiedadesEspectrales.ts
│   │   └── index.ts              # Exportaciones
│   │   
│   │   📊 Módulos:
│   │   
│   │   🔄 Dualidad
│   │   • Calcular dual H*
│   │   • Verificar autoduales
│   │   • Período de dualidad
│   │   
│   │   ⭐ Centralidad (5 métricas)
│   │   • Grado
│   │   • Ponderada
│   │   • Betweenness
│   │   • Eigenvector
│   │   • Closeness
│   │   
│   │   🔗 Clustering (5 métricas)
│   │   • Clustering local
│   │   • Clustering global
│   │   • Clustering promedio
│   │   • Homofilia
│   │   • Modularidad
│   │   
│   │   📊 Propiedades Espectrales
│   │   • Matriz de adyacencia
│   │   • Matriz de grados
│   │   • Matriz Laplaciana
│   │   • Energía espectral
│   │   • Spectral gap
│   │   • Índice de Wiener
│   │
│   ├── 🧪 PRUEBAS (src/__tests__/)
│   │   ├── Hipergrafo.test.ts           # 8 tests
│   │   ├── MapeoRedNeuronal.test.ts     # 5 tests
│   │   ├── Persistencia.test.ts         # 7 tests
│   │   └── AnalisisAvanzado.test.ts    # 16 tests
│   │   
│   │   ✅ Total: 36 pruebas unitarias
│   │
│   └── index.ts                  # Punto de entrada principal
│
├── 📦 Archivos Generados (no versionados)
│   ├── dist/                     # Salida compilada
│   └── node_modules/             # Dependencias
│
└── 📝 Licencia
    └── MIT
```

## 📊 Estadísticas por Módulo

### Core (~450 LOC)
- **Clases**: 3 (Nodo, Hiperedge, Hipergrafo)
- **Métodos públicos**: 25+
- **Pruebas**: 8
- **Responsabilidad**: Abstracciones fundamentales

### Neural (~200 LOC)
- **Clases**: 1 (MapeoRedNeuronalAHipergrafo)
- **Estrategias**: 4 (conexiones, capas, patrones, dualidad)
- **Pruebas**: 5
- **Responsabilidad**: Mapeo de redes neuronales

### Persistencia (~180 LOC)
- **Clases**: 2 (ServicioPersistencia, GestorAlmacenamiento)
- **Formatos**: 2 (JSON, CSV)
- **Pruebas**: 7
- **Responsabilidad**: Almacenamiento y reportes

### Análisis (~550 LOC)
- **Clases**: 4 (Dualidad, Centralidad, Clustering, Espectral)
- **Métricas**: 16+
- **Pruebas**: 16
- **Responsabilidad**: Análisis avanzado y rigor matemático

## 🎯 Flujos de Trabajo Principales

### Flujo 1: Crear Hipergrafo Manual
```
Crear Nodos
    ↓
Agregar Nodos a Hipergrafo
    ↓
Crear Hiperedges
    ↓
Agregar Hiperedges a Hipergrafo
    ↓
Consultar Propiedades
    ↓
Analizar (Dualidad, Centralidad, etc.)
```

### Flujo 2: Mapear Red Neuronal
```
Crear RedNeuronal (1024 neuronas)
    ↓
Instanciar MapeoRedNeuronalAHipergrafo
    ↓
Ejecutar mapeo.mapear()
    ↓
Hipergrafo generado con:
    • Nodos = Neuronas
    • Hiperedges = Conexiones + Patrones + Capas
    ↓
Persistir o Analizar
```

### Flujo 3: Análisis Completo
```
Hipergrafo
    ├─→ DualidadHipergrafo
    ├─→ CentralidadHipergrafo
    ├─→ ClusteringHipergrafo
    └─→ PropiedadesEspectrales
         ↓
    Generación de Reporte
```

## 🔗 Dependencias

### Producción
```json
{
  "uuid": "^9.0.0"  // Generación de IDs únicos
}
```

### Desarrollo
```json
{
  "typescript": "^5.0.0",
  "@types/node": "^20.0.0",
  "@types/jest": "^29.5.0",
  "jest": "^29.5.0",
  "ts-jest": "^29.1.0",
  "eslint": "^8.0.0",
  "@typescript-eslint/parser": "^6.0.0",
  "@typescript-eslint/eslint-plugin": "^6.0.0"
}
```

## 📈 Camino de Crecimiento

```
Fase 1 (Core)           ✅
    ↓
Fase 2 (Mapeo + Persist) ✅
    ↓
Fase 3 (Análisis Avanz.)  ✅ ← ACTUAL
    ↓
Fase 4 (CLI + Visualiz.)  ⏳ (próxima)
    ↓
Fase 5 (Escala + Integr.) ⏳
```

## 🚀 Cómo Usar Este Árbol

1. **Para entender la estructura**: Ver los módulos principales
2. **Para encontrar una función**: Buscar en el módulo correspondiente
3. **Para agregar una característica**: Crear en el módulo apropiado
4. **Para escribir pruebas**: Crear en `src/__tests__/`
5. **Para documentar**: Actualizar en `docs/`

## ✨ Puntos Clave

- ✅ Código completamente tipado (TypeScript)
- ✅ Pruebas exhaustivas (36/36 passing)
- ✅ Sin dependencias complejas (solo uuid)
- ✅ Documentación en 3 niveles: README, TECNICA, MATEMATICA
- ✅ Ejemplos funcionales ejecutables
- ✅ Linting y formateo automático

---

**Generado**: 20 de Diciembre, 2025  
**Versión**: 0.1.0  
**Fase**: 3/5 Completada ✅
