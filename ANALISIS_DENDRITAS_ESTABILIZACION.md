# ANÁLISIS COMPLETO: FLUJO DE DENDRITAS Y ESTABILIZACIÓN DE ÁTOMOS

## 🎯 PROBLEMA IDENTIFICADO

**Las dendritas NO están recibiendo los campos D001-D056 del vector 256D para estabilizar los átomos antes de generar datos de entrenamiento.**

---

## 📊 ARQUITECTURA REAL DEL SISTEMA

### 1. Vector 256D de Entrada

```
┌─────────────────────────────────────────────────────────┐
│                    VECTOR 256D (Entrada)                │
│                                                         │
│  D001-D016 → S1 (Criptografía)          16 dims        │
│  D017-D032 → S2 (Fenomenología)         16 dims        │
│  D033-D048 → S3 (Histograma)            16 dims        │
│  D049-D056 → S4 (Streaming)              8 dims        │
│  D057-D072 → S5 (Seguridad)             16 dims        │
│  D073-D080 → S6 (Análisis Relacional)    8 dims        │
│  ...                                                    │
│  D241-D256 → S25 (Membrana/Reservoir)   16 dims        │
└─────────────────────────────────────────────────────────┘
         │
         │
         ├──────────┬──────────────────────────────────┐
         │          │                                  │
         ▼          ▼                                  ▼
   DENDRITAS    ÁTOMOS S1-S25              ENTRENAMIENTO COLAB
   (D001-D056)  (Capa 1)                   (Capa 2-5)
```

### 2. Las 16 Dendritas y su Rol

**CRÍTICO:** Las dendritas reciben un **SUBSET** del vector 256D (específicamente D001-D056, los primeros 56 campos) para **ESTABILIZAR** el átomo antes de que procese la telemetría completa.

#### 2.1 Definición de Dendritas (Schema.ts)

```typescript
export const DENDRITE_DEFINITIONS = [
  { id: 'D001', nombre: 'Voltage', tipo: 'electrico', rango: [0, 100] },
  { id: 'D002', nombre: 'Current', tipo: 'electrico', rango: [0, 1000] },
  { id: 'D003', nombre: 'Power', tipo: 'electrico', rango: [0, 5000] },
  { id: 'D004', nombre: 'Altitude', tipo: 'ambiental', rango: [0, 2000] },
  { id: 'D005', nombre: 'DewTemp', tipo: 'ambiental', rango: [-40, 60] },
  { id: 'D006', nombre: 'Velocity', tipo: 'mecanico', rango: [0, 200] },
  { id: 'D007', nombre: 'Phase', tipo: 'onda', rango: [0, 360] },
  { id: 'D008', nombre: 'Frequency', tipo: 'onda', rango: [0, 200] },
  { id: 'D009', nombre: 'Delay', tipo: 'temporal', rango: [0, 1000] },
  { id: 'D010', nombre: 'Memory', tipo: 'cognitivo', rango: [0, 1] },
  { id: 'D011', nombre: 'Decay', tipo: 'temporal', rango: [0, 1] },
  { id: 'D012', nombre: 'Michaelis', tipo: 'bioquimico', rango: [0, 100] },
  { id: 'D013', nombre: 'Hill', tipo: 'bioquimico', rango: [0.5, 4] },
  { id: 'D014', nombre: 'Capacitor', tipo: 'electrico', rango: [0, 100] },
  { id: 'D015', nombre: 'Entropy', tipo: 'informacional', rango: [0, 8] },
  { id: 'D016', nombre: 'Soma', tipo: 'neuronal', rango: [-90, 40] }
];
```

**TOTAL: 16 Dendritas**

#### 2.2 Datos de Telemetría Actuales

En `Omega21Telemetry`, las dendritas reciben **11 valores** (no 16):

```typescript
dendrites: {
    voltage: number;      // D001
    current: number;      // D002
    power: number;        // D003
    altitude: number;     // D004
    dew_temp: number;     // D005
    velocity: number;     // D006
    phase: number;        // D007
    freq: number;         // D008
    soma_v: number;       // D016
    spike: number;        // ??
    loss: number;         // ??
}
```

**PROBLEMA 1:** Faltan D009-D015 (Delay, Memory, Decay, Michaelis, Hill, Capacitor, Entropy)

**PROBLEMA 2:** Las dendritas están recibiendo datos simulados aleatorios, NO del vector 256D real

---

## 🔄 FLUJO CORRECTO (Como Debería Ser)

### Fase 1: Extracción de Datos para Dendritas

```typescript
// Vector 256D entrada (desde métricas físicas reales)
const vector256D: Vector256D = {
    D001: 45.2,    // Voltage
    D002: 520.7,   // Current
    D003: 2340,    // Power
    D004: 500,     // Altitude
    D005: 18.3,    // DewTemp
    D006: 85.4,    // Velocity
    D007: 120,     // Phase
    D008: 87.2,    // Frequency
    D009: 245,     // Delay
    D010: 0.67,    // Memory
    D011: 0.12,    // Decay
    D012: 42.1,    // Michaelis
    D013: 2.3,     // Hill
    D014: 68.5,    // Capacitor
    D015: 3.2,     // Entropy
    D016: -68.4,   // Soma_V
    // ... D017-D056 (resto de los primeros 56 campos)
    // ... D057-D256 (resto del vector)
};

// Extraer SOLO los primeros 56 campos para las dendritas
const camposDendritas = extraerCampos(vector256D, 1, 56); // D001-D056
```

### Fase 2: Estabilización del Átomo

```typescript
// 1. Aplicar valores dendríticos al átomo
dendriteController.aplicarValoresDesdVector256D(camposDendritas);

// 2. Ajustar parámetros del simulador/hardware
simulador.ajustarParametrosDendritas(dendriteController.getEstadoActual());

// 3. Esperar estabilización (varios ciclos de simulación)
await simulador.estabilizar(ciclos: 100);
```

### Fase 3: Generación de Datos del Átomo Estabilizado

```typescript
// 4. Ahora el átomo está estabilizado, generar telemetría REAL
const telemetria = simulador.generarMuestra();

// 5. Pasar telemetría completa al átomo ONNX
const inferencia = await atom.percibir(telemetria);

// 6. El átomo genera embeddings estables (64D)
const embedding = inferencia.estadoOculto; // [64]
```

### Fase 4: Envío a Colab para Entrenamiento

```typescript
// 7. Procesar embedding a través de 25 subespacios → 1600D
const vector1600D = await procesarPor25Subespacios(embedding, vector256D);

// 8. Enviar a Colab para entrenar Capa 2-5
await streamingBridge.enviarVector(vector1600D);
```

---

## ❌ FLUJO ACTUAL (Incorrecto)

### Qué está pasando ahora:

```typescript
// 1. Simulador genera datos ALEATORIOS sin considerar vector 256D
const telemetria = simulador.generarMuestra(); 
// dendrites: { voltage: random(), current: random(), ... }

// 2. Átomo procesa telemetría aleatoria SIN estabilización dendrítica
const inferencia = await atom.percibir(telemetria);

// 3. Se genera embedding inestable/aleatorio
const embedding = inferencia.estadoOculto;

// 4. Se envía a Colab datos basados en ruido, no en física real
```

**RESULTADO:** Los átomos generan datos aleatorios, NO datos físicos reales estabilizados por dendritas.

---

## 🛠️ SOLUCIÓN PROPUESTA

### 1. Crear Interfaz para Mapear Vector256D → Dendritas

```typescript
// src/control/MapeoVector256DaDendritas.ts

export interface EstadoDendritico {
    D001_voltage: number;
    D002_current: number;
    D003_power: number;
    D004_altitude: number;
    D005_dewTemp: number;
    D006_velocity: number;
    D007_phase: number;
    D008_frequency: number;
    D009_delay: number;
    D010_memory: number;
    D011_decay: number;
    D012_michaelis: number;
    D013_hill: number;
    D014_capacitor: number;
    D015_entropy: number;
    D016_soma_v: number;
    // ... D017-D056 (otros 40 campos para estabilización extendida)
}

export class MapeoVector256DaDendritas {
    /**
     * Extrae los campos D001-D056 del vector 256D para las dendritas
     */
    extraerCamposDendriticos(vector256D: Vector256D): EstadoDendritico {
        return {
            D001_voltage: vector256D.D001,
            D002_current: vector256D.D002,
            D003_power: vector256D.D003,
            D004_altitude: vector256D.D004,
            D005_dewTemp: vector256D.D005,
            D006_velocity: vector256D.D006,
            D007_phase: vector256D.D007,
            D008_frequency: vector256D.D008,
            D009_delay: vector256D.D009 || 0,
            D010_memory: vector256D.D010 || 0,
            D011_decay: vector256D.D011 || 0,
            D012_michaelis: vector256D.D012 || 0,
            D013_hill: vector256D.D013 || 0,
            D014_capacitor: vector256D.D014 || 0,
            D015_entropy: vector256D.D015 || 0,
            D016_soma_v: vector256D.D016 || -70,
            // ... extraer D017-D056
        };
    }

    /**
     * Convierte estado dendrítico a formato de telemetría
     */
    convertirATelemetriaDendritas(estado: EstadoDendritico): any {
        return {
            voltage: estado.D001_voltage,
            current: estado.D002_current,
            power: estado.D003_power,
            altitude: estado.D004_altitude,
            dew_temp: estado.D005_dewTemp,
            velocity: estado.D006_velocity,
            phase: estado.D007_phase,
            freq: estado.D008_frequency,
            // AGREGAR los campos faltantes:
            delay: estado.D009_delay,
            memory: estado.D010_memory,
            decay: estado.D011_decay,
            michaelis: estado.D012_michaelis,
            hill: estado.D013_hill,
            capacitor: estado.D014_capacitor,
            entropy: estado.D015_entropy,
            soma_v: estado.D016_soma_v,
            spike: 0,
            loss: 0
        };
    }
}
```

### 2. Modificar Simulador para Aceptar Parámetros Dendríticos

```typescript
// src/hardware/Simulador.ts

export class Omega21Simulador {
    private parametrosDendriticos: any | null = null;

    /**
     * Configura los parámetros dendríticos desde el vector 256D
     */
    configurarDendritas(parametros: any): void {
        this.parametrosDendriticos = parametros;
    }

    /**
     * Genera telemetría basada en parámetros dendríticos
     */
    generarMuestra(): Omega21Telemetry {
        const base = this.generarMuestraBase();
        
        // Si hay parámetros dendríticos, USARLOS en lugar de aleatorios
        if (this.parametrosDendriticos) {
            base.dendrites = {
                voltage: this.parametrosDendriticos.voltage + (Math.random() * 2 - 1), // ±1 ruido
                current: this.parametrosDendriticos.current + (Math.random() * 10 - 5),
                power: this.parametrosDendriticos.power + (Math.random() * 20 - 10),
                altitude: this.parametrosDendriticos.altitude,
                dew_temp: this.parametrosDendriticos.dew_temp + (Math.random() * 0.5 - 0.25),
                velocity: this.parametrosDendriticos.velocity + (Math.random() * 5 - 2.5),
                phase: this.parametrosDendriticos.phase,
                freq: this.parametrosDendriticos.freq + (Math.random() * 2 - 1),
                soma_v: this.parametrosDendriticos.soma_v + (Math.random() * 100 - 50),
                spike: Math.random() > 0.95 ? 1 : 0,
                loss: Math.random() * 1000000
            };
        }
        
        return base;
    }

    /**
     * Ciclo de estabilización: ejecuta N iteraciones para que las dendritas
     * estabilicen el comportamiento del átomo
     */
    async estabilizar(ciclos: number = 100): Promise<void> {
        for (let i = 0; i < ciclos; i++) {
            this.generarMuestra(); // Simular sin retornar
            await new Promise(resolve => setTimeout(resolve, 1)); // 1ms por ciclo
        }
    }
}
```

### 3. Actualizar Schema.ts para Incluir Campos Faltantes

```typescript
// src/omega21/Schema.ts

export interface DendritesData {
    // Campos existentes
    voltage: number;      // D001
    current: number;      // D002
    power: number;        // D003
    altitude: number;     // D004
    dew_temp: number;     // D005
    velocity: number;     // D006
    phase: number;        // D007
    freq: number;         // D008
    
    // AGREGAR campos faltantes:
    delay: number;        // D009
    memory: number;       // D010
    decay: number;        // D011
    michaelis: number;    // D012
    hill: number;         // D013
    capacitor: number;    // D014
    entropy: number;      // D015
    
    soma_v: number;       // D016
    spike: number;
    loss: number;
}
```

### 4. Crear Script de Entrenamiento Completo

```typescript
// src/run_entrenamiento_con_dendritas.ts

import { Omega21Simulador } from './hardware/Simulador';
import { SistemaOmnisciente, AtomoTopologico } from './SistemaOmnisciente';
import { StreamingBridge } from './neural/StreamingBridge';
import { MapeoVector256DaDendritas } from './control/MapeoVector256DaDendritas';
import { Vector256D } from './neural/CapaSensorial';

class EntrenadorConAtomosEstabilizados {
    private sistema: SistemaOmnisciente;
    private bridge: StreamingBridge;
    private mapeador: MapeoVector256DaDendritas;
    
    constructor() {
        this.sistema = new SistemaOmnisciente(3); // 3 macro átomos
        this.bridge = new StreamingBridge();
        this.mapeador = new MapeoVector256DaDendritas();
    }

    /**
     * Ciclo de entrenamiento con estabilización dendrítica
     */
    async entrenarConEstabilizacion(duracionSegundos: number): Promise<void> {
        const inicio = Date.now();
        const fin = inicio + (duracionSegundos * 1000);
        
        let ciclo = 0;
        
        while (Date.now() < fin) {
            ciclo++;
            console.log(`\n=== CICLO ${ciclo} ===`);
            
            // PASO 1: Generar vector 256D (simulado o desde sensores reales)
            const vector256D = this.generarVector256DSimulado();
            
            // PASO 2: Extraer campos dendríticos (D001-D056)
            const estadoDendritico = this.mapeador.extraerCamposDendriticos(vector256D);
            
            // PASO 3: Para cada átomo macro (ALFA, BETA, GAMMA)
            for (const atom of this.sistema.atomos) {
                console.log(`  Procesando átomo: ${atom.id}`);
                
                // 3.1 Configurar dendritas en el simulador del átomo
                const telemetriaDendritas = this.mapeador.convertirATelemetriaDendritas(estadoDendritico);
                atom.simulador.configurarDendritas(telemetriaDendritas);
                
                // 3.2 Estabilizar átomo (100 ciclos internos)
                console.log(`  Estabilizando átomo...`);
                await atom.simulador.estabilizar(100);
                
                // 3.3 Generar telemetría estabilizada
                const telemetria = atom.simulador.generarMuestra();
                console.log(`  Telemetría generada. Soma: ${telemetria.dendrites.soma_v}mV`);
                
                // 3.4 Procesar con red ONNX
                const inferencia = await atom.percibir(telemetria);
                console.log(`  Embedding: [${inferencia.estadoOculto.slice(0,5).join(', ')}...]`);
                
                // 3.5 Expandir a 1600D
                const vector1600D = this.expandirA1600D(inferencia.estadoOculto, vector256D);
                
                // 3.6 Enviar a Colab
                const esAnomalia = telemetria.neuro.nov > 200;
                await this.bridge.enviarVector(vector1600D, esAnomalia);
                console.log(`  Enviado a Colab (anomalía: ${esAnomalia})`);
            }
            
            // Esperar entre ciclos
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        console.log(`\n✅ Entrenamiento completado. Total ciclos: ${ciclo}`);
    }

    /**
     * Genera un vector 256D simulado
     */
    private generarVector256DSimulado(): Vector256D {
        const vec: Vector256D = {};
        for (let i = 1; i <= 256; i++) {
            const clave = `D${i.toString().padStart(3, '0')}`;
            
            // Campos dendríticos críticos con valores más realistas
            if (i === 1) vec[clave] = 20 + Math.random() * 50;  // Voltage
            else if (i === 2) vec[clave] = 100 + Math.random() * 500; // Current
            else if (i === 3) vec[clave] = 1000 + Math.random() * 2000; // Power
            else if (i === 4) vec[clave] = 500; // Altitude
            else if (i === 5) vec[clave] = 18 + Math.random() * 10; // DewTemp
            else if (i === 6) vec[clave] = Math.random() * 100; // Velocity
            else if (i === 7) vec[clave] = Math.random() * 360; // Phase
            else if (i === 8) vec[clave] = 50 + Math.random() * 100; // Frequency
            else if (i === 9) vec[clave] = Math.random() * 500; // Delay
            else if (i === 10) vec[clave] = Math.random(); // Memory
            else if (i === 11) vec[clave] = Math.random() * 0.5; // Decay
            else if (i === 12) vec[clave] = 20 + Math.random() * 50; // Michaelis
            else if (i === 13) vec[clave] = 1 + Math.random() * 2; // Hill
            else if (i === 14) vec[clave] = 30 + Math.random() * 40; // Capacitor
            else if (i === 15) vec[clave] = Math.random() * 5; // Entropy
            else if (i === 16) vec[clave] = -70 + Math.random() * 10; // Soma_V
            else vec[clave] = Math.random() * 100; // Resto
        }
        return vec;
    }

    /**
     * Expande embedding 64D a 1600D (25 subespacios × 64D)
     */
    private expandirA1600D(embedding: number[], vector256D: Vector256D): number[] {
        // Simplificado: repetir embedding 25 veces con variaciones
        const resultado: number[] = [];
        for (let i = 0; i < 25; i++) {
            for (let j = 0; j < 64; j++) {
                resultado.push(embedding[j] * (1 + Math.random() * 0.1 - 0.05));
            }
        }
        return resultado;
    }
}

// Ejecutar
const entrenador = new EntrenadorConAtomosEstabilizados();
entrenador.entrenarConEstabilizacion(300); // 5 minutos
```

---

## 📋 CHECKLIST DE IMPLEMENTACIÓN

### ✅ Fase 1: Preparación
- [ ] Actualizar `Schema.ts` con campos D009-D015 en `DendritesData`
- [ ] Crear `MapeoVector256DaDendritas.ts`
- [ ] Modificar `Omega21Simulador` para aceptar `configurarDendritas()`
- [ ] Agregar método `estabilizar()` al simulador

### ✅ Fase 2: Integración
- [ ] Crear `run_entrenamiento_con_dendritas.ts`
- [ ] Probar estabilización de 1 átomo
- [ ] Validar que telemetría refleja parámetros dendríticos
- [ ] Verificar que embeddings son consistentes (no aleatorios)

### ✅ Fase 3: Validación
- [ ] Ejecutar 10 ciclos de entrenamiento
- [ ] Inspeccionar datos enviados a Colab
- [ ] Confirmar que los valores dendríticos afectan la salida del átomo
- [ ] Medir estabilidad de embeddings (varianza baja con mismos parámetros)

### ✅ Fase 4: Escalamiento
- [ ] Extender a los 3 macro átomos (ALFA, BETA, GAMMA)
- [ ] Implementar infección entre átomos con dendritas estabilizadas
- [ ] Conectar con hardware real (cuando esté disponible)
- [ ] Habilitar feedback: Colab → ajustes dendríticos → nueva estabilización

---

## 🔥 CONCLUSIÓN

**EL PROBLEMA:** Las dendritas no están recibiendo los campos D001-D056 del vector 256D, por lo que:
1. Los átomos no se estabilizan antes de procesar
2. Generan embeddings aleatorios/ruidosos
3. Los datos enviados a Colab no reflejan física real
4. El entrenamiento de la Corteza Cognitiva usa datos sintéticos sin sentido físico

**LA SOLUCIÓN:** Implementar el flujo completo:
```
Vector 256D → Extraer D001-D056 → Dendritas → Estabilizar Átomo → 
Generar Telemetría Real → Procesar con ONNX → Embedding Estable → 
Expandir a 1600D → Entrenar Corteza Cognitiva en Colab
```

**PRÓXIMO PASO:** Implementar `MapeoVector256DaDendritas.ts` y modificar `Omega21Simulador` para estabilización.
