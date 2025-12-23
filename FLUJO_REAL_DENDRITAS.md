# FLUJO REAL: DENDRITAS Y ÁTOMOS

## 🎯 ARQUITECTURA CORRECTA

### 1. Los Átomos se Auto-Estabilizan

```
Vector 256D → Átomo (procesa con ONNX) → Se estabiliza solo → 
Genera embedding + análisis físico
```

**NO HAY** un proceso externo de estabilización. El átomo es autónomo.

### 2. Las Dendritas Reciben 256D Completo

Las **dendritas NO reciben solo D001-D056**. Reciben el **vector 256D COMPLETO** y generan señales de control para **alterar/ajustar** los átomos.

### 3. Red de Comunicación entre Átomos

Los átomos se comunican mediante **protocolo de infección LSH**:

```typescript
// Átomo A detecta patrón importante
const senal = atomoA.emitirSenal(); // Devuelve firmas LSH

// Átomo B recibe la señal
atomoB.recibirSenal(senal); // Añade a memoria colectiva

// Átomo B ahora "conoce" los patrones de A
// Si ve una firma conocida, aumenta su alerta
```

### 4. Salida Colectiva para Entrenamiento

Los átomos generan embeddings individuales que se combinan:

```
Átomo ALFA → Embedding 64D + Análisis Físico
Átomo BETA → Embedding 64D + Análisis Físico
Átomo GAMMA → Embedding 64D + Análisis Físico
         ↓
   Fusión/Agregación
         ↓
   Salida Colectiva (1600D)
         ↓
   Entrenar Red Neuronal en Colab
```

---

## 🔄 FLUJO COMPLETO PASO A PASO

### Entrada: Vector 256D

```json
{
  "D001": 45.2,    // Voltage
  "D002": 520.7,   // Current
  "D003": 2340,    // Power
  ...
  "D256": 18.3     // Último campo
}
```

### PASO 1: Procesar en Múltiples Átomos

```typescript
// Los 3 macro átomos procesan EL MISMO vector 256D
const telemetria = generarTelemetriaDesde256D(vector256D);

// Átomo ALFA
const resultadoALFA = await atomoALFA.percibir(telemetria);
// Átomo BETA
const resultadoBETA = await atomoBETA.percibir(telemetria);
// Átomo GAMMA
const resultadoGAMMA = await atomoGAMMA.percibir(telemetria);
```

### PASO 2: Red de Comunicación (Infección)

```typescript
// Cada átomo emite sus patrones detectados
const senalALFA = atomoALFA.emitirSenal();  // ["SIG_abc123", "SIG_def456"]
const senalBETA = atomoBETA.emitirSenal();  // ["SIG_ghi789"]
const senalGAMMA = atomoGAMMA.emitirSenal(); // []

// Broadcast: todos reciben señales de todos
atomoALFA.recibirSenal([...senalBETA, ...senalGAMMA]);
atomoBETA.recibirSenal([...senalALFA, ...senalGAMMA]);
atomoGAMMA.recibirSenal([...senalALFA, ...senalBETA]);

// Ahora cada átomo tiene memoria colectiva compartida
console.log(atomoALFA.memoria); // 3 firmas (1 propia + 1 de BETA + 0 de GAMMA)
console.log(atomoBETA.memoria); // 3 firmas
console.log(atomoGAMMA.memoria); // 3 firmas
```

### PASO 3: Dendritas Reciben 256D y Generan Control

```typescript
// DendriteController analiza el hipergrafo y telemetría
const estadoDendritas = dendriteController.analizarEstado(
    hipergrafo,
    telemetria
);

// Las dendritas generan ajustes basados en el análisis
// Ejemplo: si densidad del grafo > 0.8, reducir ganancia
if (estadoDendritas.densidad > 0.8) {
    const ajustes = [
        { indiceDendrita: 0, parametro: 'ganancia', valor: 0.9 },
        { indiceDendrita: 5, parametro: 'umbral', valor: 0.6 }
    ];
    
    // Aplicar ajustes al hardware/simulador
    await dendriteController.aplicarAjustesVector(
        convertirAjustesAVector(ajustes) // [0.9, 0, 0, 0, 0, 0.6, 0, ...]
    );
}
```

### PASO 4: Fusión de Salidas

```typescript
// Combinar embeddings de los 3 átomos
const embeddings = [
    resultadoALFA.neuronal.estadoOculto,  // 64D
    resultadoBETA.neuronal.estadoOculto,  // 64D
    resultadoGAMMA.neuronal.estadoOculto  // 64D
];

// Procesar por 25 subespacios → 1600D
const vector1600D = await procesarPor25Subespacios(
    embeddings, 
    vector256D
);
```

### PASO 5: Enviar a Colab

```typescript
// Detectar si hay anomalía colectiva
const esAnomalia = 
    resultadoALFA.neuronal.prediccion_anomalia > 0.7 ||
    resultadoBETA.neuronal.prediccion_anomalia > 0.7 ||
    resultadoGAMMA.neuronal.prediccion_anomalia > 0.7;

// Enviar a Colab para entrenar Capa 2-5
await streamingBridge.enviarVector(vector1600D, esAnomalia);
```

---

## 📊 DATOS QUE FLUYEN

### Vector 256D → Telemetría Omega21

```typescript
function generarTelemetriaDesde256D(v256: Vector256D): Omega21Telemetry {
    return {
        meta: {
            ts: Date.now(),
            blk: blockCounter++,
            sz: 256
        },
        logic: {
            h: v256.D001,      // Ejemplo: mapeo directo
            lz: v256.D002,
            chi: v256.D003,
            pad: [0, 0, 0]
        },
        neuro: {
            id: Math.floor(v256.D008 % 1024),
            sim: v256.D009,
            nov: v256.D010,
            cat: Math.floor(v256.D011 % 8)
        },
        sig: {
            fp: generarFingerprint(v256),
            lsh: Math.floor(v256.D012 % 255),
            eq: Math.floor(v256.D013 % 8),
            sc: Math.floor(v256.D014 % 3)
        },
        dendrites: {
            voltage: v256.D015,
            current: v256.D016,
            power: v256.D017,
            altitude: v256.D018,
            dew_temp: v256.D019,
            velocity: v256.D020,
            phase: v256.D021,
            freq: v256.D022,
            soma_v: v256.D023,
            spike: v256.D024 > 0.9 ? 1 : 0,
            loss: v256.D025
        },
        vector_72d: Array.from({ length: 72 }, (_, i) => v256[`D${i+26}`] || 0),
        metrics_256: Array.from({ length: 256 }, (_, i) => v256[`D${i+1}`] || 0)
    };
}
```

---

## 🛠️ IMPLEMENTACIÓN CORRECTA

### Script de Entrenamiento

```typescript
// src/run_entrenamiento_con_red_atomos.ts

import { SistemaOmnisciente } from './SistemaOmnisciente';
import { StreamingBridge } from './neural/StreamingBridge';
import { DendriteController } from './control/DendriteController';
import { Vector256D } from './neural/CapaSensorial';

class EntrenadorRedAtomos {
    private sistema: SistemaOmnisciente;
    private bridge: StreamingBridge;
    private dendriteController: DendriteController;
    
    constructor() {
        this.sistema = new SistemaOmnisciente();
        this.bridge = new StreamingBridge();
        this.dendriteController = new DendriteController({
            autoAjuste: false // Sin hardware real por ahora
        });
    }

    async entrenar(duracionSegundos: number) {
        // Crear 3 macro átomos
        const atomoALFA = await this.sistema.crearAtomo('ALFA');
        const atomoBETA = await this.sistema.crearAtomo('BETA');
        const atomoGAMMA = await this.sistema.crearAtomo('GAMMA');
        
        const inicio = Date.now();
        const fin = inicio + (duracionSegundos * 1000);
        let ciclo = 0;
        
        while (Date.now() < fin) {
            ciclo++;
            console.log(`\n=== CICLO ${ciclo} ===`);
            
            // 1. Generar Vector 256D (desde sensores o simulado)
            const vector256D = this.generarVector256D();
            
            // 2. Convertir a Telemetría Omega21
            const telemetria = this.vectorATelemetria(vector256D);
            
            // 3. Procesar en los 3 átomos (en paralelo)
            const [resALFA, resBETA, resGAMMA] = await Promise.all([
                atomoALFA.percibir(telemetria),
                atomoBETA.percibir(telemetria),
                atomoGAMMA.percibir(telemetria)
            ]);
            
            // 4. Red de comunicación (infección LSH)
            const senalALFA = atomoALFA.emitirSenal();
            const senalBETA = atomoBETA.emitirSenal();
            const senalGAMMA = atomoGAMMA.emitirSenal();
            
            // Broadcast
            atomoALFA.recibirSenal([...senalBETA, ...senalGAMMA]);
            atomoBETA.recibirSenal([...senalALFA, ...senalGAMMA]);
            atomoGAMMA.recibirSenal([...senalALFA, ...senalBETA]);
            
            console.log(`  🦠 Memoria colectiva: ${resALFA.memoria + resBETA.memoria + resGAMMA.memoria} firmas`);
            
            // 5. Dendritas analizan y generan ajustes
            const estadoDendritas = this.dendriteController.analizarEstado(
                atomoALFA.hipergrafo, // Usar hipergrafo de ALFA como referencia
                telemetria
            );
            
            console.log(`  📊 Densidad grafo: ${estadoDendritas.densidad.toFixed(3)}`);
            
            // 6. Fusionar embeddings de los 3 átomos
            const embeddings = [
                resALFA.neuronal.estadoOculto,
                resBETA.neuronal.estadoOculto,
                resGAMMA.neuronal.estadoOculto
            ];
            
            // 7. Expandir a 1600D (25 subespacios × 64D)
            const vector1600D = this.fusionarEmbeddings(embeddings);
            
            // 8. Detectar anomalía colectiva
            const esAnomalia = Math.max(
                resALFA.neuronal.prediccion_anomalia,
                resBETA.neuronal.prediccion_anomalia,
                resGAMMA.neuronal.prediccion_anomalia
            ) > 0.7;
            
            // 9. Enviar a Colab
            await this.bridge.enviarVector(vector1600D, esAnomalia);
            console.log(`  ✅ Enviado a Colab (anomalía: ${esAnomalia})`);
            
            // Esperar entre ciclos
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        console.log(`\n✅ Entrenamiento completado. Total ciclos: ${ciclo}`);
    }

    private generarVector256D(): Vector256D {
        const vec: Vector256D = {};
        for (let i = 1; i <= 256; i++) {
            vec[`D${i.toString().padStart(3, '0')}`] = Math.random() * 100;
        }
        return vec;
    }

    private vectorATelemetria(v: Vector256D): any {
        return {
            meta: { ts: Date.now(), blk: 0, sz: 256 },
            logic: { h: v.D001, lz: 0, chi: 0, pad: [0,0,0] },
            neuro: { id: 0, sim: v.D009, nov: v.D010, cat: 0 },
            sig: { fp: "fake", lsh: 0, eq: 0, sc: 0 },
            dendrites: {
                voltage: v.D015 || 50,
                current: v.D016 || 300,
                power: v.D017 || 2000,
                altitude: 500,
                dew_temp: 20,
                velocity: 50,
                phase: 90,
                freq: 100,
                soma_v: -70,
                spike: 0,
                loss: 0
            },
            metrics_256: Array.from({ length: 256 }, (_, i) => v[`D${i+1}`] || 0)
        };
    }

    private fusionarEmbeddings(embeddings: number[][]): number[] {
        // Repetir cada embedding ~8 veces para llegar a 1600D
        const resultado: number[] = [];
        const repeticiones = Math.ceil(1600 / (embeddings.length * embeddings[0].length));
        
        for (let r = 0; r < repeticiones; r++) {
            for (const emb of embeddings) {
                resultado.push(...emb);
                if (resultado.length >= 1600) break;
            }
            if (resultado.length >= 1600) break;
        }
        
        return resultado.slice(0, 1600);
    }
}

// Ejecutar
const entrenador = new EntrenadorRedAtomos();
entrenador.entrenar(300); // 5 minutos
```

---

## ✅ RESUMEN DEL FLUJO REAL

1. **Vector 256D** entra al sistema
2. **3 Macro Átomos** (ALFA, BETA, GAMMA) procesan el MISMO vector
3. Cada átomo **se auto-estabiliza** y genera:
   - Embedding 64D (de ONNX)
   - Análisis físico (conservación energía, entropía, etc.)
   - Firma LSH (patrón detectado)
4. **Red de comunicación**: Los átomos comparten firmas LSH (infección)
5. **Dendritas** analizan el hipergrafo y generan ajustes de control
6. **Fusión de embeddings**: Los 3 embeddings se combinan → 1600D
7. **Envío a Colab**: Se entrena la Corteza Cognitiva (Capas 2-5)

**CRÍTICO:** Los átomos SON autónomos. Las dendritas NO estabilizan, sino que ALTERAN/AJUSTAN basándose en el análisis del vector 256D completo.
