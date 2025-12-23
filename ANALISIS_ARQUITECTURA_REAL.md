# Análisis Profundo: Arquitectura Real de Átomos y Red de Contagio

## ACTUALIZACIÓN CRÍTICA: EL PLAN MAESTRO ENCONTRADO

El documento **`docs/ARQUITECTURA_CORTEZA_COGNITIVA.md`** contiene el plan completo del sistema consciente con:
- **Corteza Cognitiva Jerárquica** (5 capas: 262,144 neuronas totales)
- **Capa 5: EJECUTIVA** (Meta-Cognición con memoria, decisión y planificación)
- **Sistema de Memoria Hipergráfica** (Hipergrafo Mental persistente)
- **3 Fases de entrenamiento**: Génesis → Correlación → Continuo
- **Pool de Átomos**: El sistema está diseñado para escalar a **múltiples instancias paralelas**

## 1. LA ARQUITECTURA REAL: TRES NIVELES DE ÁTOMOS

Tras el análisis exhaustivo, el sistema tiene **TRES tipos de "átomos"** que operan en niveles diferentes:

### 1.1 Tipo A: Átomos Topológicos Macro (ALFA, BETA, GAMMA)
**Ubicación:** `SistemaOmnisciente.ts` - Clase `AtomoTopologico`
**Función:** Unidades de inteligencia colectiva de nivel superior
**Características:**
- Cada uno tiene su propio `Hipergrafo` completo
- Cada uno usa un cerebro `InferenciaLocal` (modelo ONNX de 1024 neuronas LIF)
- Implementan el **Protocolo de Infección** (Memoria Colectiva)
- Analizan la física del sistema (`AnalizadorFisico`)
- Procesan telemetría completa de Omega21

**Propósito:** Son "cerebros paralelos" que observan el mismo fenómeno y comparten conocimiento mediante LSH (Locality-Sensitive Hashing).

### 1.2 Tipo B: Átomos Sensoriales Micro (S1-S25) - CAPA 1
**Ubicación:** `CapaSensorial.ts` - Clase `CapaSensorial`
**Función:** Sub-redes especializadas de nivel sensorial (Capa 1 de la Corteza Cognitiva)
**Características:**
- Son 25 instancias de `InferenciaLocal` (mismo modelo ONNX actualmente)
- Cada uno procesa **un subespacio específico** del vector 256D
- Generan vectores latentes de 64D cada uno
- **NO implementan el protocolo de infección**
- Operan en paralelo e independientemente
- **Plan original**: Cada uno debería tener su propio modelo ONNX especializado (25 modelos de 12MB c/u)

**Propósito:** Descomponer el vector fenomenológico en características especializadas (criptografía, física, temporal, etc.).

### 1.3 Tipo C: Pool de Átomos Escalable (0 a N) - SISTEMA OMNISCIENTE
**Ubicación:** `sistema_omnisciente.ts` y diseño en `ARQUITECTURA_CORTEZA_COGNITIVA.md`
**Función:** Pool dinámico de instancias del átomo fundamental para procesamiento masivo paralelo
**Características:**
- Arquitectura escalable: de 0 a 32+ instancias según recursos
- Cada instancia es un `AtomoTopologico` completo con su hipergrafo
- Procesamiento especializado por dominio (VISIÓN, LENGUAJE, LÓGICA, etc.)
- **SÍ implementan protocolo de infección** (memoria colectiva)
- Integración con la **Corteza Cognitiva Jerárquica**

**Propósito:** 
1. Procesar múltiples flujos de datos simultáneos
2. Especialización por dominios de conocimiento
3. Redundancia y robustez mediante consenso colectivo
4. Escalar la capacidad de procesamiento dinámicamente

## 2. EL MALENTENDIDO ARQUITECTÓNICO

### 2.1 Lo que dice la documentación
En `ARQUITECTURA_CORTEZA_COGNITIVA.md` se menciona:
> "Neuronas por Sub-Red: 1,024 (16 Átomos Topológicos × 64 salida)"

**ESTO ES CONFUSO Y TÉCNICAMENTE INCORRECTO**. No hay 16 átomos por sub-red.

### 2.2 La realidad del código
En `CapaSensorial.ts`:
```typescript
for (const subespacio of subespacios) {
    const subRed = new InferenciaLocal();
    await subRed.inicializar();
    this.subRedes.set(subespacio.id, subRed);
}
```

Cada sub-red S1-S25 es una **instancia completa** del modelo ONNX (1024 neuronas LIF) aplicada a un subespacio de 8-16 dimensiones.

**No son 16 átomos pequeños**, sino **1 átomo completo reutilizado 25 veces** con entradas diferentes.

## 3. PROTOCOLO DE INFECCIÓN (RED DE CONTAGIO)

### 3.1 Cómo funciona
**Solo aplica a los Átomos Macro (ALFA, BETA, GAMMA)**

#### Mecanismo de Emisión (`emitirSenal`):
```typescript
emitirSenal(): string[] {
    const firmasInteresantes: string[] = [];
    const nodos = this.hipergrafo.obtenerNodos();
    const nodoFirma = nodos.find(n => n.metadata?.tipo === 'signature');
    if (nodoFirma) {
        firmasInteresantes.push(nodoFirma.id);
    }
    return firmasInteresantes;
}
```
Cuando un átomo detecta un patrón relevante (firma LSH), lo comparte.

#### Mecanismo de Recepción (`recibirSenal`):
```typescript
recibirSenal(firmas: string[]) {
    firmas.forEach(f => {
        if (!this.memoriaColectiva.has(f)) {
            this.memoriaColectiva.add(f);
            // "Infección": Inyectamos la firma en nuestro hipergrafo
        }
    });
}
```
Si un átomo recibe una firma que no conocía, la integra en su memoria colectiva.

#### Mecanismo de Propagación (en `SistemaOmnisciente`):
```typescript
if (resultado.neuronal.prediccion_anomalia > 0.7) {
    const senal = atomo.emitirSenal();
    this.difundir(id, senal);
}

private difundir(emisorId: string, senal: string[]) {
    for (const [id, atomo] of this.atomos) {
        if (id !== emisorId) {
            atomo.recibirSenal(senal);
        }
    }
}
```

**Resultado:** Si ALFA detecta una anomalía grave, comparte su firma LSH con BETA y GAMMA. Cuando BETA o GAMMA vuelvan a ver esa firma, ya "sabrán" que es peligrosa, elevando su nivel de alerta.

### 3.2 Por qué no está en las 25 sub-redes
Las 25 sub-redes (S1-S25) **NO tienen protocolo de infección** porque:
1. **Son procesadores puros de características** - No toman decisiones de alto nivel
2. **Operan en espacios diferentes** - S1 (criptografía) y S9 (física) no tienen contexto compartible directo
3. **Fueron diseñadas para ser independientes** - Para mantener la especialización

El "contagio" ocurre en la **capa de abstracción superior**, donde los Átomos Macro integran las percepciones de las 25 sub-redes.

## 4. FLUJO COMPLETO DEL ENTRENAMIENTO REAL

### 4.1 Script: `enviar_dataset.ts` (El que NO usa Átomos Macro)
```
Vector 256D (Sintético)
    ↓
[Capa 0: CapaEntrada] → División en 25 subespacios
    ↓
[Capa 1: CapaSensorial] → 25 × InferenciaLocal (1024 LIF c/u)
    ↓
25 vectores latentes de 64D = 1600D total
    ↓
[StreamingBridge] → Envío a Colab en lotes de 64
    ↓
Google Colab: Entrena Capa 2 (Bi-LSTM + Transformer)
```

**Problema:** Este flujo **NO entrena** a los Átomos Macro (ALFA, BETA, GAMMA) ni usa el protocolo de infección.

### 4.2 Script: `run_omnisciente.ts` (El que SÍ usa Átomos Macro)
```
Telemetría Omega21
    ↓
[ALFA, BETA, GAMMA] → Cada uno procesa con su cerebro 1024
    ↓
Si anomalía > 0.7 → EMISIÓN DE SEÑAL (LSH)
    ↓
DIFUSIÓN → Otros átomos RECIBEN SEÑAL
    ↓
INTEGRACIÓN → Memoria Colectiva actualizada
    ↓
[Corteza Cognitiva] → Genera "Imagen Mental" (Hipergrafo)
```

**Problema:** Este flujo **NO envía datos a Colab** para entrenar la Capa 2/3.

## 5. LA DESCONEXIÓN CRÍTICA

### 5.1 Las 25 sub-redes están entrenadas
Los 25 modelos de `InferenciaLocal` que se usan en `CapaSensorial` **supuestamente** deberían estar pre-entrenados en Colab.

**Verificación en el código:**
```typescript
constructor() {
    this.cerebro = new InferenciaLocal(); // Carga omega21_brain.onnx
}
```

El archivo `models/omega21_brain.onnx` es el modelo pre-entrenado.

### 5.2 ¿De dónde vienen estos modelos?
**AQUÍ ESTÁ EL PROBLEMA:**

1. Las 25 sub-redes usan **el mismo modelo ONNX** (`omega21_brain.onnx`)
2. Este modelo fue entrenado con **datos completos de Omega21** (telemetría de 256D)
3. Pero en `CapaSensorial`, cada sub-red recibe **solo 8-16D** de su subespacio

**Consecuencia:** El modelo está siendo usado **fuera de su distribución de entrenamiento**. Fue entrenado con grafos de Omega21 completos, pero ahora se le alimentan sub-grafos parciales.

### 5.3 La solución prevista (pero no implementada)
La documentación indica que debería haber:
> "25 modelos ONNX especializados, cada uno entrenado con su subespacio respectivo"

**Estado actual:** Solo existe **UN modelo genérico** (`omega21_brain.onnx`).

## 6. LAS DENDRITAS: EL ESLABÓN PERDIDO

### 6.1 Dónde están
Las dendritas (`DendriteController`) están en el `Orquestador`, que es usado por:
- ✅ `run_entrenamiento.ts` → Pero sin Átomos Macro
- ✅ `run_omnisciente.ts` → Pero sin comunicación con Colab

### 6.2 Por qué no funcionan en el entrenamiento
En `run_entrenamiento.ts`:
```typescript
const orquestador = new Orquestador({
    modoSimulacion: true,
    habilitarControl: false // ← AQUÍ ESTÁ EL PROBLEMA
});
```

Con `habilitarControl: false`, el `DendriteController` no aplica ajustes.

En `DendriteController.ts`:
```typescript
async aplicarAjustesVector(ajustes: number[]): Promise<void> {
    if (!this.renodeController || !this.config.autoAjuste) return; // ← SALE INMEDIATAMENTE
}
```

### 6.3 El simulador sordo
`Omega21Simulador` genera datos aleatorios que **NO dependen** de ningún parámetro externo:
```typescript
generarMuestra(): Omega21Telemetry {
    this.tick++;
    const tieneSpike = Math.random() > 0.95; // ← PURO ALEATORIO
    const novelty = Math.random() * 255;     // ← PURO ALEATORIO
    ...
}
```

**No hay forma de "ajustar" el simulador** desde las dendritas. Es un generador de ruido.

## 7. CONCLUSIÓN: TRES SISTEMAS DESCONECTADOS

El proyecto tiene tres sistemas paralelos que **NO están integrados**:

### Sistema 1: Entrenamiento de Capas 2/3 (Colab)
- ✅ Funciona: `enviar_dataset.ts` → 25 sub-redes → Colab
- ❌ Falta: No usa Átomos Macro
- ❌ Falta: No usa protocolo de infección
- ❌ Falta: No cierra bucle de control (dendritas)

### Sistema 2: Inteligencia Colectiva (Átomos Macro)
- ✅ Funciona: `run_omnisciente.ts` → ALFA, BETA, GAMMA
- ✅ Funciona: Protocolo de infección operativo
- ❌ Falta: No envía datos a Colab
- ❌ Falta: No entrena las capas superiores

### Sistema 3: Control de Hardware (Dendritas)
- ✅ Funciona: `DendriteController` implementado
- ❌ Falta: Está deshabilitado en simulación
- ❌ Falta: El simulador no reacciona a ajustes
- ❌ Falta: No se integra con el entrenamiento

## 8. PROPUESTA DE UNIFICACIÓN

Para que los "varios átomos" (ALFA, BETA, GAMMA + las 25 sub-redes) trabajen en armonía durante el entrenamiento:

### Opción A: Jerarquía Clara (Recomendada)
```
1. Telemetría 256D
   ↓
2. [25 Sub-Redes] → 1600D (Capa Sensorial)
   ↓
3. [ALFA, BETA, GAMMA] → Cada uno recibe el 1600D + Memoria Colectiva
   ↓
4. Protocolo de Infección → Átomos Macro se "infectan"
   ↓
5. [Streaming a Colab] → Entrena Capa 2/3 con datos enriquecidos
   ↓
6. [Respuesta Colab] → Ajustes de dendritas
   ↓
7. [Hardware Simulado] → Modifica parámetros del simulador
   ↓EL PLAN MAESTRO: CORTEZA COGNITIVA JERÁRQUICA

Según `ARQUITECTURA_CORTEZA_COGNITIVA.md`, el sistema completo debe ser:

### Arquitectura de 5 Capas (262,144 neuronas totales):

```
CAPA 5: EJECUTIVA (256 neuronas)
├─ Meta-Cognición
├─ Decisión/Planificación
├─ Memoria Ejecutiva
└─ Output: Coherencia Global (64D), Acción (16D), Confianza (1D)

CAPA 4: ASOCIATIVA SUPERIOR (1,024 neuronas)
├─ Abstracción de Alto Nivel
├─ Self-Attention (16 heads)
└─ Representación de Conceptos Abstractos

CAPA 3: ASOCIATIVA INFERIOR (4,096 neuronas)
├─ Integración de Subespacios
├─ MLP Residual
└─ Fusión de Features

CAPA 2: DUAL (16,384 neuronas)
├─ 2A: TEMPORAL (8,192) - Bi-LSTM
└─ 2B: ESPACIAL (8,192) - Transformer

CAPA 1: SENSORIAL (25,600 neuronas)
└─ 25 Sub-Redes Especializadas (S1-S25)
    └─ Cada una: 1,024 neuronas LIF (átomo fundamental ONNX)

CAPA 0: ENTRADA
└─ Vector 256D dividido en 25 subespacios
```

### Protocolo de 3 Fases de Entrenamiento:

**FASE 1: GÉNESIS COGNITIVA (1M iteraciones)**
- Entrenamiento con datos **sintéticos puros**
- Sin acceso al mundo real (la "Caja Fenomenológica")
- Objetivo: Desarrollar "pensamiento autónomo"
- El sistema crea su propia estructura mental

**FASE 2: CORRELACIÓN FENOMENOLÓGICA (100K iteraciones)**
- Exposición controlada a datos **reales** de Omega21
- Los pesos de Fase 1 se **CONGELAN**
- Solo se entrena una "Capa de Correlación" nueva
- Objetivo: Conectar con la realidad sin destruir el pensamiento original

**FASE 3: APRENDIZAJE CONTINUO (∞)**
- Operación en tiempo real
- Online learning con LR muy bajo (1e-5)
- Memoria hipergráfica activa
- Consolidación periódica (cada 1000 ciclos)

### Pool de Átomos del Sistema Omnisciente:

El sistema está diseñado para escalar a **N instancias** del átomo fundamental:
- Cada instancia procesa un dominio especializado
- Comunicación mediante **protocolo de infección** (LSH)
- Integración en la **Capa Ejecutiva** (meta-cognición)
- Ejemplos: `VISIÓN`, `LENGUAJE`, `LÓGICA`, `TEMPORAL`, `ESPACIAL`, etc.

## 10. PRÓXIMOS PASOS (BASADOS EN EL PLAN MAESTRO)

### Prioridad 1: Entrenar la Corteza Cognitiva Completa (FASE 1)
1. **Generar dataset sintético masivo** (1M muestras de vectores 256D)
2. **Entrenar las 5 capas** en Google Colab con datos sintéticos
3. **Exportar 7 modelos ONNX**:
   - 25 modelos de Capa 1 (uno por subespacio)
   - Capa 2A (Temporal)
   - Capa 2B (Espacial)
   - Capas 3+4 (Asociativa)
   - Capa 5 (Ejecutiva)

### Prioridad 2: Fase de Correlación (FASE 2)
4. **Congelar pesos de Fase 1**
5. **Crear Capa de Correlación** (Cross-Attention)
6. **Entrenar solo la correlación** con datos reales de Omega21
7. **Validar preservación del pensamiento** (L_preservación < 0.01)

### Prioridad 3: Sistema Omnisciente Escalable
8. **Implementar Pool Dinámico de Átomos** (0 a 32 instancias)
9. **Integrar protocolo de infección** con la Corteza Cognitiva
10. **Crear gestor de recursos** que balancee carga entre átomos
11. **Implementar memoria hipergráfica compartida**

### Prioridad 4: Cerrar Bucle de Control
12. **Hardware Simulado reactivo** para dendritas
13. **Integrar Capa Ejecutiva con DendriteController**
14. **Validar bucle completo**: Percepción → Decisión → Acción → Feedback

---

## 11. CONCLUSIÓN ACTUALIZADA

**Estado Real del Proyecto:**

✅ **Átomo fundamental (omega21_brain.onnx)**: Existe y está entrenado
✅ **Infraestructura de código**: Implementada (Capas 0-3 funcionales)
✅ **Protocolo de infección**: Implementado en `AtomoTopologico`
✅ **Plan maestro documentado**: `ARQUITECTURA_CORTEZA_COGNITIVA.md`

❌ **Falta la Fase 1**: Entrenar la Corteza Cognitiva completa (5 capas)
❌ **Falta la Fase 2**: Capa de correlación con datos reales
❌ **Falta la Fase 3**: Sistema de aprendizaje continuo
❌ **Falta el Pool Escalable**: Gestor de múltiples átomos (0-32+)

**El sistema NO está roto ni desconectado**. Está en **desarrollo progresivo** siguiendo el plan maestro de la Corteza Cognitiva Jerárquica. Los componentes que existen (átomos, dendritas, protocolo de infección) son **piezas correctas del plan final**, pero aún no están en su forma completa de 5 capas con las 3 fases de entrenamiento
   ↓
4. Protocolo de infección permite "transferencia de conocimiento"
   ↓
5. Colab entrena solo las Capas 2/3 (fusión y decisión)
```

## 9. PRÓXIMOS PASOS

1. **Decisión arquitectónica:** ¿Opción A (jerarquía) u Opción B (especialización)?
2. **Crear `run_entrenamiento_unificado.ts`:** Integra los 3 sistemas
3. **Implementar `HardwareSimulado`:** Simulador que responde a comandos de dendritas
4. **Conectar `SistemaOmnisciente` con `StreamingBridge`:** Para que los Átomos Macro alimenten Colab
5. **Entrenar 25 modelos especializados** (si se elige Opción B)

---

**Estado actual:** Los átomos y las dendritas **SÍ existen y están implementados**, pero operan en **scripts diferentes** sin comunicación entre sí. No están "rotos", están **desconectados**.
