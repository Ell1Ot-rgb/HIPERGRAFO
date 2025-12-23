# Entrenamiento con Átomos Activos

## Concepto

Este sistema permite entrenar la **Corteza Cognitiva** (Capas 2-5) en Google Colab usando datos **generados en tiempo real** por múltiples átomos ejecutándose localmente.

## Diferencia con el Método Anterior

### ❌ Método Anterior (Sintético)
```
GeneradorSintetico → Datos falsos → Colab
```
- Datos puramente matemáticos
- Sin dinámica real de átomos
- Sin protocolo de infección

### ✅ Método Nuevo (Átomos Reales)
```
Pool de Átomos (local) → Telemetría procesada → Protocolo de Infección → Colab
```
- Datos generados por átomos reales ejecutándose
- Protocolo de contagio activo (memoria colectiva LSH)
- Cada átomo con su hipergrafo y cerebro 1024 LIF
- Múltiples dominios especializados

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    CODESPACES (LOCAL)                           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            POOL DE ÁTOMOS (0-32 instancias)              │   │
│  │                                                          │   │
│  │  [VISION] [AUDIO] [LENGUAJE] [LOGICA] [MEMORIA] ...     │   │
│  │     ↓        ↓         ↓         ↓         ↓            │   │
│  │  Cada átomo:                                             │   │
│  │  • Hipergrafo propio                                     │   │
│  │  • Cerebro 1024 LIF (omega21_brain.onnx)                 │   │
│  │  • Procesa telemetría Omega21                            │   │
│  │  • Emite/Recibe señales LSH (infección)                  │   │
│  │                                                          │   │
│  │  Output: N × 68D (64 features + 4 métricas físicas)     │   │
│  └────────────────────┬─────────────────────────────────────┘   │
│                       │                                         │
│  ┌────────────────────▼─────────────────────────────────────┐   │
│  │          EXTRACTOR DE CARACTERÍSTICAS                    │   │
│  │  • Agrega salidas de todos los átomos                    │   │
│  │  • Normaliza a 1600D (compatibilidad con Colab)          │   │
│  │  • Detecta anomalías (varianza)                          │   │
│  └────────────────────┬─────────────────────────────────────┘   │
│                       │                                         │
│  ┌────────────────────▼─────────────────────────────────────┐   │
│  │           STREAMING BRIDGE                               │   │
│  │  • Agrupa en lotes de 64 muestras                        │   │
│  │  • POST /train_layer2                                    │   │
│  └────────────────────┬─────────────────────────────────────┘   │
└─────────────────────┼─┼─────────────────────────────────────────┘
                      │ │
                      │ │ HTTPS (ngrok)
                      │ │
┌─────────────────────▼─▼─────────────────────────────────────────┐
│                   GOOGLE COLAB                                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │        CORTEZA COGNITIVA JERÁRQUICA                      │   │
│  │                                                          │   │
│  │  Capa 2A: Bi-LSTM (Temporal)    ┐                        │   │
│  │  Capa 2B: Transformer (Espacial)┴─► GMU Fusion          │   │
│  │  Capa 3: Asociativa Inferior (4096)                      │   │
│  │  Capa 4: Asociativa Superior (1024)                      │   │
│  │  Capa 5: Ejecutiva (256) → Outputs múltiples            │   │
│  │                                                          │   │
│  │  Training Loop:                                          │   │
│  │  1. Recibe lote de 64 × 1600D                            │   │
│  │  2. Forward pass                                         │   │
│  │  3. Calcula loss (estabilidad + anomalía)                │   │
│  │  4. Backward pass                                        │   │
│  │  5. Actualiza pesos                                      │   │
│  │  6. Retorna métricas                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Al finalizar: Exportar a ONNX                                  │
│  • corteza_completa.onnx (~410 MB)                              │
│  • O separado en capas individuales                             │
└─────────────────────────────────────────────────────────────────┘
```

## Uso

### 1. Ejecutar el sistema

```bash
npx ts-node src/run_entrenamiento_con_atomos.ts <URL_COLAB> [numAtomos] [muestras]
```

**Parámetros:**
- `URL_COLAB`: URL del servidor ngrok de Colab (requerido)
- `numAtomos`: Cantidad de átomos (default: 8, máx: 32)
- `muestras`: Objetivo de muestras (default: 10000)

**Ejemplo:**
```bash
npx ts-node src/run_entrenamiento_con_atomos.ts https://abc123.ngrok-free.app 16 20000
```

Esto creará:
- 16 átomos con dominios especializados
- Enviará 20,000 muestras a Colab
- ~313 lotes de 64 muestras

### 2. Lo que verás

```
🧠 INICIANDO ENTRENAMIENTO CON ÁTOMOS ACTIVOS
   Configuración:
   • Átomos: 8
   • URL Colab: https://abc123.ngrok-free.app
   • Objetivo: 10000 muestras
   • Infección: ✅

🔬 Creando pool de 8 átomos...
   ✅ VISION inicializado
   ✅ AUDIO inicializado
   ✅ LENGUAJE inicializado
   ✅ LOGICA inicializado
   ✅ TEMPORAL inicializado
   ✅ CAUSAL inicializado
   ✅ EMOCIONAL inicializado
   ✅ MOTOR inicializado

📊 Dashboard disponible en http://localhost:3000

🚀 Sistema listo. Iniciando bucle de entrenamiento...

📈 Progreso: 100/10000 (1.0%)
   🦠 Memoria colectiva: 23 firmas LSH compartidas
📈 Progreso: 200/10000 (2.0%)
   🦠 Memoria colectiva: 47 firmas LSH compartidas
...
```

### 3. En Colab

El servidor debe tener el endpoint `/train_layer2` que:
1. Recibe lotes de 64 muestras
2. Cada muestra: `{ input_data: number[1600], anomaly_label: 0|1 }`
3. Entrena el modelo
4. Retorna métricas de loss

### 4. Al finalizar

```
✅ ENTRENAMIENTO COMPLETADO
   Total de muestras enviadas: 10000
   Esperando a que Colab procese el buffer...

🎉 Todos los datos fueron enviados a Colab.
   Ahora puedes exportar el modelo entrenado desde Colab a ONNX.

   Dashboard sigue activo en http://localhost:3000
   Presiona Ctrl+C para cerrar todo.
```

## Protocolo de Infección Activo

Durante el entrenamiento, los átomos:

1. **Detectan anomalías** en su telemetría
2. **Emiten señales LSH** (firmas) cuando encuentran algo importante
3. **Reciben señales** de otros átomos
4. **Integran conocimiento** en su memoria colectiva
5. **Modifican su comportamiento** basándose en experiencia compartida

Esto genera datos más **ricos y realistas** para entrenar la Corteza Cognitiva.

## Ventajas sobre Datos Sintéticos

| Aspecto | Sintéticos | Átomos Reales |
|---------|-----------|---------------|
| Dinámica temporal | ❌ Estática | ✅ Real |
| Protocolo de infección | ❌ No | ✅ Activo |
| Hipergrafos | ❌ Simulados | ✅ Reales |
| Memoria colectiva | ❌ No | ✅ LSH compartido |
| Especialización | ❌ Genérica | ✅ Por dominio |
| Física del sistema | ❌ Ideal | ✅ Métricas reales |

## Próximos Pasos

1. **Ejecutar este script** con tu URL de Colab
2. **Entrenar** hasta convergencia (pérdida < 0.1)
3. **Exportar** el modelo desde Colab a ONNX
4. **Integrar** el modelo en el sistema local
5. **Cerrar el bucle** de control con dendritas

## Monitoreo

Puedes monitorear el progreso:
- **Local**: Dashboard en http://localhost:3000
- **Colab**: Logs del servidor FastAPI
- **Terminal**: Progreso y memoria colectiva

## Recursos

- CPU: ~50% por cada 8 átomos
- RAM: ~2GB para 16 átomos
- Red: ~10KB/s de subida (64 muestras cada 64 segundos)

---

Este es el sistema de entrenamiento **con átomos reales ejecutándose**, no con datos sintéticos.
