# 🎯 GUÍA RÁPIDA: Entrenar con Colab desde VS Code

## ¿Qué necesitas?

1. **Google Colab** (gratis, con GPU)
2. **Este workspace** (VS Code)
3. **URL ngrok** (se genera automáticamente)

---

## 3 Pasos Simples

### Paso 1️⃣: Ejecutar Servidor en Colab

```python
# En Google Colab (https://colab.research.google.com/)
# Copia TODO el contenido de:
#   COLAB_SERVER_OMEGA21_V4_UNIFICADO.py
# Y ejecútalo en una celda

# Verás:
# 📡 NGROK TUNNEL:
#    ✅ https://xxxxx-xxxxx-xxxxx.ngrok-free.app
```

⭐ **Copia esa URL** (es temporal, válida ~8 horas)

---

### Paso 2️⃣: Ejecutar en VS Code Terminal

```bash
# Terminal en VS Code
cd /workspaces/HIPERGRAFO

# Instalar dependencias (primera vez)
npm install

# Ejecutar con tu URL
./conectar_colab.sh https://xxxxx-xxxxx-xxxxx.ngrok-free.app
```

---

### Paso 3️⃣: Ver Resultados

```
📤 Enviando lote de 64 muestras...
✅ Entrenamiento completado (0.52s)
   Loss: 0.234567
   Anomalía detectada: 45.23%

📈 RESUMEN DE ENTRENAMIENTOS:
   Lotes enviados: 8
   Total muestras: 500
   Loss promedio: 0.245612
```

✅ **¡Listo!**

---

## Opciones Disponibles

```bash
# Forma básica
./conectar_colab.sh <URL>

# Con opciones
./conectar_colab.sh <URL> \
  --muestras 1000 \
  --lote 64 \
  --tipo neuronal \
  --anomalias 15 \
  --diagnostico \
  --metricas
```

| Opción | Defecto | Descripción |
|--------|---------|-------------|
| `--muestras` | 500 | Cuántos datos entrenar |
| `--lote` | 64 | Muestras por batch |
| `--tipo` | simple | Tipo de datos: simple/temporal/neuronal |
| `--anomalias` | 10% | Porcentaje de datos anómalos |
| `--diagnostico` | no | Test del servidor |
| `--metricas` | no | Mostrar gráficos |

---

## 🎬 Ejemplos Prácticos

### Prueba Rápida (< 1 minuto)
```bash
./conectar_colab.sh https://tu-url.ngrok-free.app \
  --muestras 100 --lote 32 --diagnostico
```

### Detección de Anomalías
```bash
./conectar_colab.sh https://tu-url.ngrok-free.app \
  --muestras 2000 --tipo temporal --anomalias 20 --metricas
```

### Entrenamiento Completo
```bash
./conectar_colab.sh https://tu-url.ngrok-free.app \
  --muestras 5000 --tipo neuronal --lote 128 --metricas
```

---

## 🔍 Monitoreo en Tiempo Real

Desde **otra terminal** (mientras entrena):

```bash
# Ver estado del servidor
curl https://tu-url.ngrok-free.app/status | jq

# Ver métricas
curl https://tu-url.ngrok-free.app/metricas | jq

# O acceder a Swagger UI
# Abre en navegador: https://tu-url.ngrok-free.app/docs
```

---

## ⚡ Flujo de Datos

```
1. GeneradorDatosEntrenamiento
   ↓ Crea 1600D vectores
   
2. ClienteColabEntrenamiento
   ↓ Envía lotes por HTTP
   
3. Servidor Colab (FastAPI)
   ↓ Procesa con GPU
   
4. CortezaCognitivaV4
   ├─ Capa2: LSTM + Transformer
   ├─ Capa3: MLP Residual
   ├─ Capa4: Self-Attention
   └─ Capa5: Decision Heads
   ↓ Retorna Loss + Anomalía + Feedback
   
5. Tu PC (VS Code)
   ↓ Recibe resultados
```

---

## 🛠️ Solución de Problemas

### "No se puede conectar"
```bash
# Verificar URL
# Asegúrate de copiar exactamente la URL de Colab
# Prueba con:
curl https://tu-url.ngrok-free.app/health

# Si funciona, deberías ver:
# {"alive": true, "timestamp": "...", ...}
```

### "Timeout"
```bash
# Colab está lento, intenta:
# - Reducir tamaño de lote: --lote 32
# - Reducir muestras: --muestras 500
```

### "Input mismatch 1600D"
```bash
# No modificar GeneradorDatosEntrenamiento
# Siempre genera exactamente 1600D
```

### "CUDA out of memory"
```bash
# En Colab, el lote es muy grande
# Opciones:
# 1. Reducir en VS Code: --lote 32
# 2. Reducir en Colab: editar batch_size
```

---

## 📊 Estructura de Archivos

```
src/colab/
├── ClienteColabEntrenamiento.ts  ← Cliente HTTP
├── GeneradorDatosEntrenamiento.ts ← Genera datos 1600D
├── entrenar_con_colab.ts          ← Script principal
├── config.colab.ts                ← Configuración
└── README.md                       ← Documentación

conectar_colab.sh                   ← Script bash helper
COLAB_SERVER_OMEGA21_V4_UNIFICADO.py ← Servidor (copiar a Colab)
```

---

## 🚀 Casos de Uso

### 1. Detect Anomalías en IoT
```bash
./conectar_colab.sh <URL> --tipo temporal --anomalias 5 --muestras 5000
```

### 2. Red Neuronal General
```bash
./conectar_colab.sh <URL> --tipo neuronal --muestras 10000
```

### 3. Testing Rápido
```bash
./conectar_colab.sh <URL> --muestras 50 --diagnostico
```

---

## 💻 Desde Código TypeScript

```typescript
import { ClienteColabEntrenamiento } from './src/colab/ClienteColabEntrenamiento';
import { GeneradorDatosEntrenamiento } from './src/colab/GeneradorDatosEntrenamiento';

async function main() {
    // Crear cliente
    const cliente = new ClienteColabEntrenamiento(
        'https://tu-url.ngrok-free.app'
    );
    
    // Conectar
    await cliente.conectar();
    
    // Generar datos
    const generador = new GeneradorDatosEntrenamiento();
    const muestras = generador.generarPatronesNeuronales(1000);
    
    // Entrenar
    const resultados = await cliente.entrenarMultiplesLotes(muestras, 64);
    
    // Mostrar resultado
    cliente.mostrarResumen();
}

main();
```

---

## ⏱️ Tiempo Estimado

| Operación | Tiempo |
|-----------|--------|
| Conectar a Colab | 1-2s |
| Generar 500 muestras | < 1s |
| Entrenar 8 lotes (GPU) | 5-10s |
| Total | ~20s |

---

## 📞 Necesitas Ayuda?

1. ✅ Verifica URL de Colab
2. ✅ Prueba `curl` al endpoint `/health`
3. ✅ Revisa logs en Colab
4. ✅ Lee `src/colab/README.md` (guía completa)
5. ✅ Abre Swagger: `{URL}/docs`

---

## 🎉 ¡Listo para Entrenar!

```bash
# Primer entrenamiento
./conectar_colab.sh https://tu-url.ngrok-free.app

# ¡Debería funcionar en ~20 segundos!
```

**Más información:** Lee `src/colab/README.md`
