# 🚀 GUÍA: Acceder al Servidor Colab desde VS Code

## Resumen Rápido

**Tu setup tendrá 2 partes:**
1. **Servidor remoto** (Colab): Corteza cognitiva con GPU/TPU
2. **Cliente local** (VS Code): Envía datos de entrenamiento, recibe resultados

---

## PASO 1: Preparar Google Colab

### 1.1 Crear/Abrir Notebook en Colab
- Ve a https://colab.research.google.com
- Crea un notebook nuevo o abre uno existente
- En `Entorno de ejecución` → `Cambiar tipo de entorno` → Selecciona **GPU** o **TPU**

### 1.2 Copiar el servidor Python
1. Copia TODO el contenido de: [`COLAB_SERVER_OMEGA21_V4_UNIFICADO.py`](../../COLAB_SERVER_OMEGA21_V4_UNIFICADO.py)
2. Pégalo en una celda de Colab
3. Ejecuta (`Shift + Enter`)

### 1.3 Instalar dependencias (si faltan)
```python
!pip install fastapi uvicorn pyngrok torch
```

### 1.4 Obtener URL pública
Cuando el servidor inicie, verás algo como:
```
🚀 INICIANDO OMEGA 21 v4.0...
📡 NGROK TUNNEL:
   ✅ https://1234-5678-90ab-cdef.ngrok.io
```

**⚠️ Copia esta URL (la necesitarás en el cliente)**

---

## PASO 2: Configurar Cliente en VS Code

### 2.1 Actualizar URL del servidor
Abre: `src/colab/cliente_colab.ts`

Busca esta línea (linea ~23):
```typescript
const CONFIG: ConfiguracionCliente = {
  serverUrl: process.env.COLAB_SERVER_URL || 'http://localhost:8000',
```

Reemplázala con tu URL de ngrok:
```typescript
const CONFIG: ConfiguracionCliente = {
  serverUrl: 'https://1234-5678-90ab-cdef.ngrok.io',  // ← TU URL AQUÍ
```

O usa variable de entorno:
```bash
export COLAB_SERVER_URL=https://1234-5678-90ab-cdef.ngrok.io
```

### 2.2 Instalar dependencias Node.js
```bash
npm install
```

### 2.3 Ejecutar cliente de prueba
```bash
npx ts-node src/colab/cliente_colab.ts
```

Debería conectarse y mostrar:
```
✅ Conexión exitosa
   Uptime: 45.23s

📋 INFORMACIÓN DEL SERVIDOR:
   Nombre: OMEGA 21 v4.0 - Corteza Cognitiva Distribuida
   ...
```

---

## PASO 3: Integrar con tu código

### Opción A: Script standalone
Crea un archivo `entrenar_con_colab.ts`:

```typescript
import { ClienteColab } from './src/colab/cliente_colab';

const cliente = new ClienteColab({
  serverUrl: 'https://tu-url-ngrok.ngrok.io'
});

async function entrenarModelo() {
  // 1. Conectar
  await cliente.conectar();

  // 2. Cargar tus datos
  const datos = cargarMiDataset();  // Tu función

  // 3. Entrenar
  const resultado = await cliente.entrenar(datos, 3);  // 3 épocas

  // 4. Procesar resultados
  console.log('Loss:', resultado?.loss);
  console.log('Anomalías detectadas:', resultado?.outputs.anomaly_prob);

  // 5. Enviar feedback (opcional)
  const feedback = calcularFeedback(resultado);
  await cliente.enviarFeedback(feedback, true);
}

entrenarModelo().catch(console.error);
```

### Opción B: Integrar en tu sistema existente

Si tienes código que carga datos localmente:

```typescript
import { ClienteColab, MuestraEntrenamiento } from './src/colab/cliente_colab';

export class EntrenadorDistribuido {
  private cliente: ClienteColab;

  constructor() {
    this.cliente = new ClienteColab({
      serverUrl: process.env.COLAB_SERVER_URL!
    });
  }

  async entrenarEnColab(datos: any[]) {
    // Convertir tu formato a MuestraEntrenamiento
    const muestras: MuestraEntrenamiento[] = datos.map(d => ({
      input_data: d.features,  // Array 1600D
      anomaly_label: d.isAnomaly ? 1 : 0
    }));

    // Entrenar
    return await this.cliente.entrenar(muestras, 1);
  }
}
```

---

## PASO 4: Flujo completo

```
┌─────────────────────────────────────────────────────────┐
│         VS CODE WORKSPACE (Local)                       │
│                                                         │
│  1. ClienteColab.conectar()                            │
│  2. Cargar datos (CSV, JSON, etc)                      │
│  3. ClienteColab.entrenar(datos)                       │
│                    ↓                                    │
│              (HTTP Request)                             │
│                    ↓                                    │
└─────────────────────────────────────────────────────────┘
                      │
              ┌───────┴────────┐
              │                │
         ┌────▼────────────┐   │
         │  ngrok tunnel   │   │
         └────┬────────────┘   │
              │                │
┌─────────────────────────────────────────────────────────┐
│  Google Colab (GPU/TPU)                                 │
│                                                         │
│  FastAPI Server (Puerto 8000)                           │
│  ├─ POST /train_layer2 ← Recibe datos                  │
│  ├─ Ejecuta modelo (Capas 2-5)                         │
│  └─ Devuelve resultados (Loss + outputs)               │
│                    ↑                                    │
│              (HTTP Response)                            │
│                    ↑                                    │
└─────────────────────────────────────────────────────────┘
                      │
              ┌───────┴────────┐
              │                │
         ┌────▼────────────┐   │
         │  ngrok tunnel   │   │
         └────┬────────────┘   │
              │                │
┌─────────────────────────────────────────────────────────┐
│  VS CODE - Procesa resultados                           │
│  4. Visualizar loss, anomalías, etc.                   │
│  5. Enviar feedback (opcional)                          │
│  6. Siguiente iteración...                             │
└─────────────────────────────────────────────────────────┘
```

---

## ENDPOINTS disponibles

| Endpoint | Método | Input | Output |
|----------|--------|-------|--------|
| `/train_layer2` | POST | Lote (muestras + épocas) | Loss + anomaly + dendrites + coherence |
| `/feedback_dendritas` | POST | Ajustes + validación | Status + estadísticas |
| `/status` | GET | - | Estado servidor + métricas |
| `/health` | GET | - | ¿Vivo? + uptime |
| `/info` | GET | - | Arquitectura detallada |
| `/diagnostico` | POST | - | Test GPU + shapes outputs |
| `/metricas` | GET | - | Histórico losses + tendencia |

---

## ⚠️ Troubleshooting

### "Connection refused" o "Cannot connect"
1. Verifica que el servidor Colab sigue ejecutándose
2. Copia nuevamente la URL de ngrok (cambia cada reinicio)
3. Verifica que no hay firewall bloqueando ngrok

### "Input mismatch: expected 1600D, got XD"
Tu array de datos no tiene 1600 elementos. Revisa:
```typescript
console.log(misDatos[0].length);  // Debe ser 1600
```

### Server responde pero muy lento
- Colab puede tener otros notebooks ejecutándose
- Reduce tamaño del lote
- Cambia a GPU en "Entorno de ejecución"

### ngrok muere después de 2 horas
Es normal (límite gratuito). Soluciones:
- Actualiza ngrok a premium
- Reinicia el servidor cada 2 horas
- Usa SSH tunneling en lugar de ngrok

---

## Ejemplo de uso completo

```typescript
import { ClienteColab } from './src/colab/cliente_colab';
import * as fs from 'fs';

async function main() {
  const cliente = new ClienteColab({
    serverUrl: 'https://tu-url.ngrok.io'
  });

  // 1. Conectar
  const ok = await cliente.conectar();
  if (!ok) {
    console.error('No se puede conectar');
    process.exit(1);
  }

  // 2. Cargar dataset
  const dataset = JSON.parse(
    fs.readFileSync('mi_dataset.json', 'utf-8')
  );

  // 3. Entrenar en batches
  const batchSize = 32;
  for (let i = 0; i < dataset.length; i += batchSize) {
    const lote = dataset.slice(i, i + batchSize);
    
    const resultado = await cliente.entrenar(lote, 1);
    
    if (resultado) {
      console.log(`Batch ${Math.floor(i / batchSize)} - Loss: ${resultado.loss}`);
    }
  }

  // 4. Ver métricas finales
  const metricas = await cliente.obtenerMetricas();
  console.log('Tendencia:', metricas.tendencia);
}

main().catch(console.error);
```

---

## 🎯 Próximos pasos

1. **Optimización de datos**: Asegúrate que tus datos estén normalizados a 1600D
2. **Monitoreo**: Usa `/metricas` para trackear progreso
3. **Feedback local**: Implementa tus ajustes dendríticos locales y envíalos via `/feedback_dendritas`
4. **Escala**: Una vez funcione, entrena con datasets más grandes

---

¿Preguntas? Revisa los tipos en `cliente_colab.ts` para más detalles.
