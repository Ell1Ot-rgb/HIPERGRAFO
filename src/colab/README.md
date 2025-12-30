# � Entrenamiento Remoto en Colab desde VS Code

## ¿Qué es esto?

**Sistema completo para entrenar tu IA en Colab directamente desde VS Code:**

- ✅ Entrenar en GPU gratis de Colab
- ✅ Controlar desde tu PC local
- ✅ Enviar datos automáticamente
- ✅ Recibir resultados en tiempo real
- ✅ Integración sin complicaciones

```
Tu PC (VS Code)          Google Colab (GPU)
    ↓                            ↑
Generar datos        ←→    Entrenar modelo
Enviar lotes         ←→    Procesamiento
Recibir loss         ←→    Métricas
```

---

## 🔧 Configuración Paso a Paso

### 1️⃣ En Google Colab (una sola vez)

1. Abre [Google Colab](https://colab.research.google.com/)
2. Copia **TODO** el contenido de: `COLAB_SERVER_OMEGA21_V4_UNIFICADO.py`
3. Pégalo en **UNA SOLA CELDA**
4. Ejecuta (Shift + Enter)

**Deberías ver:**
```
🚀 INICIANDO OMEGA 21 v4.0...
📡 NGROK TUNNEL:
   ✅ https://tu-id-unico.ngrok-free.app
```

⭐ **Guarda esta URL** ← La necesitarás

### 2️⃣ En tu PC (VS Code)

```bash
# Terminal de VS Code
cd /workspaces/HIPERGRAFO

# Instalar dependencias (solo primera vez)
npm install

# Compilar TypeScript
npm run build

# Ejecuta prueba rápida
npx ts-node src/colab/cliente_colab.ts
```

Verás:
```
✅ Conexión exitosa
📋 INFORMACIÓN DEL SERVIDOR:
   Nombre: OMEGA 21 v4.0 - Corteza Cognitiva Distribuida
   ...
```

### 3️⃣ Para entrenar con tus datos

```bash
npx ts-node src/colab/ejemplo_entrenamiento_colab.ts
```

---

## 🎯 Casos de uso

### Caso 1: Entrenar dataset local con GPU remota
```typescript
import { ClienteColab } from './src/colab/cliente_colab';

const cliente = new ClienteColab({
  serverUrl: 'https://tu-url.ngrok.io'
});

// Cargar datos locales
const datos = cargarMiDataset();

// Entrenar en Colab
const resultado = await cliente.entrenar(datos);
```

### Caso 2: Monitoreo en tiempo real
```typescript
// Ver estado mientras entrenas
const estado = await cliente.obtenerEstado();
console.log('GPU:', estado.cuda_available);
console.log('Loss actual:', estado.estadisticas.loss_promedio_global);
```

### Caso 3: Feedback dendrítico bidireccional
```typescript
// Entrenar en Colab
const resultado = await cliente.entrenar(batch);

// Procesar localmente
const ajustes = procesarResultados(resultado);

// Enviar feedback al servidor
await cliente.enviarFeedback(ajustes, true);
```

---

## 📡 Arquitectura de la Conexión

```
┌─────────────────────────────────────────┐
│     VS CODE WORKSPACE (Local)           │
│                                         │
│  cliente_colab.ts                       │
│  ├─ conectar()                          │
│  ├─ entrenar(datos)                     │
│  ├─ enviarFeedback()                    │
│  └─ obtenerEstado()                     │
└──────────────────┬──────────────────────┘
                   │
              HTTPS (ngrok)
                   │
┌──────────────────▼──────────────────────┐
│   Google Colab (Python)                 │
│                                         │
│  FastAPI Server (puerto 8000)           │
│  ├─ POST /train_layer2                  │
│  ├─ POST /feedback_dendritas            │
│  ├─ GET  /status                        │
│  ├─ GET  /info                          │
│  └─ GET  /metricas                      │
│                                         │
│  CortezaCognitivaV4 (PyTorch)           │
│  ├─ Capa2 (LSTM + Transformer)          │
│  ├─ Capa3 (MLP Residual)                │
│  ├─ Capa4 (Self-Attention)              │
│  └─ Capa5 (Decision Heads)              │
│                                         │
│  GPU/TPU disponible                     │
└─────────────────────────────────────────┘
```

---

## 🔌 Endpoints Disponibles

Todos los endpoints están documentados con Swagger en: `{SERVER_URL}/docs`

| Endpoint | Método | Propósito |
|----------|--------|----------|
| `/train_layer2` | POST | Entrenar el modelo con un lote |
| `/feedback_dendritas` | POST | Enviar ajustes de feedback |
| `/status` | GET | Estado y estadísticas |
| `/health` | GET | Health check |
| `/info` | GET | Arquitectura del modelo |
| `/diagnostico` | POST | Test del sistema |
| `/metricas` | GET | Histórico de métricas |

---

## 🛠️ Ejemplo Completo

Archivo: `src/colab/ejemplo_entrenamiento_colab.ts`

Demuestra:
- ✅ Conectar al servidor remoto
- ✅ Cargar dataset local o generar datos sintéticos
- ✅ Entrenar en batches
- ✅ Monitorear progreso con barra de avance
- ✅ Enviar feedback cada N batches
- ✅ Recopilar estadísticas finales

Ejecutar:
```bash
npx ts-node src/colab/ejemplo_entrenamiento_colab.ts
```

---

## ⚠️ Problemas Comunes

### "Cannot connect to server"
```bash
# 1. Verifica que Colab sigue ejecutándose
# 2. Copia la URL de ngrok nuevamente (cambia cada reinicio)
# 3. Actualiza:
export COLAB_SERVER_URL=https://nueva-url.ngrok.io
```

### "Input dimension mismatch: expected 1600D, got XD"
Tu array no tiene 1600 elementos. Revisa:
```typescript
console.log(misDatos[0].input_data.length);  // Debe ser 1600
```

### "ngrok disconnected after 2 hours"
Es normal con ngrok gratuito. Soluciones:
- Actualiza a ngrok premium
- Reinicia el servidor cada 2 horas
- Usa SSH tunneling en lugar de ngrok

---

## 📊 Monitoreo

Desde VS Code puedes monitore en tiempo real:

```typescript
const cliente = new ClienteColab({ serverUrl: 'https://...' });

setInterval(async () => {
  const estado = await cliente.obtenerEstado();
  const metricas = await cliente.obtenerMetricas();
  
  console.clear();
  console.log('Loss:', estado.estadisticas.loss_promedio_global);
  console.log('Tendencia:', metricas.tendencia);
  console.log('GPU:', estado.cuda_available ? '✅' : '❌');
}, 5000);  // Cada 5 segundos
```

---

## 🚀 Próximos Pasos

1. **Optimizar datos**: Asegúrate que tus datos estén normalizados a 1600D
2. **Escalar**: Entrena con datasets más grandes en Colab
3. **Feedback**: Implementa tus propios ajustes dendríticos locales
4. **Persistencia**: Guarda el modelo entrenado después de entrenar
5. **Automatizar**: Integra con tu pipeline CI/CD

---

## 📚 Documentación Completa

Ver: [`docs/GUIA_ACCESO_COLAB.md`](../docs/GUIA_ACCESO_COLAB.md)

---

## 💡 Casos Avanzados

### Entrenamiento Distribuido Multi-Nodo
Si tienes múltiples instancias de Colab, puedes:
```typescript
const cliente1 = new ClienteColab({ serverUrl: 'https://url1.ngrok.io' });
const cliente2 = new ClienteColab({ serverUrl: 'https://url2.ngrok.io' });

// Entrenar en paralelo
await Promise.all([
  cliente1.entrenar(batchA),
  cliente2.entrenar(batchB)
]);
```

### Monitoreo con Webhooks
Enviar notificaciones cuando el loss mejora:
```typescript
const estado = await cliente.obtenerEstado();
if (estado.estadisticas.loss_promedio_global < umbral) {
  enviarNotificacion('¡Loss mejoró! Nueva métrica: ' + ...);
}
```

### Persistencia del Modelo
Después de entrenar, el modelo en Colab está actualizado. Para guardar:
```python
# En Colab
torch.save(model.state_dict(), 'modelo_entrenado.pt')
# Descárgalo manualmente o usa Google Drive
```

---

## 📞 Soporte

¿Preguntas? Revisa:
- `cliente_colab.ts` - Tipos y métodos disponibles
- `ejemplo_entrenamiento_colab.ts` - Caso de uso completo
- `docs/GUIA_ACCESO_COLAB.md` - Guía detallada
- Swagger en `{SERVER_URL}/docs` - API completa

---

**¡Disfruta entrenando con GPU gratis en Colab! 🚀**
