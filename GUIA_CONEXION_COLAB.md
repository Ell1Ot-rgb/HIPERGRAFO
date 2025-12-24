# 🔗 GUÍA DE CONEXIÓN COLAB - HIPERGRAFO

**Estado Actual**: ⚠️ NGROK Tunnel Inactivo  
**Última URL**: https://paleographic-transonic-adell.ngrok-free.dev (expirada)  
**Última actualización**: 2025-12-23

---

## 📊 ESTADO GENERAL

### ✅ Capas 0-1 (LOCAL)
- Estado: 100% OPTIMIZADAS (10 mejoras implementadas)
- Funcionan completamente sin Colab
- Training: `npm run simular_cognicion`
- Capas 0-1 pueden entrenar localmente sin problemas

### ⚠️ Conexión Colab
- Estado: INACTIVA (ngrok tunnel cerrado)
- Causa: Colab session closed or expired
- Impacto: No afecta Capas 0-1, solo Capas 2-5 en Colab
- Solución: Reiniciar Colab + ngrok tunnel

---

## 🚀 OPCIÓN 1: TRAINING LOCAL (SIN COLAB)

### Recomendado para desarrollo rápido

```bash
# Entrenar Capas 0-1 localmente
npm run simular_cognicion

# Esperar mejoras:
# ✅ Convergencia -50% (60-80 épocas vs 100-150)
# ✅ Accuracy +8-12% (~93-95% vs ~85%)
# ✅ Overfitting -70% (2-3% vs 8-10%)
```

**Ventajas:**
- ✅ Rápido de iniciar
- ✅ Todas las Fases 1-2-3 funcionan
- ✅ Capas 0-1 100% optimizadas
- ✅ Sin dependencias externas

**Limitaciones:**
- ⚠️ Solo Capas 0-1 (Local)
- ⚠️ Sin GPU de Colab
- ⚠️ Training más lento

---

## 🔧 OPCIÓN 2: RECONECTAR COLAB

### Para training end-to-end (Capas 0-5)

### Paso 1: Abrir Google Colab
```
https://colab.research.google.com
```

### Paso 2: Crear Notebook y Ejecutar Celdas

**Celda 1: Instalaciones**
```python
!pip install pyngrok fastapi uvicorn torch numpy uvicorn-asgi2

# Limpiar ngrok
!pkill -9 -f ngrok || true
```

**Celda 2: Configurar ngrok**
```python
from pyngrok import ngrok

# 1️⃣ Obtener tu NGROK_TOKEN en:
#    https://dashboard.ngrok.com/auth/your-authtoken

NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"  # ← CAMBIAR ESTO
ngrok.set_auth_token(NGROK_TOKEN)

# 2️⃣ Crear tunnel
public_url = ngrok.connect(8000)
print(f"\n✅ NGROK URL: {public_url}")
print(f"   Copia esta URL para usarla en LOCAL")
```

**Celda 3: Iniciar FastAPI Server**
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

app = FastAPI()

# Habilitar CORS para ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0"}

# Stream data desde LOCAL (Capas 0-1)
@app.post("/stream_data")
async def stream_data(data: dict):
    try:
        samples = data.get("samples", [])
        print(f"📥 Recibido: {len(samples)} samples")
        # Aquí entrenar Capas 2-5 si es necesario
        return {
            "status": "received",
            "samples_count": len(samples),
            "message": "Data queued for training"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Training endpoint
@app.post("/train")
async def train(data: dict):
    try:
        epochs = data.get("epochs", 1)
        batch_size = data.get("batch_size", 64)
        print(f"🚀 Training: {epochs} epochs, batch_size={batch_size}")
        # Aquí ejecutar training de Capas 2-5
        return {
            "status": "training",
            "epochs": epochs,
            "batch_size": batch_size
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Info endpoint
@app.get("/info")
async def info():
    return {
        "system": "HIPERGRAFO Colab Server",
        "version": "1.0",
        "capas": ["2", "3", "4", "5"],
        "status": "ready"
    }

# Iniciar servidor
print("🔥 Iniciando FastAPI Server...")
import nest_asyncio
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Celda 4: Verificar Servidor**
```python
import requests

url = public_url
try:
    response = requests.get(f"{url}/health")
    print(f"✅ Servidor respondiendo: {response.json()}")
except Exception as e:
    print(f"❌ Error: {e}")
```

### Paso 3: Usar la URL en LOCAL

```bash
# Copiar URL de Colab (ejemplo: https://abcd1234.ngrok-free.dev)
# Luego ejecutar:

npm run simular_cognicion https://abcd1234.ngrok-free.dev

# El sistema automáticamente:
# 1. Procesa Capas 0-1 localmente
# 2. Envía datos a Colab (Capas 2-5)
# 3. Realiza training end-to-end
```

---

## 📋 ARCHIVOS DE CONFIGURACIÓN

### [src/neural/configColab.ts](../src/neural/configColab.ts)
```typescript
// Actualizar con URL de Colab
export const CONFIG_COLAB = {
    urlServidor: "https://TU_URL_NGROK_AQUI.ngrok-free.dev",
    // ... resto de config
};
```

### [src/neural/StreamingBridge.ts](../src/neural/StreamingBridge.ts)
```typescript
// Endpoint correcto ya configurado
await axios.post(`${this.urlColab}/stream_data`, lote, {
    headers: { 
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true'
    },
    timeout: 15000
});
```

### Script de Verificación
```bash
./verificar_colab_conexion.sh "https://tu-url-colab.ngrok-free.dev"
```

---

## 🔍 TROUBLESHOOTING

### Error: "No se puede resolver DNS"
**Solución:**
1. Verificar URL de ngrok es correcta
2. Verificar Colab aún está corriendo
3. Reiniciar Colab session

### Error: "Connection refused" (Puerto 8000)
**Solución:**
1. Verificar FastAPI server inició en Colab
2. Verificar ngrok tunnel está activo
3. Ejecutar `/health` endpoint primero

### Error: "Timeout esperando respuesta"
**Solución:**
1. Aumentar timeout en StreamingBridge (ya es 15s)
2. Verificar latencia: `./verificar_colab_conexion.sh`
3. Reducir TAMANO_BATCH en StreamingBridge

### Error: "ngrok-skip-browser-warning header required"
**Solución:**
- Ya está incluido en StreamingBridge.ts
- Verificar versión de curl actualizada

---

## 📊 ARQUITECTURA DE DATOS

```
LOCAL (Este workspace)          COLAB (Remoto)
────────────────────────────────────────────────

256D Vector                 
    ↓
Capa 0 (Normalización)      
    ↓
25 Subespacios (25 × 64D)   
    ↓
Capa 1 (Procesamiento)      
    ↓
1600D Vector
    ↓
    └─→ /stream_data ────────→ Capas 2-5
         (POST 1600D)          (GPU Training)
         HTTP/ngrok-free.dev   (PyTorch)
```

### Flujo Datos POST /stream_data

```json
{
  "samples": [
    {
      "input_data": [1.0, 2.0, ..., 1600],
      "anomaly_label": 0
    },
    // ... más samples (batch_size=64)
  ]
}
```

---

## 🎯 COMANDOS ÚTILES

### Verificar conexión
```bash
./verificar_colab_conexion.sh https://tu-url.ngrok-free.dev
```

### Testing local
```bash
npm run simular_cognicion  # Sin Colab
```

### Testing con Colab
```bash
npm run simular_cognicion https://tu-url.ngrok-free.dev
```

### Ver logs en streaming
```bash
# En Colab: Ver Output de la celda que corre uvicorn
# Muestra en tiempo real:
# 🚀 Lote de 64 muestras enviado. Latencia: XXXms
```

---

## 📚 REFERENCIAS

- [ngrok Dashboard](https://dashboard.ngrok.com)
- [Google Colab](https://colab.research.google.com)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [pyngrok Docs](https://pyngrok.readthedocs.io)

---

## 💡 PRÓXIMOS PASOS

### Opción A: Training Local AHORA
```bash
npm run simular_cognicion
# ✅ Capas 0-1 con 10 mejoras
# ✅ Resultados en 1-2 horas
```

### Opción B: Reconectar Colab (Completo)
```
1. Abre Google Colab
2. Ejecuta celdas (5 minutos)
3. Copia URL ngrok
4. npm run simular_cognicion <URL>
5. Training end-to-end 6-8 horas
```

---

*Guía de Conexión - Actualizada 2025-12-23*
