# 🌉 Puente Hipergrafo ↔️ Google Colab

## Descripción
Este puente permite que tu **Hipergrafo (TypeScript en Codespaces)** se comunique directamente con **Modelos de IA en Google Colab (Python)**.

Es una arquitectura **IA ↔️ IA**: Copilot controla Codespaces, Gemini controla Colab, y ambas IAs coordinan a través de un servidor REST.

---

## 🚀 Cómo Usar

### Paso 1: Configurar Colab (Una sola vez)

1. Abre un nuevo cuaderno en [Google Colab](https://colab.research.google.com)
2. En la **primera celda**, pega y ejecuta este código:

```python
# ==========================================
# CÓDIGO PARA GOOGLE COLAB
# ==========================================

!pip install fastapi uvicorn pyngrok nest_asyncio

import nest_asyncio
from pyngrok import ngrok
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "online", "mensaje": "Hipergrafo Neural Server Activo"}

@app.post("/procesar")
def procesar_datos(data: dict = Body(...)):
    """
    Aquí va tu modelo de IA.
    Por ahora, devolvemos un análisis dummy.
    """
    print(f"Recibido paquete de datos con {len(data)} claves.")
    
    resultado = {
        "analisis": "Procesado por Colab",
        "nodos_afectados": list(data.keys()),
        "prediccion": 0.95
    }
    
    return resultado

ngrok.kill()
tunnel = ngrok.connect(8000)
public_url = tunnel.public_url

print(f"\n✅ PUENTE ESTABLECIDO EXITOSAMENTE")
print(f"🔗 COPIA ESTA URL Y PÉGALA EN CODESPACES: {public_url}")

nest_asyncio.apply()
uvicorn.run(app, port=8000)
```

3. **Copia la URL** que aparece (ej: `https://overexpressive-percy-unapportioned.ngrok-free.dev`)

### Paso 2: Actualizar URL en Codespaces

Abre el archivo [src/neural/configColab.ts](src/neural/configColab.ts) y reemplaza:

```typescript
urlServidor: "https://overexpressive-percy-unapportioned.ngrok-free.dev",
```

Con la URL que copiaste de Colab.

### Paso 3: Ejecutar Prueba

```bash
cd /workspaces/HIPERGRAFO
npx ts-node src/pruebas/prueba_colab.ts
```

Deberías ver:
```
✅ Puente con Colab ACTIVO
📤 Enviando Hipergrafo a Colab...
📥 Respuesta recibida de Colab: {...}
✅ FLUJO COMPLETADO EXITOSAMENTE
```

---

## 📦 Arquitectura

### Clases principales

| Archivo | Propósito |
|---------|-----------|
| [ColabBridge.ts](src/neural/ColabBridge.ts) | Cliente HTTP que habla con el servidor Colab |
| [IntegradorHipergrafoColo.ts](src/neural/IntegradorHipergrafoColo.ts) | Orquestador: crea Hipergrafos, los serializa y envía a Colab |
| [configColab.ts](src/neural/configColab.ts) | Configuración (URL del servidor, timeouts, etc) |

### Flujo de datos

```
Codespaces (TypeScript)          Google Colab (Python)
┌─────────────────────┐          ┌──────────────────┐
│ Hipergrafo          │          │ Modelo de IA     │
│ {nodos, edges}      │          │ (PyTorch, etc)   │
└──────────┬──────────┘          └────────┬─────────┘
           │                              ▲
           │ JSON                         │ JSON
           │ (HTTP POST)                  │ (HTTP Response)
           └──────────────────────────────┘
                 ngrok tunnel
```

---

## 🔧 Personalización

### Agregar lógica de IA en Colab

En el endpoint `/procesar` de Colab, puedes agregar tu modelo:

```python
@app.post("/procesar")
def procesar_datos(data: dict = Body(...)):
    # Tu modelo aquí
    import torch
    modelo = cargar_modelo("mi_modelo.pt")
    
    # Procesar datos del Hipergrafo
    resultado = modelo.predict(data)
    
    return {"analisis": resultado, "confianza": 0.95}
```

### Usar desde tu código TypeScript

```typescript
import { IntegradorHipergrafoColo } from './src/neural';

const integrador = new IntegradorHipergrafoColo("Mi Proyecto");

// Crear hipergrafo manualmente
integrador.crearEjemplo();

// Enviar a Colab
const resultado = await integrador.procesarEnColab();

console.log(resultado);
```

---

## 🐛 Troubleshooting

### ❌ "No se pudo conectar con Colab"

1. ¿Está corriendo el cuaderno de Colab?
2. ¿La URL de ngrok es correcta en `configColab.ts`?
3. ¿ngrok cambió la URL? (Sucede cuando reinicias Colab)

**Solución:** Vuelve a ejecutar el código en Colab y copia la nueva URL.

### 🔗 La URL de ngrok expiró

Las URLs de ngrok gratuitas expiran cada 2 horas. Si la conexión falla:

1. Recarga el cuaderno en Colab
2. Ejecuta el código nuevamente
3. Copia la nueva URL y actualiza `configColab.ts`

---

## 📚 Referencias

- **FastAPI**: https://fastapi.tiangolo.com/
- **ngrok**: https://ngrok.com/
- **Google Colab**: https://colab.research.google.com/
- **TypeScript Fetch API**: https://developer.mozilla.org/en-US/docs/Web/API/fetch

---

## ✅ Checklist

- [ ] Código compilado en Codespaces (`npm run build`)
- [ ] Servidor corriendo en Colab
- [ ] URL de ngrok copiada a `configColab.ts`
- [ ] Prueba ejecutada: `npx ts-node src/pruebas/prueba_colab.ts`
- [ ] IA ↔️ IA comunicándose correctamente
