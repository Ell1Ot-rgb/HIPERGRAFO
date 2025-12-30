# 🏠 ENTRENAMIENTO LOCAL EN VS CODE

## Análisis de Viabilidad

¿Es posible entrenar OMEGA 21 directamente en este entorno (VS Code/Codespaces)?
**SÍ, es posible, pero con limitaciones importantes.**

### 📊 Comparativa: Local (VS Code) vs. Remoto (Colab)

| Característica | 🏠 VS Code Local (CPU) | ☁️ Google Colab (GPU) |
|----------------|------------------------|-----------------------|
| **Velocidad** | Lenta (x1) | Muy Rápida (x50 - x100) |
| **Hardware** | CPU (AMD EPYC 64-Core) | GPU (NVIDIA T4/A100) |
| **RAM** | ~8 GB (Compartida) | ~12-25 GB (Dedicada) |
| **Uso Ideal** | Depuración, Pruebas, Datasets pequeños | Entrenamiento masivo, Datasets grandes |
| **Persistencia**| Alta (Archivos se guardan) | Baja (Se borra al cerrar) |

### 🛠️ Estrategia Híbrida (Recomendada)

1.  **Desarrollo y Pruebas (AQUÍ):** Usa el servidor local para verificar que tu código funciona, probar la arquitectura y entrenar con pocos datos (ej: 100 muestras).
2.  **Entrenamiento Pesado (COLAB):** Cuando todo funcione, cambia la URL a Colab para entrenar con miles de datos.

---

## 🚀 Cómo usar el Servidor Local

He creado una versión optimizada para CPU del servidor (`src/local_server/servidor_local.py`).

### 1. Instalar dependencias
```bash
pip install fastapi uvicorn psutil torch
```

### 2. Iniciar el servidor
Abre una terminal y ejecuta:
```bash
python3 src/local_server/servidor_local.py
```
Verás:
```
🏠 SERVIDOR LOCAL OMEGA 21 - INICIANDO
   • URL: http://localhost:8000
   • CPU: 64 cores
```

### 3. Configurar el Cliente
En otra terminal, configura la variable de entorno para apuntar a `localhost`:

```bash
export COLAB_SERVER_URL=http://localhost:8000
```

### 4. Ejecutar el entrenamiento
Ahora ejecuta el mismo script de cliente que ya tenías:

```bash
npx ts-node src/colab/ejemplo_entrenamiento_colab.ts
```

---

## ⚠️ Optimizaciones Realizadas (Versión CPU)

Para que funcione fluido en VS Code, he modificado el modelo en `servidor_local.py`:
1.  **Reducción de Capas:** LSTM y Transformer tienen menos capas (1 en vez de 2).
2.  **Menos Neuronas:** Capas densas reducidas (1024 en vez de 4096).
3.  **Sin CUDA:** Forzado a usar `device='cpu'`.
4.  **Monitor de RAM:** El servidor rechazará peticiones si la RAM supera el 90% para evitar que se cuelgue el entorno.
