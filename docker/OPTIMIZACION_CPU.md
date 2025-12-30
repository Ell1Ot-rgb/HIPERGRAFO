# 🚀 MAXIMIZANDO EL ENTRENAMIENTO EN CPU (DOCKER)

Dado que tu entorno de Codespaces tiene **2 núcleos (cores)** disponibles, hemos aplicado una serie de optimizaciones de bajo nivel para exprimir cada ciclo de reloj sin saturar el sistema.

## 🛠️ Optimizaciones Aplicadas

### 1. Gestión de Hilos (Threading)
En CPUs con pocos núcleos, el mayor enemigo es el "Context Switching" (cuando el procesador pierde tiempo saltando entre demasiados hilos).
- **Configuración:** Hemos limitado PyTorch y OpenMP a exactamente **2 hilos** (`OMP_NUM_THREADS=2`).
- **Resultado:** El procesador se mantiene enfocado en el cálculo matricial sin distracciones.

### 2. Aceleración OneDNN (MKL-DNN)
Hemos habilitado el backend de **OneDNN** en el Dockerfile.
- **¿Qué hace?:** Utiliza instrucciones vectoriales avanzadas de tu CPU AMD EPYC (como **AVX2** y **FMA**) para acelerar las multiplicaciones de matrices.
- **Configuración:** `TORCH_CPU_BACKEND=onednn`.

### 3. Bibliotecas de Álgebra Lineal
Hemos cambiado las librerías estándar por **OpenBLAS** y **libomp**, que están mejor optimizadas para arquitecturas Linux modernas.

### 4. Límites de Recursos en Docker
El archivo `docker-compose.yml` ahora tiene reservas y límites estrictos:
- **CPUs:** 2.0 (Uso total de los núcleos disponibles).
- **Memoria:** Reserva de 1GB, límite de 4GB.

---

## 📈 Consejos para mejorar la velocidad en 2 Cores

Si sientes que el entrenamiento sigue siendo lento, aplica estos cambios en tu lógica de entrenamiento:

1.  **Batch Size Pequeño:** Usa un batch size de **8 o 16**. Esto permite que los datos quepan en la memoria caché (L2/L3) del procesador, que es miles de veces más rápida que la RAM.
2.  **Num Workers = 0:** En tus `DataLoaders` de PyTorch, establece `num_workers=0`. En sistemas de 2 núcleos, crear procesos adicionales para cargar datos suele ser más lento que cargarlos en el proceso principal.
3.  **Precisión Simple (Float32):** No intentes usar Double (Float64). Float32 es el "punto dulce" para CPUs con AVX2.

---

## 🛠️ Cómo aplicar los cambios

Si ya habías construido la imagen antes, debes reconstruirla para aplicar las nuevas optimizaciones:

```bash
# Detener y borrar lo anterior
docker stop omega21_local_trainer
docker rm omega21_local_trainer

# Reconstruir con las optimizaciones de CPU
./scripts/run_docker_training.sh
```

Puedes verificar que las optimizaciones están activas viendo los logs:
```bash
docker logs omega21_local_trainer
```
Deberías ver: `💻 MODO LOCAL: Usando cpu con 2 hilos`.
