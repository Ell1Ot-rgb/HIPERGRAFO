# 🐳 ENTRENAMIENTO AISLADO CON DOCKER

¡Buenas noticias! Tu entorno de VS Code soporta Docker perfectamente. Esto te permite correr el servidor de entrenamiento en un contenedor aislado, sin ensuciar tu entorno principal.

## 🏗️ Estructura del Entorno Docker

- **Imagen:** Basada en Python 3.10-slim (ligera).
- **Aislamiento:** El entrenamiento corre en su propio espacio de memoria (limitado a 4GB por defecto).
- **Persistencia:** Los modelos se guardan en la carpeta `/models` de tu workspace mediante volúmenes.

## 🚀 Cómo Lanzar el Entrenamiento

He creado un script simplificado para gestionar todo:

```bash
./scripts/run_docker_training.sh
```

Este script hará lo siguiente:
1. Construirá la imagen de Docker (solo la primera vez).
2. Levantará el contenedor en segundo plano.
3. Mapeará el puerto `8000` para que tu cliente local pueda conectarse.

## 🛠️ Comandos Útiles de Docker

Si prefieres gestionar los contenedores manualmente:

| Acción | Comando |
|--------|---------|
| **Ver Logs** | `docker logs -f omega21_local_trainer` |
| **Detener** | `docker stop omega21_local_trainer` |
| **Reiniciar** | `docker restart omega21_local_trainer` |
| **Estado** | `docker ps` |

## 🔗 Conexión desde el Cliente

Una vez que el contenedor esté corriendo, puedes usar el mismo cliente de siempre:

```bash
export COLAB_SERVER_URL=http://localhost:8000
npx ts-node src/colab/ejemplo_entrenamiento_colab.ts
```

## ⚠️ Notas sobre Rendimiento
Aunque Docker ofrece aislamiento, sigue usando la **CPU** de tu entorno de VS Code. El rendimiento será similar al del servidor local directo, pero con la ventaja de que si el entrenamiento falla o consume demasiada memoria, no afectará a tu editor de VS Code.
