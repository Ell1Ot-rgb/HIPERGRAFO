#!/bin/bash

# Script para lanzar el entrenamiento en Docker

echo "🐳 INICIANDO ENTORNO DE ENTRENAMIENTO DOCKER"
echo "============================================"

# Ir al directorio del script
cd "$(dirname "$0")/../docker"

# Verificar si Docker está corriendo
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker no parece estar corriendo. Por favor, inícialo primero."
    exit 1
fi

echo "🏗️  Construyendo imagen de entrenamiento (esto puede tardar la primera vez)..."
docker compose build

echo "🚀 Lanzando contenedor..."
docker compose up -d

echo "✅ Servidor de entrenamiento corriendo en: http://localhost:8000"
echo "📊 Puedes ver los logs con: docker compose logs -f"
echo "🛑 Para detenerlo: docker compose down"

# Volver al directorio original
cd - > /dev/null
