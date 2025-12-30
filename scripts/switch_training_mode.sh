#!/bin/bash

# Script para gestionar el entorno de entrenamiento (Local vs Colab)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       🎛️  SELECTOR DE ENTORNO DE ENTRENAMIENTO               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

echo "1. 🏠 MODO LOCAL (VS Code CPU)"
echo "   - Ideal para: Pruebas, depuración, datasets pequeños"
echo "   - Velocidad: Baja"
echo "   - Costo: Gratis/Incluido"
echo ""
echo "2. ☁️  MODO REMOTO (Google Colab GPU)"
echo "   - Ideal para: Entrenamiento real, datasets grandes"
echo "   - Velocidad: Alta"
echo "   - Requiere: Configurar ngrok en Colab"
echo ""

read -p "Selecciona una opción (1/2): " OPCION

if [ "$OPCION" == "1" ]; then
    echo -e "\n✅ Configurando entorno LOCAL..."
    export COLAB_SERVER_URL="http://localhost:8000"
    
    echo "📦 Verificando dependencias Python..."
    pip install fastapi uvicorn psutil torch > /dev/null 2>&1
    
    echo "🚀 Iniciando servidor local en segundo plano..."
    nohup python3 src/local_server/servidor_local.py > local_server.log 2>&1 &
    SERVER_PID=$!
    echo "   PID: $SERVER_PID"
    
    echo "⏳ Esperando a que el servidor inicie..."
    sleep 5
    
    echo "🧪 Ejecutando prueba de entrenamiento..."
    npx ts-node src/colab/ejemplo_entrenamiento_colab.ts
    
    echo -e "\n⚠️  NOTA: El servidor sigue corriendo en segundo plano (PID $SERVER_PID)"
    echo "   Para detenerlo: kill $SERVER_PID"

elif [ "$OPCION" == "2" ]; then
    echo -e "\n✅ Configurando entorno REMOTO..."
    read -p "Ingresa la URL de ngrok (ej: https://xxxx.ngrok.io): " NGROK_URL
    export COLAB_SERVER_URL="$NGROK_URL"
    
    echo "🧪 Ejecutando prueba de conexión..."
    npx ts-node src/colab/cliente_colab.ts

else
    echo "❌ Opción no válida"
fi
