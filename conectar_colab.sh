#!/bin/bash

# Script: conectar_colab.sh
# Uso: ./conectar_colab.sh <URL_COLAB> [opciones]
# 
# Ejemplos:
#   ./conectar_colab.sh https://tu-id.ngrok-free.app
#   ./conectar_colab.sh https://tu-id.ngrok-free.app --muestras 1000 --diagnostico

set -e

echo "=================================================="
echo "🧠 CONEXIÓN A COLAB - OMEGA 21"
echo "=================================================="

# Validar URL
if [ -z "$1" ]; then
    echo ""
    echo "❌ Error: Debes proporcionar la URL de Colab"
    echo ""
    echo "Uso: ./conectar_colab.sh <URL_COLAB> [opciones]"
    echo ""
    echo "Ejemplo:"
    echo "  ./conectar_colab.sh https://tu-id-unico.ngrok-free.app"
    echo ""
    echo "Opciones:"
    echo "  --muestras <num>      Número de muestras (default: 500)"
    echo "  --lote <num>          Tamaño del lote (default: 64)"
    echo "  --tipo <tipo>         simple|temporal|neuronal (default: simple)"
    echo "  --anomalias <pct>     Porcentaje de anomalías (default: 10)"
    echo "  --diagnostico         Ejecutar diagnóstico"
    echo "  --metricas            Mostrar métricas"
    echo ""
    exit 1
fi

URL="$1"
shift  # Descartar primer argumento

echo ""
echo "📡 URL del servidor: $URL"
echo ""
echo "✅ Iniciando entrenamiento..."
echo ""

# Ejecutar script TypeScript con parámetros
npx ts-node src/colab/entrenar_con_colab.ts "$URL" "$@"

echo ""
echo "=================================================="
echo "✅ Completado"
echo "=================================================="
