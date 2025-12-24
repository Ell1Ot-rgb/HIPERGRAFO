#!/bin/bash

# SCRIPT: Verificación de Conexión con Colab
# Propósito: Validar que el ngrok tunnel está activo y que Colab responde correctamente

NGROK_URL="${1:-https://paleographic-transonic-adell.ngrok-free.dev}"
TIMEOUT=5

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                  VERIFICACIÓN DE CONEXIÓN COLAB - HIPERGRAFO                 ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🔗 URL NGROK: $NGROK_URL"
echo ""

# 1. Verificar DNS
echo "1️⃣  VERIFICANDO RESOLUCIÓN DNS..."
if nslookup $(echo $NGROK_URL | sed 's|https://||' | cut -d/ -f1) &>/dev/null; then
    echo "   ✅ DNS resuelto correctamente"
else
    echo "   ❌ No se puede resolver DNS"
    exit 1
fi
echo ""

# 2. Verificar conexión TCP
echo "2️⃣  VERIFICANDO CONEXIÓN TCP..."
HOST=$(echo $NGROK_URL | sed 's|https://||' | cut -d/ -f1)
if timeout $TIMEOUT bash -c "echo > /dev/tcp/$HOST/443" 2>/dev/null; then
    echo "   ✅ Puerto 443 (HTTPS) accesible"
else
    echo "   ❌ No se puede conectar al puerto 443"
    exit 1
fi
echo ""

# 3. Verificar endpoint /health
echo "3️⃣  VERIFICANDO ENDPOINT /health..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "ngrok-skip-browser-warning: true" \
    "$NGROK_URL/health" 2>/dev/null)

echo "   Código HTTP: $HEALTH_RESPONSE"
if [ "$HEALTH_RESPONSE" = "200" ] || [ "$HEALTH_RESPONSE" = "404" ]; then
    echo "   ✅ Servidor responde (ngrok activo)"
else
    echo "   ❌ Servidor no responde o error"
    echo "   💡 Posible causa: Colab no está ejecutando o ngrok tunnel cerrado"
fi
echo ""

# 4. Verificar endpoint /stream_data (POST vacío)
echo "4️⃣  VERIFICANDO ENDPOINT /stream_data..."
STREAM_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "ngrok-skip-browser-warning: true" \
    -d '{}' \
    -o /dev/null -w "%{http_code}" \
    "$NGROK_URL/stream_data" 2>/dev/null)

echo "   Código HTTP: $STREAM_RESPONSE"
if [ "$STREAM_RESPONSE" = "200" ] || [ "$STREAM_RESPONSE" = "400" ] || [ "$STREAM_RESPONSE" = "422" ]; then
    echo "   ✅ Endpoint /stream_data accesible"
else
    echo "   ⚠️  Endpoint /stream_data no responde como se esperaba (HTTP $STREAM_RESPONSE)"
fi
echo ""

# 5. Verificar endpoint /train (POST vacío)
echo "5️⃣  VERIFICANDO ENDPOINT /train..."
TRAIN_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "ngrok-skip-browser-warning: true" \
    -d '{}' \
    -o /dev/null -w "%{http_code}" \
    "$NGROK_URL/train" 2>/dev/null)

echo "   Código HTTP: $TRAIN_RESPONSE"
if [ "$TRAIN_RESPONSE" = "200" ] || [ "$TRAIN_RESPONSE" = "400" ] || [ "$TRAIN_RESPONSE" = "422" ]; then
    echo "   ✅ Endpoint /train accesible"
else
    echo "   ⚠️  Endpoint /train no responde como se esperaba (HTTP $TRAIN_RESPONSE)"
fi
echo ""

# 6. Verificar latencia
echo "6️⃣  VERIFICANDO LATENCIA..."
START=$(date +%s%N)
curl -s -o /dev/null \
    -H "ngrok-skip-browser-warning: true" \
    "$NGROK_URL/health" 2>/dev/null
END=$(date +%s%N)
LATENCIA_MS=$(( (END - START) / 1000000 ))

echo "   Latencia: ${LATENCIA_MS}ms"
if [ $LATENCIA_MS -lt 500 ]; then
    echo "   ✅ Latencia excelente (< 500ms)"
elif [ $LATENCIA_MS -lt 2000 ]; then
    echo "   ✅ Latencia aceptable (< 2s)"
else
    echo "   ⚠️  Latencia alta (> 2s)"
fi
echo ""

# 7. Resumen Final
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                           RESUMEN FINAL                                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

if [ "$HEALTH_RESPONSE" = "200" ] || [ "$HEALTH_RESPONSE" = "404" ]; then
    echo "✅ ESTADO: CONECTADO A COLAB"
    echo ""
    echo "📝 Comandos disponibles:"
    echo ""
    echo "   Entrenar local (Capas 0-1):"
    echo "   npm run simular_cognicion"
    echo ""
    echo "   Entrenar con Colab (Capas 0-1-2-3-4-5):"
    echo "   npm run simular_cognicion $NGROK_URL"
    echo ""
    echo "   Verificación detallada:"
    echo "   curl -H 'ngrok-skip-browser-warning: true' $NGROK_URL/health"
else
    echo "❌ ESTADO: NO CONECTADO A COLAB"
    echo ""
    echo "🔧 Acciones para reconectar:"
    echo ""
    echo "   1. Verifica que Colab está corriendo:"
    echo "      → Abre: https://colab.research.google.com"
    echo ""
    echo "   2. Ejecuta en Colab:"
    echo "      !pip install pyngrok fastapi uvicorn"
    echo ""
    echo "   3. Inicia ngrok tunnel en Colab:"
    echo "      from pyngrok import ngrok"
    echo "      public_url = ngrok.connect(8000)"
    echo "      print(f'URL: {public_url}')"
    echo ""
    echo "   4. Inicia servidor FastAPI en Colab"
    echo ""
    echo "   5. Repite verificación con nueva URL:"
    echo "      ./verificar_colab_conexion.sh <NUEVA_URL>"
fi
echo ""

exit 0
