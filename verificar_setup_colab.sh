#!/bin/bash

# verificar_setup_colab.sh
# Script para verificar que todo está configurado correctamente

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   🔍 VERIFICACIÓN DE SETUP - ENTRENAMIENTO CON COLAB         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

verificar() {
    local nombre="$1"
    local comando="$2"
    
    if eval "$comando" &> /dev/null; then
        echo -e "${GREEN}✅${NC} $nombre"
        return 0
    else
        echo -e "${RED}❌${NC} $nombre"
        return 1
    fi
}

advertencia() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

info() {
    echo -e "ℹ️  $1"
}

# ============ VERIFICACIONES ============

echo "📦 DEPENDENCIAS DEL SISTEMA:"
verificar "Node.js instalado" "node --version &> /dev/null"
verificar "npm instalado" "npm --version &> /dev/null"
verificar "TypeScript disponible" "npx tsc --version &> /dev/null"

echo ""
echo "📁 ESTRUCTURA DE ARCHIVOS:"
verificar "Carpeta src/colab/" "test -d src/colab"
verificar "ClienteColabEntrenamiento.ts" "test -f src/colab/ClienteColabEntrenamiento.ts"
verificar "GeneradorDatosEntrenamiento.ts" "test -f src/colab/GeneradorDatosEntrenamiento.ts"
verificar "entrenar_con_colab.ts" "test -f src/colab/entrenar_con_colab.ts"
verificar "config.colab.ts" "test -f src/colab/config.colab.ts"
verificar "ejemplo_integracion_completa.ts" "test -f src/colab/ejemplo_integracion_completa.ts"

echo ""
echo "📜 DOCUMENTACIÓN:"
verificar "GUIA_RAPIDA_COLAB.md" "test -f GUIA_RAPIDA_COLAB.md"
verificar "INSTALACION_RAPIDA.md" "test -f INSTALACION_RAPIDA.md"
verificar "src/colab/README.md" "test -f src/colab/README.md"

echo ""
echo "🔧 SERVIDOR COLAB:"
verificar "COLAB_SERVER_OMEGA21_V4_UNIFICADO.py" "test -f COLAB_SERVER_OMEGA21_V4_UNIFICADO.py"

echo ""
echo "📝 SCRIPTS:"
verificar "conectar_colab.sh" "test -f conectar_colab.sh"
if test -x conectar_colab.sh; then
    echo -e "${GREEN}✅${NC} conectar_colab.sh es ejecutable"
else
    advertencia "conectar_colab.sh no es ejecutable"
    info "Usa: chmod +x conectar_colab.sh"
fi

echo ""
echo "🎯 VERIFICACIONES OPCIONALES:"

# Verificar Node modules
if test -d node_modules; then
    echo -e "${GREEN}✅${NC} Dependencias npm instaladas"
    echo "   Paquetes: $(ls node_modules | wc -l)"
else
    advertencia "npm packages no instalados (ejecuta: npm install)"
fi

# Verificar dist
if test -d dist; then
    echo -e "${GREEN}✅${NC} TypeScript compilado en dist/"
    if test -f dist/colab/ClienteColabEntrenamiento.js; then
        echo -e "${GREEN}✅${NC} ClienteColabEntrenamiento.js compilado"
    fi
else
    advertencia "dist/ no existe (ejecuta: npm run build)"
fi

echo ""
echo "═════════════════════════════════════════════════════════════════"
echo ""

# Instrucciones finales
echo "📝 PRÓXIMOS PASOS:"
echo ""

if ! test -d node_modules; then
    echo "1. Instalar dependencias:"
    echo "   $ npm install"
    echo ""
fi

if ! test -d dist; then
    echo "2. Compilar TypeScript:"
    echo "   $ npm run build"
    echo ""
fi

echo "3. Ejecutar servidor en Google Colab:"
echo "   • Visita: https://colab.research.google.com/"
echo "   • Copia COLAB_SERVER_OMEGA21_V4_UNIFICADO.py"
echo "   • Pégalo en una celda de Colab y ejecuta"
echo "   • Copia la URL de ngrok que aparece"
echo ""

echo "4. Ejecutar entrenamiento desde VS Code:"
echo "   $ ./conectar_colab.sh https://tu-url-colab.ngrok-free.app"
echo ""

echo "5. (Opcional) Ejecutar ejemplo completo:"
echo "   $ COLAB_SERVER_URL=https://tu-url npx ts-node src/colab/ejemplo_integracion_completa.ts"
echo ""

echo "═════════════════════════════════════════════════════════════════"
echo ""
echo "✨ ¡Verificación completada!"
echo ""
