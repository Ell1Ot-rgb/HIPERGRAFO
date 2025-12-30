#!/bin/bash

# 🚀 SCRIPT DE CONFIGURACIÓN - ACCESO A SERVIDOR COLAB
# Este script facilita la configuración de la conexión con Colab

set -e

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║           🔗 CONFIGURACIÓN: ACCESO REMOTO A SERVIDOR COLAB                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==========================================
# PASO 1: VERIFICAR DEPENDENCIAS
# ==========================================

echo -e "\n${BLUE}📋 Verificando dependencias...${NC}"

if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js no encontrado${NC}"
    echo "   Instala desde: https://nodejs.org/"
    exit 1
fi
echo -e "${GREEN}✅ Node.js: $(node --version)${NC}"

if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm no encontrado${NC}"
    exit 1
fi
echo -e "${GREEN}✅ npm: $(npm --version)${NC}"

if ! command -v npx &> /dev/null; then
    echo -e "${RED}❌ npx no encontrado${NC}"
    exit 1
fi
echo -e "${GREEN}✅ npx disponible${NC}"

# ==========================================
# PASO 2: OBTENER URL DE COLAB
# ==========================================

echo -e "\n${BLUE}🌐 Configurar URL del servidor Colab${NC}"
echo -e "${YELLOW}Necesitas la URL pública que genera ngrok en Colab${NC}"
echo ""
echo "Para obtenerla:"
echo "1. Ejecuta el servidor en Colab: COLAB_SERVER_OMEGA21_V4_UNIFICADO.py"
echo "2. Busca la línea con ngrok:"
echo "   📡 NGROK TUNNEL:"
echo "      ✅ https://xxxx-xxxx-xxxx-xxxx.ngrok.io"
echo "3. Copia esa URL"
echo ""

# Opción A: Variable de entorno
echo -e "${YELLOW}Opción 1: Usar variable de entorno (recomendado)${NC}"
echo ""
echo "Ejecuta en tu terminal:"
echo -e "${GREEN}export COLAB_SERVER_URL=https://tu-url-aqui.ngrok.io${NC}"
echo ""
echo "Después verifica:"
echo -e "${GREEN}echo \$COLAB_SERVER_URL${NC}"
echo ""

# Opción B: Editar archivo
echo -e "${YELLOW}Opción 2: Editar archivo (alternativa)${NC}"
echo ""
echo "Edita: src/colab/cliente_colab.ts"
echo "Línea ~23, reemplaza:"
echo -e "${RED}serverUrl: process.env.COLAB_SERVER_URL || 'http://localhost:8000',${NC}"
echo "Por:"
echo -e "${GREEN}serverUrl: 'https://tu-url-aqui.ngrok.io',${NC}"
echo ""

read -p "¿Ya tienes la URL de Colab? (s/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    read -p "Ingresa la URL (ej: https://1234-5678.ngrok.io): " COLAB_URL
    
    if [[ ! $COLAB_URL =~ ^https?:// ]]; then
        echo -e "${RED}❌ URL inválida (debe empezar con http:// o https://)${NC}"
        exit 1
    fi
    
    export COLAB_SERVER_URL="$COLAB_URL"
    echo -e "${GREEN}✅ URL configurada: $COLAB_SERVER_URL${NC}"
else
    echo -e "${YELLOW}⚠️ Necesitarás la URL de Colab para continuar${NC}"
fi

# ==========================================
# PASO 3: INSTALAR DEPENDENCIAS NODE
# ==========================================

echo -e "\n${BLUE}📦 Instalando dependencias Node.js...${NC}"

if [ ! -d "node_modules" ]; then
    npm install
    echo -e "${GREEN}✅ Dependencias instaladas${NC}"
else
    echo -e "${GREEN}✅ node_modules ya existe${NC}"
fi

# ==========================================
# PASO 4: PROBAR CONEXIÓN
# ==========================================

echo -e "\n${BLUE}🧪 Probando conexión con servidor Colab...${NC}"

if [ -z "$COLAB_SERVER_URL" ]; then
    echo -e "${YELLOW}⚠️ Skipping test: URL no configurada${NC}"
else
    echo ""
    npx ts-node src/colab/cliente_colab.ts
fi

# ==========================================
# RESUMEN
# ==========================================

echo -e "\n${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ CONFIGURACIÓN COMPLETADA${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}Próximos pasos:${NC}"
echo ""
echo "1️⃣ Asegúrate de que el servidor Colab está ejecutándose:"
echo "   • COLAB_SERVER_OMEGA21_V4_UNIFICADO.py en Colab"
echo "   • URL de ngrok disponible"
echo ""
echo "2️⃣ Configura la URL de Colab:"
echo "   export COLAB_SERVER_URL=https://tu-url.ngrok.io"
echo ""
echo "3️⃣ Ejecuta ejemplos:"
echo "   • Prueba rápida:"
echo "     npx ts-node src/colab/cliente_colab.ts"
echo ""
echo "   • Ejemplo completo:"
echo "     npx ts-node src/colab/ejemplo_entrenamiento_colab.ts"
echo ""
echo "4️⃣ Para uso en tu código:"
echo "   import { ClienteColab } from './src/colab/cliente_colab';"
echo ""
echo "   const cliente = new ClienteColab({"
echo "     serverUrl: process.env.COLAB_SERVER_URL!"
echo "   });"
echo ""
echo "   await cliente.conectar();"
echo "   const resultado = await cliente.entrenar(datos);"
echo ""

echo -e "${YELLOW}Documentación:${NC}"
echo "   • Guía completa: docs/GUIA_ACCESO_COLAB.md"
echo "   • Cliente API: src/colab/cliente_colab.ts"
echo "   • Ejemplo: src/colab/ejemplo_entrenamiento_colab.ts"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
