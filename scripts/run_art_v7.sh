#!/bin/bash

# 🚀 Script para lanzar el Reactor ART V7 en Docker

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║          ⚛️  LANZADOR REACTOR ART V7 - SERVIDOR DOCKER                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"

cd "$(dirname "$0")/../docker" || exit 1

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Verificar Docker
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker no está corriendo${NC}"
    exit 1
fi

echo -e "\n${BLUE}🏗️  Construyendo imagen ART V7...${NC}"
docker compose build

echo -e "\n${BLUE}🚀 Lanzando Reactor ART V7...${NC}"
docker compose up -d

echo -e "\n${GREEN}✅ REACTOR ONLINE${NC}"
echo -e "${YELLOW}   URL: http://localhost:8000${NC}"
echo -e "${YELLOW}   Docs: http://localhost:8000/docs${NC}"
echo -e "${YELLOW}   Logs: docker compose logs -f${NC}"
echo -e "\n${BLUE}Esperando a que el Reactor inicie...${NC}"
sleep 3

# Intentar conectar
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}   ✓ Reactor respondiendo${NC}"
        
        # Ejecutar cliente de prueba
        echo -e "\n${BLUE}🧪 Ejecutando prueba de cliente...${NC}"
        cd ../.. || exit 1
        export COLAB_SERVER_URL=http://localhost:8000
        npx ts-node src/colab/cliente_art_v7.ts
        
        exit 0
    fi
    echo -e "   Intento $i/10..."
    sleep 1
done

echo -e "${YELLOW}⚠️  Reactor tardó en iniciar. Revisa los logs:${NC}"
echo "   docker compose logs"
