#!/bin/bash

# BUCLE DE ENTRENAMIENTO INFINITO PARA EL REACTOR ART V7
# Este script mantiene al Reactor ocupado entrenando continuamente.

# Colores para la terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}⚛️  INICIANDO BUCLE DE ENTRENAMIENTO CONTINUO - ART V7${NC}"
echo -e "${BLUE}============================================================${NC}"

# Verificar si el servidor está arriba
echo -e "${YELLOW}🔍 Verificando conexión con el Reactor...${NC}"
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${YELLOW}⚠️  El Reactor no parece estar corriendo.${NC}"
    echo -e "${YELLOW}Intentando iniciar el contenedor Docker...${NC}"
    ./scripts/run_docker_training.sh
    sleep 10
fi

ITERACION=1

echo -e "\n${BLUE}MODO: Entrenamiento con datos REALISTAS (patrones neuronales)${NC}\n"

while true; do
    echo -e "\n${GREEN}🔄 Iteración del Bucle: $ITERACION${NC}"
    echo -e "${GREEN}────────────────────────────────────────────────────────────────${NC}"
    
    # Generar datos realistas usando Python/bash (sin dependencias TypeScript)
    DATOS_FILE="/tmp/datos_iter_$ITERACION.json"
    python3 << 'PYEOF' > "$DATOS_FILE"
import json
import math
import random

def seeded_random(seed):
    s = seed
    while True:
        s = (s * 9301 + 49297) % 233280
        yield s / 233280

rng = seeded_random(42)
muestras = []

for muestra in range(8):
    es_anomalia = random.random() < 0.20
    datos = [0.0] * 1600
    
    if es_anomalia:
        # Patrón anómalo: muchas activaciones fuertes
        num_neuronas = 100
        for j in range(num_neuronas):
            idx_neurona = int(next(rng) * 1600)
            amplitud = 1.5 + next(rng) * 0.5
            ancho = 20
            
            for k in range(idx_neurona, min(idx_neurona + ancho, 1600)):
                dist = k - idx_neurona
                gaussiana = math.exp(-((dist - ancho/2)**2) / (2 * 5**2))
                datos[k] += amplitud * gaussiana
    else:
        # Patrón normal: pocas activaciones suaves
        num_neuronas = 50
        for j in range(num_neuronas):
            idx_neurona = int(next(rng) * 1600)
            amplitud = 0.5 + next(rng) * 0.3
            ancho = 10
            
            for k in range(idx_neurona, min(idx_neurona + ancho, 1600)):
                dist = k - idx_neurona
                gaussiana = math.exp(-((dist - ancho/2)**2) / (2 * 5**2))
                datos[k] += amplitud * gaussiana
    
    # Normalizar
    max_val = max(datos) if datos else 1.0
    if max_val > 0:
        datos = [min(1.0, max(0.0, v/max_val)) for v in datos]
    
    muestras.append({
        "input_data": datos,
        "anomaly_label": 1 if es_anomalia else 0
    })

print(json.dumps({"samples": muestras, "epochs": 1}))
PYEOF
    
    # Enviar datos al Reactor
    echo "🧬 Datos realistas generados y enviados al Reactor..."
    RESULTADO=$(curl -s -X POST http://localhost:8000/train_reactor \
      -H "Content-Type: application/json" \
      -d @"$DATOS_FILE")
    
    LOSS=$(echo "$RESULTADO" | grep -o '"loss":[0-9.]*' | cut -d: -f2)
    EPOCH=$(echo "$RESULTADO" | grep -o '"epoch":[0-9]*' | cut -d: -f2)
    
    if [ ! -z "$LOSS" ]; then
        echo -e "${GREEN}✅ Entrenamiento completado${NC}"
        echo "   Loss: $LOSS"
        echo "   Epoch: $EPOCH"
        echo -e "${BLUE}Pausando 5 segundos antes del siguiente ciclo...${NC}"
        sleep 5
    else
        echo -e "${YELLOW}⚠️  Error en la respuesta del Reactor. Reintentando en 10s...${NC}"
        sleep 10
    fi
    
    ITERACION=$((ITERACION + 1))
    
    # Mostrar estado del Reactor
    echo -e "${YELLOW}📊 Estado actual del Reactor:${NC}"
    curl -s http://localhost:8000/status | grep -o '"epoch":[0-9]*' | head -1
done
