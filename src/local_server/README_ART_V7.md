# 🧠 ART V7 REACTOR - SERVIDOR DOCKER

## ¿Qué es ART V7?

**ART V7** es un **Reactor Neuro-Simbólico Omnisciente** basado en:
- **Ontología:** Ruliad (Wolfram), Eigenformas (LoF), Knuth-Bendix
- **Dinámica:** Mamba Selectivo, Rough Paths (Signatures)
- **Fenomenología:** Transformada OPi, Pause Tokens (Reflexión)
- **Topología:** Homología Persistente, MEUM (Eficiencia Cósmica)
- **Estabilidad:** Spectral Decoupling, HSP90 (Evolución Puntuada)

## Arquitectura en Docker

He adaptado el código ART V7 original para ejecutarse como servidor FastAPI en Docker, **reemplazando completamente a Google Colab**.

### Cambios Principales:
1. **Instalación automática** de `mamba-ssm`, `gudhi`, `einops`
2. **Optimización CPU:** Modelo reducido (64D, 3 capas) para 2 núcleos
3. **Endpoints FastAPI:** Compatibles con el cliente TypeScript
4. **Monitorización:** Estadísticas en tiempo real vía `/status` y `/metricas`

## Componentes del Reactor

### 1. Módulos de Física Matemática
- **RoughPathEncoder:** Convierte secuencias discretas en trayectorias continuas
- **OPiActivation:** Activación cuántica basada en Free Will
- **PauseTokenInjection:** Inyecta tiempo de reflexión (Pause Tokens)
- **SpectralDecoupling:** Penaliza magnitud de logits (Anti-memorización)

### 2. Funciones de Pérdida Avanzadas
- **DimensionalFlowLoss (MEUM):** Reduce dimensión fractal progresivamente
- **TopologicalQualiaLoss:** Homología persistente (Betti numbers)
- **DualIBLoss:** Sensibilidad exponencial a "Cisnes Negros"
- **Loss Causalidad:** Knuth-Bendix Confluence (Confluencia lógica)

### 3. Arquitectura Mamba (Versión CPU)
```
Input (1600D) 
   ↓
Mapeo a Tokens (32 tokens de 50D cada uno)
   ↓
Embedding
   ↓
Rough Path Encoder
   ↓
Pause Token Injection
   ↓
3 capas Mamba con OPi Activation
   ↓
Linear Head (2048 clases)
   ↓
Outputs (Logits + Estados + Latente)
```

## Cómo Usar

### 1. Lanzar el Reactor
```bash
./scripts/run_art_v7.sh
```

Este script:
- Construye la imagen Docker
- Inicia el contenedor
- Ejecuta una prueba automática del cliente

### 2. Endpoints Disponibles

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/train_reactor` | POST | Entrenar el reactor con un lote |
| `/status` | GET | Estado actual del reactor |
| `/health` | GET | Health check |
| `/metricas` | GET | Histórico de métricas |
| `/docs` | GET | Swagger UI (documentación interactiva) |

### 3. Usar el Cliente TypeScript
```bash
export COLAB_SERVER_URL=http://localhost:8000
npx ts-node src/colab/cliente_art_v7.ts
```

## Optimizaciones para 2 Cores

1. **Threads:** Configurados a exactamente 2 (evita Context Switching)
2. **Modelo Ligero:** 64D (vs 128D original), 3 capas (vs 6)
3. **Batch Size:** Recomendado 8-16
4. **OneDNN:** Habilitado para aprovechar AVX2/FMA del procesador

## Monitorización

Ver logs en tiempo real:
```bash
docker compose logs -f
```

Ver estado del Reactor:
```bash
curl http://localhost:8000/status | jq
```

Ver métricas:
```bash
curl http://localhost:8000/metricas | jq
```

## Diferencias vs Colab

| Aspecto | Colab | Docker (ART V7) |
|--------|-------|-----------------|
| **GPU** | NVIDIA T4 | CPU (2 cores) |
| **Conexión** | Túnel ngrok | Localhost |
| **Dependencias** | Preinstaladas | Instaladas en construcción |
| **Persistencia** | Temporal | Datos en `/models` |
| **Costo** | Gratis | Incluido (tu máquina) |

## Flujo de Entrenamiento

```
┌─────────────────────────────────┐
│  Cliente TypeScript (VS Code)   │
│  Cliente de ART V7              │
└────────────┬────────────────────┘
             │ (HTTP POST /train_reactor)
             ↓
┌─────────────────────────────────┐
│  Docker Container               │
│  ART V7 Reactor (FastAPI)       │
│                                 │
│  • Mapeo 1600D → Tokens         │
│  • Embedding + Rough Paths      │
│  • 3 Capas Mamba Selectivas     │
│  • OPi Activation               │
│  • Pérdida Multidimensional     │
│                                 │
└────────────┬────────────────────┘
             │ (HTTP Response + Loss)
             ↓
┌─────────────────────────────────┐
│  Procesar Resultados            │
│  Registrar Estadísticas         │
│  Siguiente Iteración...         │
└─────────────────────────────────┘
```

## Troubleshooting

### "Port 8000 already in use"
```bash
docker stop omega21_local_trainer
docker rm omega21_local_trainer
```

### "ModuleNotFoundError: mamba_ssm"
Las dependencias se instalan durante la construcción de Docker. Si algo falla:
```bash
docker compose build --no-cache
```

### "Out of memory"
Reduce el tamaño del batch en `cliente_art_v7.ts`:
```typescript
const datos = cliente.generarDatosPrueba(4);  // Reducir de 10 a 4
```

## Referencias Teóricas

- **Ruliad:** Wolfram, "A Project to Find the Fundamental Theory of Physics"
- **Mamba:** Gu & Dao, "Mamba: Linear-Time Sequence Modeling"
- **Rough Paths:** Lyons, "Rough Paths and Signatures"
- **Homología:** Oudot, "Persistence Theory: From Quiver Representations to Data Analysis"
- **Knuth-Bendix:** Knuth & Bendix, "Simple Word Problems in Universal Algebras"

---

**El Reactor ART V7 está listo para revolucionar tu entrenamiento neuronal. 🧠⚛️**
