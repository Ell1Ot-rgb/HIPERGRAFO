# 📑 ÍNDICE COMPLETO: Sistema de Entrenamiento Remoto en Colab

## 🎯 Punto de Entrada

**👉 COMIENZA AQUÍ:**  
[GUIA_RAPIDA_COLAB.md](GUIA_RAPIDA_COLAB.md) - 3 pasos simples (5 minutos)

---

## 📚 Documentación

### Guías Principales

| Archivo | Propósito | Tiempo |
|---------|-----------|--------|
| [GUIA_RAPIDA_COLAB.md](GUIA_RAPIDA_COLAB.md) | ⭐ **COMIENZA AQUÍ** - 3 pasos simples | 5 min |
| [INSTALACION_RAPIDA.md](INSTALACION_RAPIDA.md) | Setup detallado con verificación | 10 min |
| [RESUMEN_SOLUCION_COLAB.md](RESUMEN_SOLUCION_COLAB.md) | Visión completa del sistema | 15 min |
| [src/colab/README.md](src/colab/README.md) | Referencia técnica completa | 30 min |

### Guías de Troubleshooting
- [GUIA_RAPIDA_COLAB.md#solución-de-problemas](GUIA_RAPIDA_COLAB.md) - Problemas comunes
- [INSTALACION_RAPIDA.md#solución-de-problemas-de-instalación](INSTALACION_RAPIDA.md) - Errores de setup
- [src/colab/README.md#-solución-de-problemas](src/colab/README.md) - Errores de runtime

---

## 💻 Código

### Cliente TypeScript
**Archivo:** [src/colab/ClienteColabEntrenamiento.ts](src/colab/ClienteColabEntrenamiento.ts)

```typescript
// Uso:
const cliente = new ClienteColabEntrenamiento(url);
await cliente.conectar();
await cliente.entrenarMultiplesLotes(datos, 64);
```

**Métodos principales:**
- `conectar()` - Conectar al servidor
- `entrenarLote(muestras)` - Entrenar un lote
- `entrenarMultiplesLotes(muestras, tamanoLote)` - Entrenar múltiples
- `obtenerEstado()` - Obtener métricas
- `enviarFeedback(ajustes, validacion)` - Enviar feedback
- `obtenerMetricas()` - Obtener histórico

### Generador de Datos
**Archivo:** [src/colab/GeneradorDatosEntrenamiento.ts](src/colab/GeneradorDatosEntrenamiento.ts)

```typescript
// Uso:
const gen = new GeneradorDatosEntrenamiento();
const datos = gen.generarMuestras({...});
```

**Métodos:**
- `generarMuestras(config)` - Datos aleatorios
- `generarSeriesTemporal(numMuestras)` - Series de tiempo
- `generarPatronesNeuronales(numMuestras)` - Patrones realistas

### Script Principal
**Archivo:** [src/colab/entrenar_con_colab.ts](src/colab/entrenar_con_colab.ts)

```bash
# Uso:
npx ts-node src/colab/entrenar_con_colab.ts <URL> [opciones]
```

### Configuración
**Archivo:** [src/colab/config.colab.ts](src/colab/config.colab.ts)

```typescript
// Contiene:
- CONFIGURACION_COLAB_DEFECTO
- PRESETS (prueba_rapida, entrenamiento_estandar, etc.)
- validarUrlColab()
- obtenerUrlColab()
```

### Ejemplo Completo
**Archivo:** [src/colab/ejemplo_integracion_completa.ts](src/colab/ejemplo_integracion_completa.ts)

```bash
# Ejecutar:
COLAB_SERVER_URL=https://tu-url npx ts-node src/colab/ejemplo_integracion_completa.ts
```

---

## 🔧 Scripts Ejecutables

### Script Principal (Recomendado)
**Archivo:** [conectar_colab.sh](conectar_colab.sh)

```bash
./conectar_colab.sh https://tu-url.ngrok-free.app [opciones]
```

**Opciones:**
```
--muestras <num>      Número de muestras (default: 500)
--lote <num>          Tamaño del lote (default: 64)
--tipo <tipo>         simple|temporal|neuronal (default: simple)
--anomalias <pct>     Porcentaje de anomalías (default: 10)
--diagnostico         Ejecutar diagnóstico del servidor
--metricas            Mostrar métricas avanzadas
```

### Script de Verificación
**Archivo:** [verificar_setup_colab.sh](verificar_setup_colab.sh)

```bash
./verificar_setup_colab.sh
```

Verifica:
- Dependencias del sistema (Node.js, npm, TypeScript)
- Estructura de archivos
- Documentación
- Compilación
- Configuración

---

## 🖥️ Servidor Colab

**Archivo:** [COLAB_SERVER_OMEGA21_V4_UNIFICADO.py](COLAB_SERVER_OMEGA21_V4_UNIFICADO.py)

**Qué hace:**
- Implementa CortezaCognitivaV4 con PyTorch
- Expone 7 endpoints FastAPI
- Tuneliza con ngrok
- GPU en Colab

**Cómo usarlo:**
```python
# En Google Colab (https://colab.research.google.com/)
# Copia TODO el contenido y ejecuta en una celda

# Verás la URL de ngrok:
# 📡 NGROK TUNNEL: https://xxxxx.ngrok-free.app
```

---

## 📊 Flujo de Trabajo

```
1. PREPARAR
   └─ Abrir Google Colab
   └─ Copiar servidor Python
   └─ Ejecutar
   └─ Copiar URL de ngrok

2. EJECUTAR
   └─ ./conectar_colab.sh <URL>
   └─ Ver progreso en terminal
   └─ Resultados en tiempo real

3. ANALIZAR
   └─ Revisar loss
   └─ Revisar anomalías
   └─ Revisar feedback
```

---

## 🎯 Casos de Uso

### Caso 1: Prueba Rápida
```bash
./conectar_colab.sh <URL> --muestras 100 --diagnostico
```
→ Verifica que todo funciona (< 1 min)

### Caso 2: Detección de Anomalías
```bash
./conectar_colab.sh <URL> --muestras 5000 --tipo temporal --anomalias 20
```
→ Entrena modelo para detectar anomalías

### Caso 3: Entrenamiento Completo
```bash
./conectar_colab.sh <URL> --muestras 10000 --tipo neuronal --metricas
```
→ Entrenamiento pesado con análisis completo

### Caso 4: Integración Programática
```typescript
// Ver: src/colab/ejemplo_integracion_completa.ts
const cliente = new ClienteColabEntrenamiento(url);
// ... hacer cosas programáticamente
```

---

## 🚀 Quick Start

```bash
# 1. Verificar setup
./verificar_setup_colab.sh

# 2. En Google Colab:
#    - Copiar COLAB_SERVER_OMEGA21_V4_UNIFICADO.py
#    - Ejecutar
#    - Copiar URL

# 3. En terminal
./conectar_colab.sh https://tu-url.ngrok-free.app

# 4. Ver resultados ✅
```

---

## 📈 Monitoreo

### Durante Entrenamiento
```bash
# En otra terminal, ver estado en vivo:
curl https://tu-url/status | jq

# O acceder a Swagger UI:
# https://tu-url/docs
```

### Después del Entrenamiento
```
📈 RESUMEN DE ENTRENAMIENTOS:
   Lotes enviados: 8
   Total muestras: 500
   Loss promedio: 0.245612
   Tiempo total: 8.34s
```

---

## 🔌 API Reference

### Endpoints Disponibles

```
POST   /train_layer2        - Entrenar lote
POST   /feedback_dendritas  - Enviar feedback
GET    /status              - Estado del servidor
GET    /health              - Health check
GET    /info                - Arquitectura del modelo
POST   /diagnostico         - Test del sistema
GET    /metricas            - Métricas avanzadas
```

### Swagger UI
Después de conectar, accede a:  
`https://tu-url-colab.ngrok-free.app/docs`

---

## 🛠️ Troubleshooting

### "No se puede conectar"
1. Verifica URL de Colab
2. Prueba con `curl`: `curl https://tu-url/health`
3. Verifica que Colab sigue ejecutándose

### "Timeout"
1. Reduce tamaño de lote: `--lote 32`
2. Reduce muestras: `--muestras 500`

### "Input mismatch 1600D"
GeneradorDatosEntrenamiento lo maneja automáticamente

### "CUDA out of memory"
Reduce el tamaño de lote en Colab o en tu comando

### npm/Node.js errores
```bash
node --version    # Debe estar instalado
npm install       # Reinstalar dependencias
npm run build     # Recompilar
```

---

## 📚 Aprender Más

| Tema | Archivo |
|------|---------|
| Visión general | [RESUMEN_SOLUCION_COLAB.md](RESUMEN_SOLUCION_COLAB.md) |
| Arquitectura | [src/colab/README.md](src/colab/README.md) |
| API completa | [src/colab/ClienteColabEntrenamiento.ts](src/colab/ClienteColabEntrenamiento.ts) |
| Tipos de datos | [src/colab/GeneradorDatosEntrenamiento.ts](src/colab/GeneradorDatosEntrenamiento.ts) |
| Configuración | [src/colab/config.colab.ts](src/colab/config.colab.ts) |

---

## 🎓 Ejemplos

```bash
# Ejemplo 1: Setup y test
./verificar_setup_colab.sh

# Ejemplo 2: Primer entrenamiento
./conectar_colab.sh https://tu-url.ngrok-free.app

# Ejemplo 3: Con opciones
./conectar_colab.sh https://tu-url.ngrok-free.app \
  --muestras 1000 --tipo neuronal --metricas

# Ejemplo 4: Ejemplo completo en TypeScript
COLAB_SERVER_URL=https://tu-url \
  npx ts-node src/colab/ejemplo_integracion_completa.ts
```

---

## 🔐 Consideraciones

⚠️ **Importante:**
- URL de Colab es pública (ngrok)
- Válida ~8 horas antes de regenerarse
- Colab se descontinúa por inactividad
- Para producción, usar servidor dedicado

✅ **Ventajas:**
- GPU gratis (T4/A100)
- Sin instalación local
- Fácil de escalar
- Bien documentado

---

## 🎉 ¡Listo!

**Todos los archivos están creados y funcionales.**

### Próximos pasos:

1. ✅ Lee [GUIA_RAPIDA_COLAB.md](GUIA_RAPIDA_COLAB.md)
2. ✅ Ejecuta `./verificar_setup_colab.sh`
3. ✅ Copia servidor a Colab
4. ✅ Ejecuta `./conectar_colab.sh <URL>`
5. ✅ ¡Disfruta entrenando! 🚀

---

**Índice actualizado:** 27 Dic 2025
