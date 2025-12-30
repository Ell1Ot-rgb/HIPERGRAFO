# ⚙️ INSTALACIÓN Y PRIMEROS PASOS

## Requisitos

- ✅ Node.js 16+ (`npm --version`)
- ✅ Google Colab (cuenta Google)
- ✅ Este workspace HIPERGRAFO
- ✅ Conexión a Internet

---

## Instalación (5 minutos)

### 1. Instalar Dependencias

```bash
cd /workspaces/HIPERGRAFO

# Instalar todos los paquetes npm
npm install

# Compilar TypeScript
npm run build

# Verificar compilación
ls dist/colab/
```

Deberías ver:
```
ClienteColabEntrenamiento.js
GeneradorDatosEntrenamiento.js
config.colab.js
entrenar_con_colab.js
ejemplo_integracion_completa.js
```

### 2. Configurar URL de Colab (Opcional)

**Opción A: Variable de entorno (recomendado)**
```bash
export COLAB_SERVER_URL=https://tu-id-unico.ngrok-free.app
```

**Opción B: Pasarla como argumento**
```bash
./conectar_colab.sh https://tu-id-unico.ngrok-free.app
```

**Opción C: Editar config.colab.ts**
```typescript
// src/colab/config.colab.ts
export const CONFIGURACION_COLAB_DEFECTO: ConfiguracionColab = {
    urlServidor: 'https://tu-id-unico.ngrok-free.app'  // Aquí
    // ...
};
```

---

## Tu Primer Entrenamiento (15 minutos)

### 1️⃣ Abrir Google Colab

1. Visita: https://colab.research.google.com/
2. Haz clic en "Archivo" → "Nuevo Cuaderno"

### 2️⃣ Copiar y Ejecutar Servidor

```python
# En una SOLA celda de Colab, copia TODO esto:

# @title 🧠 OMEGA 21 v4.0 - SERVIDOR UNIFICADO OPTIMIZADO
# Copia este CÓDIGO COMPLETO en una celda de Google Colab y ejecútalo.
# ESTE ES EL SERVIDOR FINAL UNIFICADO...

[... contenido completo de COLAB_SERVER_OMEGA21_V4_UNIFICADO.py ...]
```

**Ejecuta (Shift + Enter)**

Espera a ver:
```
📡 NGROK TUNNEL:
   ✅ https://xxxxx-xxxxx-xxxxx.ngrok-free.app
```

⭐ **Copia esa URL**

### 3️⃣ Abrir Terminal en VS Code

```bash
# Navega a la carpeta
cd /workspaces/HIPERGRAFO

# Ejecuta con tu URL
./conectar_colab.sh https://xxxxx-xxxxx-xxxxx.ngrok-free.app \
  --muestras 500 --diagnostico
```

### 4️⃣ Ver Resultados

```
✅ Servidor Colab conectado
📊 Modelo: OMEGA 21 v4.0
📈 Parámetros: 12,345,678

🔧 Diagnóstico del servidor:
   Status: diagnostico_ok
   GPU: Tesla T4

📊 GENERANDO DATOS DE ENTRENAMIENTO...
   Tipo: Simple (500 muestras)
   Total muestras: 500
   Normales: 450 (90.00%)
   Anomalías: 50 (10.00%)

🚀 INICIANDO ENTRENAMIENTO...
   Lote 1/8...
   Lote 2/8...
   [... etc ...]

✅ ENTRENAMIENTO COMPLETADO
   Tiempo total: 8.45s
   Lotes procesados: 8

📈 RESUMEN DE ENTRENAMIENTOS:
   Lotes enviados: 8
   Total muestras: 500
   Loss promedio: 0.245612
   Tiempo total: 8.45s
```

✅ **¡Funciona!**

---

## Verificación de Instalación

### Test de Conexión

```bash
# Verificar que npm funciona
npm --version

# Verificar que TypeScript compila
npx tsc --version

# Compilar todo el proyecto
npm run build

# Debería terminar sin errores
```

### Test de Generador de Datos

```bash
npx ts-node -e "
import { GeneradorDatosEntrenamiento } from './src/colab/GeneradorDatosEntrenamiento';
const gen = new GeneradorDatosEntrenamiento();
const datos = gen.generarMuestras({
    numMuestras: 10,
    numCaracteristicas: 1600,
    porcentajeAnomalias: 10
});
console.log('✅ Generador funciona');
console.log('Dimensión:', datos[0].input_data.length);
"
```

Deberías ver:
```
✅ Generador funciona
Dimensión: 1600
```

### Test sin Colab (Simulación)

```bash
# Prueba el cliente sin conectar a Colab
npx ts-node -e "
import { ClienteColabEntrenamiento } from './src/colab/ClienteColabEntrenamiento';
const cliente = new ClienteColabEntrenamiento('http://localhost:8000');
console.log('✅ Cliente creado');
console.log('URL:', 'http://localhost:8000');
"
```

---

## Solución de Problemas de Instalación

### Error: "npm: command not found"
```bash
# Node.js no está instalado
# Instalar desde: https://nodejs.org/

# Verificar
node --version
npm --version
```

### Error: "TypeScript compilation error"
```bash
# Limpiar caché y reinstalar
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Error: "Port 3000 already in use"
```bash
# Si intentas usar un servidor local
# Mata el proceso anterior
lsof -ti:3000 | xargs kill -9

# O usa otro puerto
PORT=3001 npm start
```

### Error: "EACCES permission denied"
```bash
# Problemas de permisos
# En Linux/Mac, puede ser necesario:
sudo chown -R $USER:$USER /workspaces/HIPERGRAFO
npm install
```

---

## Estructura Post-Instalación

```
HIPERGRAFO/
├── node_modules/          ← Instalado por npm
├── dist/                  ← Compilado por TypeScript
│   └── colab/
│       ├── ClienteColabEntrenamiento.js
│       ├── GeneradorDatosEntrenamiento.js
│       └── ...
├── src/
│   └── colab/
│       ├── ClienteColabEntrenamiento.ts
│       ├── GeneradorDatosEntrenamiento.ts
│       ├── entrenar_con_colab.ts
│       ├── config.colab.ts
│       ├── ejemplo_integracion_completa.ts
│       └── README.md
├── package.json
├── tsconfig.json
├── conectar_colab.sh      ← Script helper
├── GUIA_RAPIDA_COLAB.md   ← Esta guía
└── COLAB_SERVER_OMEGA21_V4_UNIFICADO.py ← Copiar a Colab
```

---

## Próximos Pasos

1. ✅ Completar instalación
2. ✅ Ejecutar primer entrenamiento
3. ✅ Leer `GUIA_RAPIDA_COLAB.md`
4. ✅ Explorar `src/colab/README.md`
5. ✅ Ejecutar ejemplo: `npx ts-node src/colab/ejemplo_integracion_completa.ts`

---

## Scripts Disponibles

```bash
# Compilar TypeScript
npm run build

# Ejecutar en modo watch
npm run dev

# Usar cliente (necesita URL de Colab)
./conectar_colab.sh https://tu-url.ngrok-free.app

# Ejecutar ejemplo completo
npx ts-node src/colab/ejemplo_integracion_completa.ts

# Linter (si está configurado)
npm run lint

# Tests (si existen)
npm test
```

---

## Validación Final

Cuando todo esté instalado, ejecuta:

```bash
./conectar_colab.sh https://tu-url.ngrok-free.app \
  --muestras 50 \
  --diagnostico

# Deberías ver ✅ en todos los puntos
```

---

## ¿Necesitas Ayuda?

- 📖 Lee: `GUIA_RAPIDA_COLAB.md`
- 📚 Lee: `src/colab/README.md`
- 🔗 GitHub: https://github.com/Ell1Ot-rgb/HIPERGRAFO
- 📊 Swagger: `{COLAB_URL}/docs` (después de conectar)

---

**¡Listo para entrenar! 🚀**
