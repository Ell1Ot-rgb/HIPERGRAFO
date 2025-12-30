/**
 * EJEMPLO PRÁCTICO: Entrenar con datos locales usando servidor Colab
 * 
 * Este script demuestra:
 * 1. Cargar datos locales
 * 2. Conectar a servidor Colab (remoto)
 * 3. Entrenar en batch
 * 4. Monitorear progreso
 * 5. Enviar feedback
 */

import { ClienteColab, MuestraEntrenamiento } from '../colab/cliente_colab';
import * as fs from 'fs';
import * as path from 'path';

// ==========================================
// CONFIGURACIÓN
// ==========================================

const CONFIG = {
  // ⚠️ REEMPLAZA CON TU URL DE NGROK
  serverUrl: process.env.COLAB_SERVER_URL || 'https://replace-with-ngrok-url.ngrok.io',
  
  // Configuración de entrenamiento
  batchSize: 32,
  epochs: 1,
  maxBatches: null  // null = entrenar todo, número = límite de batches
};

// ==========================================
// UTILIDADES
// ==========================================

/**
 * Generar datos sintéticos para demostración
 */
function generarDatasetSintetico(cantidad: number): MuestraEntrenamiento[] {
  const muestras: MuestraEntrenamiento[] = [];

  for (let i = 0; i < cantidad; i++) {
    // Generar 1600D aleatorio
    const features = Array(1600)
      .fill(0)
      .map(() => Math.random() * 2 - 1);

    // Inyectar patrón de anomalía (20% de probabilidad)
    let isAnomaly = Math.random() < 0.2;
    if (isAnomaly) {
      // Las anomalías tienen valores más altos
      for (let j = 0; j < 100; j++) {
        features[Math.floor(Math.random() * 1600)] = Math.random() > 0.5 ? 3 : -3;
      }
    }

    muestras.push({
      input_data: features,
      anomaly_label: isAnomaly ? 1 : 0
    });
  }

  return muestras;
}

/**
 * Cargar dataset desde archivo JSON (si existe)
 */
function cargarDataset(rutaArchivo: string): MuestraEntrenamiento[] | null {
  try {
    if (!fs.existsSync(rutaArchivo)) {
      console.log(`⚠️ Archivo no encontrado: ${rutaArchivo}`);
      return null;
    }

    const contenido = fs.readFileSync(rutaArchivo, 'utf-8');
    const datos = JSON.parse(contenido);

    // Validar estructura
    if (!Array.isArray(datos) || datos.length === 0) {
      console.log('⚠️ Dataset vacío o no es un array');
      return null;
    }

    // Validar primera muestra
    const primeraMuestra = datos[0];
    if (
      !primeraMuestra.input_data ||
      primeraMuestra.input_data.length !== 1600 ||
      !Number.isInteger(primeraMuestra.anomaly_label)
    ) {
      console.log('⚠️ Formato de dataset incorrecto');
      console.log('   Esperado: { input_data: number[1600], anomaly_label: 0|1 }');
      return null;
    }

    console.log(`✅ Dataset cargado: ${datos.length} muestras`);
    return datos;

  } catch (error: any) {
    console.log(`❌ Error cargando dataset: ${error.message}`);
    return null;
  }
}

/**
 * Dividir dataset en batches
 */
function crearBatches(
  muestras: MuestraEntrenamiento[],
  tamañoBatch: number
): MuestraEntrenamiento[][] {
  const batches: MuestraEntrenamiento[][] = [];

  for (let i = 0; i < muestras.length; i += tamañoBatch) {
    batches.push(muestras.slice(i, i + tamañoBatch));
  }

  return batches;
}

/**
 * Calcular estadísticas de un batch
 */
interface EstadisticasBatch {
  loss: number;
  anomalyProb: number;
  tiempo: number;
}

interface EstadisticasGlobales {
  totalMuestras: number;
  totalBatches: number;
  lossPromedio: number;
  anomaliasTotales: number;
  anomaliasPromedio: number;
  tiempoTotal: number;
  tasa_anomalias: number;
}

// ==========================================
// FLUJO PRINCIPAL
// ==========================================

async function entrenarConColab() {
  console.log('\n' + '='.repeat(80));
  console.log('🧠 ENTRENAMIENTO DISTRIBUIDO - COLAB + WORKSPACE LOCAL');
  console.log('='.repeat(80));

  // 1. Crear cliente
  console.log('\n📱 Inicializando cliente Colab...');
  const cliente = new ClienteColab({
    serverUrl: CONFIG.serverUrl
  });

  // 2. Conectar al servidor
  console.log('\n🔗 Conectando a servidor remoto...');
  const conectado = await cliente.conectar();

  if (!conectado) {
    console.log('\n❌ ERROR: No se pudo conectar');
    console.log('\nSOLUCIONES:');
    console.log('1. Verifica que el servidor Python está ejecutándose en Colab');
    console.log('2. Copia la URL de ngrok que aparece en Colab');
    console.log('3. Actualiza CONFIG.serverUrl en este archivo');
    console.log('4. O usa: export COLAB_SERVER_URL=https://tu-url.ngrok.io');
    process.exit(1);
  }

  // 3. Obtener información del servidor
  console.log('\n📋 Información del servidor:');
  const info = await cliente.obtenerInfo();
  if (info) {
    console.log(`   Modelo: ${info.nombre}`);
    console.log(`   Parámetros: ${info.arquitectura.parametros_totales.toLocaleString()}`);
  }

  // 4. Obtener estado inicial
  console.log('\n📊 Estado inicial:');
  let estado = await cliente.obtenerEstado();
  if (estado) {
    console.log(`   Muestras entrenadas: ${estado.estadisticas.total_muestras}`);
    console.log(`   Batches procesados: ${estado.estadisticas.total_batches}`);
    console.log(`   Loss promedio: ${estado.estadisticas.loss_promedio_global.toFixed(6)}`);
  }

  // 5. Cargar o generar dataset
  console.log('\n📚 Cargando dataset...');
  let dataset = cargarDataset('./dataset.json');
  
  if (!dataset) {
    console.log('   Generando dataset sintético de prueba...');
    dataset = generarDatasetSintetico(1000);
    console.log(`   ✅ Dataset sintético: ${dataset.length} muestras`);
  }

  // 6. Crear batches
  const batches = crearBatches(dataset, CONFIG.batchSize);
  const maxBatches = CONFIG.maxBatches || batches.length;
  const batchesAProcesar = batches.slice(0, maxBatches);

  console.log(`\n🎓 Configuración de entrenamiento:`);
  console.log(`   Total muestras: ${dataset.length}`);
  console.log(`   Tamaño de batch: ${CONFIG.batchSize}`);
  console.log(`   Total batches: ${batchesAProcesar.length}`);
  console.log(`   Épocas por batch: ${CONFIG.epochs}`);

  // 7. Entrenar en batches
  console.log('\n🚀 Iniciando entrenamiento...\n');

  const estadisticasGlobales: EstadisticasGlobales = {
    totalMuestras: 0,
    totalBatches: 0,
    lossPromedio: 0,
    anomaliasTotales: 0,
    anomaliasPromedio: 0,
    tiempoTotal: 0,
    tasa_anomalias: 0
  };

  const tiempoInicio = Date.now();
  const losses: number[] = [];
  const anomalias: number[] = [];

  for (let i = 0; i < batchesAProcesar.length; i++) {
    const batch = batchesAProcesar[i];
    const tiemeInicioBatch = Date.now();

    // Entrenar batch
    const resultado = await cliente.entrenar(batch, CONFIG.epochs);

    const tiempoFin = Date.now();
    const tiempoBatch = (tiempoFin - tiemeInicioBatch) / 1000;

    if (resultado) {
      losses.push(resultado.loss);
      anomalias.push(resultado.outputs.anomaly_prob);

      estadisticasGlobales.totalMuestras += resultado.batch_size;
      estadisticasGlobales.totalBatches += 1;
      estadisticasGlobales.lossPromedio =
        losses.reduce((a, b) => a + b, 0) / losses.length;
      estadisticasGlobales.anomaliasPromedio =
        anomalias.reduce((a, b) => a + b, 0) / anomalias.length;
      estadisticasGlobales.tiempoTotal += tiempoBatch;
      estadisticasGlobales.anomaliasTotales = batch.filter(m => m.anomaly_label === 1).length;
      estadisticasGlobales.tasa_anomalias =
        (estadisticasGlobales.anomaliasTotales / estadisticasGlobales.totalMuestras) * 100;

      // Mostrar progreso
      const progreso = ((i + 1) / batchesAProcesar.length * 100).toFixed(1);
      const barraProgreso = '█'.repeat(Math.floor((i + 1) / batchesAProcesar.length * 40));
      const barraVacia = '░'.repeat(40 - Math.floor((i + 1) / batchesAProcesar.length * 40));

      console.log(
        `[${progreso}%] ${barraProgreso}${barraVacia} | ` +
        `Loss: ${resultado.loss.toFixed(6)} | ` +
        `Anomaly: ${resultado.outputs.anomaly_prob.toFixed(4)} | ` +
        `Tiempo: ${tiempoBatch.toFixed(2)}s`
      );

      // Cada 5 batches, enviar feedback
      if ((i + 1) % 5 === 0) {
        console.log(`   📤 Enviando feedback...\n`);
        
        // Generar ajustes dendríticos basados en anomalías detectadas
        const ajustes = Array(16)
          .fill(0)
          .map(() => (Math.random() - 0.5) * 0.2);  // -0.1 a 0.1

        await cliente.enviarFeedback(ajustes, true);
      }
    }
  }

  // 8. Resumen final
  console.log('\n' + '='.repeat(80));
  console.log('✅ ENTRENAMIENTO COMPLETADO');
  console.log('='.repeat(80));

  const tiempoTotalS = (Date.now() - tiempoInicio) / 1000;

  console.log('\n📊 ESTADÍSTICAS FINALES:');
  console.log(`   Total muestras procesadas: ${estadisticasGlobales.totalMuestras}`);
  console.log(`   Total batches: ${estadisticasGlobales.totalBatches}`);
  console.log(`   Loss final: ${estadisticasGlobales.lossPromedio.toFixed(6)}`);
  console.log(`   Anomalía media detectada: ${estadisticasGlobales.anomaliasPromedio.toFixed(4)}`);
  console.log(`   Tasa de anomalías: ${estadisticasGlobales.tasa_anomalias.toFixed(2)}%`);
  console.log(`   Tiempo total: ${tiempoTotalS.toFixed(2)}s`);
  console.log(
    `   Velocidad: ${(estadisticasGlobales.totalMuestras / tiempoTotalS).toFixed(2)} muestras/s`
  );

  // 9. Obtener métricas finales del servidor
  console.log('\n📈 MÉTRICAS DEL SERVIDOR:');
  const metricas = await cliente.obtenerMetricas();
  if (metricas) {
    console.log(`   Tendencia: ${metricas.tendencia}`);
    console.log(`   Anomalías en histórico: ${metricas.anomalias_detectadas}`);
    console.log(`   Tasa éxito feedback: ${(metricas.feedback_tasa_exito * 100).toFixed(1)}%`);
  }

  // 10. Estado final
  console.log('\n🎯 ESTADO FINAL DEL SERVIDOR:');
  estado = await cliente.obtenerEstado();
  if (estado) {
    console.log(`   Muestras totales entrenadas: ${estado.estadisticas.total_muestras}`);
    console.log(`   Loss promedio global: ${estado.estadisticas.loss_promedio_global.toFixed(6)}`);
    console.log(`   Dispositivo: ${estado.estadisticas.dispositivo}`);
  }

  console.log('\n' + '='.repeat(80));
}

// ==========================================
// EJECUTAR
// ==========================================

entrenarConColab().catch((error) => {
  console.error('❌ Error fatal:', error);
  process.exit(1);
});
