/**
 * ejemplo_integracion_completa.ts
 * 
 * Ejemplo completo de integración:
 * - Conectar a Colab
 * - Generar datos
 * - Entrenar
 * - Monitorear
 * - Enviar feedback
 */

import { ClienteColabEntrenamiento } from './ClienteColabEntrenamiento';
import { GeneradorDatosEntrenamiento, ConfiguracionDatos } from './GeneradorDatosEntrenamiento';
import { PRESETS, obtenerUrlColab } from './config.colab';

async function main() {
    console.log('\n' + '='.repeat(80));
    console.log('📚 EJEMPLO COMPLETO DE INTEGRACIÓN COLAB');
    console.log('='.repeat(80));

    try {
        // ===== 1. OBTENER URL =====
        console.log('\n1️⃣ OBTENIENDO URL DE COLAB...');
        const urlColab = await obtenerUrlColab();
        console.log(`   ✅ URL: ${urlColab}`);

        // ===== 2. CREAR CLIENTE =====
        console.log('\n2️⃣ CREANDO CLIENTE...');
        const cliente = new ClienteColabEntrenamiento(urlColab);
        console.log('   ✅ Cliente creado');

        // ===== 3. CONECTAR =====
        console.log('\n3️⃣ CONECTANDO AL SERVIDOR...');
        const conectado = await cliente.conectar();
        if (!conectado) {
            throw new Error('No se pudo conectar al servidor Colab');
        }

        // ===== 4. EJECUTAR DIAGNÓSTICO =====
        console.log('\n4️⃣ EJECUTANDO DIAGNÓSTICO...');
        await cliente.diagnostico();

        // ===== 5. OBTENER INFORMACIÓN =====
        console.log('\n5️⃣ OBTENIENDO INFORMACIÓN DEL MODELO...');
        const info = await cliente.obtenerInfo();
        console.log(`   Nombre: ${info.nombre}`);
        console.log(`   Parámetros: ${info.arquitectura.parametros_totales.toLocaleString()}`);
        console.log(`   Entrada: ${info.flujo.entrada}`);
        console.log(`   Salida:`, info.flujo.salida);

        // ===== 6. GENERAR DATOS =====
        console.log('\n6️⃣ GENERANDO DATOS DE ENTRENAMIENTO...');
        const generador = new GeneradorDatosEntrenamiento(123); // Semilla para reproducibilidad
        
        // Usar preset
        const config: ConfiguracionDatos = {
            numMuestras: PRESETS.entrenamiento_estandar.numMuestras,
            numCaracteristicas: 1600,
            porcentajeAnomalias: PRESETS.entrenamiento_estandar.porcentajeAnomalias,
            semilla: 123
        };

        const muestras = generador.generarMuestras(config);
        GeneradorDatosEntrenamiento.mostrarEstadisticas(muestras);

        // ===== 7. ENTRENAR =====
        console.log('\n7️⃣ ENTRENANDO MODELO...');
        console.log('   (Esto puede tomar 10-30 segundos)\n');
        
        const resultados = await cliente.entrenarMultiplesLotes(
            muestras,
            PRESETS.entrenamiento_estandar.tamanoLote
        );

        // ===== 8. ANALIZAR RESULTADOS =====
        console.log('\n8️⃣ ANALIZANDO RESULTADOS...');
        if (resultados.length > 0) {
            const primerEntrenamiento = resultados[0];
            const ultimoEntrenamiento = resultados[resultados.length - 1];

            console.log(`   Primer loss: ${primerEntrenamiento.loss.toFixed(6)}`);
            console.log(`   Último loss: ${ultimoEntrenamiento.loss.toFixed(6)}`);
            
            const mejora = ((primerEntrenamiento.loss - ultimoEntrenamiento.loss) / 
                           primerEntrenamiento.loss * 100);
            console.log(`   Mejora: ${mejora.toFixed(2)}%`);

            // Análisis de anomalías
            const anomaliaPromedio = resultados.reduce(
                (sum, r) => sum + r.outputs.anomaly_prob,
                0
            ) / resultados.length;
            console.log(`   Anomalía detectada (promedio): ${(anomaliaPromedio * 100).toFixed(2)}%`);
        }

        // ===== 9. OBTENER ESTADO ACTUAL =====
        console.log('\n9️⃣ ESTADO ACTUAL DEL SERVIDOR...');
        const estado = await cliente.obtenerEstado();
        console.log(`   Total muestras procesadas: ${estado.estadisticas.total_muestras}`);
        console.log(`   Loss promedio global: ${estado.estadisticas.loss_promedio_global.toFixed(6)}`);
        console.log(`   Dispositivo: ${estado.estadisticas.dispositivo}`);
        console.log(`   GPU: ${estado.estadisticas.gpu_memoria_mb} MB`);

        // ===== 10. OBTENER MÉTRICAS =====
        console.log('\n🔟 MÉTRICAS AVANZADAS...');
        const metricas = await cliente.obtenerMetricas();
        console.log(`   Tendencia: ${metricas.tendencia}`);
        console.log(`   Anomalías detectadas: ${metricas.anomalias_detectadas}`);
        console.log(`   Últimas 5 losses:`);
        metricas.ultimos_20_losses.slice(-5).forEach((loss, idx) => {
            console.log(`     ${idx + 1}. ${loss.toFixed(6)}`);
        });

        // ===== 11. ENVIAR FEEDBACK =====
        console.log('\n1️⃣1️⃣ ENVIANDO FEEDBACK DENDRÍTICO...');
        const ajustes = new Array(16)
            .fill(0)
            .map(() => (Math.random() - 0.5) * 0.1); // Pequeños ajustes aleatorios
        
        await cliente.enviarFeedback(ajustes, true);

        // ===== 12. MOSTRAR RESUMEN FINAL =====
        console.log('\n1️⃣2️⃣ RESUMEN FINAL...');
        cliente.mostrarResumen();

        // ===== 13. GENERAR REPORTE =====
        console.log('\n1️⃣3️⃣ REPORTE TÉCNICO...');
        console.log('\n📊 ESTADÍSTICAS DE ENTRENAMIENTO:');
        console.log(`{
  "total_entrenamientos": ${resultados.length},
  "total_muestras": ${muestras.length},
  "tiempo_total_segundos": "~${(resultados.length * 0.52).toFixed(2)}",
  "loss_inicial": ${resultados[0]?.loss.toFixed(6) || 'N/A'},
  "loss_final": ${resultados[resultados.length - 1]?.loss.toFixed(6) || 'N/A'},
  "anomalia_detectada": ${(resultados.reduce((s, r) => s + r.outputs.anomaly_prob, 0) / resultados.length * 100).toFixed(2)}%,
  "arquitectura": {
    "capas": 5,
    "parametros_totales": ${info.arquitectura.parametros_totales},
    "entrada": "1600D",
    "salida": {
      "anomaly": "1D",
      "dendrites": "16D",
      "coherence": "64D"
    }
  },
  "servidor": {
    "dispositivo": "${estado.estadisticas.dispositivo}",
    "cuda_disponible": ${estado.cuda_available},
    "feedback_tasa_exito": ${estado.estadisticas.feedback.tasa_exito}
  }
}`);

        console.log('\n' + '='.repeat(80));
        console.log('✅ EJEMPLO COMPLETADO EXITOSAMENTE');
        console.log('='.repeat(80));

    } catch (error) {
        console.error('\n❌ ERROR:', error);
        process.exit(1);
    }
}

main();
