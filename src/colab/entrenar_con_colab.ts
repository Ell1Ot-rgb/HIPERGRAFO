/**
 * entrenar_con_colab.ts
 * 
 * Script principal para entrenar modelos usando Colab remoto desde VS Code
 * 
 * Uso:
 *   npx ts-node src/colab/entrenar_con_colab.ts <URL_COLAB> [opciones]
 * 
 * Ejemplos:
 *   npx ts-node src/colab/entrenar_con_colab.ts https://tu-url.ngrok-free.app
 *   npx ts-node src/colab/entrenar_con_colab.ts https://tu-url.ngrok-free.app --muestras 1000 --lote 64
 *   npx ts-node src/colab/entrenar_con_colab.ts https://tu-url.ngrok-free.app --tipo neuronal --anomalias 15
 */

import { ClienteColabEntrenamiento } from './ClienteColabEntrenamiento';
import { GeneradorDatosEntrenamiento } from './GeneradorDatosEntrenamiento';

interface OpcionesEntrenamiento {
    urlColab: string;
    numMuestras: number;
    tamanoLote: number;
    tipo: 'simple' | 'temporal' | 'neuronal';
    porcentajeAnomalias: number;
    mostrarDiagnostico: boolean;
    mostrarMetricas: boolean;
}

function parseArgumentos(): OpcionesEntrenamiento {
    const args = process.argv.slice(2);
    
    if (args.length === 0) {
        console.error('❌ Error: Debes proporcionar la URL de Colab');
        console.log('\nUso: npx ts-node src/colab/entrenar_con_colab.ts <URL_COLAB> [opciones]');
        console.log('\nOpciones:');
        console.log('  --muestras <num>      Número de muestras (default: 500)');
        console.log('  --lote <num>          Tamaño del lote (default: 64)');
        console.log('  --tipo <tipo>         simple|temporal|neuronal (default: simple)');
        console.log('  --anomalias <pct>     Porcentaje de anomalías (default: 10)');
        console.log('  --diagnostico         Ejecutar diagnóstico del servidor');
        console.log('  --metricas            Mostrar métricas al final');
        process.exit(1);
    }

    const opciones: OpcionesEntrenamiento = {
        urlColab: args[0],
        numMuestras: 500,
        tamanoLote: 64,
        tipo: 'simple',
        porcentajeAnomalias: 10,
        mostrarDiagnostico: false,
        mostrarMetricas: false
    };

    for (let i = 1; i < args.length; i++) {
        const arg = args[i];
        
        if (arg === '--muestras' && i + 1 < args.length) {
            opciones.numMuestras = parseInt(args[++i]);
        } else if (arg === '--lote' && i + 1 < args.length) {
            opciones.tamanoLote = parseInt(args[++i]);
        } else if (arg === '--tipo' && i + 1 < args.length) {
            opciones.tipo = args[++i] as any;
        } else if (arg === '--anomalias' && i + 1 < args.length) {
            opciones.porcentajeAnomalias = parseInt(args[++i]);
        } else if (arg === '--diagnostico') {
            opciones.mostrarDiagnostico = true;
        } else if (arg === '--metricas') {
            opciones.mostrarMetricas = true;
        }
    }

    return opciones;
}

async function main() {
    const opciones = parseArgumentos();

    console.log('\n' + '='.repeat(80));
    console.log('🧠 ENTRENAMIENTO EN COLAB REMOTO - OMEGA 21');
    console.log('='.repeat(80));

    // 1. Crear cliente
    const cliente = new ClienteColabEntrenamiento(opciones.urlColab);

    // 2. Conectar
    console.log('\n🔗 CONECTANDO...');
    const conectado = await cliente.conectar();
    
    if (!conectado) {
        console.error('\n❌ No se pudo conectar al servidor Colab');
        console.log('\nVerifica que:');
        console.log('  1. Colab esté ejecutando el servidor');
        console.log('  2. La URL ngrok sea correcta');
        console.log('  3. Tengas conexión a Internet');
        process.exit(1);
    }

    // 3. Diagnóstico opcional
    if (opciones.mostrarDiagnostico) {
        console.log('\n🔧 EJECUTANDO DIAGNÓSTICO...');
        try {
            await cliente.diagnostico();
        } catch (error) {
            console.error('Error en diagnóstico:', error);
        }
    }

    // 4. Generar datos
    console.log('\n📊 GENERANDO DATOS DE ENTRENAMIENTO...');
    const generador = new GeneradorDatosEntrenamiento();
    
    let muestras;
    switch (opciones.tipo) {
        case 'temporal':
            console.log(`   Tipo: Series temporales (${opciones.numMuestras} muestras)`);
            muestras = generador.generarSeriesTemporal(opciones.numMuestras);
            break;
        case 'neuronal':
            console.log(`   Tipo: Patrones neuronales (${opciones.numMuestras} muestras)`);
            muestras = generador.generarPatronesNeuronales(opciones.numMuestras);
            break;
        default:
            console.log(`   Tipo: Simple (${opciones.numMuestras} muestras)`);
            muestras = generador.generarMuestras({
                numMuestras: opciones.numMuestras,
                numCaracteristicas: 1600,
                porcentajeAnomalias: opciones.porcentajeAnomalias
            });
    }

    GeneradorDatosEntrenamiento.mostrarEstadisticas(muestras);

    // 5. Entrenar
    console.log('\n🚀 INICIANDO ENTRENAMIENTO...');
    const tiempoInicio = Date.now();
    
    try {
        const resultados = await cliente.entrenarMultiplesLotes(
            muestras,
            opciones.tamanoLote
        );

        const tiempoTotal = (Date.now() - tiempoInicio) / 1000;

        console.log('\n✅ ENTRENAMIENTO COMPLETADO');
        console.log(`   Tiempo total: ${tiempoTotal.toFixed(2)}s`);
        console.log(`   Lotes procesados: ${resultados.length}`);

        // 6. Mostrar resumen
        cliente.mostrarResumen();

        // 7. Obtener estado final
        console.log('\n📈 ESTADO FINAL DEL SERVIDOR:');
        const estado = await cliente.obtenerEstado();
        console.log(`   Total muestras entrenadas: ${estado.estadisticas.total_muestras}`);
        console.log(`   Loss promedio global: ${estado.estadisticas.loss_promedio_global.toFixed(6)}`);
        console.log(`   Loss últimas 100: ${estado.estadisticas.loss_promedio_ultimos_100.toFixed(6)}`);
        console.log(`   Anomalías media: ${estado.estadisticas.anomalia_media.toFixed(3)}`);
        console.log(`   Dispositivo: ${estado.estadisticas.dispositivo}`);
        console.log(`   Feedback recibido: ${estado.estadisticas.feedback.recibido}`);
        console.log(`   Tasa de éxito feedback: ${(estado.estadisticas.feedback.tasa_exito * 100).toFixed(2)}%`);

        // 8. Métricas avanzadas
        if (opciones.mostrarMetricas) {
            console.log('\n📊 MÉTRICAS AVANZADAS:');
            const metricas = await cliente.obtenerMetricas();
            console.log(`   Tendencia: ${metricas.tendencia}`);
            console.log(`   Anomalías detectadas: ${metricas.anomalias_detectadas}`);
            console.log(`   Últimas 20 losses:`);
            
            metricas.ultimos_20_losses.forEach((loss, idx) => {
                const marca = loss < metricas.ultimos_20_losses[0] ? '📈' : '📉';
                console.log(`     ${idx + 1}. ${loss.toFixed(6)} ${marca}`);
            });
        }

        // 9. Enviar feedback de ejemplo
        console.log('\n💬 ENVIANDO FEEDBACK DENDRÍTICO DE EJEMPLO...');
        const ajustesEjemplo = new Array(16).fill(0).map(() => Math.random() * 0.1 - 0.05);
        await cliente.enviarFeedback(ajustesEjemplo, true);

        console.log('\n' + '='.repeat(80));
        console.log('🎉 PROCESO COMPLETADO EXITOSAMENTE');
        console.log('='.repeat(80));

    } catch (error) {
        console.error('\n❌ Error durante entrenamiento:', error);
        process.exit(1);
    } finally {
        cliente.desconectar();
    }
}

main();
