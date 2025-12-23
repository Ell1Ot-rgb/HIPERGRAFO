/**
 * Test de Estrés Topológico - Omega 21
 * 
 * Este script inyecta impulsos extremos (Vectores Wolfram) para observar
 * la respuesta del modelo ONNX y la deformación del hipergrafo.
 */

import { Orquestador } from './orquestador';
import { EntrenadorDistribuido } from './neural/EntrenadorDistribuido';
import { Visualizador } from './visualizacion/Visualizador';
import { Omega21Simulador } from './hardware/Simulador';

async function main() {
    console.log("🔥 INICIANDO TEST DE ESTRÉS: IMPULSOS WOLFRAM EXTREMOS");
    
    const orquestador = new Orquestador({ modoSimulacion: true });
    const entrenador = new EntrenadorDistribuido(orquestador, {
        muestrasParaLote: 1, // Respuesta inmediata
        intervaloEnvioMs: 1000
    });
    const visualizador = new Visualizador(3000);
    visualizador.iniciar();
    const simulador = new Omega21Simulador();

    await orquestador.iniciar();
    await entrenador.iniciar();

    orquestador.on('procesado', (res) => visualizador.actualizarEstado(res));
    entrenador.on('feedback', (f) => visualizador.actualizarFeedback(f));
    entrenador.on('fisica', (fis) => {
        visualizador.actualizarFisica(fis);
        console.log(`[FÍSICA] Tensión: ${fis.metricas.tension.toFixed(1)}% | Curvatura: ${fis.metricas.curvatura.toFixed(2)} | Diagnóstico: ${fis.diagnostico}`);
    });

    console.log("⏳ Fase 1: Estabilización (5 segundos)...");
    
    let fase = 1;
    let contador = 0;

    const interval = setInterval(async () => {
        contador++;
        let muestra = simulador.generarMuestra();

        if (fase === 2) {
            // INYECCIÓN DE ESTRÉS: Vector Wolfram Extremo
            console.log("⚡ INYECTANDO IMPULSO WOLFRAM EXTREMO...");
            muestra.neuro.novelty = 5000; // 5x el máximo normal
            muestra.metrics_256 = new Array(256).fill(0).map(() => Math.random() * 100); // Ruido de alta energía
            muestra.dendrites.voltage = 500; // Sobrecarga
        }

        if (contador === 5 && fase === 1) {
            console.log("🚀 FASE 2: ESTRÉS CRÍTICO ACTIVADO");
            fase = 2;
        }

        if (contador === 15 && fase === 2) {
            console.log("🧘 FASE 3: RELAJACIÓN (Homeostasis)");
            fase = 3;
        }

        if (contador === 25) {
            console.log("✅ Test de estrés completado.");
            clearInterval(interval);
            process.exit(0);
        }

        await orquestador.procesar(muestra);
    }, 1000);
}

main().catch(console.error);
