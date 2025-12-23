/**
 * Script de Entrenamiento (Simulado)
 * 
 * Este script valida el flujo completo:
 * 1. Genera telemetría simulada de Omega 21.
 * 2. El Orquestador crea los hipergrafos y los analiza.
 * 3. El EntrenadorDistribuido empaqueta los datos y los envía a Colab vía ngrok.
 */

import { Orquestador } from './orquestador';
import { Omega21Simulador } from './hardware/Simulador';
import { EntrenadorDistribuido } from './neural/EntrenadorDistribuido';
import { Visualizador } from './visualizacion/Visualizador';

async function main() {
    console.log("🏋️ INICIANDO FLUJO DE ENTRENAMIENTO DISTRIBUIDO (SIMULADO)");
    
    // 1. Inicializar Orquestador
    const orquestador = new Orquestador({
        modoSimulacion: true,
        habilitarControl: false // No queremos modificar el simulador durante el entrenamiento
    });

    // 2. Inicializar Entrenador (conectado a Colab)
    const entrenador = new EntrenadorDistribuido(orquestador, {
        muestrasParaLote: 10, // Lotes pequeños para ver resultados rápido
        intervaloEnvioMs: 10000
    });

    // 3. Inicializar Visualizador
    const visualizador = new Visualizador(3000);
    visualizador.iniciar();

    // 4. Inicializar Simulador
    const simulador = new Omega21Simulador();

    try {
        // Conectar eventos al visualizador
        orquestador.on('procesado', (resultado) => {
            visualizador.actualizarEstado(resultado);
        });

        entrenador.on('feedback', (feedback) => {
            visualizador.actualizarFeedback(feedback);
        });

        entrenador.on('fisica', (fisica) => {
            visualizador.actualizarFisica(fisica);
        });

        // Iniciar componentes
        await orquestador.iniciar();
        await entrenador.iniciar();

        console.log("📡 Generando datos y enviando a Colab...");

        // Iniciar flujo
        simulador.iniciarFlujo((telemetria) => {
            orquestador.procesar(telemetria);
        }, 1000); // 1 muestra por segundo

    } catch (error) {
        console.error("❌ Error en el flujo de entrenamiento:", error);
        process.exit(1);
    }

    // Manejar cierre
    process.on('SIGINT', async () => {
        console.log("\n🛑 Finalizando sesión de entrenamiento...");
        entrenador.detener();
        visualizador.detener();
        await orquestador.detener();
        process.exit();
    });
}

main().catch(console.error);
