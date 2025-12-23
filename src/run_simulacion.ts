/**
 * Script de Ejecución en Modo Simulación
 * 
 * Permite probar todo el flujo (Mapeo -> Análisis -> Control)
 * sin necesidad de conexión física con Omega 21.
 */

import { Orquestador } from './orquestador';
import { Omega21Simulador } from './hardware/Simulador';

async function main() {
    console.log("🚀 INICIANDO HIPERGRAFO EN MODO SIMULACIÓN");
    
    const orquestador = new Orquestador({
        modoSimulacion: true,
        habilitarControl: true // El control se aplicará al estado interno
    });

    const simulador = new Omega21Simulador();

    // Suscribirse a eventos de interés
    orquestador.on('procesado', (resultado) => {
        const { estado, ajustesAplicados } = resultado;
        console.log(`[${new Date().toLocaleTimeString()}] 📊 Nodos: ${estado.numNodos} | Densidad: ${estado.densidad.toFixed(4)} | Cat: ${estado.categoria}`);
        
        if (ajustesAplicados.length > 0) {
            console.log(`   ⚙️ Ajustes: ${ajustesAplicados.map((a: any) => `${a.parametro}=${a.valor}`).join(', ')}`);
        }
    });

    orquestador.on('spike', () => {
        console.log("   ⚡ SPIKE DETECTADO");
    });

    // Iniciar orquestador
    await orquestador.iniciar();

    // Iniciar flujo de datos simulados
    console.log("📡 Generando telemetría sintética...");
    simulador.iniciarFlujo((telemetria) => {
        orquestador.procesar(telemetria);
    }, 500); // Una muestra cada 500ms

    // Manejar cierre
    process.on('SIGINT', async () => {
        console.log("\n🛑 Deteniendo simulación...");
        await orquestador.detener();
        process.exit();
    });
}

main().catch(console.error);
