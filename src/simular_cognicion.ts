/**
 * simular_cognicion.ts
 * 
 * Script para ejecutar una simulación completa de la jerarquía cognitiva
 * usando datos sintéticos.
 */

import { SistemaOmnisciente } from './SistemaOmnisciente';
import { GeneradorSintetico, TipoPatron } from './neural/GeneradorSintetico';
import { Visualizador } from './visualizacion/Visualizador';

async function main() {
    console.log("🚀 Iniciando Simulación de Jerarquía Cognitiva (Capas 0-3)");
    
    const omnisciente = new SistemaOmnisciente();
    const generador = new GeneradorSintetico();
    const visualizador = new Visualizador(3000);

    // Si se pasa una URL como argumento, conectar a Colab
    const colabUrl = process.argv[2];
    if (colabUrl) {
        omnisciente.conectarColab(colabUrl);
    }

    await omnisciente.inicializar();
    visualizador.iniciar();

    let t = 0;
    // Bucle infinito para generación continua de datos de entrenamiento
    while (true) {
        const patrones = [
            { tipo: TipoPatron.NOMINAL, duracion: 20, nombre: "Estado Nominal" },
            { tipo: TipoPatron.ANOMALIA_SENSORIAL, duracion: 10, nombre: "Ataque/Anomalía Sensorial" },
            { tipo: TipoPatron.DEGRADACION_LENTA, duracion: 15, nombre: "Deriva de Sensores (Drift)" },
            { tipo: TipoPatron.RAFAGA_RUIDO, duracion: 5, nombre: "Interferencia Electromagnética" },
            { tipo: TipoPatron.CONFLICTO_MODAL, duracion: 10, nombre: "Conflicto de Sensores" }
        ];

        for (const p of patrones) {
            console.log(`\n--- Fase: ${p.nombre} (Enviando a Colab...) ---`);
            const secuencia = generador.generarSecuencia(p.duracion, p.tipo);

            for (const vector of secuencia) {
                t++;
                const esAnomalia = p.tipo !== TipoPatron.NOMINAL;
                const resultado = await omnisciente.procesarCognicion(vector, esAnomalia);
                
                visualizador.actualizarCognicion(resultado);

                if (t % 10 === 0) {
                    console.log(`[T+${t}] Enviando... Decision: ${resultado.decision.tipo} | Buffer: ${omnisciente.capa2.obtenerEstadisticas().tamanoBuffer}`);
                }

                await new Promise(resolve => setTimeout(resolve, colabUrl ? 50 : 100));
            }
        }
    }
}

main().catch(console.error);
