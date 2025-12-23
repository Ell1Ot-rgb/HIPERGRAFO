/**
 * run_omnisciente.ts
 * 
 * Punto de entrada para el Sistema Omnisciente (Jerarquía de Inteligencia).
 * Integra:
 * 1. Múltiples Átomos Topológicos (1024 neuronas LIF cada uno).
 * 2. Corteza Cognitiva (Red Neuronal Convencional para Imágenes Mentales).
 * 3. Memoria Colectiva (Protocolo de Infección).
 * 4. Visualización Avanzada (Cerebro Wolfram + Espacio Latente).
 */

import { SistemaOmnisciente } from './SistemaOmnisciente';
import { Omega21Simulador } from './hardware/Simulador';
import { Visualizador } from './visualizacion/Visualizador';
import { Orquestador } from './orquestador';

async function main() {
    console.log("🌌 INICIANDO SISTEMA OMNISCIENTE (HIPERGRAFO V3)");
    
    const visualizador = new Visualizador(3000);
    visualizador.iniciar();

    const omnisciente = new SistemaOmnisciente();
    await omnisciente.inicializar(); // Inicializa Capas 0 y 1
    
    // Creamos los 25 átomos de la Capa 1 (S1-S25)
    console.log("🧬 Desplegando Capa 1: 25 Átomos Topológicos...");
    for (let i = 1; i <= 25; i++) {
        await omnisciente.crearAtomo(`S${i}`);
    }

    console.log("🧠 Jerarquía de Inteligencia configurada (25 Átomos + Corteza)");

    // Usamos un orquestador base para el análisis topológico
    const orquestador = new Orquestador({ modoSimulacion: true });
    await orquestador.iniciar();

    // 3. Iniciar ciclo de vida autónomo para cada átomo
    console.log("⚡ Iniciando ciclo de vida autónomo de los átomos...");
    
    // Simulamos un loop principal que orquesta todos los átomos
    let ciclo = 0;
    setInterval(async () => {
        ciclo++;
        
        // FASE 1: Procesamiento distribuido
        for (const [id, atom] of omnisciente.atomos) {
            // A. Generar telemetría propia (simulada/percibida)
            const telemetria = atom.simulador.generarMuestra();
            
            // B. Procesar flujo cognitivo
            const resultadoOmni = await omnisciente.procesarFlujo(id, telemetria);
            
            // C. Visualización (solo actualizamos con el átomo S1 para no saturar la UI por ahora)
            if (id === 'S1') {
                // Procesar con el Orquestador para métricas topológicas globales
                const resultadoTopologico = await orquestador.procesar(telemetria);
                
                if (resultadoTopologico) {
                    visualizador.actualizarNeuronal(resultadoOmni.neuronal);
                    visualizador.actualizarCoherencia(resultadoOmni.coherencia);
                    visualizador.actualizarFisica(resultadoOmni.fisica);
                    visualizador.actualizarEstado(resultadoTopologico);
                    
                    if (visualizador['ultimoEstado']) {
                        visualizador['ultimoEstado'].memoria = resultadoOmni.memoria;
                    }
                }
            }
        }
        
        // FASE 2: Protocolo de Infección (cada 10 ciclos)
        if (ciclo % 10 === 0) {
            console.log(`\n🦠 CICLO ${ciclo}: Ejecutando Protocolo de Infección`);
            await omnisciente.propagarInfeccion();
        }
        
        // FASE 3: Reporte de estadísticas (cada 50 ciclos)
        if (ciclo % 50 === 0) {
            console.log(`\n📊 ESTADÍSTICAS DEL CICLO ${ciclo}:`);
            for (const [id, atom] of omnisciente.atomos) {
                const stats = (atom as any).getEstadisticasMemoria?.();
                if (stats) {
                    console.log(`  [${id}] Firmas aprendidas: ${stats.firmasAprendidas}`);
                }
            }
        }
    }, 100); // 10Hz ciclo global

    console.log("📊 Dashboard disponible en http://localhost:3000");

    process.on('SIGINT', async () => {
        console.log("\n🛑 Apagando Sistema Omnisciente...");
        visualizador.detener();
        await orquestador.detener();
        process.exit();
    });
}

main().catch(console.error);
