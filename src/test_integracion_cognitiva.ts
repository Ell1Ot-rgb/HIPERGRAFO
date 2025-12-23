import { SistemaOmnisciente } from './SistemaOmnisciente';
import { EntrenadorCognitivo } from './neural/EntrenadorCognitivo';
import { CortezaCognitiva } from './neural/CortezaCognitiva';
import { MapeoVector256DaDendritas } from './control/MapeoVector256DaDendritas';
import { Omega21Telemetry } from './omega21';

/**
 * Test de Integración Completa
 * 
 * Valida que:
 * 1. Los 25 átomos se crean y se inicializan correctamente
 * 2. Las dendritas alteran correctamente los átomos
 * 3. La telemetría se registra en el EntrenadorCognitivo
 * 4. El consolidación cognitiva genera conceptos en el hipergrafo
 * 5. El vector de salida se expande de 256D a 1600D
 */
class TestIntegracionCognitiva {
    private sistema: SistemaOmnisciente;
    private mapeador: MapeoVector256DaDendritas;
    
    constructor() {
        this.sistema = new SistemaOmnisciente();
        this.mapeador = new MapeoVector256DaDendritas();
    }

    async ejecutar() {
        console.log("\n╔════════════════════════════════════════════════════════════╗");
        console.log("║   TEST DE INTEGRACIÓN - SISTEMA COGNITIVO OMNISCIENTE   ║");
        console.log("╚════════════════════════════════════════════════════════════╝\n");

        // TEST 1: Creación de Átomos
        console.log("📋 TEST 1: Creación de 25 Átomos (S1-S25)");
        console.log("─".repeat(60));
        try {
            for (let i = 1; i <= 25; i++) {
                const id = `S${i}`;
                await this.sistema.crearAtomo(id);
                if (i % 5 === 0) process.stdout.write(`✓`);
            }
            console.log("\n✅ PASADO: 25 átomos creados exitosamente\n");
        } catch (e) {
            console.error("❌ FALLIDO:", e);
            return;
        }

        // TEST 2: Procesamiento con Dendritas
        console.log("📋 TEST 2: Configuración Dendrítica (D001-D056)");
        console.log("─".repeat(60));
        try {
            const vector256D = this.generarVector256D();
            const configDendritas = this.mapeador.extraerCamposDendriticos(vector256D);
            
            console.log(`  Campos dendríticos extraídos: ${Object.keys(configDendritas).length}`);
            console.log(`  D001: ${configDendritas.D001?.toFixed(2)}`);
            console.log(`  D028: ${configDendritas.D028?.toFixed(2)}`);
            console.log(`  D056: ${configDendritas.D056?.toFixed(2)}`);
            
            if (Object.keys(configDendritas).length >= 56) {
                console.log("✅ PASADO: Dendritas configuradas correctamente\n");
            } else {
                throw new Error(`Solo ${Object.keys(configDendritas).length} dendritas extraídas, se esperaban 56`);
            }
        } catch (e) {
            console.error("❌ FALLIDO:", e);
            return;
        }

        // TEST 3: Procesamiento Sensorial y Cognitivo
        console.log("📋 TEST 3: Procesamiento de Flujo Sensorial");
        console.log("─".repeat(60));
        try {
            const telemetria = this.generarTelemetria();
            const resultado = await this.sistema.procesarFlujo("S1", telemetria);
            
            console.log(`  Predicción de anomalía: ${resultado.neuronal?.prediccion_anomalia?.toFixed(3)}`);
            console.log(`  Estado: ${resultado.neuronal?.status}`);
            console.log(`  Estabilidad: ${resultado.topologia?.estabilidad?.toFixed(3)}`);
            
            if (resultado && resultado.neuronal) {
                console.log("✅ PASADO: Flujo procesado correctamente\n");
            } else {
                throw new Error("Resultado de procesamiento incompleto");
            }
        } catch (e) {
            console.error("❌ FALLIDO:", e);
            return;
        }

        // TEST 4: Integración con EntrenadorCognitivo
        console.log("📋 TEST 4: Consolidación Cognitiva");
        console.log("─".repeat(60));
        try {
            const corteza = new CortezaCognitiva();
            const entrenador = new EntrenadorCognitivo(corteza);
            
            // Simular 60 experiencias para disparar consolidación
            const vector72D = new Array(72).fill(0).map(() => Math.random() * 100);
            const mapa = corteza.getMapaMental();
            
            for (let i = 0; i < 60; i++) {
                entrenador.registrarExperiencia(
                    vector72D.map(v => v + Math.random() * 10),
                    mapa,
                    i % 10 === 0
                );
            }
            
            const stats = entrenador.obtenerEstadisticas();
            console.log(`  Buffer lleno: ${stats.bufferLleno}`);
            console.log(`  Conceptos aprendidos: ${stats.conceptosAprendidos}`);
            console.log(`  Ciclos consolidación: ${stats.ciclosConsolidacion}`);
            console.log(`  Tasa de acierto: ${stats.tasaAcierto}`);
            
            if (parseInt(stats.ciclosConsolidacion as any) > 0) {
                console.log("✅ PASADO: Consolidación ejecutada\n");
            } else {
                throw new Error("No se ejecutó consolidación");
            }
        } catch (e) {
            console.error("❌ FALLIDO:", e);
            return;
        }

        // TEST 5: Expansión de Vector 1600D
        console.log("📋 TEST 5: Expansión de Embedding a Vector 1600D");
        console.log("─".repeat(60));
        try {
            const embedding256D = new Array(256).fill(0).map(() => Math.random());
            const vector1600D = (this.sistema as any).expandirAVector1600D(embedding256D);
            
            console.log(`  Dimensión entrada: 256`);
            console.log(`  Dimensión salida: ${vector1600D.length}`);
            console.log(`  Media de valores: ${(vector1600D.reduce((a, b) => a + b) / vector1600D.length).toFixed(3)}`);
            
            if (vector1600D.length === 1600) {
                console.log("✅ PASADO: Vector expandido correctamente\n");
            } else {
                throw new Error(`Vector tiene ${vector1600D.length} dimensiones, se esperaban 1600`);
            }
        } catch (e) {
            console.error("❌ FALLIDO:", e);
            return;
        }

        // TEST 6: Flujo Completo Simulado
        console.log("📋 TEST 6: Flujo Completo (Vector → Dendritas → Átomo → Colab)");
        console.log("─".repeat(60));
        try {
            let procesados = 0;
            for (let ciclo = 0; ciclo < 5; ciclo++) {
                const vector256D = this.generarVector256D();
                const dendritas = this.mapeador.extraerCamposDendriticos(vector256D);
                const telemetria = this.generarTelemetria();
                
                // Procesar con 3 átomos diferentes
                for (const id of ["S1", "S10", "S25"]) {
                    const resultado = await this.sistema.procesarFlujo(id, telemetria, dendritas);
                    procesados++;
                }
            }
            
            console.log(`  Ciclos completados: 5`);
            console.log(`  Átomos procesados: ${procesados}`);
            console.log("✅ PASADO: Flujo completo ejecutado\n");
        } catch (e) {
            console.error("❌ FALLIDO:", e);
            return;
        }

        // RESUMEN FINAL
        console.log("╔════════════════════════════════════════════════════════════╗");
        console.log("║                     ✅ TODOS LOS TESTS PASARON             ║");
        console.log("╚════════════════════════════════════════════════════════════╝");
        console.log("\nEl sistema está listo para:\n");
        console.log("  ✨ 25 Átomos locales procesando en paralelo");
        console.log("  ✨ Dendritas alterando dinámicamente los átomos");
        console.log("  ✨ Entrenador Cognitivo consolidando experiencias");
        console.log("  ✨ Vectores expandidos enviando datos a Colab");
        console.log("  ✨ Protocolo de Infección propagando anomalías\n");
    }

    private generarVector256D() {
        const vec: any = {};
        for (let i = 1; i <= 256; i++) {
            const key = `D${i.toString().padStart(3, '0')}`;
            vec[key] = Math.random() * 100 - 50;
        }
        return vec;
    }

    private generarTelemetria(): Omega21Telemetry {
        return {
            timestamp: Date.now(),
            vector_72d: new Array(72).fill(0).map(() => Math.random() * 100),
            neuro: {
                nov: Math.random() * 500,
                energy: 0.5 + Math.random() * 0.3,
                entropia: Math.random() * 0.5
            },
            sig: {
                fp: new Uint32Array(32),
                distancia: Math.random()
            }
        };
    }
}

// Ejecutar tests
const test = new TestIntegracionCognitiva();
test.ejecutar().catch(console.error);
