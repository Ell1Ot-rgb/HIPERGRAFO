/**
 * run_entrenamiento_con_atomos.ts
 * 
 * Sistema de entrenamiento en tiempo real usando ÁTOMOS ACTIVOS.
 * 
 * Flujo:
 * 1. Múltiples átomos (pool de 0-32) ejecutándose en paralelo
 * 2. Cada átomo procesa telemetría simulada de Omega21
 * 3. Los estados de los átomos se envían a Colab para entrenar la Corteza Cognitiva
 * 4. Protocolo de infección activo entre átomos
 * 5. Feedback de Colab aplicado a dendritas (opcional)
 */

import { AtomoTopologico, SistemaOmnisciente } from './SistemaOmnisciente';
import { Omega21Simulador } from './hardware/Simulador';
import { Omega21Telemetry } from './omega21';
import { StreamingBridge } from './neural/StreamingBridge';
import { Visualizador } from './visualizacion/Visualizador';

interface ConfiguracionEntrenamiento {
    numAtomos: number;                // Cantidad de átomos en el pool (0-32)
    urlColab: string;                 // URL del servidor Colab
    intervaloMs: number;              // Intervalo entre muestras (ms)
    muestrasObjetivo: number;         // Cuántas muestras enviar
    habilitarVisualizacion: boolean;  // Mostrar dashboard
    habilitarInfeccion: boolean;      // Activar protocolo de contagio
}

export class EntrenadorConAtomos {
    private sistema: SistemaOmnisciente;
    private simulador: Omega21Simulador;
    private bridge: StreamingBridge;
    private visualizador: Visualizador | null = null;
    private config: ConfiguracionEntrenamiento;
    private muestrasEnviadas: number = 0;
    private activo: boolean = false;

    constructor(config: Partial<ConfiguracionEntrenamiento> = {}) {
        this.config = {
            numAtomos: config.numAtomos || 8,
            urlColab: config.urlColab || '',
            intervaloMs: config.intervaloMs || 1000,
            muestrasObjetivo: config.muestrasObjetivo || 10000,
            habilitarVisualizacion: config.habilitarVisualizacion ?? true,
            habilitarInfeccion: config.habilitarInfeccion ?? true
        };

        this.sistema = new SistemaOmnisciente();
        this.simulador = new Omega21Simulador();
        this.bridge = new StreamingBridge(this.config.urlColab);

        if (this.config.habilitarVisualizacion) {
            this.visualizador = new Visualizador(3000);
        }
    }

    /**
     * Inicializa el sistema completo
     */
    async iniciar(): Promise<void> {
        console.log("🧠 INICIANDO ENTRENAMIENTO CON ÁTOMOS ACTIVOS");
        console.log(`   Configuración:`);
        console.log(`   • Átomos: ${this.config.numAtomos}`);
        console.log(`   • URL Colab: ${this.config.urlColab}`);
        console.log(`   • Objetivo: ${this.config.muestrasObjetivo} muestras`);
        console.log(`   • Infección: ${this.config.habilitarInfeccion ? '✅' : '❌'}`);

        // 1. Inicializar Sistema Omnisciente (Capas 0-1)
        await this.sistema.inicializar();
        
        // 2. Conectar a Colab
        this.sistema.conectarColab(this.config.urlColab);

        // 3. Crear pool de átomos
        console.log(`\n🔬 Creando pool de ${this.config.numAtomos} átomos...`);
        const nombresDominios = [
            'VISION', 'AUDIO', 'TACTO', 'OLFATO',
            'LENGUAJE', 'LOGICA', 'MATEMATICA', 'ESPACIAL',
            'TEMPORAL', 'CAUSAL', 'EMOCIONAL', 'MOTOR',
            'MEMORIA', 'ATENCION', 'PLANIFICACION', 'EJECUTIVO',
            'SENSORIAL_A', 'SENSORIAL_B', 'SENSORIAL_C', 'SENSORIAL_D',
            'COGNITIVO_A', 'COGNITIVO_B', 'COGNITIVO_C', 'COGNITIVO_D',
            'ASOCIATIVO_A', 'ASOCIATIVO_B', 'ASOCIATIVO_C', 'ASOCIATIVO_D',
            'META_A', 'META_B', 'META_C', 'META_D'
        ];

        for (let i = 0; i < this.config.numAtomos; i++) {
            const nombreAtomo = nombresDominios[i] || `ATOMO_${i}`;
            await this.sistema.crearAtomo(nombreAtomo);
            console.log(`   ✅ ${nombreAtomo} inicializado`);
        }

        // 4. Iniciar visualizador
        if (this.visualizador) {
            this.visualizador.iniciar();
            console.log(`\n📊 Dashboard disponible en http://localhost:3000`);
        }

        console.log(`\n🚀 Sistema listo. Iniciando bucle de entrenamiento...\n`);
        this.activo = true;
    }

    /**
     * Bucle principal de entrenamiento
     */
    async ejecutarEntrenamiento(): Promise<void> {
        const nombresAtomos = Array.from(this.sistema['atomos'].keys());
        let ciclo = 0;

        const interval = setInterval(async () => {
            if (!this.activo || this.muestrasEnviadas >= this.config.muestrasObjetivo) {
                clearInterval(interval);
                await this.finalizar();
                return;
            }

            ciclo++;
            
            // 1. Generar telemetría simulada
            const telemetria = this.simulador.generarMuestra();

            // 2. Procesar con TODOS los átomos en paralelo
            const resultadosAtomos = await Promise.all(
                nombresAtomos.map(nombre => 
                    this.sistema.procesarFlujo(nombre, telemetria)
                )
            );

            // 3. Extraer datos para entrenamiento
            const datosEntrenamiento = this.extraerDatosParaColab(
                resultadosAtomos,
                telemetria
            );

            // 4. Enviar a Colab
            await this.enviarAColab(datosEntrenamiento);

            // 5. Actualizar visualización
            if (this.visualizador && ciclo % 5 === 0) {
                this.actualizarVisualizacion(resultadosAtomos);
            }

            // 6. Log de progreso
            if (this.muestrasEnviadas % 100 === 0) {
                const progreso = (this.muestrasEnviadas / this.config.muestrasObjetivo * 100).toFixed(1);
                console.log(`📈 Progreso: ${this.muestrasEnviadas}/${this.config.muestrasObjetivo} (${progreso}%)`);
                
                // Mostrar estadísticas de infección
                if (this.config.habilitarInfeccion) {
                    const memorias = resultadosAtomos.map(r => r.memoria || 0);
                    const memoriaTotal = memorias.reduce((a, b) => a + b, 0);
                    console.log(`   🦠 Memoria colectiva: ${memoriaTotal} firmas LSH compartidas`);
                }
            }

        }, this.config.intervaloMs);
    }

    /**
     * Extrae los datos relevantes de los átomos para enviar a Colab
     */
    private extraerDatosParaColab(resultadosAtomos: any[], telemetria: Omega21Telemetry): number[] {
        // Formato esperado: Vector de características de TODOS los átomos
        // Cada átomo contribuye con su vector latente (64D si usa omega21_brain.onnx)
        
        const vectorCompleto: number[] = [];

        for (const resultado of resultadosAtomos) {
            // Extraer las características neuronales del átomo
            if (resultado.neuronal && resultado.neuronal.ajustes_dendritas) {
                // Usar los primeros 64 valores de ajustes de dendritas como features
                const features = resultado.neuronal.ajustes_dendritas.slice(0, 64);
                vectorCompleto.push(...features);
            } else {
                // Fallback: vector de 64 ceros
                vectorCompleto.push(...new Array(64).fill(0));
            }

            // Agregar métricas físicas como features adicionales
            if (resultado.fisica && resultado.fisica.metricas) {
                const m = resultado.fisica.metricas;
                vectorCompleto.push(
                    m.tension || 0,
                    m.curvatura || 0,
                    m.entropia || 0,
                    m.energia || 0
                );
            } else {
                vectorCompleto.push(0, 0, 0, 0);
            }
        }

        // El vector completo es: numAtomos × (64 features + 4 métricas) = numAtomos × 68D
        // Para 8 átomos: 544D
        // Esto es diferente del formato 1600D (25 × 64) que se esperaba

        // SOLUCIÓN: Rellenar o truncar a 1600D para mantener compatibilidad con Colab
        while (vectorCompleto.length < 1600) {
            vectorCompleto.push(0);
        }
        
        return vectorCompleto.slice(0, 1600);
    }

    /**
     * Envía los datos a Colab para entrenamiento
     */
    private async enviarAColab(vectorCaracteristicas: number[]): Promise<void> {
        // Determinar si es anomalía (heurística simple basada en varianza)
        const media = vectorCaracteristicas.reduce((a, b) => a + b, 0) / vectorCaracteristicas.length;
        const varianza = vectorCaracteristicas.reduce((acc, val) => acc + Math.pow(val - media, 2), 0) / vectorCaracteristicas.length;
        const esAnomalia = varianza > 0.5; // Umbral ajustable

        await this.bridge.enviarVector(vectorCaracteristicas, esAnomalia);
        this.muestrasEnviadas++;
    }

    /**
     * Actualiza el visualizador con datos de los átomos
     */
    private actualizarVisualizacion(resultadosAtomos: any[]): void {
        if (!this.visualizador) return;

        // Calcular métricas agregadas
        const estabilidadPromedio = resultadosAtomos.reduce(
            (acc, r) => acc + (r.fisica?.metricas?.estabilidad || 0), 0
        ) / resultadosAtomos.length;

        const tensionPromedio = resultadosAtomos.reduce(
            (acc, r) => acc + (r.fisica?.metricas?.tension || 0), 0
        ) / resultadosAtomos.length;

        // Enviar al visualizador
        // (El visualizador necesitaría un nuevo método para esto)
        this.visualizador['ultimoEstado'] = {
            timestamp: new Date(),
            numAtomos: resultadosAtomos.length,
            estabilidad: estabilidadPromedio,
            tension: tensionPromedio,
            muestrasEnviadas: this.muestrasEnviadas
        };
    }

    /**
     * Finaliza el entrenamiento
     */
    private async finalizar(): Promise<void> {
        console.log("\n✅ ENTRENAMIENTO COMPLETADO");
        console.log(`   Total de muestras enviadas: ${this.muestrasEnviadas}`);
        console.log(`   Esperando a que Colab procese el buffer...`);

        // Esperar a que el bridge vacíe su buffer
        while (this.bridge.obtenerEstadoBuffer() > 0) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            console.log(`   ⏳ Muestras restantes en buffer: ${this.bridge.obtenerEstadoBuffer()}`);
        }

        console.log("\n🎉 Todos los datos fueron enviados a Colab.");
        console.log("   Ahora puedes exportar el modelo entrenado desde Colab a ONNX.\n");

        if (this.visualizador) {
            console.log("   Dashboard sigue activo en http://localhost:3000");
            console.log("   Presiona Ctrl+C para cerrar todo.\n");
        }

        this.activo = false;
    }

    /**
     * Detiene el entrenamiento
     */
    detener(): void {
        this.activo = false;
        if (this.visualizador) {
            this.visualizador.detener();
        }
    }
}

// ==========================================
// SCRIPT PRINCIPAL
// ==========================================

async function main() {
    const urlColab = process.argv[2];
    
    if (!urlColab) {
        console.error("❌ Error: Debes proporcionar la URL de Colab.");
        console.log("\nUso:");
        console.log("  npx ts-node src/run_entrenamiento_con_atomos.ts <URL_COLAB> [numAtomos] [muestras]");
        console.log("\nEjemplo:");
        console.log("  npx ts-node src/run_entrenamiento_con_atomos.ts https://abc123.ngrok-free.app 8 10000");
        console.log("\nParámetros:");
        console.log("  URL_COLAB  : URL del servidor Colab (requerido)");
        console.log("  numAtomos  : Cantidad de átomos (default: 8, máx: 32)");
        console.log("  muestras   : Muestras objetivo (default: 10000)");
        process.exit(1);
    }

    const numAtomos = parseInt(process.argv[3]) || 8;
    const muestras = parseInt(process.argv[4]) || 10000;

    const entrenador = new EntrenadorConAtomos({
        urlColab,
        numAtomos: Math.min(numAtomos, 32),
        muestrasObjetivo: muestras,
        habilitarVisualizacion: true,
        habilitarInfeccion: true
    });

    try {
        await entrenador.iniciar();
        await entrenador.ejecutarEntrenamiento();
    } catch (error) {
        console.error("❌ Error durante el entrenamiento:", error);
        entrenador.detener();
        process.exit(1);
    }

    // Manejar cierre
    process.on('SIGINT', () => {
        console.log("\n🛑 Deteniendo entrenamiento...");
        entrenador.detener();
        process.exit(0);
    });
}

if (require.main === module) {
    main();
}

export default EntrenadorConAtomos;
