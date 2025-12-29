import { Hipergrafo } from './core';
import { InferenciaLocal, CortezaCognitiva, CapaEspacioTemporal, CapaCognitiva } from './neural';
import { EntrenadorCognitivo } from './neural/EntrenadorCognitivo';
import { ProcesadorSensorial, Vector256D } from './neural/CapaSensorial';
import { AnalizadorFisico } from './analisis';
import { MapeoOmegaAHipergrafo, Omega21Telemetry } from './omega21';
import { StreamingBridge } from './neural/StreamingBridge';
import { HipergrafoBridge } from './neural/HipergrafoBridge';
import { Omega21Simulador } from './hardware/Simulador';

/**
 * AtomoTopologico - La unidad fundamental del Sistema Omnisciente.
 * Combina estructura (Hipergrafo), procesamiento (GNN 1024) y leyes físicas.
 */
export class AtomoTopologico {
    public id: string;
    public hipergrafo: Hipergrafo;
    public cerebro: InferenciaLocal;
    public analista: AnalizadorFisico;
    public mapeador: MapeoOmegaAHipergrafo;
    public simulador: Omega21Simulador;
    private memoriaColectiva: Set<string> = new Set(); // Firmas aprendidas de otros átomos

    constructor(id: string) {
        this.id = id;
        this.cerebro = new InferenciaLocal();
        this.analista = new AnalizadorFisico();
        this.mapeador = new MapeoOmegaAHipergrafo();
        this.hipergrafo = new Hipergrafo(`Atomo_${id}`);
        this.simulador = new Omega21Simulador();
    }

    async inicializar() {
        await this.cerebro.inicializar();
    }

    /**
     * Emite una señal con los patrones más relevantes detectados (Firmas LSH)
     * Protocolo de Infección: Si anomalía detectada, propaga la firma a otros átomos
     */
    emitirSenal(): { firma: string, intensidad: number, timestamp: number }[] {
        const senalesEmitidas: { firma: string, intensidad: number, timestamp: number }[] = [];
        const nodos = this.hipergrafo.obtenerNodos();
        
        // Buscar nodos de anomalía o signature crítica
        nodos.forEach(n => {
            if (n.metadata?.tipo === 'signature' || n.metadata?.anomalia) {
                const intensidad = n.metadata?.similitud || 0.5;
                const firma = n.id;
                
                if (intensidad > 0.7) { // Solo propagar anomalías fuertes
                    senalesEmitidas.push({
                        firma,
                        intensidad,
                        timestamp: Date.now()
                    });
                }
            }
        });
        
        return senalesEmitidas;
    }

    /**
     * Recibe señales de otros átomos e integra el conocimiento (Infección LSH)
     */
    recibirSenal(senales: { firma: string, intensidad: number, timestamp: number }[]) {
        senales.forEach(s => {
            if (!this.memoriaColectiva.has(s.firma)) {
                this.memoriaColectiva.add(s.firma);
                console.log(`  [${this.id}] Infección recibida: ${s.firma} (intensidad: ${s.intensidad.toFixed(2)})`);
            }
        });
    }

    async percibir(telemetria: Omega21Telemetry) {
        // 1. Mapear a estructura topológica
        this.hipergrafo = this.mapeador.mapear(telemetria);

        // 2. Inferencia Neuronal (Cerebro 1024) - Primero para alimentar el análisis físico
        const { nodeFeatures, edgeIndex, globalVector } = this.mapeador.extraerTensores(this.hipergrafo);
        const prediccion = await this.cerebro.predecir(nodeFeatures, edgeIndex, globalVector, telemetria);

        // 4. Integrar Memoria Colectiva en la predicción
        const coincidenciaMemoria = Array.from(this.memoriaColectiva).some(f => 
            telemetria.sig.fp.toString().includes(f.replace('SIG_', ''))
        );

        if (coincidenciaMemoria) {
            prediccion.prediccion_anomalia = Math.max(prediccion.prediccion_anomalia, 0.8);
            prediccion.status = 'ALERTA_COLECTIVA';
        }

        // 5. Análisis físico (Leyes de conservación, entropía, etc.)
        const analisisFisico = this.analista.analizar(this.hipergrafo, telemetria, prediccion);

        return {
            id: this.id,
            fisica: analisisFisico,
            neuronal: prediccion,
            topologia: {
                nodos: this.hipergrafo.obtenerNodos().length,
                edges: this.hipergrafo.obtenerHiperedges().length
            },
            memoria: this.memoriaColectiva.size
        };
    }
}

/**
 * SistemaOmnisciente - Orquestador de múltiples átomos topológicos.
 */
export class SistemaOmnisciente {
    public atomos: Map<string, AtomoTopologico> = new Map();
    public corteza: CortezaCognitiva = new CortezaCognitiva();
    public entrenador: EntrenadorCognitivo = new EntrenadorCognitivo(this.corteza);
    public sensorial: ProcesadorSensorial = new ProcesadorSensorial();
    public capa2: CapaEspacioTemporal = new CapaEspacioTemporal();
    public capa3: CapaCognitiva = new CapaCognitiva();
    private bridge: StreamingBridge | null = null;
    private hipergrafoBridge: HipergrafoBridge = new HipergrafoBridge();
    private inicializado: boolean = false;

    async inicializar() {
        await this.sensorial.inicializar();
        this.inicializado = true;
        console.log("🌌 Sistema Omnisciente: Capas 0 y 1 (Sensorial) inicializadas.");
        console.log("🧠 Sistema Omnisciente: Capa 2 (Espacio-Temporal con GMU) lista.");
        console.log("💭 Sistema Omnisciente: Capa 3 (Cognitiva con umbrales adaptativos) lista.");
        this.verificarEstructura();
    }

    /**
     * Conecta el sistema a un puente de streaming (ej. Google Colab)
     */
    conectarColab(url: string) {
        this.bridge = new StreamingBridge(url);
        
        // Registrar callback para procesar feedback de Colab
        this.bridge.registrarCallbackFeedback((decision) => {
            this.procesarRespuestaColab(decision);
        });
        
        console.log(`🔗 Sistema Omnisciente: Conectado a Colab Bridge en ${url}`);
    }

    /**
     * Verifica que los 25 subespacios y sus átomos hayan sido creados correctamente
     */
    verificarEstructura() {
        const stats = this.sensorial.getCapa1().getEstadisticas();
        const statsCapa2 = this.capa2.obtenerEstadisticas();
        const statsCapa3 = this.capa3.obtenerEstadisticas();
        
        console.log(`✅ Capa 1: ${stats.subRedesActivas}/25 sub-redes activas.`);
        console.log(`✅ Capa 2: Buffer=${statsCapa2.tamanoBuffer}, Timestep=${statsCapa2.timestep}`);
        console.log(`✅ Capa 3: Umbrales adaptativos=[${statsCapa3.umbralesActuales.leve.toFixed(2)}, ${statsCapa3.umbralesActuales.grave.toFixed(2)}]`);
        
        if (stats.subRedesActivas < 25) {
            console.error("⚠️ Error: No se crearon todos los subespacios sensoriales.");
        }
    }

    async crearAtomo(id: string) {
        const atomo = new AtomoTopologico(id);
        await atomo.inicializar();
        this.atomos.set(id, atomo);
        return atomo;
    }

    /**
     * Protocolo de Infección: Difunde señales críticas entre todos los átomos
     */
    async propagarInfeccion() {
        const atomosArray = Array.from(this.atomos.values());
        
        // Cada átomo emite sus señales
        for (const atomo of atomosArray) {
            const senales = atomo.emitirSenal();
            
            if (senales.length > 0) {
                console.log(`🦠 [${atomo.id}] Emitiendo ${senales.length} señales de anomalía`);
                
                // Propagar a todos los demás átomos
                for (const otroAtomo of atomosArray) {
                    if (otroAtomo.id !== atomo.id) {
                        otroAtomo.recibirSenal(senales);
                    }
                }
            }
        }
    }

    async procesarFlujo(id: string, telemetria: Omega21Telemetry, dendritasConfig?: any) {
        const atomo = this.atomos.get(id);
        if (!atomo) throw new Error(`Átomo ${id} no encontrado`);
        
        // 1. Aplicar configuración dendrítica si está disponible
        if (dendritasConfig) {
            atomo.simulador.configurarDendritas(dendritasConfig);
        }
        
        // 2. Procesar a través del átomo
        const resultado = await atomo.percibir(telemetria);

        // 3. Propagar anomalías detectadas a otros átomos (Mecanismo de Infección)
        if (resultado.neuronal.prediccion_anomalia > 0.7) {
            const senales = atomo.emitirSenal();
            const atomosArray = Array.from(this.atomos.values());
            for (const otroAtomo of atomosArray) {
                if (otroAtomo.id !== atomo.id) {
                    otroAtomo.recibirSenal(senales);
                }
            }
        }

        // 4. Registrar en el entrenador cognitivo para consolidación
        const percepciones = telemetria.vector_72d || new Array(72).fill(0);
        const fueFalla = telemetria.neuro.nov > 200; // Heurística simple
        this.entrenador.registrarExperiencia(percepciones, atomo.hipergrafo, fueFalla);

        // 5. Enviar a Colab si está disponible
        if (this.bridge) {
            const vector1600D = this.expandirAVector1600D(resultado.neuronal.ajustes_dendritas);
            await this.bridge.enviarVector(vector1600D, fueFalla);
        }

        return resultado;
    }

    /**
     * Procesa un vector 256D a través de toda la jerarquía cognitiva
     */
    async procesarCognicion(vector: Vector256D, esAnomalia: boolean = false) {
        if (!this.inicializado) await this.inicializar();

        // 1. Capa 0 y 1: Procesamiento Sensorial (25 Sub-redes)
        const salidaSensorial = await this.sensorial.procesar(vector);

        // 1.5 Streaming a Colab (si está conectado)
        if (this.bridge) {
            const vector1600d = Object.values(salidaSensorial).flat();
            this.bridge.enviarVector(vector1600d, esAnomalia);
        }

        // 2. Capa 2: Procesamiento Espacio-Temporal (Bi-LSTM + Transformer)
        const salidaContextual = await this.capa2.procesar(salidaSensorial);

        // 3. Capa 3: Cognición y Consenso
        const decision = await this.capa3.procesar(salidaContextual);

        // 4. Generar Coherencia Mental (Imagen Mental en Hipergrafo)
        // Mantenemos la compatibilidad con la versión anterior
        const percepcionesArray = Object.values(salidaSensorial).flat() as number[];
        
        const imagenMental = this.corteza.generarCoherencia([]);
        this.entrenador.registrarExperiencia(percepcionesArray, imagenMental, false);

        const ultimoConcepto = imagenMental.obtenerNodos().slice(-1)[0];

        return {
            sensorial: salidaSensorial,
            contexto: salidaContextual,
            decision: decision,
            coherencia: {
                idConcepto: ultimoConcepto?.id || 'CONCEPT_NULL',
                estabilidadGlobal: Math.min(1, imagenMental.obtenerNodos().length / 100),
                numConceptos: imagenMental.obtenerNodos().length,
                imagenMental: imagenMental // El hipergrafo resultante
            }
        };
    }

    /**
     * Procesa respuesta de Colab con feedback dinámico
     */
    private procesarRespuestaColab(decision: any): void {
        // Obtener el primer átomo para aplicar feedback (en futuro, distribuir entre todos)
        const atomos = Array.from(this.atomos.values());
        if (atomos.length === 0) return;
        
        const atomo = atomos[0];
        
        // Procesar decisión a través del HipergrafoBridge
        this.hipergrafoBridge.procesarDecision(decision, atomo);
        
        console.log(`📊 [HipergrafoBridge] Decisión procesada: anomaly=${decision.anomaly_prob?.toFixed(2)}, dendrites=${decision.dendrite_adjustments?.length ?? 0}D, coherence=${decision.coherence_state?.length ?? 0}D`);
    }

    /**
     * Obtiene reporte de estadísticas del Hipergrafo
     */
    obtenerReporteHipergrafoBridge() {
        return this.hipergrafoBridge.generarReporte();
    }

    /**
     * Expande un embedding de 256D a 1600D usando los 25 subespacios
     * Cada subespacio recibe una versión del embedding modulada
     */
    private expandirAVector1600D(embedding256D: number[]): number[] {
        const vector1600D: number[] = [];
        const DIMENSIONES_SUBESPACIO = 64; // 1600 / 25 = 64
        const NUM_SUBESPACIOS = 25;

        // Garantizar que el embedding sea 256D
        const emb = embedding256D || new Array(256).fill(0);
        const embAjustado = emb.length === 256 ? emb : emb.slice(0, 256);

        for (let s = 0; s < NUM_SUBESPACIOS; s++) {
            // Para cada subespacio, crear una versión modulada del embedding
            for (let i = 0; i < DIMENSIONES_SUBESPACIO; i++) {
                // Aplicar una transformación harmónica basada en la frecuencia del subespacio
                const idxEmb = (s * 10 + i) % 256;
                const modulacion = Math.sin((s + 1) * Math.PI / 25) * Math.cos((i + 1) * Math.PI / 64);
                const valor = (embAjustado[idxEmb] || 0) * (1 + modulacion * 0.3);
                vector1600D.push(valor);
            }
        }

        return vector1600D;
    }
}
