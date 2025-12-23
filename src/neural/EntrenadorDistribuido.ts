/**
 * Entrenador Distribuido
 * 
 * Coordina la recolección de estados del hipergrafo y su envío a Google Colab
 * para el entrenamiento de la red neuronal LIF 1024.
 */

import { EventEmitter } from 'events';
import { Orquestador, ResultadoAnalisis } from '../orquestador';
import { ColabBridge } from '../neural/ColabBridge';
import { InferenciaLocal } from '../neural/InferenciaLocal';
import { CONFIG_COLAB } from '../neural/configColab';
import { AnalizadorFisico } from '../analisis/AnalizadorFisico';
import * as fs from 'fs';

export interface ConfiguracionEntrenamiento {
    muestrasParaLote: number;    // Cuántas muestras acumular antes de enviar
    intervaloEnvioMs: number;    // Tiempo máximo entre envíos
    habilitarAumento: boolean;   // Aumentar datos localmente
    modoLocal: boolean;          // Usar ONNX localmente
}

export class EntrenadorDistribuido extends EventEmitter {
    private orquestador: Orquestador;
    private colab: ColabBridge;
    private inferenciaLocal: InferenciaLocal | null = null;
    private loteActual: any[] = [];
    private config: ConfiguracionEntrenamiento;
    private timerEnvio: NodeJS.Timeout | null = null;
    private historialEstados: any[] = []; // Últimos 10 estados para predicción
    private contadorAnomalias: number = 0;
    private prediccionAnomalia: boolean = false;
    private analizadorFisico: AnalizadorFisico;
    private ultimoResultado: ResultadoAnalisis | null = null;

    constructor(orquestador: Orquestador, config: Partial<ConfiguracionEntrenamiento> = {}) {
        super();
        this.orquestador = orquestador;
        this.colab = new ColabBridge(CONFIG_COLAB.urlServidor);
        this.analizadorFisico = new AnalizadorFisico();
        this.config = {
            muestrasParaLote: config.muestrasParaLote || 20,
            intervaloEnvioMs: config.intervaloEnvioMs || 5000,
            habilitarAumento: config.habilitarAumento || false,
            modoLocal: config.modoLocal || fs.existsSync('/workspaces/HIPERGRAFO/models/omega21_brain.onnx')
        };

        if (this.config.modoLocal) {
            console.log('🚀 Detectado modelo ONNX. Iniciando motor de inferencia local...');
            this.inferenciaLocal = new InferenciaLocal();
            this.inferenciaLocal.inicializar().catch(err => {
                console.error('⚠️ Falló la inicialización local, volviendo a modo Colab:', err);
                this.config.modoLocal = false;
            });
        }

        this.configurarEscucha();
    }

    /**
     * Configura la escucha de eventos del orquestador
     */
    private configurarEscucha() {
        this.orquestador.on('procesado', (resultado: ResultadoAnalisis) => {
            this.acumularMuestra(resultado);
        });

        this.orquestador.on('spike', (telemetria) => {
            // Los spikes son eventos críticos para el entrenamiento
            this.marcarEventoCritico('SPIKE', telemetria);
        });
    }

    /**
     * Acumula una muestra en el lote actual
     */
    private acumularMuestra(resultado: ResultadoAnalisis) {
        // Serialización completa para GNN
        const muestraSerializada = this.serializarHipergrafo(resultado);
        this.ultimoResultado = resultado;
        
        // Detectar si esta muestra es una anomalía
        const esAnomalia = resultado.estadoAnalisis.novelty > 200 || 
                          resultado.estadoAnalisis.densidad > 0.9 ||
                          resultado.estadoAnalisis.ultimoSpike;
        
        if (esAnomalia) {
            this.contadorAnomalias++;
            console.log(`[Entrenador] ⚠️ Anomalía detectada (#${this.contadorAnomalias}): novelty=${resultado.estadoAnalisis.novelty.toFixed(0)}, densidad=${resultado.estadoAnalisis.densidad.toFixed(3)}`);
        }
        
        const muestra = {
            timestamp: resultado.timestamp,
            ...muestraSerializada, // Expande node_features, edge_index, global_vector
            telemetria_raw: resultado.telemetriaOriginal,
            es_anomalia: esAnomalia, // Label para aprendizaje supervisado
            estado_topologico: resultado.estadoAnalisis // Métricas completas
        };

        this.loteActual.push(muestra);
        
        // Guardar en historial (últimos 10 estados)
        this.historialEstados.push({
            timestamp: resultado.timestamp,
            estado: resultado.estadoAnalisis,
            es_anomalia: esAnomalia
        });
        if (this.historialEstados.length > 10) {
            this.historialEstados.shift();
        }

        if (this.loteActual.length >= this.config.muestrasParaLote) {
            this.enviarAColab();
        }
    }

    /**
     * Serializa el hipergrafo a un formato compatible con PyTorch Geometric
     */
    private serializarHipergrafo(resultado: ResultadoAnalisis) {
        const hg = resultado.hipergrafo;
        const nodos = hg.obtenerNodos();
        const nodoIdAIndice = new Map(nodos.map((n, i) => [n.id, i]));

        // 1. Node Features
        // Vector de características: [valor, es_soma, es_dendrita, es_spike]
        const node_features = nodos.map(n => {
            const valor = n.metadata.valor || 0;
            const tipo = n.metadata.tipo || 'UNKNOWN';
            return [
                typeof valor === 'number' ? valor : 0,
                tipo === 'SOMA' ? 1 : 0,
                tipo === 'DENDRITA' ? 1 : 0,
                tipo === 'SPIKE' ? 1 : 0
            ];
        });

        // 2. Edge Index (Clique Expansion)
        const sources: number[] = [];
        const targets: number[] = [];

        hg.obtenerHiperedges().forEach(edge => {
            const indices = Array.from(edge.nodos)
                .map(id => nodoIdAIndice.get(id))
                .filter(idx => idx !== undefined) as number[];
            
            // Conectar todos con todos (Clique)
            for (let i = 0; i < indices.length; i++) {
                for (let j = 0; j < indices.length; j++) {
                    if (i !== j) {
                        sources.push(indices[i]);
                        targets.push(indices[j]);
                    }
                }
            }
        });

        // 3. Vector Global (256D)
        let global_vector = (resultado.telemetriaOriginal as any).metrics_256 || [];
        if (!global_vector || global_vector.length === 0) {
            // Fallback: Usar valores de dendritas como vector base
            const dendritas = Object.values(resultado.telemetriaOriginal.dendrites);
            // Rellenar hasta 256 con ceros si es necesario
            global_vector = Array(256).fill(0).map((_, i) => dendritas[i] || 0);
        }

        return {
            node_features,
            edge_index: [sources, targets],
            global_vector,
            target_stability: resultado.estadoAnalisis.densidad
        };
    }

    private marcarEventoCritico(tipo: string, _datos: any) {
        console.log(`[Entrenador] Evento crítico detectado: ${tipo}. Forzando envío...`);
        this.enviarAColab();
    }

    /**
     * Envía el lote acumulado a Colab o procesa localmente
     */
    async enviarAColab() {
        if (this.loteActual.length === 0) return;

        const datosAEnviar = [...this.loteActual];
        this.loteActual = []; // Limpiar lote inmediatamente

        // Calcular estadísticas del lote
        const anomaliasEnLote = datosAEnviar.filter((m: any) => m.es_anomalia).length;
        const tasaAnomalias = (anomaliasEnLote / datosAEnviar.length) * 100;
        
        let respuesta: any = null;

        if (this.config.modoLocal && this.inferenciaLocal) {
            // MODO LOCAL: Usar el modelo ONNX
            const ultimaMuestra = datosAEnviar[datosAEnviar.length - 1];
            respuesta = await this.inferenciaLocal.predecir(
                ultimaMuestra.node_features,
                ultimaMuestra.edge_index,
                ultimaMuestra.global_vector,
                ultimaMuestra.telemetria_raw
            );
            
            if (respuesta) {
                console.log(`[Entrenador] 🧠 Inferencia LOCAL (ONNX) completada.`);
            }
        } 
        
        // Si no hay modo local o falló, intentar Colab
        if (!respuesta) {
            console.log(`[Entrenador] Enviando lote de ${datosAEnviar.length} muestras a Colab...`);
            console.log(`[Entrenador] 📊 Estadísticas: ${anomaliasEnLote}/${datosAEnviar.length} anomalías (${tasaAnomalias.toFixed(1)}%)`);
            
            try {
                respuesta = await this.colab.ejecutarModelo({
                    accion: 'entrenar',
                    datos: datosAEnviar,
                    historial: this.historialEstados,
                    config: {
                        tipo_red: 'LIF_1024',
                        learning_rate: 0.001
                    }
                });
                console.log('✅ Respuesta de Colab:', respuesta);
            } catch (error) {
                console.error('❌ Error enviando a Colab:', error);
                return; // No podemos continuar sin respuesta
            }
        }

        // Procesar respuesta (común para Local y Colab)
        if (respuesta) {
            // Si hay predicción de anomalía
            if (respuesta.prediccion_anomalia !== undefined) {
                this.prediccionAnomalia = respuesta.prediccion_anomalia > 0.5;
                if (this.prediccionAnomalia) {
                    console.log(`[Entrenador] 🔮 Predicción: Anomalía inminente (confianza: ${(respuesta.prediccion_anomalia * 100).toFixed(1)}%)`);
                }
            }
            
            this.emit('feedback', respuesta);

            // --- CAPA FÍSICA ---
            if (this.ultimoResultado) {
                const metricasFisicas = this.analizadorFisico.calcular(
                    this.ultimoResultado.estadoAnalisis, 
                    respuesta,
                    this.ultimoResultado.telemetriaOriginal
                );
                const diagnostico = this.analizadorFisico.generarDiagnostico(metricasFisicas);
                
                this.emit('fisica', {
                    metricas: metricasFisicas,
                    diagnostico: diagnostico,
                    timestamp: new Date(),
                    origen: respuesta.modo || 'COLAB'
                });
            }

            // Aplicar feedback
            if (respuesta.ajustes_dendritas && Array.isArray(respuesta.ajustes_dendritas)) {
                await this.orquestador.aplicarAjustesIA(respuesta.ajustes_dendritas);
            }
        }
    }

    /**
     * Inicia el proceso de entrenamiento
     */
    async iniciar() {
        if (this.config.modoLocal) {
            console.log('🚀 Iniciando en MODO LOCAL (ONNX).');
            this.timerEnvio = setInterval(() => this.enviarAColab(), this.config.intervaloEnvioMs);
            return;
        }

        console.log('[Entrenador] Verificando conexión con Colab...');
        const conectado = await this.colab.verificarConexion();
        
        if (conectado) {
            console.log('🚀 Entrenamiento distribuido iniciado.');
            this.timerEnvio = setInterval(() => this.enviarAColab(), this.config.intervaloEnvioMs);
        } else {
            console.warn('⚠️ No se pudo conectar con Colab. El sistema funcionará en modo pasivo hasta que se detecte un modelo local o conexión.');
        }
    }

    /**
     * Detiene el entrenamiento
     */
    detener() {
        if (this.timerEnvio) clearInterval(this.timerEnvio);
        console.log('[Entrenador] Entrenamiento detenido.');
    }
}
