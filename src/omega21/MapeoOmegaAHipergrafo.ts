/**
 * MapeoOmegaAHipergrafo - Transforma telemetría Omega 21 a Hipergrafo
 */

import { Hipergrafo, Nodo, Hiperedge } from '../core';
import { Omega21Telemetry, DENDRITE_DEFINITIONS } from './Schema';

export enum TipoNodoOmega {
    SUBESPACIO = 'subespacio',
    DENDRITA = 'dendrita',
    SOMA = 'soma',
    SPIKE = 'spike',
    ANOMALIA = 'anomalia'
}

export interface ConfiguracionMapeoOmega {
    umbralCorrelacion: number;
    incluirDendritas: boolean;
    pesoMinimo: number;
}

const CONFIG_DEFAULT: ConfiguracionMapeoOmega = {
    umbralCorrelacion: 0.3,
    incluirDendritas: true,
    pesoMinimo: 0.1
};

export class MapeoOmegaAHipergrafo {
    private config: ConfiguracionMapeoOmega;

    constructor(config: Partial<ConfiguracionMapeoOmega> = {}) {
        this.config = { ...CONFIG_DEFAULT, ...config };
    }

    mapear(telemetry: Omega21Telemetry): Hipergrafo {
        const hipergrafo = new Hipergrafo(`Omega21_${telemetry.meta.blk}`);
        
        // 1. Crear nodo SOMA central
        const nodeSoma = new Nodo('SOMA', {
            tipo: TipoNodoOmega.SOMA,
            voltage: telemetry.dendrites.soma_v,
            spike: telemetry.dendrites.spike,
            patron_id: telemetry.neuro.id,
            similitud: telemetry.neuro.sim,
            novelty: telemetry.neuro.nov,
            categoria: telemetry.neuro.cat
        });
        hipergrafo.agregarNodo(nodeSoma);
        
        // 2. Crear nodos de dendritas
        const nodosDendritas: Nodo[] = [];
        if (this.config.incluirDendritas) {
            DENDRITE_DEFINITIONS.forEach((def, idx) => {
                const valor = this.obtenerValorDendrita(telemetry, def.tipo);
                const nodoDendrita = new Nodo(def.nombre, {
                    tipo: TipoNodoOmega.DENDRITA,
                    tipoFisico: def.tipoFisico,
                    modelo: def.modelo,
                    valor,
                    unidad: def.unidad,
                    indice: idx
                });
                hipergrafo.agregarNodo(nodoDendrita);
                nodosDendritas.push(nodoDendrita);
            });
        }
        
        // 3. Crear hiperedge conectando SOMA con dendritas
        if (nodosDendritas.length > 0) {
            const edgeIntegracion = new Hiperedge(
                'integracion_soma',
                [nodeSoma, ...nodosDendritas],
                telemetry.neuro.sim / 255
            );
            hipergrafo.agregarHiperedge(edgeIntegracion);
        }
        
        // 4. Crear clusters de dendritas relacionadas
        this.crearClusters(hipergrafo, nodosDendritas);
        
        // 5. Marcar eventos especiales
        if (telemetry.dendrites.spike) {
            const nodoSpike = new Nodo('SPIKE_EVENT', {
                tipo: TipoNodoOmega.SPIKE,
                timestamp: telemetry.meta.ts,
                voltage: telemetry.dendrites.soma_v
            });
            hipergrafo.agregarNodo(nodoSpike);
            
            const edgeSpike = new Hiperedge('spike_connection', [nodeSoma, nodoSpike], 1.0);
            hipergrafo.agregarHiperedge(edgeSpike);
        }
        
        if (telemetry.neuro.nov > 150) {
            const nodoAnomalia = new Nodo('ANOMALIA', {
                tipo: TipoNodoOmega.ANOMALIA,
                novelty: telemetry.neuro.nov,
                timestamp: telemetry.meta.ts
            });
            hipergrafo.agregarNodo(nodoAnomalia);
            
            const edgeAnomalia = new Hiperedge('anomalia_detected', [nodeSoma, nodoAnomalia], 0.9);
            hipergrafo.agregarHiperedge(edgeAnomalia);
        }

        // 6. Nodo de Firma (Signature)
        const nodoFirma = new Nodo(`SIG_${telemetry.sig.fp}`, {
            tipo: 'signature',
            lsh: telemetry.sig.lsh,
            quantile: telemetry.sig.eq,
            scale: telemetry.sig.sc
        });
        hipergrafo.agregarNodo(nodoFirma);
        hipergrafo.agregarHiperedge(new Hiperedge('signature_link', [nodeSoma, nodoFirma], 0.5));
        
        return hipergrafo;
    }

    /**
     * Extrae tensores para el motor de inferencia GNN
     */
    extraerTensores(hipergrafo: Hipergrafo): { nodeFeatures: number[][], edgeIndex: number[][], globalVector: number[] } {
        const nodos = hipergrafo.obtenerNodos();
        const nodeFeatures: number[][] = nodos.map(n => {
            const props = n.metadata;
            // Vector de características: [valor_normalizado, tipo_id, similitud, novelty]
            let tipoId = 0;
            switch(props.tipo) {
                case TipoNodoOmega.SOMA: tipoId = 1; break;
                case TipoNodoOmega.DENDRITA: tipoId = 2; break;
                case TipoNodoOmega.SPIKE: tipoId = 3; break;
                case TipoNodoOmega.ANOMALIA: tipoId = 4; break;
                case 'signature': tipoId = 5; break;
            }

            return [
                (props.valor || props.voltage || 0) / 10000, // Normalización básica
                tipoId,
                (props.similitud || 0) / 255,
                (props.novelty || 0) / 255
            ];
        });

        const edgeIndex: number[][] = [[], []];
        hipergrafo.obtenerHiperedges().forEach(edge => {
            const targetNodos = Array.from(edge.nodos); 
            if (targetNodos.length > 0) {
                const sourceIdx = nodos.findIndex(n => n.id === targetNodos[0]);
                for (let i = 1; i < targetNodos.length; i++) {
                    const targetIdx = nodos.findIndex(n => n.id === targetNodos[i]);
                    if (sourceIdx !== -1 && targetIdx !== -1) {
                        edgeIndex[0].push(sourceIdx);
                        edgeIndex[1].push(targetIdx);
                    }
                }
            }
        });

        // El global vector se llenará en InferenciaLocal usando la telemetría cruda
        return { nodeFeatures, edgeIndex, globalVector: new Array(256).fill(0) };
    }

    private crearClusters(hipergrafo: Hipergrafo, nodos: Nodo[]): void {
        // Cluster eléctrico: voltage, current, power
        const nodosElectricos = nodos.filter(n => 
            ['Voltage', 'Current', 'Power', 'Capacitor'].includes(n.label)
        );
        if (nodosElectricos.length >= 2) {
            const edge = new Hiperedge('cluster_electrico', nodosElectricos, 0.8);
            hipergrafo.agregarHiperedge(edge);
        }
        
        // Cluster temporal: delay, decay, memory
        const nodosTemporales = nodos.filter(n => 
            ['Delay', 'Decay', 'Memory'].includes(n.label)
        );
        if (nodosTemporales.length >= 2) {
            const edge = new Hiperedge('cluster_temporal', nodosTemporales, 0.7);
            hipergrafo.agregarHiperedge(edge);
        }
        
        // Cluster bioquímico: michaelis, hill
        const nodosBio = nodos.filter(n => 
            ['Michaelis', 'Hill'].includes(n.label)
        );
        if (nodosBio.length >= 2) {
            const edge = new Hiperedge('cluster_bioquimico', nodosBio, 0.75);
            hipergrafo.agregarHiperedge(edge);
        }
    }

    private obtenerValorDendrita(telemetry: Omega21Telemetry, tipo: string): number {
        const d = telemetry.dendrites as any;
        if (tipo === 'dew') return d.dew_temp;
        if (tipo === 'frequency') return d.freq;
        return typeof d[tipo] === 'number' ? d[tipo] : 0;
    }

    reset(): void {
        // No hay estado persistente por ahora
    }
}

export default MapeoOmegaAHipergrafo;
