/**
 * Controlador de Dendritas
 */

import { Hipergrafo } from '../core';
import { 
    CentralidadHipergrafo, 
    ClusteringHipergrafo
} from '../analisis';
import { DendriteType, DENDRITE_DEFINITIONS, Omega21Telemetry } from '../omega21/Schema';
import { RenodeController } from '../hardware/RenodeController';

export interface ParametrosDendrita {
    indice: number;
    tipo: DendriteType;
    nombre: string;
    ganancia: number;
    offset: number;
    umbral: number;
    decaimiento: number;
    minimo: number;
    maximo: number;
}

export interface AjusteDendrita {
    indiceDendrita: number;
    parametro: 'ganancia' | 'offset' | 'umbral' | 'decaimiento';
    operacion: 'set' | 'add' | 'multiply';
    valor: number;
}

export interface EstadoAnalisis {
    numNodos: number;
    numAristas: number;
    densidad: number;
    centralidadMaxima: number;
    coeficienteClusteringGlobal: number;
    numComunidades: number;
    radioEspectral: number;
    conectividad: number;
    ultimoSpike: boolean;
    categoria: number;
    novelty: number;
    score: number;
    anomaliaDetectada?: boolean;
    confianza?: number;
}

export interface ConfiguracionControlador {
    intervaloEvaluacion: number;
    factorSuavizado: number;
    autoAjuste: boolean;
    maxCambiosPorCiclo: number;
    dendritasHabilitadas: number[];
}

export class DendriteController {
    private config: ConfiguracionControlador;
    private parametros: ParametrosDendrita[];
    private renodeController: RenodeController | null = null;
    private historialEstados: EstadoAnalisis[] = [];

    constructor(config: Partial<ConfiguracionControlador> = {}) {
        this.config = {
            intervaloEvaluacion: config.intervaloEvaluacion || 100,
            factorSuavizado: config.factorSuavizado || 0.3,
            autoAjuste: config.autoAjuste ?? false,
            maxCambiosPorCiclo: config.maxCambiosPorCiclo || 3,
            dendritasHabilitadas: config.dendritasHabilitadas || Array.from({ length: 16 }, (_, i) => i)
        };
        
        this.parametros = DENDRITE_DEFINITIONS.map((def, indice) => ({
            indice,
            tipo: def.tipo,
            nombre: def.nombre,
            ganancia: 1.0,
            offset: 0.0,
            umbral: 0.5,
            decaimiento: 0.1,
            minimo: 0.01,
            maximo: 10.0
        }));
    }

    async conectarHardware(renodeController: RenodeController): Promise<void> {
        this.renodeController = renodeController;
    }

    analizarEstado(hipergrafo: Hipergrafo, telemetria: Omega21Telemetry): EstadoAnalisis {
        const nodos = hipergrafo.obtenerNodos();
        const numNodos = nodos.length;
        const numAristas = hipergrafo.obtenerHiperedges().length;
        
        const maxAristas = numNodos > 1 ? (numNodos * (numNodos - 1)) / 2 : 1;
        const densidad = numAristas / maxAristas;
        
        let centralidadMaxima = 0;
        nodos.forEach(n => {
            const c = CentralidadHipergrafo.centralidadGrado(hipergrafo, n.id);
            if (c > centralidadMaxima) centralidadMaxima = c;
        });
        
        const coefGlobal = ClusteringHipergrafo.coeficienteClusteringGlobal(hipergrafo);
        
        const estado: EstadoAnalisis = {
            numNodos,
            numAristas,
            densidad,
            centralidadMaxima,
            coeficienteClusteringGlobal: coefGlobal,
            numComunidades: 1, // Simplificado
            radioEspectral: 0, // Simplificado
            conectividad: 0, // Simplificado
            ultimoSpike: telemetria.dendrites.spike === 1,
            categoria: telemetria.neuro.cat,
            novelty: (telemetria.neuro as any).novelty || 100,
            score: (telemetria.neuro as any).score || 0.5
        };
        
        this.historialEstados.push(estado);
        if (this.historialEstados.length > 100) this.historialEstados.shift();
        
        return estado;
    }

    async ejecutarCiclo(hipergrafo: Hipergrafo, telemetria: Omega21Telemetry): Promise<{
        estado: EstadoAnalisis;
        ajustes: AjusteDendrita[];
        aplicadoHardware: boolean;
    }> {
        const estado = this.analizarEstado(hipergrafo, telemetria);
        const ajustes: AjusteDendrita[] = [];
        
        // Lógica de control simplificada
        if (estado.densidad > 0.8) {
            ajustes.push({ indiceDendrita: 0, parametro: 'ganancia', operacion: 'multiply', valor: 0.9 });
        }
        
        if (this.config.autoAjuste && this.renodeController && ajustes.length > 0) {
            for (const ajuste of ajustes.slice(0, this.config.maxCambiosPorCiclo)) {
                await this.renodeController.modificarDendrita(ajuste.indiceDendrita, 'param_a', 1000);
            }
            return { estado, ajustes, aplicadoHardware: true };
        }
        
        return { estado, ajustes, aplicadoHardware: false };
    }

    obtenerParametros(): ParametrosDendrita[] {
        return this.parametros;
    }

    /**
     * Aplica ajustes recibidos de una IA externa (vector de 16 valores)
     */
    async aplicarAjustesVector(ajustes: number[]): Promise<void> {
        if (!this.renodeController || !this.config.autoAjuste) return;

        for (let i = 0; i < Math.min(ajustes.length, 16); i++) {
            const valorAjuste = ajustes[i];
            if (Math.abs(valorAjuste) > 0.1) { // Umbral de acción
                // Mapear salida de IA (-1 a 1) a parámetros de hardware
                // Ejemplo: Modificar ganancia (param_a)
                // Valor base 1000, ajuste +/- 100
                const nuevoValor = 1000 + (valorAjuste * 100);
                await this.renodeController.modificarDendrita(i, 'param_a', Math.floor(nuevoValor));
            }
        }
    }

    obtenerHistorial(): EstadoAnalisis[] {
        return this.historialEstados;
    }
}

export default DendriteController;
