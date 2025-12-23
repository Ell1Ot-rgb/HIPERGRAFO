/**
 * FusionMultimodal.ts
 * 
 * Implementa estrategias avanzadas de fusión de datos multimodales.
 * Basado en el informe "Estrategias de Fusión de Datos y Optimización de Modelos".
 * 
 * Incluye:
 * - Gated Multimodal Unit (GMU): Fusión con compuertas adaptativas
 * - Multi-Head GMU: Fusión por grupos de características
 * - Fusión Híbrida: Combinación de Early + Late Fusion
 */

/**
 * Resultado de una operación de fusión
 */
export interface ResultadoFusion {
    vectorFusionado: number[];
    pesosCompuerta: number[];      // Valores z del GMU (interpretabilidad)
    modalidadDominante: 'temporal' | 'espacial' | 'equilibrado';
    confianzaFusion: number;
}

/**
 * Configuración para el GMU
 */
export interface ConfiguracionGMU {
    dimensionEntrada: number;      // Dimensión de cada modalidad
    dimensionSalida: number;       // Dimensión del vector fusionado
    numCabezas: number;            // Número de cabezas para Multi-Head GMU
    dropout: number;               // Probabilidad de dropout (0-1)
    temperaturaCompuerta: number;  // Controla la "dureza" de la decisión sigmoid
}

const CONFIG_GMU_DEFAULT: ConfiguracionGMU = {
    dimensionEntrada: 512,
    dimensionSalida: 512,
    numCabezas: 8,
    dropout: 0.1,
    temperaturaCompuerta: 1.0
};

/**
 * Gated Multimodal Unit (GMU)
 * 
 * Implementa fusión con compuerta adaptativa que aprende a ponderar
 * dinámicamente la importancia de cada modalidad basándose en la entrada.
 * 
 * Ecuaciones:
 * h_temp' = tanh(W_temp · H_temporal + b_temp)
 * h_spat' = tanh(W_spat · H_espacial + b_spat)
 * z = σ((W_z · [H_temporal, H_espacial] + b_z) / T)
 * H_fused = z ⊙ h_temp' + (1 - z) ⊙ h_spat'
 * 
 * Donde T es la temperatura que controla la "dureza" de la compuerta.
 */
export class GatedMultimodalUnit {
    private config: ConfiguracionGMU;
    
    // Pesos simulados (en producción, estos vienen del modelo ONNX)
    private W_temporal!: number[][];
    private W_espacial!: number[][];
    private W_gate!: number[][];
    private b_temporal!: number[];
    private b_espacial!: number[];
    private b_gate!: number[];

    // Estado interno para debugging/explicabilidad
    private ultimoZ: number[] = [];
    private historialZ: number[][] = [];

    constructor(config: Partial<ConfiguracionGMU> = {}) {
        this.config = { ...CONFIG_GMU_DEFAULT, ...config };
        this.inicializarPesos();
    }

    /**
     * Inicializa pesos con distribución Xavier/Glorot
     */
    private inicializarPesos() {
        const { dimensionEntrada, dimensionSalida } = this.config;
        
        // Xavier initialization: sqrt(6 / (fan_in + fan_out))
        const escalaXavier = Math.sqrt(6 / (dimensionEntrada + dimensionSalida));
        
        this.W_temporal = this.crearMatriz(dimensionSalida, dimensionEntrada, escalaXavier);
        this.W_espacial = this.crearMatriz(dimensionSalida, dimensionEntrada, escalaXavier);
        this.W_gate = this.crearMatriz(dimensionSalida, dimensionEntrada * 2, escalaXavier);
        
        this.b_temporal = new Array(dimensionSalida).fill(0);
        this.b_espacial = new Array(dimensionSalida).fill(0);
        this.b_gate = new Array(dimensionSalida).fill(0);
    }

    private crearMatriz(filas: number, columnas: number, escala: number): number[][] {
        return Array.from({ length: filas }, () =>
            Array.from({ length: columnas }, () => (Math.random() * 2 - 1) * escala)
        );
    }

    /**
     * Ejecuta la fusión GMU
     */
    public fusionar(hTemporal: number[], hEspacial: number[]): ResultadoFusion {
        const { dimensionSalida, temperaturaCompuerta } = this.config;

        // 1. Proyección no-lineal de cada modalidad
        const hTempProyectado = this.tanh(this.matmulVector(this.W_temporal, hTemporal, this.b_temporal));
        const hSpatProyectado = this.tanh(this.matmulVector(this.W_espacial, hEspacial, this.b_espacial));

        // 2. Calcular coeficiente de compuerta z
        const concatenado = [...hTemporal, ...hEspacial];
        const preActivacion = this.matmulVector(this.W_gate, concatenado, this.b_gate);
        const z = preActivacion.map(v => this.sigmoid(v / temperaturaCompuerta));

        // 3. Fusión ponderada dinámica
        const vectorFusionado = new Array(dimensionSalida);
        for (let i = 0; i < dimensionSalida; i++) {
            vectorFusionado[i] = z[i] * hTempProyectado[i] + (1 - z[i]) * hSpatProyectado[i];
        }

        // 4. Análisis de la compuerta para explicabilidad
        this.ultimoZ = z;
        this.historialZ.push(z);
        if (this.historialZ.length > 100) this.historialZ.shift();

        const promedioZ = z.reduce((a, b) => a + b, 0) / z.length;
        let modalidadDominante: 'temporal' | 'espacial' | 'equilibrado';
        if (promedioZ > 0.6) modalidadDominante = 'temporal';
        else if (promedioZ < 0.4) modalidadDominante = 'espacial';
        else modalidadDominante = 'equilibrado';

        // 5. Calcular confianza basada en la consistencia del gate
        const varianzaZ = this.calcularVarianza(z);
        const confianzaFusion = 1 - Math.min(varianzaZ * 4, 0.5); // Alta varianza = baja confianza

        return {
            vectorFusionado,
            pesosCompuerta: z,
            modalidadDominante,
            confianzaFusion
        };
    }

    /**
     * Multiplicación matriz-vector con bias
     */
    private matmulVector(W: number[][], x: number[], b: number[]): number[] {
        const resultado = new Array(W.length).fill(0);
        for (let i = 0; i < W.length; i++) {
            for (let j = 0; j < Math.min(W[i].length, x.length); j++) {
                resultado[i] += W[i][j] * x[j];
            }
            resultado[i] += b[i];
        }
        return resultado;
    }

    private tanh(x: number[]): number[] {
        return x.map(v => Math.tanh(v));
    }

    private sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
    }

    private calcularVarianza(arr: number[]): number {
        const media = arr.reduce((a, b) => a + b, 0) / arr.length;
        return arr.reduce((acc, val) => acc + Math.pow(val - media, 2), 0) / arr.length;
    }

    /**
     * Obtiene estadísticas de explicabilidad
     */
    public obtenerEstadisticasCompuerta(): {
        promedioActual: number;
        tendenciaHistorica: number;
        estabilidad: number;
    } {
        const promedioActual = this.ultimoZ.reduce((a, b) => a + b, 0) / (this.ultimoZ.length || 1);
        
        const promediosHistoricos = this.historialZ.map(z => 
            z.reduce((a, b) => a + b, 0) / z.length
        );
        const tendenciaHistorica = promediosHistoricos.length > 0
            ? promediosHistoricos.reduce((a, b) => a + b, 0) / promediosHistoricos.length
            : 0.5;
        
        const estabilidad = 1 - this.calcularVarianza(promediosHistoricos);

        return { promedioActual, tendenciaHistorica, estabilidad };
    }
}

/**
 * Multi-Head GMU
 * 
 * Divide el vector de entrada en múltiples "cabezas" y aplica un GMU
 * independiente a cada una. Permite que diferentes partes del vector
 * tengan diferentes pesos de fusión.
 * 
 * Útil para vectores 256D donde diferentes rangos tienen semánticas distintas.
 */
export class MultiHeadGMU {
    private cabezas: GatedMultimodalUnit[];
    private numCabezas: number;
    private dimensionPorCabeza: number;

    constructor(dimensionTotal: number, numCabezas: number = 8) {
        this.numCabezas = numCabezas;
        this.dimensionPorCabeza = Math.floor(dimensionTotal / numCabezas);
        
        this.cabezas = Array.from({ length: numCabezas }, () => 
            new GatedMultimodalUnit({
                dimensionEntrada: this.dimensionPorCabeza,
                dimensionSalida: this.dimensionPorCabeza,
                numCabezas: 1,
                temperaturaCompuerta: 1.0
            })
        );
    }

    public fusionar(hTemporal: number[], hEspacial: number[]): ResultadoFusion {
        const resultadosPorCabeza: ResultadoFusion[] = [];

        for (let i = 0; i < this.numCabezas; i++) {
            const inicio = i * this.dimensionPorCabeza;
            const fin = inicio + this.dimensionPorCabeza;
            
            const segmentoTemporal = hTemporal.slice(inicio, fin);
            const segmentoEspacial = hEspacial.slice(inicio, fin);
            
            resultadosPorCabeza.push(this.cabezas[i].fusionar(segmentoTemporal, segmentoEspacial));
        }

        // Concatenar resultados de todas las cabezas
        const vectorFusionado = resultadosPorCabeza.flatMap(r => r.vectorFusionado);
        const pesosCompuerta = resultadosPorCabeza.flatMap(r => r.pesosCompuerta);

        // Determinar modalidad dominante global
        const promedioGlobal = pesosCompuerta.reduce((a, b) => a + b, 0) / pesosCompuerta.length;
        let modalidadDominante: 'temporal' | 'espacial' | 'equilibrado';
        if (promedioGlobal > 0.6) modalidadDominante = 'temporal';
        else if (promedioGlobal < 0.4) modalidadDominante = 'espacial';
        else modalidadDominante = 'equilibrado';

        // Confianza promedio
        const confianzaFusion = resultadosPorCabeza.reduce((a, r) => a + r.confianzaFusion, 0) / this.numCabezas;

        return {
            vectorFusionado,
            pesosCompuerta,
            modalidadDominante,
            confianzaFusion
        };
    }

    /**
     * Análisis por cabeza: útil para debugging
     */
    public obtenerAnalisisPorCabeza(): { cabeza: number; modalidad: string; confianza: number }[] {
        return this.cabezas.map((cabeza, i) => {
            const stats = cabeza.obtenerEstadisticasCompuerta();
            return {
                cabeza: i,
                modalidad: stats.promedioActual > 0.5 ? 'temporal' : 'espacial',
                confianza: stats.estabilidad
            };
        });
    }
}

/**
 * Estrategia de Fusión Híbrida
 * 
 * Combina características de Early Fusion (interacción a nivel de features)
 * con Late Fusion (modularidad y robustez).
 */
export class FusionHibrida {
    private gmuPrincipal: GatedMultimodalUnit;
    private multiHeadGMU: MultiHeadGMU;
    private pesoEarlyFusion: number = 0.6;  // Peso para la fusión temprana
    private pesoLateFusion: number = 0.4;   // Peso para la fusión tardía

    constructor(dimension: number, numCabezas: number = 8) {
        this.gmuPrincipal = new GatedMultimodalUnit({
            dimensionEntrada: dimension,
            dimensionSalida: dimension
        });
        this.multiHeadGMU = new MultiHeadGMU(dimension, numCabezas);
    }

    /**
     * Ejecuta fusión híbrida combinando GMU global y Multi-Head
     */
    public fusionar(hTemporal: number[], hEspacial: number[]): ResultadoFusion {
        // Early Fusion: GMU global (captura interacciones globales)
        const resultadoEarly = this.gmuPrincipal.fusionar(hTemporal, hEspacial);

        // Late Fusion: Multi-Head GMU (preserva especificidad por región)
        const resultadoLate = this.multiHeadGMU.fusionar(hTemporal, hEspacial);

        // Combinar ambos resultados
        const vectorFusionado = resultadoEarly.vectorFusionado.map((v, i) =>
            v * this.pesoEarlyFusion + (resultadoLate.vectorFusionado[i] || 0) * this.pesoLateFusion
        );

        // Consolidar métricas
        const confianzaFusion = 
            resultadoEarly.confianzaFusion * this.pesoEarlyFusion +
            resultadoLate.confianzaFusion * this.pesoLateFusion;

        return {
            vectorFusionado,
            pesosCompuerta: resultadoEarly.pesosCompuerta, // Usamos el global para interpretabilidad
            modalidadDominante: resultadoEarly.modalidadDominante,
            confianzaFusion
        };
    }

    /**
     * Ajusta dinámicamente los pesos de fusión basado en el rendimiento
     */
    public ajustarPesos(errorEarly: number, errorLate: number) {
        const total = errorEarly + errorLate;
        if (total > 0) {
            // Dar más peso al que tiene menor error
            this.pesoEarlyFusion = 1 - (errorEarly / total);
            this.pesoLateFusion = 1 - (errorLate / total);
        }
    }
}
