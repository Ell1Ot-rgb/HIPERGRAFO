/**
 * CapaEspacioTemporal.ts
 * 
 * Implementa la Capa 2 de la Corteza Cognitiva.
 * Arquitectura Híbrida: Bi-LSTM (Temporal) + Transformer (Espacial) + GMU (Fusión).
 * 
 * Mejoras basadas en:
 * - "Estrategias de Fusión de Datos y Optimización de Modelos"
 * - Informe "Diseño y Optimización de Capa 2 Híbrida"
 * 
 * Responsabilidades:
 * 1. Gestión de Memoria a Corto Plazo (Sliding Window con Overlap).
 * 2. Inferencia Stateful (Mantenimiento de estados h/c de LSTM).
 * 3. Fusión GMU (Gated Multimodal Unit) adaptativa.
 * 4. Detección de anomalías con múltiples umbrales.
 * 5. Métricas de explicabilidad (qué modalidad domina).
 */

import { SalidaCapa1 } from './CapaSensorial';
import { FusionHibrida, ResultadoFusion } from './FusionMultimodal';

/**
 * Estado persistente para inferencia Stateful
 * En ONNX Runtime, estos se pasan como inputs/outputs explícitos
 */
export interface EstadoLSTM {
    h_forward: number[];   // Estado oculto dirección forward
    h_backward: number[];  // Estado oculto dirección backward  
    c_forward: number[];   // Estado celda dirección forward
    c_backward: number[];  // Estado celda dirección backward
    timestep: number;      // Contador de pasos de tiempo
}

/**
 * Resultado del procesamiento espacio-temporal
 */
export interface SalidaEspacioTemporal {
    vectorContextual: number[];           // Vector fusionado de 512D
    anomaliaDetectada: boolean;
    scoreAnomalia: number;                // Score continuo 0-1
    confianza: number;
    modalidadDominante: 'temporal' | 'espacial' | 'equilibrado';
    metricas: {
        magnitudTemporal: number;
        magnitudEspacial: number;
        ratioFusion: number;              // z promedio del GMU
        estabilidadSecuencia: number;     // Varianza en la ventana
    };
}

/**
 * Configuración de la Capa 2
 */
export interface ConfiguracionCapa2 {
    longitudVentana: number;           // Tamaño del buffer de secuencia
    solapamientoVentana: number;       // Overlap para Bi-LSTM (porcentaje)
    dimensionAtomo: number;            // Dimensión de salida de cada átomo Capa 1
    dimensionOculta: number;           // Hidden size de la LSTM
    dimensionModelo: number;           // d_model del Transformer
    numCabezasGMU: number;             // Cabezas para Multi-Head GMU
    umbralAnomaliaAlto: number;        // Umbral para anomalía grave
    umbralAnomaliaBajo: number;        // Umbral para anomalía leve
}

const CONFIG_DEFAULT: ConfiguracionCapa2 = {
    longitudVentana: 32,
    solapamientoVentana: 0.25,         // 25% overlap
    dimensionAtomo: 64,
    dimensionOculta: 256,              // Bi-LSTM: 256*2 = 512
    dimensionModelo: 512,
    numCabezasGMU: 8,
    umbralAnomaliaAlto: 0.8,
    umbralAnomaliaBajo: 0.5
};

/**
 * Capa 2: Procesamiento Espacio-Temporal Híbrido
 */
export class CapaEspacioTemporal {
    private config: ConfiguracionCapa2;
    
    // Buffer de secuencia con ventana deslizante
    private bufferSecuencia: number[][] = [];
    private bufferTimestamps: number[] = [];
    
    // Estado persistente para la rama LSTM (Stateful)
    private estadoLSTM: EstadoLSTM;
    
    // Módulo de fusión GMU
    private fusionGMU: FusionHibrida;
    
    // Historial para análisis de estabilidad
    private historialVectores: number[][] = [];
    private historialAnomalias: number[] = [];

    constructor(config: Partial<ConfiguracionCapa2> = {}) {
        this.config = { ...CONFIG_DEFAULT, ...config };
        this.estadoLSTM = this.inicializarEstadoLSTM();
        this.fusionGMU = new FusionHibrida(this.config.dimensionModelo, this.config.numCabezasGMU);
    }

    /**
     * Inicializa estados LSTM con ceros
     * En producción, esto debe coincidir exactamente con las dimensiones del modelo ONNX
     */
    private inicializarEstadoLSTM(): EstadoLSTM {
        const hidden = this.config.dimensionOculta;
        return {
            h_forward: new Array(hidden).fill(0),
            h_backward: new Array(hidden).fill(0),
            c_forward: new Array(hidden).fill(0),
            c_backward: new Array(hidden).fill(0),
            timestep: 0
        };
    }

    /**
     * Resetea el estado LSTM (útil al inicio de una nueva secuencia)
     */
    public resetearEstado(): void {
        this.estadoLSTM = this.inicializarEstadoLSTM();
        this.bufferSecuencia = [];
        this.historialVectores = [];
        console.log("🔄 Estado LSTM reseteado");
    }

    /**
     * Procesa la salida de la Capa Sensorial
     */
    public async procesar(entrada: SalidaCapa1): Promise<SalidaEspacioTemporal> {
        // 1. Aplanar entrada: 25 átomos × 64D = 1600D
        const vectorT = this.aplanarEntrada(entrada);
        
        // 2. Adaptar dimensionalidad: 1600D → 512D (Input Adapter)
        const vectorAdaptado = this.adaptarDimensionalidad(vectorT);

        // 3. Actualizar buffer de secuencia (Sliding Window)
        this.actualizarBuffer(vectorAdaptado);

        // 4. Simular rama Temporal (Bi-LSTM)
        const salidaTemporal = await this.procesarRamaTemporal(vectorAdaptado);

        // 5. Simular rama Espacial (Transformer sobre ventana)
        const salidaEspacial = await this.procesarRamaEspacial();

        // 6. Fusión GMU
        const resultadoFusion = this.fusionGMU.fusionar(salidaTemporal, salidaEspacial);

        // 7. Detección de anomalías multi-criterio
        const analisisAnomalia = this.analizarAnomalias(
            resultadoFusion.vectorFusionado,
            salidaTemporal,
            salidaEspacial,
            resultadoFusion
        );

        // 8. Actualizar estado temporal
        this.estadoLSTM.timestep++;

        return {
            vectorContextual: resultadoFusion.vectorFusionado,
            anomaliaDetectada: analisisAnomalia.detectada,
            scoreAnomalia: analisisAnomalia.score,
            confianza: resultadoFusion.confianzaFusion,
            modalidadDominante: resultadoFusion.modalidadDominante,
            metricas: {
                magnitudTemporal: this.calcularMagnitud(salidaTemporal),
                magnitudEspacial: this.calcularMagnitud(salidaEspacial),
                ratioFusion: this.calcularPromedio(resultadoFusion.pesosCompuerta),
                estabilidadSecuencia: this.calcularEstabilidadSecuencia()
            }
        };
    }

    /**
     * Aplana la entrada de 25 subespacios a un vector único
     */
    private aplanarEntrada(entrada: SalidaCapa1): number[] {
        let vector: number[] = [];
        for (let i = 1; i <= 25; i++) {
            const id = `S${i}`;
            const atomo = entrada[id] || new Array(this.config.dimensionAtomo).fill(0);
            vector = vector.concat(atomo);
        }
        return vector;
    }

    /**
     * Adapta dimensionalidad: Proyección 1600D → 512D
     * Simula el "Input Adapter" del reporte: Linear + LayerNorm + GELU
     */
    private adaptarDimensionalidad(vector: number[]): number[] {
        const dimSalida = this.config.dimensionModelo;
        const dimEntrada = vector.length;
        const ratio = Math.floor(dimEntrada / dimSalida);
        
        // Pooling por grupos + no-linealidad
        const adaptado = new Array(dimSalida);
        for (let i = 0; i < dimSalida; i++) {
            const inicio = i * ratio;
            const fin = Math.min(inicio + ratio, dimEntrada);
            let suma = 0;
            for (let j = inicio; j < fin; j++) {
                suma += vector[j];
            }
            // GELU aproximado: x * sigmoid(1.702 * x)
            const promedio = suma / (fin - inicio);
            adaptado[i] = this.gelu(promedio);
        }
        
        // Layer Normalization
        return this.layerNorm(adaptado);
    }

    private gelu(x: number): number {
        return x * (1 / (1 + Math.exp(-1.702 * x)));
    }

    private layerNorm(x: number[], epsilon: number = 1e-5): number[] {
        const media = x.reduce((a, b) => a + b, 0) / x.length;
        const varianza = x.reduce((a, b) => a + Math.pow(b - media, 2), 0) / x.length;
        const std = Math.sqrt(varianza + epsilon);
        return x.map(v => (v - media) / std);
    }

    /**
     * Actualiza buffer con ventana deslizante
     */
    private actualizarBuffer(vector: number[]): void {
        this.bufferSecuencia.push([...vector]);
        this.bufferTimestamps.push(Date.now());
        
        if (this.bufferSecuencia.length > this.config.longitudVentana) {
            this.bufferSecuencia.shift();
            this.bufferTimestamps.shift();
        }

        // Mantener historial para análisis
        this.historialVectores.push([...vector]);
        if (this.historialVectores.length > 100) {
            this.historialVectores.shift();
        }
    }

    /**
     * Procesa rama temporal (simula Bi-LSTM)
     * En producción: ONNX session.run con h_in, c_in → h_out, c_out
     */
    private async procesarRamaTemporal(vectorActual: number[]): Promise<number[]> {
        // Simulación: Combina el vector actual con el estado previo
        // En ONNX real: output, h_new, c_new = lstm(input, h_old, c_old)
        
        const hidden = this.config.dimensionOculta;
        const salida = new Array(hidden * 2).fill(0); // Bi = forward + backward
        
        // Forward: depende del estado previo y entrada actual
        for (let i = 0; i < hidden; i++) {
            const inputContrib = vectorActual[i % vectorActual.length] || 0;
            const stateContrib = this.estadoLSTM.h_forward[i] * 0.9; // Decaimiento
            salida[i] = Math.tanh(inputContrib * 0.1 + stateContrib);
            
            // Actualizar estado (simulación simplificada)
            this.estadoLSTM.h_forward[i] = salida[i];
        }
        
        // Backward: en streaming real, esto usa contexto futuro del buffer
        if (this.bufferSecuencia.length > 1) {
            const ultimoBuffer = this.bufferSecuencia[this.bufferSecuencia.length - 1];
            for (let i = 0; i < hidden; i++) {
                const inputContrib = ultimoBuffer[i % ultimoBuffer.length] || 0;
                const stateContrib = this.estadoLSTM.h_backward[i] * 0.9;
                salida[hidden + i] = Math.tanh(inputContrib * 0.1 + stateContrib);
                this.estadoLSTM.h_backward[i] = salida[hidden + i];
            }
        }

        return salida;
    }

    /**
     * Procesa rama espacial (simula Transformer Self-Attention)
     * Opera sobre toda la ventana para capturar relaciones globales
     */
    private async procesarRamaEspacial(): Promise<number[]> {
        if (this.bufferSecuencia.length === 0) {
            return new Array(this.config.dimensionModelo).fill(0);
        }

        // Simulación de Self-Attention: promedio ponderado por similitud
        const n = this.bufferSecuencia.length;
        const dim = this.config.dimensionModelo;
        
        // Query: último elemento
        const query = this.bufferSecuencia[n - 1];
        
        // Keys/Values: toda la secuencia
        const attentionWeights = new Array(n).fill(0);
        
        // Calcular scores de atención (dot product simplificado)
        for (let i = 0; i < n; i++) {
            const key = this.bufferSecuencia[i];
            let dotProduct = 0;
            for (let j = 0; j < Math.min(dim, key.length, query.length); j++) {
                dotProduct += query[j] * key[j];
            }
            attentionWeights[i] = dotProduct / Math.sqrt(dim);
        }
        
        // Softmax
        const maxScore = Math.max(...attentionWeights);
        const expScores = attentionWeights.map(s => Math.exp(s - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const normalizedWeights = expScores.map(s => s / sumExp);
        
        // Weighted sum de values
        const output = new Array(dim).fill(0);
        for (let i = 0; i < n; i++) {
            const value = this.bufferSecuencia[i];
            for (let j = 0; j < dim; j++) {
                output[j] += normalizedWeights[i] * (value[j] || 0);
            }
        }

        return output;
    }

    /**
     * Análisis de anomalías multi-criterio
     */
    private analizarAnomalias(
        vectorFusionado: number[],
        temporal: number[],
        espacial: number[],
        fusion: ResultadoFusion
    ): { detectada: boolean; score: number } {
        const scores: number[] = [];

        // Criterio 1: Magnitud del vector fusionado
        const magnitud = this.calcularMagnitud(vectorFusionado);
        const umbralMagnitud = 50; // Calibrar empíricamente
        scores.push(Math.min(magnitud / umbralMagnitud, 1));

        // Criterio 2: Divergencia temporal-espacial
        const divergencia = this.calcularDivergencia(temporal, espacial);
        scores.push(Math.min(divergencia / 10, 1));

        // Criterio 3: Inestabilidad de la secuencia
        const inestabilidad = 1 - this.calcularEstabilidadSecuencia();
        scores.push(inestabilidad);

        // Criterio 4: Baja confianza en fusión
        scores.push(1 - fusion.confianzaFusion);

        // Score final: promedio ponderado
        const pesos = [0.3, 0.25, 0.25, 0.2];
        const scoreFinal = scores.reduce((acc, s, i) => acc + s * pesos[i], 0);

        // Actualizar historial
        this.historialAnomalias.push(scoreFinal);
        if (this.historialAnomalias.length > 50) {
            this.historialAnomalias.shift();
        }

        return {
            detectada: scoreFinal > this.config.umbralAnomaliaBajo,
            score: scoreFinal
        };
    }

    private calcularMagnitud(vector: number[]): number {
        return Math.sqrt(vector.reduce((acc, v) => acc + v * v, 0));
    }

    private calcularDivergencia(a: number[], b: number[]): number {
        const minLen = Math.min(a.length, b.length);
        let suma = 0;
        for (let i = 0; i < minLen; i++) {
            suma += Math.abs(a[i] - b[i]);
        }
        return suma / minLen;
    }

    private calcularEstabilidadSecuencia(): number {
        if (this.historialVectores.length < 2) return 1;
        
        const n = Math.min(10, this.historialVectores.length);
        const recientes = this.historialVectores.slice(-n);
        
        // Calcular varianza de magnitudes
        const magnitudes = recientes.map(v => this.calcularMagnitud(v));
        const mediaMag = magnitudes.reduce((a, b) => a + b, 0) / n;
        const varianza = magnitudes.reduce((a, m) => a + Math.pow(m - mediaMag, 2), 0) / n;
        
        // Normalizar: alta varianza = baja estabilidad
        return Math.max(0, 1 - varianza / 100);
    }

    private calcularPromedio(arr: number[]): number {
        return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    }

    /**
     * Obtiene el estado actual del LSTM (para persistencia o debugging)
     */
    public obtenerEstado(): EstadoLSTM {
        return { ...this.estadoLSTM };
    }

    /**
     * Carga un estado previo (para reanudar procesamiento)
     */
    public cargarEstado(estado: EstadoLSTM): void {
        this.estadoLSTM = { ...estado };
    }

    /**
     * Estadísticas para monitoreo
     */
    public obtenerEstadisticas(): {
        timestep: number;
        tamanoBuffer: number;
        promedioAnomalia: number;
        estadoCompuerta: ReturnType<FusionHibrida['fusionar']>['modalidadDominante'];
    } {
        return {
            timestep: this.estadoLSTM.timestep,
            tamanoBuffer: this.bufferSecuencia.length,
            promedioAnomalia: this.historialAnomalias.length > 0
                ? this.historialAnomalias.reduce((a, b) => a + b, 0) / this.historialAnomalias.length
                : 0,
            estadoCompuerta: 'equilibrado' // Se actualizaría con la última fusión
        };
    }
}
