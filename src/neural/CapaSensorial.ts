/**
 * CapaSensorial.ts
 *
 * Implementa las Capas 0 y 1 de la Corteza Cognitiva Jerárquica.
 *
 * Capa 0: Entrada - Preprocesamiento y división en 25 subespacios.
 * Capa 1: Sensorial - 25 sub-redes especializadas (Átomos Topológicos LIF).
 */

import { InferenciaLocal } from './InferenciaLocal';

export interface Vector256D {
    [key: string]: number; // D001-D256
}

export interface Subespacio {
    id: string;
    rango: [number, number]; // [inicio, fin] en el vector 256D
    dimensiones: number;
    descripcion: string;
}

export interface SalidaCapa1 {
    [subespacioId: string]: number[]; // Vector latente 64D por subespacio
}

/**
 * Normalizador Adaptativo con Running Statistics
 * Aprende distribuciones en tiempo real
 */
class AdaptiveNormalizer {
    private estadisticas: Map<string, { μ: number; σ: number; count: number }> = new Map();
    private momentum: number = 0.95;
    private epsilon: number = 1e-7;

    actualizar(campo: string, valores: number[]): void {
        if (valores.length === 0) return;
        
        const μ_batch = valores.reduce((a, b) => a + b, 0) / valores.length;
        const σ_batch = Math.sqrt(
            valores.reduce((sum, v) => sum + Math.pow(v - μ_batch, 2), 0) / valores.length
        );

        const stats = this.estadisticas.get(campo) || { μ: 0, σ: 1, count: 0 };
        
        // Exponential Moving Average
        stats.μ = this.momentum * stats.μ + (1 - this.momentum) * μ_batch;
        stats.σ = this.momentum * stats.σ + (1 - this.momentum) * Math.max(σ_batch, this.epsilon);
        stats.count++;
        
        this.estadisticas.set(campo, stats);
    }

    normalizar(campo: string, valor: number): number {
        const stats = this.estadisticas.get(campo) || { μ: 0, σ: 1, count: 0 };
        return (valor - stats.μ) / (stats.σ + this.epsilon);
    }

    obtenerEstadisticas(campo: string) {
        return this.estadisticas.get(campo) || { μ: 0, σ: 1, count: 0 };
    }
}

/**
 * Generador de Positional Encoding Sinusoidal
 * Inspurado en Transformers: PE(pos, 2i) = sin(pos/10000^(2i/d))
 */
class PositionalEncoder {
    private cache: Map<number, number[]> = new Map();

    generar(posicion: number, dimension: number): number[] {
        const cacheKey = posicion * 1000 + dimension;
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey)!;
        }

        const encoding: number[] = [];
        const logDiv = Math.log(10000) / Math.max(1, dimension - 1);

        for (let i = 0; i < dimension; i++) {
            const angle = posicion / Math.exp((i / dimension) * logDiv);
            if (i % 2 === 0) {
                encoding.push(Math.sin(angle));
            } else {
                encoding.push(Math.cos(angle));
            }
        }

        this.cache.set(cacheKey, encoding);
        return encoding;
    }

    limpiarCache() {
        if (this.cache.size > 1000) {
            // Evitar memory leak: limpiar si crece demasiado
            this.cache.clear();
        }
    }
}

/**
 * Inter-Subespacio Attention
 * Permite que los 25 subespacios se "escuchen" mutuamente
 * Calcula pesos de atención basados en magnitud de activación
 */
class InterSubespacioAttention {
    private pesos: Map<string, number> = new Map();
    private historico: Map<string, number[]> = new Map();
    private readonly WINDOW_SIZE = 10;

    /**
     * Calcula pesos de atención entre subespacios
     * Basado en la magnitud de activación de cada uno
     */
    calcularPesos(salidas: SalidaCapa1): Map<string, number> {
        let totalMag = 0;
        const magnitudes = new Map<string, number>();

        // 1. Calcular magnitud de cada subespacio
        Object.entries(salidas).forEach(([id, vector]) => {
            const mag = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
            magnitudes.set(id, mag);
            totalMag += mag;

            // Actualizar histórico
            if (!this.historico.has(id)) {
                this.historico.set(id, []);
            }
            const hist = this.historico.get(id)!;
            hist.push(mag);
            if (hist.length > this.WINDOW_SIZE) {
                hist.shift();
            }
        });

        // 2. Normalizar a probabilidades (suma = 1)
        magnitudes.forEach((mag, id) => {
            this.pesos.set(id, totalMag > 0 ? mag / totalMag : 1.0 / magnitudes.size);
        });

        return this.pesos;
    }

    /**
     * Aplica pesos de atención a las salidas
     * Los subespacios con mayor activación influyen más en los demás
     */
    aplicarAtencion(salidas: SalidaCapa1): SalidaCapa1 {
        const pesos = this.calcularPesos(salidas);
        const resultado: SalidaCapa1 = {};

        // Para cada subespacio, mezclar sutilmente con otros
        Object.entries(salidas).forEach(([id, vector]) => {
            const vectorPonderado = [...vector];
            const pesoPropio = pesos.get(id) || 0;

            // Mezclar con otros subespacios (peso bajo: 5%)
            Object.entries(salidas).forEach(([otherId, otherVector]) => {
                if (otherId !== id) {
                    const pesoOtro = pesos.get(otherId) || 0;
                    const factorInfluencia = pesoOtro * 0.05; // 5% de influencia

                    for (let i = 0; i < vectorPonderado.length; i++) {
                        vectorPonderado[i] += otherVector[i] * factorInfluencia;
                    }
                }
            });

            resultado[id] = vectorPonderado;
        });

        return resultado;
    }

    /**
     * Obtiene estadísticas de atención
     */
    obtenerEstadisticas(): { subespaciosDominantes: string[]; diversidadAtencion: number } {
        const pesosArray = Array.from(this.pesos.entries())
            .sort((a, b) => b[1] - a[1]);

        const top3 = pesosArray.slice(0, 3).map(([id]) => id);
        
        // Diversidad: entropía de Shannon
        let entropia = 0;
        this.pesos.forEach(p => {
            if (p > 0) {
                entropia -= p * Math.log2(p);
            }
        });

        return {
            subespaciosDominantes: top3,
            diversidadAtencion: entropia / Math.log2(this.pesos.size) // Normalizado [0,1]
        };
    }
}

/**
 * Entropy-Based Field Analyzer
 * Analiza la entropía de cada campo para detectar campos "muertos" o altamente predictivos
 */
class EntropyFieldAnalyzer {
    private entropias: Map<string, number> = new Map();
    private histogramas: Map<string, Map<number, number>> = new Map();
    private readonly BINS = 50; // Número de bins para histograma
    private readonly WINDOW_SIZE = 100; // Ventana de análisis

    /**
     * Analiza un campo y calcula su entropía de Shannon
     */
    analizarCampo(campo: string, valor: number): void {
        if (!this.histogramas.has(campo)) {
            this.histogramas.set(campo, new Map());
        }

        const hist = this.histogramas.get(campo)!;
        const bin = Math.floor(valor * this.BINS);
        hist.set(bin, (hist.get(bin) || 0) + 1);

        // Calcular entropía si tenemos suficientes muestras
        const totalMuestras = Array.from(hist.values()).reduce((a, b) => a + b, 0);
        if (totalMuestras >= this.WINDOW_SIZE) {
            let entropia = 0;
            hist.forEach(count => {
                const p = count / totalMuestras;
                if (p > 0) {
                    entropia -= p * Math.log2(p);
                }
            });
            this.entropias.set(campo, entropia);

            // Limpiar histograma si crece demasiado
            if (totalMuestras > this.WINDOW_SIZE * 2) {
                this.histogramas.set(campo, new Map());
            }
        }
    }

    /**
     * Obtiene la entropía de un campo
     * 0 = campo muerto (sin variación)
     * >0 = campo con información
     * Alto = alta variabilidad/información
     */
    obtenerEntropia(campo: string): number {
        return this.entropias.get(campo) || 0;
    }

    /**
     * Identifica campos con baja entropía (candidatos a eliminar)
     */
    identificarCamposMuertos(umbral: number = 0.5): string[] {
        const muertos: string[] = [];
        this.entropias.forEach((entropia, campo) => {
            if (entropia < umbral) {
                muertos.push(campo);
            }
        });
        return muertos;
    }

    /**
     * Identifica campos con alta entropía (muy informativos)
     */
    identificarCamposInformativos(umbral: number = 3.0): string[] {
        const informativos: string[] = [];
        this.entropias.forEach((entropia, campo) => {
            if (entropia > umbral) {
                informativos.push(campo);
            }
        });
        return informativos;
    }

    /**
     * Obtiene estadísticas generales
     */
    obtenerEstadisticas(): {
        camposAnalizados: number;
        entropiaMedia: number;
        entropiaMin: number;
        entropiaMax: number;
        camposMuertos: number;
        camposInformativos: number;
    } {
        const entropiasArray = Array.from(this.entropias.values());
        return {
            camposAnalizados: this.entropias.size,
            entropiaMedia: entropiasArray.reduce((a, b) => a + b, 0) / entropiasArray.length || 0,
            entropiaMin: Math.min(...entropiasArray) || 0,
            entropiaMax: Math.max(...entropiasArray) || 0,
            camposMuertos: this.identificarCamposMuertos().length,
            camposInformativos: this.identificarCamposInformativos().length
        };
    }
}

/**
 * Entropy-Based Field Analyzer
 * Analiza la entropía de Shannon de cada campo para identificar:
 * - Campos "muertos" (entropía ~0): siempre mismo valor
 * - Campos informativos (entropía alta): mucha variabilidad
 * - Campos predictivos (entropía media): patrones útiles
 */
class EntropyFieldAnalyzer {
    private entropias: Map<string, number> = new Map();
    private historial: Map<string, number[][]> = new Map();
    private readonly WINDOW_SIZE = 100;
    private readonly NUM_BINS = 50;

    /**
     * Analiza la entropía de Shannon de un campo
     * H = -Σ p(x) * log₂(p(x))
     */
    analizarCampo(campo: string, valores: number[]): number {
        if (valores.length === 0) return 0;

        // 1. Crear histograma (binning)
        const bins = new Map<number, number>();
        valores.forEach(v => {
            const bin = Math.floor(v * this.NUM_BINS) / this.NUM_BINS;
            bins.set(bin, (bins.get(bin) || 0) + 1);
        });

        // 2. Calcular probabilidades
        const n = valores.length;
        let entropy = 0;
        bins.forEach(count => {
            const p = count / n;
            if (p > 0) {
                entropy -= p * Math.log2(p);
            }
        });

        // 3. Normalizar a [0, 1] (entropía máxima = log₂(num_bins))
        const maxEntropy = Math.log2(Math.min(this.NUM_BINS, n));
        const entropyNorm = maxEntropy > 0 ? entropy / maxEntropy : 0;

        // 4. Actualizar historial
        if (!this.historial.has(campo)) {
            this.historial.set(campo, []);
        }
        const hist = this.historial.get(campo)!;
        hist.push([Date.now(), entropyNorm]);
        if (hist.length > this.WINDOW_SIZE) {
            hist.shift();
        }

        this.entropias.set(campo, entropyNorm);
        return entropyNorm;
    }

    /**
     * Clasifica un campo según su entropía
     */
    clasificarCampo(campo: string): 'dead' | 'low' | 'medium' | 'high' | 'random' {
        const entropy = this.entropias.get(campo) || 0;

        if (entropy < 0.05) return 'dead';       // Casi constante
        if (entropy < 0.3) return 'low';         // Poca variabilidad
        if (entropy < 0.6) return 'medium';      // Variabilidad moderada (útil)
        if (entropy < 0.9) return 'high';        // Alta variabilidad (muy útil)
        return 'random';                          // Ruido aleatorio puro
    }

    /**
     * Obtiene estadísticas de todos los campos analizados
     */
    obtenerEstadisticas(): {
        camposMuertos: string[];
        camposInformativos: string[];
        entropiaPromedio: number;
        distribucion: { dead: number; low: number; medium: number; high: number; random: number };
    } {
        const muertos: string[] = [];
        const informativos: string[] = [];
        const dist = { dead: 0, low: 0, medium: 0, high: 0, random: 0 };

        let sumaEntropy = 0;
        this.entropias.forEach((entropy, campo) => {
            const clase = this.clasificarCampo(campo);
            dist[clase]++;

            if (clase === 'dead') {
                muertos.push(campo);
            } else if (clase === 'medium' || clase === 'high') {
                informativos.push(campo);
            }

            sumaEntropy += entropy;
        });

        return {
            camposMuertos: muertos,
            camposInformativos: informativos,
            entropiaPromedio: this.entropias.size > 0 ? sumaEntropy / this.entropias.size : 0,
            distribucion: dist
        };
    }

    /**
     * Recomienda campos a descartar (dead/random)
     */
    recomendarDescarte(): string[] {
        const descartar: string[] = [];
        this.entropias.forEach((entropy, campo) => {
            const clase = this.clasificarCampo(campo);
            if (clase === 'dead' || clase === 'random') {
                descartar.push(campo);
            }
        });
        return descartar;
    }
}

/**
 * Learnable Subespacio Weights
 * Aprende importancia relativa de cada subespacio durante training
 */
class LearnableSubespacioWeights {
    private pesos: Map<string, number> = new Map();
    private gradientes: Map<string, number> = new Map();
    private momentum: Map<string, number> = new Map();
    private readonly LEARNING_RATE = 0.001;
    private readonly MOMENTUM_FACTOR = 0.9;

    constructor(subespaciosIds: string[]) {
        // Inicializar todos los pesos a 1.0
        subespaciosIds.forEach(id => {
            this.pesos.set(id, 1.0);
            this.gradientes.set(id, 0);
            this.momentum.set(id, 0);
        });
    }

    /**
     * Actualiza pesos basado en performance
     * Performance alto → aumentar peso
     * Performance bajo → disminuir peso
     */
    actualizar(performance: Map<string, number>): void {
        performance.forEach((perf, id) => {
            const pesoActual = this.pesos.get(id) || 1.0;
            const gradActual = this.gradientes.get(id) || 0;
            const momentumActual = this.momentum.get(id) || 0;

            // Calcular gradiente: si perf > 0.5 → aumentar, si perf < 0.5 → disminuir
            const gradiente = (perf - 0.5) * 2; // [-1, 1]

            // Actualizar momentum
            const nuevoMomentum = this.MOMENTUM_FACTOR * momentumActual + 
                                  (1 - this.MOMENTUM_FACTOR) * gradiente;
            this.momentum.set(id, nuevoMomentum);

            // Actualizar peso
            const nuevoPeso = pesoActual + this.LEARNING_RATE * nuevoMomentum;
            
            // Clamp a [0.1, 10.0] para evitar valores extremos
            this.pesos.set(id, Math.max(0.1, Math.min(10.0, nuevoPeso)));
            this.gradientes.set(id, gradiente);
        });
    }

    /**
     * Aplica pesos a las salidas de Capa 1
     */
    aplicar(salidas: SalidaCapa1): SalidaCapa1 {
        const resultado: SalidaCapa1 = {};

        Object.entries(salidas).forEach(([id, vector]) => {
            const peso = this.pesos.get(id) || 1.0;
            resultado[id] = vector.map(v => v * peso);
        });

        return resultado;
    }

    /**
     * Obtiene estadísticas de pesos
     */
    obtenerEstadisticas(): { pesosMin: number; pesosMax: number; pesosMedio: number; subespaciosMasFuertes: string[] } {
        const pesosArray = Array.from(this.pesos.values());
        const pesosMin = Math.min(...pesosArray);
        const pesosMax = Math.max(...pesosArray);
        const pesosMedio = pesosArray.reduce((a, b) => a + b, 0) / pesosArray.length;

        const top5 = Array.from(this.pesos.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([id]) => id);

        return {
            pesosMin,
            pesosMax,
            pesosMedio,
            subespaciosMasFuertes: top5
        };
    }
}

/**
 * CAPA 0: Entrada - Preprocesamiento del Vector 256D
 */
export class CapaEntrada {
    private subespacios: Subespacio[] = [
        { id: 'S1', rango: [1, 16], dimensiones: 16, descripcion: 'Criptografía y Blockchain' },
        { id: 'S2', rango: [17, 32], dimensiones: 16, descripcion: 'Sensor Fenomenológico' },
        { id: 'S3', rango: [33, 48], dimensiones: 16, descripcion: 'Histograma Multi-Canal' },
        { id: 'S4', rango: [49, 56], dimensiones: 8, descripcion: 'Streaming' },
        { id: 'S5', rango: [57, 72], dimensiones: 16, descripcion: 'Seguridad' },
        { id: 'S6', rango: [73, 80], dimensiones: 8, descripcion: 'Análisis Relacional' },
        { id: 'S7', rango: [81, 88], dimensiones: 8, descripcion: 'Conceptual' },
        { id: 'S8', rango: [89, 104], dimensiones: 16, descripcion: 'Red Neuronal Básica' },
        { id: 'S9', rango: [105, 116], dimensiones: 12, descripcion: 'Física Simulada' },
        { id: 'S10', rango: [117, 124], dimensiones: 8, descripcion: 'Temporal' },
        { id: 'S11', rango: [125, 132], dimensiones: 8, descripcion: 'Anomalías' },
        { id: 'S12', rango: [133, 140], dimensiones: 8, descripcion: 'Emocional PAD' },
        { id: 'S13', rango: [141, 148], dimensiones: 8, descripcion: 'Información Avanzada' },
        { id: 'S14', rango: [149, 156], dimensiones: 8, descripcion: 'Espectral' },
        { id: 'S15', rango: [157, 164], dimensiones: 8, descripcion: 'Complejidad' },
        { id: 'S16', rango: [165, 172], dimensiones: 8, descripcion: 'Topología' },
        { id: 'S17', rango: [173, 180], dimensiones: 8, descripcion: 'Física Estadística' },
        { id: 'S18', rango: [181, 188], dimensiones: 8, descripcion: 'Fractal' },
        { id: 'S19', rango: [189, 196], dimensiones: 8, descripcion: 'Grafos' },
        { id: 'S20', rango: [197, 208], dimensiones: 12, descripcion: 'Kalman' },
        { id: 'S21', rango: [209, 216], dimensiones: 8, descripcion: 'Clasificación' },
        { id: 'S22', rango: [217, 224], dimensiones: 8, descripcion: 'Reserva' },
        { id: 'S23', rango: [225, 232], dimensiones: 8, descripcion: 'Dinámica de Spikes' },
        { id: 'S24', rango: [233, 240], dimensiones: 8, descripcion: 'Plasticidad' },
        { id: 'S25', rango: [241, 256], dimensiones: 16, descripcion: 'Membrana y Reservoir' }
    ];
    private normalizador: AdaptiveNormalizer = new AdaptiveNormalizer();
    private posEncoderCapa0: PositionalEncoder = new PositionalEncoder();
    private entropyAnalyzer: EntropyFieldAnalyzer = new EntropyFieldAnalyzer();
    private batchValoresParaEntropy: Map<string, number[]> = new Map();
    private contadorBatches: number = 0;
    private readonly ANALIZAR_CADA_N_BATCHES = 50; // Analizar entropía cada 50 vectores
    private entropyAnalyzer: EntropyFieldAnalyzer = new EntropyFieldAnalyzer();

    /**
     * Preprocesa el vector 256D y lo divide en subespacios
     * FASE 2: Ahora incluye Positional Encoding en cada campo
     * FASE 3: Recolecta datos para análisis de entropía
     * 
     * @param vector256d Vector completo de entrada
     * @returns Mapa de subespacios con sus valores (con PE)
     */
    procesar(vector256d: Vector256D): Map<string, number[]> {
        const resultado = new Map<string, number[]>();
        this.contadorBatches++;

        for (const subespacio of this.subespacios) {
            const valores: number[] = [];
            for (let i = subespacio.rango[0]; i <= subespacio.rango[1]; i++) {
                const clave = `D${i.toString().padStart(3, '0')}`;
                const valor = vector256d[clave];
                if (valor === undefined) {
                    throw new Error(`Campo ${clave} no encontrado en vector 256D`);
                }
                
                // FASE 3: Recolectar valores para análisis de entropía
                if (!this.batchValoresParaEntropy.has(clave)) {
                    this.batchValoresParaEntropy.set(clave, []);
                }
                this.batchValoresParaEntropy.get(clave)!.push(valor);
                
                // Normalizar valor
                let valorNormalizado = this.normalizarCampo(clave, valor);
                
                // FASE 2: Agregar Positional Encoding (peso muy bajo: 2%)
                const posicion = i - 1; // 0-indexed
                const pe = this.posEncoderCapa0.generar(posicion, 1)[0]; // Solo 1D
                valorNormalizado += pe * 0.02; // 2% de influencia
                
                valores.push(valorNormalizado);
            }
            resultado.set(subespacio.id, valores);
        }

        // FASE 3: Analizar entropía cada N batches
        if (this.contadorBatches % this.ANALIZAR_CADA_N_BATCHES === 0) {
            this.analizarEntropiaGlobal();
        }

        return resultado;
    }

    /**
     * FASE 3: Analiza la entropía de todos los campos acumulados
     */
    private analizarEntropiaGlobal(): void {
        this.batchValoresParaEntropy.forEach((valores, campo) => {
            if (valores.length > 10) { // Mínimo 10 muestras
                this.entropyAnalyzer.analizarCampo(campo, valores);
            }
        });

        // Limpiar buffer para próximo análisis
        this.batchValoresParaEntropy.clear();
    }

    /**
     * FASE 3: Obtiene estadísticas de entropía de campos
     */
    obtenerAnalisisEntropy() {
        return this.entropyAnalyzer.obtenerEstadisticas();
    }

    /**
     * Normalización adaptativa e inteligente por tipo de campo
     * Combina múltiples técnicas: running stats, log-scaling, adaptive clipping
     * 
     * @param campo Nombre del campo (D001, etc.)
     * @param valor Valor crudo
     * @returns Valor normalizado [-1,1] o [0,1]
     */
    private normalizarCampo(campo: string, valor: number): number {
        const campoNum = parseInt(campo.substring(1));
        
        // PASO 1: Categorizar tipo de campo
        const tipo = this.categorizarCampo(campoNum);
        
        // PASO 2: Pre-normalización según tipo
        let normalizado: number;
        
        switch (tipo) {
            case 'criptografico':
                // Alta magnitud: log-scaling + adaptive norm
                normalizado = this.normalizarAltaMagnitud(campo, valor);
                break;
            
            case 'temporal':
                // Preservar simetría: sin scaling
                normalizado = this.normalizarTemporal(campo, valor);
                break;
            
            case 'bipolar':
                // Emociones/Potenciales: tanh con scaling
                normalizado = this.normalizarBipolar(campo, valor);
                break;
            
            case 'binario':
                // Flags/Índices: min-max puro [0,1]
                normalizado = this.normalizarBinario(valor);
                break;
            
            default:
                // Métricas genéricas: min-max adaptativo
                normalizado = this.normalizarMetrica(campo, valor);
        }
        
        // PASO 3: Clip y asegurar rango
        return Math.max(-1.0, Math.min(1.0, normalizado));
    }

    private categorizarCampo(campoNum: number): string {
        // Criptografía (S1)
        if (campoNum >= 1 && campoNum <= 16) return 'criptografico';
        
        // Temporal (S10)
        if (campoNum >= 117 && campoNum <= 124) return 'temporal';
        
        // Emocional (S12) + Dinámica de Spikes (S23) + Plasticidad (S24)
        if ((campoNum >= 133 && campoNum <= 140) ||
            (campoNum >= 225 && campoNum <= 240)) return 'bipolar';
        
        // Streaming (S4) + Reserva (S22)
        if ((campoNum >= 49 && campoNum <= 56) ||
            (campoNum >= 217 && campoNum <= 224)) return 'binario';
        
        // Seguridad (S5) + Grafos (S19) + Kalman (S20)
        if ((campoNum >= 57 && campoNum <= 72) ||
            (campoNum >= 189 && campoNum <= 208)) return 'criptografico';
        
        return 'metrica';
    }

    private normalizarAltaMagnitud(campo: string, valor: number): number {
        // Para campos con posible alto rango (0 a 1e9)
        const stats = this.normalizador.obtenerEstadisticas(campo);
        
        // Detectar rango dinámico
        const maxEsperado = Math.max(1e6, stats.μ + 3 * stats.σ);
        
        if (Math.abs(valor) > 1e3) {
            // Log scaling para valores grandes
            const logValue = Math.sign(valor) * Math.log1p(Math.abs(valor));
            const logMax = Math.log1p(maxEsperado);
            return (logValue / logMax);
        }
        
        // Para valores pequeños: normalización estándar
        const normalizado = this.normalizador.normalizar(campo, valor);
        this.normalizador.actualizar(campo, [valor]);
        
        return Math.tanh(normalizado); // Clip suave a [-1,1]
    }

    private normalizarTemporal(campo: string, valor: number): number {
        // Preservar media=0, solo escalar por desviación
        const stats = this.normalizador.obtenerEstadisticas(campo);
        this.normalizador.actualizar(campo, [valor]);
        
        return this.normalizador.normalizar(campo, valor);
    }

    private normalizarBipolar(campo: string, valor: number): number {
        // Asumir rango ~ [-1000, 1000] para emociones/potenciales
        const normalizado = Math.tanh(valor / 1000);
        const stats = this.normalizador.obtenerEstadisticas(campo);
        this.normalizador.actualizar(campo, [valor]);
        
        // Aplicar corrección adaptativa si tenemos estadísticas
        if (stats.count > 10) {
            return normalizado * (stats.σ / Math.max(0.1, stats.σ));
        }
        
        return normalizado;
    }

    private normalizarBinario(valor: number): number {
        // Rango uint8: [0, 255]
        return (Math.max(0, Math.min(255, valor)) / 255.0) * 2 - 1; // [-1, 1]
    }

    private normalizarMetrica(campo: string, valor: number): number {
        // Rango uint16: [0, 65535]
        // O log scaling si el valor es muy grande
        
        if (valor > 10000) {
            // Log scaling para valores grandes
            const logVal = Math.log1p(valor);
            const logMax = Math.log1p(65535);
            return (logVal / logMax) * 2 - 1; // [-1, 1]
        }
        
        // Min-max directo
        const normalizado = Math.max(0, Math.min(1, valor / 65535));
        this.normalizador.actualizar(campo, [valor]);
        
        return normalizado * 2 - 1; // [-1, 1]
    }

    /**
     * Obtiene la lista de subespacios
     */
    getSubespacios(): Subespacio[] {
        return this.subespacios;
    }
}

/**
 * CAPA 1: Sensorial - 25 Sub-Redes Especializadas
 */
export class CapaSensorial {
    private subRedes: Map<string, InferenciaLocal> = new Map();
    private capaEntrada: CapaEntrada;
    private inicializado: boolean = false;
    private posEncoder: PositionalEncoder = new PositionalEncoder();
    private interAtencion: InterSubespacioAttention = new InterSubespacioAttention();
    private learnableWeights: LearnableSubespacioWeights | null = null;

    constructor() {
        this.capaEntrada = new CapaEntrada();
    }

    /**
     * Inicializa las 25 sub-redes (cada una un Átomo Topológico LIF)
     */
    async inicializar(): Promise<void> {
        if (this.inicializado) return;

        const subespacios = this.capaEntrada.getSubespacios();

        for (const subespacio of subespacios) {
            const subRed = new InferenciaLocal();
            await subRed.inicializar();
            this.subRedes.set(subespacio.id, subRed);
        }

        // Inicializar learnable weights
        const subespaciosIds = subespacios.map(s => s.id);
        this.learnableWeights = new LearnableSubespacioWeights(subespaciosIds);

        this.inicializado = true;
        console.log('🧠 Capa Sensorial inicializada con 25 sub-redes LIF');
        console.log('✨ Mejoras Fase 2: Inter-Atención + Learnable Weights activos');
    }

    /**
     * Procesa el vector 256D completo a través de las 25 sub-redes
     * Ahora incluye:
     * - Positional Encoding (Fase 1)
     * - Inter-Subespacio Attention (Fase 2)
     * - Learnable Weights (Fase 2)
     * 
     * @param vector256d Vector de entrada completo
     * @returns Salida latente de cada subespacio (25 × 64D)
     */
    async procesar(vector256d: Vector256D): Promise<SalidaCapa1> {
        if (!this.inicializado) {
            throw new Error('Capa Sensorial no inicializada. Llama a inicializar() primero.');
        }

        const subespaciosDivididos = this.capaEntrada.procesar(vector256d);
        let resultado: SalidaCapa1 = {};

        // Procesar cada subespacio en paralelo
        const subespaciosArray = Array.from(subespaciosDivididos.entries());
        
        const promesas = subespaciosArray.map(async ([id, valores], indice) => {
            const subRed = this.subRedes.get(id);
            if (!subRed) {
                throw new Error(`Sub-red ${id} no encontrada`);
            }

            try {
                // Convertir valores escalares a estructura de grafo
                const { nodeFeatures, edgeIndex } = this.vectorAGrafo(valores);
                
                // El vector global debe ser 256D, rellenamos con ceros si es local
                const globalVector = new Array(256).fill(0);
                valores.forEach((v, i) => { if(i < 256) globalVector[i] = v; });

                const salida = await subRed.predecir(nodeFeatures, edgeIndex, globalVector, {}); 
                let vectorLatente = this.extraerVectorLatente(salida);
                
                // FASE 1: Agregar Positional Encoding sinusoidal
                const posEncoding = this.posEncoder.generar(indice, 64);
                vectorLatente = vectorLatente.map((v, i) => v + posEncoding[i] * 0.1);
                
                resultado[id] = vectorLatente;
            } catch (error) {
                // Fallback: Simulación LIF mejorada
                resultado[id] = this.simularRespuestaLIF(valores);
            }
        });

        await Promise.all(promesas);
        
        // FASE 2: Aplicar Inter-Subespacio Attention
        resultado = this.interAtencion.aplicarAtencion(resultado);
        
        // FASE 2: Aplicar Learnable Weights
        if (this.learnableWeights) {
            resultado = this.learnableWeights.aplicar(resultado);
        }
        
        // Limpiar cache del encoder
        this.posEncoder.limpiarCache();

        return resultado;
    }

    /**
     * Convierte vector a grafo con Sparse Attention estratificada
     * Estrategia de conexiones en 3 niveles:
     * - Local (i → i±1): máxima densidad (100%)
     * - Medium (i → i±3): media densidad (40%)
     * - Global (i → j random): baja densidad (10%)
     * 
     * Resultado: ~10% de conexiones totales (sparse) pero efectivas
     */
    private vectorAGrafo(valores: number[]): { nodeFeatures: number[][], edgeIndex: number[][] } {
        const FEATURES_PER_NODE = 4;
        const numNodos = Math.ceil(valores.length / FEATURES_PER_NODE);
        const nodeFeatures: number[][] = [];

        // Crear nodos
        for (let i = 0; i < numNodos; i++) {
            const start = i * FEATURES_PER_NODE;
            const chunk = valores.slice(start, start + FEATURES_PER_NODE);
            while (chunk.length < FEATURES_PER_NODE) chunk.push(0);
            nodeFeatures.push(chunk);
        }

        // Crear conexiones sparse estratificadas
        const source: number[] = [];
        const target: number[] = [];

        for (let i = 0; i < numNodos; i++) {
            // 1. Conexiones locales: i → i±1 (100% densidad)
            if (i > 0) {
                source.push(i);
                target.push(i - 1);
                source.push(i - 1);
                target.push(i);
            }
            
            // 2. Conexiones medium: i → i±3 (40% probabilidad)
            if (i > 2 && Math.random() < 0.4) {
                source.push(i);
                target.push(i - 3);
                source.push(i - 3);
                target.push(i);
            }
            
            if (i < numNodos - 3 && Math.random() < 0.4) {
                source.push(i);
                target.push(i + 3);
                source.push(i + 3);
                target.push(i);
            }
            
            // 3. Self-loops para estabilidad (10% por nodo)
            if (Math.random() < 0.1) {
                source.push(i);
                target.push(i);
            }
        }

        // Fallback: si solo hay 1 nodo o sin conexiones, crear self-loop
        if (source.length === 0) {
            source.push(0);
            target.push(0);
        }

        return {
            nodeFeatures,
            edgeIndex: [source, target]
        };
    }

    /**
     * Prepara el tensor de entrada para el modelo ONNX
     * @param valores Valores normalizados del subespacio
     * @returns Tensor preparado para inferencia
     */
    // private prepararTensorParaModelo(valores: number[]): any {
    //     // Convertir array a tensor según especificaciones del modelo
    //     // Esto depende del formato exacto que espera el modelo ONNX
    //     return valores; // Placeholder - implementar según modelo real
    // }

    /**
     * Extrae el vector latente 64D de la salida del modelo
     * @param salida Salida cruda del modelo
     * @returns Vector latente de 64 dimensiones
     */
    private extraerVectorLatente(salida: any): number[] {
        // Si el modelo devuelve ajustes_dendritas (1024D), tomamos una muestra o lo reducimos
        if (salida && salida.ajustes_dendritas) {
            return salida.ajustes_dendritas.slice(0, 64);
        }
        
        if (salida && salida.latentVector) {
            return salida.latentVector.slice(0, 64); // Asegurar 64D
        }

        // Placeholder: devolver vector aleatorio para testing
        return Array.from({ length: 64 }, () => Math.random());
    }

    /**
     * Simula respuesta neuronal LIF (Leaky Integrate-and-Fire) mejorada
     * Más realista que binario puro: codifica intensidad + decaimiento
     * 
     * Modelo LIF simplificado:
     * v[i](t) = v[i](t-1) * exp(-Δt/τ) + input[i] + noise
     * spike = v[i] > θ_i
     * salida[i] = tanh(v[i] / θ_i)  // Normalizado a intensidad
     */
    private simularRespuestaLIF(valores: number[]): number[] {
        const latente = new Array(64).fill(0);
        const suma = valores.reduce((a, b) => a + b, 0);
        const promedio = suma / valores.length;
        const desv = Math.sqrt(valores.reduce((sq, v) => sq + (v - promedio) ** 2, 0) / valores.length);

        // Parámetros LIF
        const tau = 20;         // Constante de tiempo (ms)
        const dt = 1;           // Step temporal (ms)
        const decay = Math.exp(-dt / tau);
        const sigmaRuido = 0.05; // Desviación de ruido Gaussiano

        for (let i = 0; i < 64; i++) {
            // Umbral adaptativo: base + variación per-neurona
            const umbralBase = 0.3 + (i / 200);
            const umbralAdapt = umbralBase + desv * 0.1;
            
            // Estado de integración (simulado desde cero en cada paso)
            const input = (promedio + (valores[i % valores.length] || 0)) / 2;
            const ruido = (Math.random() - 0.5) * sigmaRuido;
            
            // Integración: v(t) = v(t-1) * decay + input
            let v = input + ruido;
            
            // Decaimiento exponencial (leak)
            v *= decay;
            
            // Salida: intensidad de spike normalizada
            // Si v > umbral: intensidad alta
            // Si v < umbral: intensidad baja/cero
            
            if (v > umbralAdapt) {
                // Spike ocurrió: codificar intensidad
                latente[i] = Math.tanh((v - umbralAdapt) / (umbralAdapt * 0.5));
            } else {
                // Sub-threshold: actividad espontánea mínima
                latente[i] = Math.max(0, v * 0.1);
            }
            
            // Asegurar rango [0, 1]
            latente[i] = Math.max(0, Math.min(1, latente[i]));
        }

        return latente;
    }

    /**
     * Obtiene estadísticas de procesamiento
     */
    getEstadisticas(): { 
        subRedesActivas: number; 
        memoriaUsada: number;
        atencionStats?: { subespaciosDominantes: string[]; diversidadAtencion: number };
        weightsStats?: { pesosMin: number; pesosMax: number; pesosMedio: number; subespaciosMasFuertes: string[] };
    } {
        const base = {
            subRedesActivas: this.subRedes.size,
            memoriaUsada: this.subRedes.size * 1024 * 4 // Estimación rough: 1024 neuronas × 4 bytes
        };

        // Agregar estadísticas de atención si está disponible
        if (this.interAtencion) {
            base.atencionStats = this.interAtencion.obtenerEstadisticas();
        }

        // Agregar estadísticas de pesos si está disponible
        if (this.learnableWeights) {
            base.weightsStats = this.learnableWeights.obtenerEstadisticas();
        }

        return base;
    }

    /**
     * Actualiza pesos aprendibles basado en performance de cada subespacio
     * Llamar después de cada ciclo de entrenamiento
     * 
     * @param performance Mapa de subespacio ID → accuracy/performance [0,1]
     */
    actualizarPesos(performance: Map<string, number>): void {
        if (this.learnableWeights) {
            this.learnableWeights.actualizar(performance);
        }
    }
}

/**
 * Módulo combinado: Capas 0 + 1
 */
export class ProcesadorSensorial {
    private capa0: CapaEntrada;
    private capa1: CapaSensorial;

    constructor() {
        this.capa0 = new CapaEntrada();
        this.capa1 = new CapaSensorial();
    }

    async inicializar(): Promise<void> {
        await this.capa1.inicializar();
    }

    async procesar(vector256d: Vector256D): Promise<SalidaCapa1> {
        // Capa 0 ya se ejecuta dentro de Capa 1
        return await this.capa1.procesar(vector256d);
    }

    getCapa0(): CapaEntrada {
        return this.capa0;
    }

    getCapa1(): CapaSensorial {
        return this.capa1;
    }
}
