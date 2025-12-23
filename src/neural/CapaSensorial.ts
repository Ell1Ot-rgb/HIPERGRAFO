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

    /**
     * Preprocesa el vector 256D y lo divide en subespacios.
     * @param vector256d Vector completo de entrada
     * @returns Mapa de subespacios con sus valores
     */
    procesar(vector256d: Vector256D): Map<string, number[]> {
        const resultado = new Map<string, number[]>();

        for (const subespacio of this.subespacios) {
            const valores: number[] = [];
            for (let i = subespacio.rango[0]; i <= subespacio.rango[1]; i++) {
                const clave = `D${i.toString().padStart(3, '0')}`;
                const valor = vector256d[clave];
                if (valor === undefined) {
                    throw new Error(`Campo ${clave} no encontrado en vector 256D`);
                }
                valores.push(this.normalizarCampo(clave, valor));
            }
            resultado.set(subespacio.id, valores);
        }

        return resultado;
    }

    /**
     * Normalización específica por tipo de campo.
     * @param campo Nombre del campo (D001, etc.)
     * @param valor Valor crudo
     * @returns Valor normalizado [0,1] o [-1,1]
     */
    private normalizarCampo(campo: string, valor: number): number {
        // Implementar normalización específica según el tipo de dato
        // Para uint32/64: log scaling
        // Para int16: tanh
        // Para uint8: min-max

        const campoNum = parseInt(campo.substring(1));

        // Campos uint32 (alta magnitud): log scaling
        if ([1, 2, 3, 4, 5, 9, 10, 11, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208].includes(campoNum)) {
            return Math.log1p(Math.max(0, valor)) / Math.log1p(1e9); // Asumiendo max ~1e9
        }

        // Campos int16 (potenciales, emociones): tanh para [-1,1]
        if ([89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 133, 134, 135, 136, 137, 138, 139, 140, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256].includes(campoNum)) {
            return Math.tanh(valor / 1000); // Asumiendo rango ~[-1000,1000]
        }

        // Campos uint8 (flags, índices): min-max [0,1]
        if ([8, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 125, 126, 127, 128, 129, 130, 131, 132, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224].includes(campoNum)) {
            return Math.max(0, Math.min(1, valor / 255)); // uint8 [0,255] → [0,1]
        }

        // Campos uint16 (métricas): min-max con rangos específicos
        return Math.max(0, Math.min(1, valor / 65535)); // uint16 [0,65535] → [0,1]
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

        this.inicializado = true;
        console.log('🧠 Capa Sensorial inicializada con 25 sub-redes LIF');
    }

    /**
     * Procesa el vector 256D completo a través de las 25 sub-redes
     * @param vector256d Vector de entrada completo
     * @returns Salida latente de cada subespacio (25 × 64D)
     */
    async procesar(vector256d: Vector256D): Promise<SalidaCapa1> {
        if (!this.inicializado) {
            throw new Error('Capa Sensorial no inicializada. Llama a inicializar() primero.');
        }

        const subespaciosDivididos = this.capaEntrada.procesar(vector256d);
        const resultado: SalidaCapa1 = {};

        // Procesar cada subespacio en paralelo
        const promesas = Array.from(subespaciosDivididos.entries()).map(async ([id, valores]) => {
            const subRed = this.subRedes.get(id);
            if (!subRed) {
                throw new Error(`Sub-red ${id} no encontrada`);
            }

            // El modelo ONNX espera un tensor específico
            // Aquí asumimos que el modelo procesa el subespacio y devuelve 64D
            try {
                // Convertir valores escalares a estructura de grafo (Node Features + Edge Index)
                const { nodeFeatures, edgeIndex } = this.vectorAGrafo(valores);
                
                // El vector global debe ser 256D, rellenamos con ceros si es local
                const globalVector = new Array(256).fill(0);
                valores.forEach((v, i) => { if(i < 256) globalVector[i] = v; });

                const salida = await subRed.predecir(nodeFeatures, edgeIndex, globalVector, {}); 
                resultado[id] = this.extraerVectorLatente(salida);
            } catch (error) {
                // Fallback: Simulación LIF (Leaky Integrate-and-Fire) si el modelo falla o no es compatible
                // console.warn(`⚠️ Fallback LIF en subred ${id}: ${error}`);
                resultado[id] = this.simularRespuestaLIF(valores);
            }
        });

        await Promise.all(promesas);

        return resultado;
    }

    /**
     * Convierte un vector plano de valores en una estructura de grafo simple
     * para ser consumida por la GNN/ONNX.
     * Mapea cada 4 valores a 1 nodo con 4 features.
     */
    private vectorAGrafo(valores: number[]): { nodeFeatures: number[][], edgeIndex: number[][] } {
        const FEATURES_PER_NODE = 4;
        const numNodos = Math.ceil(valores.length / FEATURES_PER_NODE);
        const nodeFeatures: number[][] = [];

        for (let i = 0; i < numNodos; i++) {
            const start = i * FEATURES_PER_NODE;
            const chunk = valores.slice(start, start + FEATURES_PER_NODE);
            // Rellenar con ceros si el chunk es menor a 4
            while (chunk.length < FEATURES_PER_NODE) chunk.push(0);
            nodeFeatures.push(chunk);
        }

        // Crear conexiones lineales simples: 0->1, 1->2, etc.
        // EdgeIndex shape: [2, num_edges]
        const source: number[] = [];
        const target: number[] = [];
        
        if (numNodos > 1) {
            for (let i = 0; i < numNodos - 1; i++) {
                source.push(i);
                target.push(i + 1);
                // Bidireccional
                source.push(i + 1);
                target.push(i);
            }
        } else {
            // Self-loop si solo hay un nodo para evitar arrays vacíos si el modelo lo requiere
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
     * Simula una respuesta de neurona LIF (Leaky Integrate-and-Fire)
     * como fallback cuando el modelo ONNX no es adecuado para el subespacio.
     */
    private simularRespuestaLIF(valores: number[]): number[] {
        const latente = new Array(64).fill(0);
        const suma = valores.reduce((a, b) => a + b, 0);
        const promedio = suma / valores.length;

        // Generar un patrón de "spikes" basado en la intensidad de la entrada
        for (let i = 0; i < 64; i++) {
            // Cada "neurona" latente tiene un umbral ligeramente diferente
            const umbral = 0.3 + (i / 100);
            const ruido = (Math.random() - 0.5) * 0.1;
            latente[i] = (promedio + ruido > umbral) ? 1.0 : 0.0;
            
            // Aplicar decaimiento (leak) - simplificado para este mock
            if (latente[i] > 0) {
                latente[i] *= Math.exp(-0.1); 
            }
        }

        return latente;
    }

    /**
     * Obtiene estadísticas de procesamiento
     */
    getEstadisticas(): { subRedesActivas: number; memoriaUsada: number } {
        return {
            subRedesActivas: this.subRedes.size,
            memoriaUsada: this.subRedes.size * 1024 * 4 // Estimación rough: 1024 neuronas × 4 bytes
        };
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
