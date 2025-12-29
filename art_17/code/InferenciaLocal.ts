import * as ort from 'onnxruntime-node';
import { EventEmitter } from 'events';

export class InferenciaLocal extends EventEmitter {
    private session: ort.InferenceSession | null = null;
    private modelPath: string;

    constructor(modelPath: string = '/workspaces/HIPERGRAFO/models/omega21_brain.onnx') {
        super();
        this.modelPath = modelPath;
    }

    async inicializar() {
        try {
            this.session = await ort.InferenceSession.create(this.modelPath);
            console.log('🧠 Modelo ONNX cargado correctamente en el motor local.');
        } catch (e) {
            console.error('❌ Error al cargar el modelo ONNX:', e);
            throw e;
        }
    }

    async predecir(nodeFeatures: number[][], edgeIndex: number[][], globalVector: number[], telemetry?: any): Promise<any> {
        if (!this.session) throw new Error('Sesión ONNX no inicializada');

        try {
            // Si tenemos el vector 72D de metrics.c, lo usamos como parte del vector global
            let inputVector = globalVector;
            if (telemetry && telemetry.vector_72d) {
                // Combinamos el vector 72D con los estados de las 16 dendritas para llegar a algo cercano a 256
                const dendriteValues = Object.values(telemetry.dendrites).filter(v => typeof v === 'number') as number[];
                inputVector = [...telemetry.vector_72d, ...dendriteValues];
                // Padeamos hasta 256 si es necesario
                while (inputVector.length < 256) inputVector.push(0);
                if (inputVector.length > 256) inputVector = inputVector.slice(0, 256);
            }

            // Preparar tensores
            const nodeFeaturesFlat = new Float32Array(nodeFeatures.flat());
            const nodeFeaturesTensor = new ort.Tensor('float32', nodeFeaturesFlat, [nodeFeatures.length, 4]);

            // Asegurar que edgeIndex tenga estructura válida [2, N]
            let safeEdgeIndex = edgeIndex;
            if (!edgeIndex || edgeIndex.length < 2 || !edgeIndex[0]) {
                safeEdgeIndex = [[0], [0]]; // Self-loop dummy por defecto
            }

            const edgeIndexFlat = new BigInt64Array(safeEdgeIndex.flat().map(n => BigInt(n)));
            const edgeIndexTensor = new ort.Tensor('int64', edgeIndexFlat, [2, safeEdgeIndex[0].length]);

            const globalVectorTensor = new ort.Tensor('float32', Float32Array.from(inputVector), [1, 256]);

            const batchData = new Int32Array(nodeFeatures.length).fill(0);
            const batchTensor = new ort.Tensor('int64', BigInt64Array.from(Array.from(batchData).map(n => BigInt(n))), [nodeFeatures.length]);

            const feeds: Record<string, ort.Tensor> = {
                'node_features': nodeFeaturesTensor,
                'edge_index': edgeIndexTensor,
                'global_vector': globalVectorTensor,
                'batch': batchTensor
            };
            
            const results = await this.session.run(feeds);
            
            // El nombre de la salida depende de cómo se exportó en PyTorch
            // Usualmente es 'output' o el nombre de la última capa
            const outputKey = Object.keys(results)[0];
            const output = results[outputKey]; 
            const data = Array.from(output.data as Float32Array);

            // Mapear la salida de 1024 a la estructura esperada
            const ajustes_dendritas = data.slice(0, 256);
            
            // Calcular una "predicción de anomalía" basada en la activación media
            const activacionMedia = data.reduce((a, b) => a + b, 0) / data.length;
            const prediccion_anomalia = activacionMedia > 0.5 ? 1.0 : 0.0;

            return {
                status: 'ok',
                ajustes_dendritas,
                prediccion_anomalia,
                prediccion_estabilidad: activacionMedia, // Usamos la media como proxy de estabilidad
                loss: 0,
                modo: 'LOCAL_ONNX'
            };
        } catch (e) {
            // Silenciamos el error para permitir el fallback a simulación LIF en CapaSensorial
            // console.error('❌ Error durante la inferencia local:', e);
            throw e; // Re-lanzar para que CapaSensorial use el fallback
        }
    }
}
