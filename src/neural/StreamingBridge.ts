/**
 * StreamingBridge.ts
 * 
 * Gestiona el flujo de datos de alta velocidad entre Codespaces y Colab.
 * Optimizado para evitar cuellos de botella mediante:
 * 1. Batching (Agrupación de vectores).
 * 2. Serialización compacta.
 * 3. Buffering asíncrono (Prefetching).
 */

import axios from 'axios';
import { DecisionColab } from './HipergrafoBridge';

export interface MuestraEntrenamiento {
    input_data: number[];
    anomaly_label: number;
}

export interface LoteEntrenamiento {
    samples: MuestraEntrenamiento[];
}

export class StreamingBridge {
    private urlColab: string;
    private buffer: MuestraEntrenamiento[] = [];
    private readonly TAMANO_BATCH = 64;
    private enviando: boolean = false;
    private feedbackCallback?: (decision: DecisionColab) => void;

    constructor(urlColab: string) {
        this.urlColab = urlColab.replace(/\/$/, "");
    }

    /**
     * Registra callback para procesar feedback de Colab
     */
    public registrarCallbackFeedback(callback: (decision: DecisionColab) => void): void {
        this.feedbackCallback = callback;
    }

    /**
     * Añade un vector procesado al flujo de streaming
     */
    public async enviarVector(vector1600d: number[], esAnomalia: boolean) {
        // 1. Agregar al buffer individual
        this.buffer.push({
            input_data: vector1600d,
            anomaly_label: esAnomalia ? 1 : 0
        });

        // 2. Si el buffer tiene suficientes muestras, intentar enviar un lote
        if (this.buffer.length >= this.TAMANO_BATCH && !this.enviando) {
            this.procesarCola();
        }
    }

    /**
     * Procesa la cola de muestras enviando lotes de TAMANO_BATCH
     */
    private async procesarCola() {
        if (this.buffer.length < this.TAMANO_BATCH) return;
        
        this.enviando = true;
        
        while (this.buffer.length >= this.TAMANO_BATCH) {
            // Extraer un lote de 64 muestras
            const samples = this.buffer.splice(0, this.TAMANO_BATCH);
            const lote: LoteEntrenamiento = { samples };
            
            try {
                // Enviar a Colab usando el endpoint correcto /train_layer2
                const inicio = Date.now();
                const respuesta = await axios.post(`${this.urlColab}/train_layer2`, lote, {
                    headers: { 
                        'Content-Type': 'application/json',
                        'ngrok-skip-browser-warning': 'true'
                    },
                    timeout: 15000 // Aumentado para batches grandes
                });
                
                const latencia = Date.now() - inicio;
                console.log(`🚀 Lote de ${this.TAMANO_BATCH} muestras enviado. Latencia: ${latencia}ms. Restantes: ${this.buffer.length}`);
                
                // Procesar feedback de Colab
                if (respuesta.data && this.feedbackCallback) {
                    const decision: DecisionColab = {
                        anomaly_prob: respuesta.data.anomaly_prob ?? 0,
                        dendrite_adjustments: respuesta.data.dendrite_adjustments ?? [],
                        coherence_state: respuesta.data.coherence_state ?? [],
                        timestamp: respuesta.data.timestamp ?? new Date().toISOString(),
                        batch_size: samples.length,
                        loss: respuesta.data.loss,
                    };
                    
                    // Invocar callback con decisión
                    this.feedbackCallback(decision);
                }
            } catch (error: any) {
                console.error(`❌ Error enviando lote a Colab: ${error.message}`);
                // Devolver las muestras al inicio del buffer para reintentar
                this.buffer.unshift(...samples);
                await new Promise(resolve => setTimeout(resolve, 5000));
                break; 
            }
        }
        
        this.enviando = false;
    }

    public obtenerEstadoBuffer(): number {
        return this.buffer.length;
    }
}
