/**
 * ClienteColabEntrenamiento.ts
 * 
 * Cliente TypeScript para conectar VS Code con servidor Colab
 * Permite enviar datos y entrenar modelos remotamente
 * 
 * Uso:
 * ```typescript
 * const cliente = new ClienteColabEntrenamiento('https://tu-ngrok-url');
 * await cliente.conectar();
 * await cliente.entrenarLote(datos);
 * ```
 */

import axios, { AxiosInstance } from 'axios';

// ==========================================
// INTERFACES DE DATOS
// ==========================================

interface MuestraEntrenamiento {
    input_data: number[];      // 1600D
    anomaly_label: number;     // 0 o 1
    timestamp?: string;
}

interface LoteEntrenamiento {
    samples: MuestraEntrenamiento[];
    epochs?: number;
}

interface RespuestaEntrenamiento {
    status: string;
    loss: number;
    batch_size: number;
    outputs: {
        anomaly_prob: number;
        dendrite_adjustments: number[];  // 16D
        coherence_state: number[];       // 64D
    };
    capa_info: {
        capa2_activations: number;
        capa3_activations: number;
        capa4_activations: number;
    };
    timestamp: string;
}

interface EstadoServidor {
    status: string;
    modelo: string;
    estadisticas: {
        total_muestras: number;
        total_batches: number;
        loss_promedio_global: number;
        loss_promedio_ultimos_100: number;
        anomalia_media: number;
        tiempo_transcurrido_seg: number;
        dispositivo: string;
        gpu_memoria_mb: number;
        feedback: {
            recibido: number;
            exitoso: number;
            tasa_exito: number;
        };
    };
    torch_version: string;
    cuda_available: boolean;
}

interface FeedbackDendritico {
    ajustes_aplicados: number[];  // 16D
    validacion: boolean;
    timestamp: string;
}

interface Metricas {
    ultimos_20_losses: number[];
    tendencia: string;
    anomalias_detectadas: number;
    feedback_tasa_exito: number;
}

// ==========================================
// CLIENTE COLAB
// ==========================================

export class ClienteColabEntrenamiento {
    private urlServidor: string;
    private cliente: AxiosInstance;
    private conectado: boolean = false;
    private historialEntrenamientos: RespuestaEntrenamiento[] = [];
    private estadisticasLocales = {
        lotesEnviados: 0,
        totalMuestras: 0,
        lossPromedio: 0,
        tiempoTotal: 0,
    };

    constructor(urlServidor: string) {
        this.urlServidor = urlServidor.endsWith('/') ? urlServidor.slice(0, -1) : urlServidor;
        
        this.cliente = axios.create({
            baseURL: this.urlServidor,
            timeout: 60000, // 60 segundos
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'HIPERGRAFO-Cliente/1.0'
            }
        });
    }

    /**
     * Conectar al servidor Colab y validar disponibilidad
     */
    async conectar(): Promise<boolean> {
        try {
            console.log(`\n🔗 Conectando a servidor Colab: ${this.urlServidor}`);
            
            const respuesta = await this.cliente.get('/health');
            
            if (respuesta.data.alive) {
                this.conectado = true;
                console.log('✅ Servidor Colab conectado');
                
                // Obtener información detallada
                const info = await this.obtenerInfo();
                console.log(`📊 Modelo: ${info.nombre} v${info.version}`);
                console.log(`📈 Parámetros: ${info.arquitectura.parametros_totales.toLocaleString()}`);
                
                return true;
            }
        } catch (error) {
            console.error('❌ Error al conectar con servidor Colab:', error);
            this.conectado = false;
            return false;
        }
        
        return false;
    }

    /**
     * Obtener estado actual del servidor
     */
    async obtenerEstado(): Promise<EstadoServidor> {
        try {
            const respuesta = await this.cliente.get('/status');
            return respuesta.data;
        } catch (error) {
            throw new Error(`Error al obtener estado: ${error}`);
        }
    }

    /**
     * Obtener información de arquitectura
     */
    async obtenerInfo(): Promise<any> {
        try {
            const respuesta = await this.cliente.get('/info');
            return respuesta.data;
        } catch (error) {
            throw new Error(`Error al obtener info: ${error}`);
        }
    }

    /**
     * Enviar un lote de entrenamiento
     */
    async entrenarLote(muestras: MuestraEntrenamiento[], epochs: number = 1): Promise<RespuestaEntrenamiento> {
        if (!this.conectado) {
            throw new Error('No estás conectado al servidor Colab');
        }

        try {
            console.log(`\n📤 Enviando lote de ${muestras.length} muestras...`);
            const tiempoInicio = Date.now();

            const lote: LoteEntrenamiento = {
                samples: muestras,
                epochs
            };

            const respuesta = await this.cliente.post<RespuestaEntrenamiento>('/train_layer2', lote);
            
            const tiempoTranscurrido = (Date.now() - tiempoInicio) / 1000;
            
            if (respuesta.data.status === 'trained') {
                this.registrarEntrenamiento(respuesta.data, tiempoTranscurrido, muestras.length);
                
                console.log(`✅ Entrenamiento completado (${tiempoTranscurrido.toFixed(2)}s)`);
                console.log(`   Loss: ${respuesta.data.loss.toFixed(6)}`);
                console.log(`   Anomalía detectada: ${(respuesta.data.outputs.anomaly_prob * 100).toFixed(2)}%`);
                
                return respuesta.data;
            } else {
                throw new Error(`Estado inesperado: ${respuesta.data.status}`);
            }
        } catch (error) {
            console.error('❌ Error en entrenamiento:', error);
            throw error;
        }
    }

    /**
     * Entrenar múltiples lotes (útil para datasets grandes)
     */
    async entrenarMultiplesLotes(
        muestras: MuestraEntrenamiento[],
        tamanoLote: number = 64
    ): Promise<RespuestaEntrenamiento[]> {
        const resultados: RespuestaEntrenamiento[] = [];
        
        const totalLotes = Math.ceil(muestras.length / tamanoLote);
        console.log(`\n📊 Entrenando ${totalLotes} lotes de ${tamanoLote} muestras c/u...`);

        for (let i = 0; i < muestras.length; i += tamanoLote) {
            const lote = muestras.slice(i, i + tamanoLote);
            const numeroLote = Math.floor(i / tamanoLote) + 1;
            
            console.log(`   Lote ${numeroLote}/${totalLotes}...`);
            
            try {
                const resultado = await this.entrenarLote(lote);
                resultados.push(resultado);
            } catch (error) {
                console.error(`   ❌ Error en lote ${numeroLote}:`, error);
            }
        }

        return resultados;
    }

    /**
     * Enviar feedback dendrítico
     */
    async enviarFeedback(
        ajustes: number[],
        validacion: boolean
    ): Promise<any> {
        if (!this.conectado) {
            throw new Error('No estás conectado al servidor Colab');
        }

        try {
            const feedback: FeedbackDendritico = {
                ajustes_aplicados: ajustes,
                validacion,
                timestamp: new Date().toISOString()
            };

            const respuesta = await this.cliente.post('/feedback_dendritas', feedback);
            
            if (respuesta.data.status === 'feedback_recibido') {
                console.log(`✅ Feedback enviado (validación: ${validacion})`);
                return respuesta.data;
            }
        } catch (error) {
            console.error('❌ Error al enviar feedback:', error);
            throw error;
        }
    }

    /**
     * Obtener métricas de entrenamiento
     */
    async obtenerMetricas(): Promise<Metricas> {
        try {
            const respuesta = await this.cliente.get('/metricas');
            return respuesta.data;
        } catch (error) {
            throw new Error(`Error al obtener métricas: ${error}`);
        }
    }

    /**
     * Ejecutar diagnóstico del sistema
     */
    async diagnostico(): Promise<any> {
        try {
            const respuesta = await this.cliente.post('/diagnostico');
            
            console.log('🔧 Diagnóstico del servidor:');
            console.log(`   Status: ${respuesta.data.status}`);
            console.log(`   Input shape: ${respuesta.data.test_input_shape}`);
            console.log(`   Outputs: ${JSON.stringify(respuesta.data.outputs_shapes, null, 2)}`);
            console.log(`   CUDA: ${respuesta.data.gpu_info.cuda}`);
            if (respuesta.data.gpu_info.cuda) {
                console.log(`   GPU: ${respuesta.data.gpu_info.device}`);
            }
            
            return respuesta.data;
        } catch (error) {
            console.error('❌ Error en diagnóstico:', error);
            throw error;
        }
    }

    /**
     * Mostrar resumen de estadísticas locales
     */
    mostrarResumen(): void {
        console.log('\n📈 RESUMEN DE ENTRENAMIENTOS:');
        console.log(`   Lotes enviados: ${this.estadisticasLocales.lotesEnviados}`);
        console.log(`   Total muestras: ${this.estadisticasLocales.totalMuestras}`);
        console.log(`   Loss promedio: ${this.estadisticasLocales.lossPromedio.toFixed(6)}`);
        console.log(`   Tiempo total: ${this.estadisticasLocales.tiempoTotal.toFixed(2)}s`);
        
        if (this.historialEntrenamientos.length > 0) {
            const ultimoEntrenamiento = this.historialEntrenamientos[this.historialEntrenamientos.length - 1];
            console.log(`\n   Último entrenamiento:`);
            console.log(`     • Loss: ${ultimoEntrenamiento.loss.toFixed(6)}`);
            console.log(`     • Anomalía: ${(ultimoEntrenamiento.outputs.anomaly_prob * 100).toFixed(2)}%`);
        }
    }

    /**
     * Registrar entrenamiento en historial local
     */
    private registrarEntrenamiento(
        resultado: RespuestaEntrenamiento,
        tiempoTranscurrido: number,
        totalMuestras: number
    ): void {
        this.historialEntrenamientos.push(resultado);
        this.estadisticasLocales.lotesEnviados++;
        this.estadisticasLocales.totalMuestras += totalMuestras;
        this.estadisticasLocales.tiempoTotal += tiempoTranscurrido;

        // Actualizar loss promedio
        const lossTotales = this.historialEntrenamientos.reduce((sum, r) => sum + r.loss, 0);
        this.estadisticasLocales.lossPromedio = lossTotales / this.historialEntrenamientos.length;
    }

    /**
     * Obtener historial de entrenamientos
     */
    obtenerHistorial(): RespuestaEntrenamiento[] {
        return this.historialEntrenamientos;
    }

    /**
     * Desconectar cliente
     */
    desconectar(): void {
        this.conectado = false;
        console.log('\n👋 Desconectado del servidor Colab');
    }
}
