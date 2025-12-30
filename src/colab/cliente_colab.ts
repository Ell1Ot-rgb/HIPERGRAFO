/**
 * CLIENTE COLAB - Conexión TypeScript/Node.js con servidor OMEGA 21 v4.0
 * 
 * Este módulo permite entrenar desde tu VS Code workspace usando:
 * - Servidor PyTorch en Google Colab (con ngrok)
 * - Cliente TypeScript que envía datos de entrenamiento
 * 
 * USO:
 * 1. Ejecuta el servidor Python en Colab: COLAB_SERVER_OMEGA21_V4_UNIFICADO.py
 * 2. Copia la URL pública de ngrok que aparece (ej: https://xxxx-xx-xx-xxx-xxx.ngrok.io)
 * 3. Configura SERVER_URL abajo
 * 4. Ejecuta: npx ts-node src/colab/cliente_colab.ts
 */

import axios, { AxiosInstance } from 'axios';

// ==========================================
// CONFIGURACIÓN
// ==========================================

interface ConfiguracionCliente {
  serverUrl: string;        // URL del servidor Colab (con ngrok)
  timeout?: number;         // Timeout en ms (default: 30000)
  retryAttempts?: number;   // Intentos de reconexión (default: 3)
}

// ⚠️ REEMPLAZA CON TU URL DE NGROK
const CONFIG: ConfiguracionCliente = {
  serverUrl: process.env.COLAB_SERVER_URL || 'http://localhost:8000',
  timeout: 30000,
  retryAttempts: 3
};

// ==========================================
// TIPOS DE DATOS
// ==========================================

interface MuestraEntrenamiento {
  input_data: number[];  // Array de 1600 números
  anomaly_label: 0 | 1;  // 0 = normal, 1 = anomalía
}

interface LoteEntrenamiento {
  samples: MuestraEntrenamiento[];
  epochs: number;
}

interface FeedbackDendritico {
  ajustes_aplicados: number[];  // 16D
  validacion: boolean;
  timestamp: string;
}

interface RespuestaEntrenamiento {
  status: string;
  loss: number;
  batch_size: number;
  outputs: {
    anomaly_prob: number;
    dendrite_adjustments: number[];
    coherence_state: number[];
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
    feedback: {
      recibido: number;
      exitoso: number;
      tasa_exito: number;
    };
  };
  torch_version: string;
  cuda_available: boolean;
}

// ==========================================
// CLIENTE COLAB
// ==========================================

export class ClienteColab {
  private cliente: AxiosInstance;
  private serverUrl: string;
  private retryAttempts: number;
  private intentosActuales: number = 0;

  constructor(config: ConfiguracionCliente) {
    this.serverUrl = config.serverUrl;
    this.retryAttempts = config.retryAttempts || 3;

    // Crear cliente Axios
    this.cliente = axios.create({
      baseURL: this.serverUrl,
      timeout: config.timeout || 30000,
      validateStatus: () => true  // No lanzar error en cualquier status
    });
  }

  /**
   * Conectar al servidor y verificar disponibilidad
   */
  async conectar(): Promise<boolean> {
    console.log(`\n🔗 Conectando a servidor: ${this.serverUrl}`);
    
    try {
      const response = await this.cliente.get('/health');
      
      if (response.status === 200) {
        console.log('✅ Conexión exitosa');
        console.log(`   Uptime: ${response.data.uptime_seconds.toFixed(2)}s`);
        return true;
      } else {
        console.log(`❌ Servidor respondió con status ${response.status}`);
        return false;
      }
    } catch (error: any) {
      console.log(`❌ No se pudo conectar: ${error.message}`);
      return false;
    }
  }

  /**
   * Obtener información del servidor y arquitectura
   */
  async obtenerInfo(): Promise<any> {
    try {
      const response = await this.cliente.get('/info');
      
      if (response.status === 200) {
        return response.data;
      } else {
        console.log(`Error al obtener info: ${response.data.error}`);
        return null;
      }
    } catch (error: any) {
      console.log(`Error en GET /info: ${error.message}`);
      return null;
    }
  }

  /**
   * Obtener estado actual del servidor
   */
  async obtenerEstado(): Promise<EstadoServidor | null> {
    try {
      const response = await this.cliente.get<EstadoServidor>('/status');
      
      if (response.status === 200) {
        return response.data;
      } else {
        console.log(`Error al obtener estado: ${response.data.error}`);
        return null;
      }
    } catch (error: any) {
      console.log(`Error en GET /status: ${error.message}`);
      return null;
    }
  }

  /**
   * Entrenar el modelo con un lote de datos
   * 
   * @param samples Array de muestras (input_data + etiqueta)
   * @param epochs Número de épocas de entrenamiento
   */
  async entrenar(
    samples: MuestraEntrenamiento[],
    epochs: number = 1
  ): Promise<RespuestaEntrenamiento | null> {
    try {
      // Validar input
      if (!samples || samples.length === 0) {
        console.log('❌ Error: No hay muestras para entrenar');
        return null;
      }

      const muestra_ejemplo = samples[0];
      if (muestra_ejemplo.input_data.length !== 1600) {
        console.log(
          `❌ Error: Dimensión incorrecta. Esperado 1600D, obtuvo ${muestra_ejemplo.input_data.length}D`
        );
        return null;
      }

      console.log(`\n📚 Entrenando: ${samples.length} muestras, ${epochs} época(s)...`);

      const lote: LoteEntrenamiento = {
        samples,
        epochs
      };

      const response = await this.cliente.post<RespuestaEntrenamiento>(
        '/train_layer2',
        lote
      );

      if (response.status === 200 && response.data.status === 'trained') {
        console.log('✅ Entrenamiento completado');
        console.log(`   Loss: ${response.data.loss.toFixed(6)}`);
        console.log(`   Batch size: ${response.data.batch_size}`);
        console.log(`   Anomaly prob: ${response.data.outputs.anomaly_prob.toFixed(4)}`);
        return response.data;
      } else {
        console.log(`❌ Error de entrenamiento: ${response.data.error}`);
        return null;
      }
    } catch (error: any) {
      console.log(`❌ Error en entrenar: ${error.message}`);
      return null;
    }
  }

  /**
   * Enviar feedback dendrítico desde entrenamiento local
   */
  async enviarFeedback(
    ajustes: number[],
    validacion: boolean
  ): Promise<any> {
    try {
      if (ajustes.length !== 16) {
        console.log(`⚠️ Advertencia: Ajustes esperan 16D, recibieron ${ajustes.length}D`);
      }

      const feedback: FeedbackDendritico = {
        ajustes_aplicados: ajustes,
        validacion,
        timestamp: new Date().toISOString()
      };

      const response = await this.cliente.post(
        '/feedback_dendritas',
        feedback
      );

      if (response.status === 200) {
        console.log(`✅ Feedback enviado (validación: ${validacion})`);
        return response.data;
      } else {
        console.log(`❌ Error al enviar feedback: ${response.data.error}`);
        return null;
      }
    } catch (error: any) {
      console.log(`❌ Error en enviarFeedback: ${error.message}`);
      return null;
    }
  }

  /**
   * Obtener métricas avanzadas
   */
  async obtenerMetricas(): Promise<any> {
    try {
      const response = await this.cliente.get('/metricas');
      
      if (response.status === 200) {
        return response.data;
      } else {
        console.log(`Error al obtener métricas: ${response.data.error}`);
        return null;
      }
    } catch (error: any) {
      console.log(`Error en GET /metricas: ${error.message}`);
      return null;
    }
  }

  /**
   * Realizar diagnóstico del sistema
   */
  async diagnostico(): Promise<any> {
    try {
      console.log('🔍 Ejecutando diagnóstico...');
      
      const response = await this.cliente.post('/diagnostico');
      
      if (response.status === 200) {
        console.log('✅ Diagnóstico OK');
        if (response.data.gpu_info?.cuda) {
          console.log(`   GPU: ${response.data.gpu_info.device}`);
        } else {
          console.log('   GPU: No disponible (usando CPU)');
        }
        return response.data;
      } else {
        console.log(`❌ Diagnóstico falló: ${response.data.error}`);
        return null;
      }
    } catch (error: any) {
      console.log(`❌ Error en diagnóstico: ${error.message}`);
      return null;
    }
  }

  /**
   * Generar datos de prueba sintéticos
   */
  generarDatosPrueba(cantidad: number = 10): MuestraEntrenamiento[] {
    const datos: MuestraEntrenamiento[] = [];

    for (let i = 0; i < cantidad; i++) {
      // Generar 1600D aleatorio
      const input = Array(1600)
        .fill(0)
        .map(() => Math.random() * 2 - 1);  // Valores entre -1 y 1

      // Etiquetar aleatoriamente (80% normal, 20% anomalía)
      const label = Math.random() < 0.8 ? 0 : 1;

      datos.push({
        input_data: input,
        anomaly_label: label as 0 | 1
      });
    }

    return datos;
  }
}

// ==========================================
// SCRIPT DE PRUEBA
// ==========================================

async function mainPrueba() {
  console.log('='.repeat(80));
  console.log('🚀 CLIENTE COLAB - PRUEBA INTERACTIVA');
  console.log('='.repeat(80));

  const cliente = new ClienteColab(CONFIG);

  // 1. Conectar
  const conectado = await cliente.conectar();
  if (!conectado) {
    console.log(
      '\n❌ No se pudo conectar. Verifica que:'
    );
    console.log(
      '   1. El servidor Python está ejecutándose en Colab'
    );
    console.log(
      '   2. Tienes la URL de ngrok correcta'
    );
    console.log(
      '   3. La URL está configurada en CONFIG.serverUrl'
    );
    console.log(
      '\n💡 Para obtener la URL de ngrok en Colab, ejecuta:'
    );
    console.log(
      '   print(ngrok_url)  # Al final del script Python'
    );
    return;
  }

  // 2. Obtener info del servidor
  console.log('\n📋 INFORMACIÓN DEL SERVIDOR:');
  const info = await cliente.obtenerInfo();
  if (info) {
    console.log(`   Nombre: ${info.nombre}`);
    console.log(`   Versión: ${info.version}`);
    console.log(`   Parámetros: ${info.arquitectura.parametros_totales.toLocaleString()}`);
  }

  // 3. Obtener estado
  console.log('\n📊 ESTADO ACTUAL:');
  const estado = await cliente.obtenerEstado();
  if (estado) {
    console.log(`   Muestras entrenadas: ${estado.estadisticas.total_muestras}`);
    console.log(`   Batches procesados: ${estado.estadisticas.total_batches}`);
    console.log(`   Loss promedio: ${estado.estadisticas.loss_promedio_global.toFixed(6)}`);
    console.log(`   GPU: ${estado.cuda_available ? '✅' : '❌'}`);
  }

  // 4. Diagnostico
  await cliente.diagnostico();

  // 5. Entrenar con datos de prueba
  console.log('\n🎓 ENTRENANDO CON DATOS DE PRUEBA:');
  const datosPrueba = cliente.generarDatosPrueba(5);
  const resultadoEntrenamiento = await cliente.entrenar(datosPrueba, 1);

  if (resultadoEntrenamiento) {
    console.log('\n📈 RESULTADOS:');
    console.log(`   Anomaly probability: ${resultadoEntrenamiento.outputs.anomaly_prob.toFixed(4)}`);
    console.log(`   Dendrite adjustments: ${resultadoEntrenamiento.outputs.dendrite_adjustments.slice(0, 3).map(v => v.toFixed(4)).join(', ')} ...`);
    console.log(`   Coherence state: ${resultadoEntrenamiento.outputs.coherence_state.slice(0, 3).map(v => v.toFixed(4)).join(', ')} ...`);

    // 6. Enviar feedback
    console.log('\n📤 ENVIANDO FEEDBACK:');
    const feedback = Array(16).fill(0).map(() => Math.random() * 0.1 - 0.05);
    await cliente.enviarFeedback(feedback, true);
  }

  // 7. Obtener métricas
  console.log('\n📊 MÉTRICAS FINALES:');
  const metricas = await cliente.obtenerMetricas();
  if (metricas) {
    console.log(`   Últimos 20 losses: ${metricas.ultimos_20_losses.map((v: number) => v.toFixed(6)).join(', ')}`);
    console.log(`   Tendencia: ${metricas.tendencia}`);
    console.log(`   Anomalías detectadas: ${metricas.anomalias_detectadas}`);
  }

  console.log('\n' + '='.repeat(80));
  console.log('✅ PRUEBA COMPLETADA');
  console.log('='.repeat(80));
}

// Ejecutar si se llama directamente
if (require.main === module) {
  mainPrueba().catch(console.error);
}

export { ClienteColab, MuestraEntrenamiento, RespuestaEntrenamiento, EstadoServidor };
