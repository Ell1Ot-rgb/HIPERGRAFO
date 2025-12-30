/**
 * CLIENTE ART V7 - Compatible con Reactor Neuro-Simbólico en Docker
 * 
 * Este cliente permite entrenar usando el servidor ART V7 en Docker
 * en lugar de Google Colab.
 */

import axios, { AxiosInstance } from 'axios';

interface ConfiguracionCliente {
  serverUrl: string;
  timeout?: number;
  retryAttempts?: number;
}

interface MuestraEntrenamiento {
  input_data: number[];
  anomaly_label: 0 | 1;
}

interface RespuestaART_V7 {
  status: string;
  loss: number;
  l_pred: number;
  l_causal: number;
  l_topo: number;
  epoch: number;
  device: string;
  error?: string;
}

const CONFIG: ConfiguracionCliente = {
  serverUrl: process.env.COLAB_SERVER_URL || 'http://localhost:8000',
  timeout: 60000,
  retryAttempts: 3
};

export class ClienteART_V7 {
  private cliente: AxiosInstance;
  private serverUrl: string;

  constructor(config: ConfiguracionCliente) {
    this.serverUrl = config.serverUrl;
    this.cliente = axios.create({
      baseURL: this.serverUrl,
      timeout: config.timeout || 60000,
      validateStatus: () => true
    });
  }

  async conectar(): Promise<boolean> {
    console.log(`\n🔗 Conectando al Reactor ART V7: ${this.serverUrl}`);
    try {
      const response = await this.cliente.get('/health');
      if (response.status === 200) {
        console.log('✅ Reactor ART V7 online');
        return true;
      }
      console.log(`❌ Status ${response.status}`);
      return false;
    } catch (error: any) {
      console.log(`❌ Error: ${error.message}`);
      return false;
    }
  }

  async entrenar(samples: MuestraEntrenamiento[], epochs: number = 1): Promise<RespuestaART_V7 | null> {
    try {
      console.log(`\n🚀 Entrenando Reactor ART V7: ${samples.length} muestras`);

      const response = await this.cliente.post<RespuestaART_V7>('/train_reactor', {
        samples,
        epochs
      });

      if (response.status === 200 && response.data.status === 'trained') {
        console.log('✅ Entrenamiento completado');
        console.log(`   Loss Total: ${response.data.loss.toFixed(6)}`);
        console.log(`   Epoch: ${response.data.epoch}`);
        return response.data;
      } else {
        console.log(`❌ Error: ${response.data.error}`);
        return null;
      }
    } catch (error: any) {
      console.log(`❌ Error: ${error.message}`);
      return null;
    }
  }

  async obtenerEstado(): Promise<any> {
    try {
      const response = await this.cliente.get('/status');
      if (response.status === 200) {
        return response.data;
      }
      return null;
    } catch (error) {
      return null;
    }
  }

  async obtenerMetricas(): Promise<any> {
    try {
      const response = await this.cliente.get('/metricas');
      if (response.status === 200) {
        return response.data;
      }
      return null;
    } catch (error) {
      return null;
    }
  }

  /**
   * Generador avanzado: Datos sintéticos realistas con patrones de anomalías
   * Basado en GeneradorSintetico (patrones neuronales)
   */
  generarDatosRealisticos(cantidad: number = 10, porcentajeAnomalias: number = 20): MuestraEntrenamiento[] {
    const muestras: MuestraEntrenamiento[] = [];
    const rng = this.seededRandom(42);

    for (let muestra = 0; muestra < cantidad; muestra++) {
      const esAnomalia = Math.random() < porcentajeAnomalias / 100;
      const datos = new Array(1600).fill(0);

      if (esAnomalia) {
        // PATRÓN DE ANOMALÍA: Activaciones neuronales anómalas
        const numNeuronasActivas = 100;
        
        for (let j = 0; j < numNeuronasActivas; j++) {
          const indiceNeurona = Math.floor(rng() * 1600);
          const amplitud = 1.5 + rng() * 0.5;  // Amplitud anómala
          const ancho = 20;

          // Pico gaussiano (activación neuronal anómala)
          for (let k = indiceNeurona; k < Math.min(indiceNeurona + ancho, 1600); k++) {
            const distancia = k - indiceNeurona;
            const gaussiana = Math.exp(-((distancia - ancho / 2) ** 2) / (2 * 5 ** 2));
            datos[k] += amplitud * gaussiana;
          }
        }
      } else {
        // PATRÓN NOMINAL: Activaciones neuronales normales
        const numNeuronasActivas = 50;
        
        for (let j = 0; j < numNeuronasActivas; j++) {
          const indiceNeurona = Math.floor(rng() * 1600);
          const amplitud = 0.5 + rng() * 0.3;  // Amplitud normal
          const ancho = 10;

          // Pico gaussiano suave
          for (let k = indiceNeurona; k < Math.min(indiceNeurona + ancho, 1600); k++) {
            const distancia = k - indiceNeurona;
            const gaussiana = Math.exp(-((distancia - ancho / 2) ** 2) / (2 * 5 ** 2));
            datos[k] += amplitud * gaussiana;
          }
        }
      }

      // Normalizar valores
      const max = Math.max(...datos);
      if (max > 0) {
        for (let j = 0; j < datos.length; j++) {
          datos[j] /= max;
          datos[j] = Math.max(-1, Math.min(1, datos[j]));  // Clamp a [-1, 1]
        }
      }

      muestras.push({
        input_data: datos,
        anomaly_label: esAnomalia ? 1 : 0
      });
    }

    return muestras;
  }

  /**
   * Random seeded para reproducibilidad
   */
  private seededRandom(seed: number): () => number {
    let s = seed;
    return () => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    };
  }

  /**
   * Generador simple (alternativa rápida)
   */
  generarDatosPrueba(cantidad: number = 10): MuestraEntrenamiento[] {
    const muestras: MuestraEntrenamiento[] = [];
    for (let i = 0; i < cantidad; i++) {
      const input = Array(1600).fill(0).map(() => Math.random() * 2 - 1);
      const label = Math.random() < 0.2 ? 1 : 0;
      muestras.push({
        input_data: input,
        anomaly_label: label as 0 | 1
      });
    }
    return muestras;
  }
}

// Script de prueba
async function main() {
  console.log('\n' + '='.repeat(70));
  console.log('⚛️  CLIENTE ART V7 - REACTOR NEURO-SIMBÓLICO');
  console.log('='.repeat(70));

  const cliente = new ClienteART_V7(CONFIG);

  const conectado = await cliente.conectar();
  if (!conectado) {
    console.log('\n❌ No se pudo conectar al Reactor');
    console.log('Asegúrate de que Docker está corriendo:');
    console.log('  ./scripts/run_docker_training.sh');
    return;
  }

  // Obtener estado
  console.log('\n📊 Estado del Reactor:');
  const estado = await cliente.obtenerEstado();
  if (estado) {
    console.log(`   Reactor: ${estado.reactor}`);
    console.log(`   Modo: ${estado.modo}`);
    console.log(`   Epoch: ${estado.estadisticas.epoch}`);
    console.log(`   Loss Promedio: ${estado.estadisticas.loss_promedio.toFixed(6)}`);
    console.log(`   Memoria: ${estado.estadisticas.memoria_mb.toFixed(1)} MB`);
  }

  // Entrenar con datos de prueba
  console.log('\n🧬 Generando datos de prueba...');
  const datos = cliente.generarDatosPrueba(8);

  console.log('\n⚛️  Ejecutando entrenamiento en Reactor...');
  for (let i = 0; i < 5; i++) {
    const resultado = await cliente.entrenar(datos, 1);
    if (resultado) {
      console.log(`\n[Iteración ${i + 1}]`);
      console.log(`   Epoch: ${resultado.epoch}`);
      console.log(`   Loss: ${resultado.loss.toFixed(6)}`);
    }
    // Pequeña pausa entre iteraciones
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  // Métricas finales
  console.log('\n📈 Métricas Finales:');
  const metricas = await cliente.obtenerMetricas();
  if (metricas) {
    console.log(`   Historico Loss (últimos 20):`, metricas.loss_history.slice(-20).map((v: number) => v.toFixed(4)));
    console.log(`   Memoria Usada: ${metricas.memoria_mb.toFixed(1)} MB`);
  }

  console.log('\n' + '='.repeat(70));
  console.log('✅ Prueba completada');
  console.log('='.repeat(70));
}

if (require.main === module) {
  main().catch(console.error);
}

