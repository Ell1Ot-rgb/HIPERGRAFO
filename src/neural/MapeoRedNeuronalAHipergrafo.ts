import { Hipergrafo, Nodo, Hiperedge } from '../core';
import { RedNeuronal, ConfiguracionMapeo, CONFIGURACION_MAPEO_DEFAULT } from './tipos';

/**
 * Adaptador para mapear redes neuronales a hipergrafos con rigor teórico
 * 
 * Estrategia de mapeo:
 * 1. Cada neurona → nodo en el hipergrafo
 * 2. Conexiones ponderadas → hiperedges (pueden conectar múltiples neuronas)
 * 3. Capas neuronales → hiperedges especializadas (opcional)
 * 4. Patrones de activación → hiperedges detectadas por análisis
 */
export class MapeoRedNeuronalAHipergrafo {
  private config: ConfiguracionMapeo;

  constructor(config: Partial<ConfiguracionMapeo> = {}) {
    this.config = { ...CONFIGURACION_MAPEO_DEFAULT, ...config };
  }

  /**
   * Mapea una red neuronal a un hipergrafo
   */
  mapear(redNeuronal: RedNeuronal): Hipergrafo {
    const hipergrafo = new Hipergrafo(`Hipergrafo de Red Neuronal (${redNeuronal.neuronas.length} neuronas)`);

    // Paso 1: Crear nodos para cada neurona
    const nodosNeuronales = this.crearNodosNeuronas(redNeuronal);
    hipergrafo.agregarNodos(nodosNeuronales);

    // Paso 2: Crear hiperedges de conexiones ponderadas significativas
    this.crearHiperedgesDeConexiones(redNeuronal, hipergrafo, nodosNeuronales);

    // Paso 3: Agrupar por capas si está configurado
    if (this.config.agruparPorCapas) {
      this.crearHiperedgesCapas(redNeuronal, hipergrafo, nodosNeuronales);
    }

    // Paso 4: Detectar y crear hiperedges de patrones de activación
    if (this.config.detectarPatrones) {
      this.crearHiperedgesPatrones(redNeuronal, hipergrafo, nodosNeuronales);
    }

    return hipergrafo;
  }

  /**
   * Crea un nodo para cada neurona
   */
  private crearNodosNeuronas(redNeuronal: RedNeuronal): Nodo[] {
    return redNeuronal.neuronas.map(neurona => {
      const nodo = new Nodo(`Neurona_${neurona.id}`, {
        idNeurona: neurona.id,
        activacion: neurona.activacion,
        sesgo: neurona.sesgo,
        activa: neurona.activacion >= this.config.umbralActivacion,
        ...neurona.metadata
      });
      return nodo;
    });
  }

  /**
   * Crea hiperedges basadas en conexiones ponderadas significativas
   */
  private crearHiperedgesDeConexiones(
    redNeuronal: RedNeuronal,
    hipergrafo: Hipergrafo,
    nodosNeuronales: Nodo[]
  ): void {
    const mapeoIdNodo = new Map<number, Nodo>();
    redNeuronal.neuronas.forEach((neurona, idx) => {
      mapeoIdNodo.set(neurona.id, nodosNeuronales[idx]);
    });

    // Procesar capas de conexiones
    redNeuronal.pesos.forEach((capaCanal, capaIdx) => {
      capaCanal.forEach((conexionesNeuron, deIdx) => {
        conexionesNeuron.forEach((peso, aIdx) => {
          // Si el peso supera el umbral, crear una hiperedge
          if (Math.abs(peso) >= this.config.umbralPeso) {
            const nodoOrigen = redNeuronal.neuronas[deIdx];
            const nodoDestino = redNeuronal.neuronas[aIdx];
            
            const nodosEdge = [
              mapeoIdNodo.get(nodoOrigen.id),
              mapeoIdNodo.get(nodoDestino.id)
            ].filter((n): n is Nodo => n !== undefined);

            if (nodosEdge.length === 2) {
              const hiperedge = new Hiperedge(
                `Conexion_${capaIdx}_${deIdx}_${aIdx}`,
                nodosEdge,
                Math.abs(peso), // Peso absoluto como peso de la hiperedge
                {
                  capaIdx,
                  deIdx,
                  aIdx,
                  pesoOriginal: peso,
                  incluirPesos: this.config.incluirPesos
                }
              );
              hipergrafo.agregarHiperedge(hiperedge);
            }
          }
        });
      });
    });
  }

  /**
   * Crea hiperedges que agrupan neuronas por capa
   */
  private crearHiperedgesCapas(
    redNeuronal: RedNeuronal,
    hipergrafo: Hipergrafo,
    nodosNeuronales: Nodo[]
  ): void {
    let indiceNeurona = 0;
    
    redNeuronal.capas.forEach((tamanoCapas, capaIdx) => {
      const nodosCapas = [];
      
      for (let i = 0; i < tamanoCapas && indiceNeurona < nodosNeuronales.length; i++) {
        nodosCapas.push(nodosNeuronales[indiceNeurona]);
        indiceNeurona++;
      }

      if (nodosCapas.length > 0) {
        const hiperedge = new Hiperedge(
          `Capa_${capaIdx}`,
          nodosCapas,
          1, // Peso uniforme para capas
          {
            tipoHiperedge: 'capa',
            capaIdx,
            tamano: nodosCapas.length
          }
        );
        hipergrafo.agregarHiperedge(hiperedge);
      }
    });
  }

  /**
   * Detecta patrones de activación similar y crea hiperedges
   */
  private crearHiperedgesPatrones(
    redNeuronal: RedNeuronal,
    hipergrafo: Hipergrafo,
    nodosNeuronales: Nodo[]
  ): void {
    // Agrupar neuronas por nivel de activación similar
    const grupos = this.agruparPorActivacion(redNeuronal.neuronas);

    grupos.forEach((activaciones, nivelActivacion) => {
      if (activaciones.length >= this.config.tamanoMinimoPatron) {
        const nodosPatron = activaciones
          .map(neurona => {
            const idx = redNeuronal.neuronas.findIndex(n => n.id === neurona.id);
            return idx !== -1 ? nodosNeuronales[idx] : undefined;
          })
          .filter((n): n is Nodo => n !== undefined);

        if (nodosPatron.length >= this.config.tamanoMinimoPatron) {
          const hiperedge = new Hiperedge(
            `Patron_Activacion_${nivelActivacion}`,
            nodosPatron,
            nodosPatron.length, // Usar cantidad de nodos como peso
            {
              tipoHiperedge: 'patron_activacion',
              nivelActivacion,
              cantidadNeuronas: nodosPatron.length
            }
          );
          hipergrafo.agregarHiperedge(hiperedge);
        }
      }
    });
  }

  /**
   * Agrupa neuronas por su nivel de activación
   */
  private agruparPorActivacion(neuronas: any[]): Map<string, any[]> {
    const grupos = new Map<string, any[]>();

    // Dividir activación en bandas
    const bandas = ['inactiva', 'baja', 'media', 'alta', 'muy_alta'];
    
    neuronas.forEach(neurona => {
      let banda: string;
      if (neurona.activacion < 0.2) banda = bandas[0];
      else if (neurona.activacion < 0.4) banda = bandas[1];
      else if (neurona.activacion < 0.6) banda = bandas[2];
      else if (neurona.activacion < 0.8) banda = bandas[3];
      else banda = bandas[4];

      if (!grupos.has(banda)) {
        grupos.set(banda, []);
      }
      grupos.get(banda)!.push(neurona);
    });

    return grupos;
  }

  /**
   * Actualiza configuración
   */
  actualizarConfiguracion(config: Partial<ConfiguracionMapeo>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Obtiene la configuración actual
   */
  obtenerConfiguracion(): ConfiguracionMapeo {
    return { ...this.config };
  }
}
