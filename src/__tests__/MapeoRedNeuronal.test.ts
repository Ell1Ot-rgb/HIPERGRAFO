import { MapeoRedNeuronalAHipergrafo } from '../neural';
import { RedNeuronal, Neurona } from '../neural/tipos';

describe('Mapeo de Red Neuronal a Hipergrafo', () => {
  let mapeador: MapeoRedNeuronalAHipergrafo;
  let redNeuronal: RedNeuronal;

  beforeEach(() => {
    mapeador = new MapeoRedNeuronalAHipergrafo();

    // Crear red neuronal simple para pruebas
    const neuronas: Neurona[] = [
      { id: 0, activacion: 0.9, sesgo: 0.1 },
      { id: 1, activacion: 0.7, sesgo: 0.2 },
      { id: 2, activacion: 0.3, sesgo: 0.15 },
      { id: 3, activacion: 0.5, sesgo: 0.25 }
    ];

    // Crear matriz de pesos (3 capas, 4 neuronas por capa)
    const pesos: number[][][] = [
      [
        [0.5, 0.3, 0.1, 0.2],
        [0.4, 0.6, 0.2, 0.3],
        [0.2, 0.1, 0.7, 0.4],
        [0.3, 0.4, 0.3, 0.8]
      ]
    ];

    redNeuronal = {
      neuronas,
      pesos,
      capas: [4, 4, 4],
      umbralActivacion: 0.5
    };
  });

  test('Mapear red neuronal a hipergrafo', () => {
    const hipergrafo = mapeador.mapear(redNeuronal);
    expect(hipergrafo.cardinalV()).toBe(4); // 4 neuronas
  });

  test('Hipergrafo contiene nodos de todas las neuronas', () => {
    const hipergrafo = mapeador.mapear(redNeuronal);
    const nodos = hipergrafo.obtenerNodos();
    expect(nodos.length).toBe(4);
  });

  test('Crear hiperedges de capas', () => {
    const config = { agruparPorCapas: true };
    const mapeadorConfigurable = new MapeoRedNeuronalAHipergrafo(config);
    
    const hipergrafo = mapeadorConfigurable.mapear(redNeuronal);
    const hiperedges = hipergrafo.obtenerHiperedges();
    
    // Debería tener hiperedges de capas
    const hiperedgesCapas = hiperedges.filter(e => e.metadata?.tipoHiperedge === 'capa');
    expect(hiperedgesCapas.length).toBeGreaterThan(0);
  });

  test('Detectar patrones de activación', () => {
    const config = { detectarPatrones: true, tamanoMinimoPatron: 1 };
    const mapeadorConfigurable = new MapeoRedNeuronalAHipergrafo(config);
    
    const hipergrafo = mapeadorConfigurable.mapear(redNeuronal);
    const hiperedges = hipergrafo.obtenerHiperedges();
    
    // Debería detectar patrones
    const hiperedgesPatrones = hiperedges.filter(e => e.metadata?.tipoHiperedge === 'patron_activacion');
    expect(hiperedgesPatrones.length).toBeGreaterThan(0);
  });

  test('Actualizar configuración', () => {
    const configInicial = mapeador.obtenerConfiguracion();
    expect(configInicial.umbralActivacion).toBe(0.5);

    mapeador.actualizarConfiguracion({ umbralActivacion: 0.7 });
    const configActualizada = mapeador.obtenerConfiguracion();
    expect(configActualizada.umbralActivacion).toBe(0.7);
  });
});
