export * from './core';
export * from './neural';
export * from './persistencia';
export * from './analisis';
export * from './zx';
export * from './omega21';
export * from './hardware';
export * from './control';
export * from './orquestador';

/**
 * HIPERGRAFO - Sistema de Mapeo de Redes Hipergráficas + Omega 21
 * 
 * Este módulo proporciona herramientas para:
 * 1. Crear y manipular hipergrafos teóricamente rigurosos
 * 2. Mapear redes neuronales a hipergrafos
 * 3. Persistir hipergrafos de forma segura
 * 4. Analizar propiedades avanzadas (dualidad, centralidad, clustering)
 * 
 * Uso básico:
 * ```typescript
 * import { Hipergrafo, Nodo, Hiperedge, MapeoRedNeuronalAHipergrafo } from 'hipergrafo';
 * 
 * // Crear hipergrafo
 * const hg = new Hipergrafo('Mi Hipergrafo');
 * 
 * // O mapear desde red neuronal
 * const mapeador = new MapeoRedNeuronalAHipergrafo();
 * const hipergrafo = mapeador.mapear(redNeuronal);
 * ```
 */
