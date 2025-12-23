/**
 * Representa una neurona en la red
 * Cada neurona será un nodo en el hipergrafo final
 */
export interface Neurona {
  id: number;
  activacion: number; // Valor entre 0 y 1
  sesgo: number;
  metadata?: Record<string, any>;
}

/**
 * Representa la estructura de una red neuronal
 * Para conversión a hipergrafo
 */
export interface RedNeuronal {
  neuronas: Neurona[];
  pesos: number[][][]; // pesos[capa][de][a]
  capas: number[];
  umbralActivacion?: number;
  metadata?: Record<string, any>;
}

/**
 * Configuración para el mapeo de red neuronal a hipergrafo
 */
export interface ConfiguracionMapeo {
  // Umbral de peso para considerar una conexión significativa
  umbralPeso: number;
  
  // Umbral de activación para considerar una neurona "activa"
  umbralActivacion: number;
  
  // Si true, agrupa neuronas por capa en hiperedges
  agruparPorCapas: boolean;
  
  // Si true, crea hiperedges para patrones de activación similares
  detectarPatrones: boolean;
  
  // Tamaño mínimo de un patrón para ser considerado
  tamanoMinimoPatron: number;
  
  // Incluir metadata sobre pesos en hiperedges
  incluirPesos: boolean;
}

/**
 * Configuración por defecto para mapeo
 */
export const CONFIGURACION_MAPEO_DEFAULT: ConfiguracionMapeo = {
  umbralPeso: 0.1,
  umbralActivacion: 0.5,
  agruparPorCapas: true,
  detectarPatrones: true,
  tamanoMinimoPatron: 3,
  incluirPesos: true
};
