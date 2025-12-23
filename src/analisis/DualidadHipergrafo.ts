import { Hipergrafo, Nodo, Hiperedge } from '../core';

/**
 * Módulo de Dualidad del Hipergrafo
 * 
 * La dualidad es una transformación fundamental en teoría de hipergrafos:
 * H* = (V*, E*) donde V* = E y E* = V
 * 
 * En otras palabras:
 * - Los nodos de H* corresponden a las hiperedges de H
 * - Las hiperedges de H* corresponden a los nodos de H
 */
export class DualidadHipergrafo {
  /**
   * Calcula el hipergrafo dual
   * H* = (V*, E*) donde V* corresponde a E y E* corresponde a V
   */
  static calcularDual(hipergrafo: Hipergrafo): Hipergrafo {
    const dual = new Hipergrafo(`${hipergrafo.label}*`);

    // Paso 1: Crear nodos en el dual a partir de hiperedges del original
    const mapeoEdgeANodoDual = new Map<string, Nodo>();
    
    hipergrafo.obtenerHiperedges().forEach(edge => {
      const nodoDual = new Nodo(
        `Edge_${edge.label}`,
        {
          edgeOriginalId: edge.id,
          edgeOriginalLabel: edge.label,
          tamanoOriginal: edge.grado(),
          pesoOriginal: edge.weight,
          ...edge.metadata
        }
      );
      mapeoEdgeANodoDual.set(edge.id, nodoDual);
      dual.agregarNodo(nodoDual);
    });

    // Paso 2: Crear hiperedges en el dual a partir de nodos del original
    hipergrafo.obtenerNodos().forEach(nodo => {
      const hiperedgesDelNodo = hipergrafo.obtenerHiperedgesDelNodo(nodo.id);
      
      if (hiperedgesDelNodo.length > 0) {
        const nodosDelDual = hiperedgesDelNodo
          .map(edge => mapeoEdgeANodoDual.get(edge.id))
          .filter((n): n is Nodo => n !== undefined);

        if (nodosDelDual.length > 0) {
          const edgeDual = new Hiperedge(
            `Node_${nodo.label}`,
            nodosDelDual,
            1.0, // Peso uniforme
            {
              nodoOriginalId: nodo.id,
              nodoOriginalLabel: nodo.label,
              gradoOriginal: hiperedgesDelNodo.length,
              ...nodo.metadata
            }
          );
          dual.agregarHiperedge(edgeDual);
        }
      }
    });

    return dual;
  }

  /**
   * Verifica si un hipergrafo es autodual (H = H*)
   * Esto requiere |V| = |E| y correspondencia entre estructuras
   */
  static esAutodual(hipergrafo: Hipergrafo): boolean {
    if (hipergrafo.cardinalV() !== hipergrafo.cardinalE()) {
      return false;
    }

    const dual = this.calcularDual(hipergrafo);
    
    // Comparar estructura básica
    if (dual.cardinalV() !== hipergrafo.cardinalV() ||
        dual.cardinalE() !== hipergrafo.cardinalE()) {
      return false;
    }

    // Comparar distribución de grados
    const gradosOriginal = hipergrafo.obtenerNodos()
      .map(n => hipergrafo.calcularGradoNodo(n.id))
      .sort((a, b) => a - b);

    const gradosDual = dual.obtenerNodos()
      .map(n => dual.calcularGradoNodo(n.id))
      .sort((a, b) => a - b);

    return gradosOriginal.every((g, i) => g === gradosDual[i]);
  }

  /**
   * Calcula el número de iteraciones hasta convergencia en la dualidad
   * Para hipergrafos finitos, aplicar dualidad N veces eventualmente retorna al original
   */
  static calcularPeriodoDualidad(hipergrafo: Hipergrafo, maxIteraciones: number = 10): number {
    const hashOriginal = this.calcularHashEstructura(hipergrafo);
    let actual = hipergrafo;

    for (let i = 1; i <= maxIteraciones; i++) {
      actual = this.calcularDual(actual);
      const hashActual = this.calcularHashEstructura(actual);
      
      if (hashActual === hashOriginal) {
        return i;
      }
    }

    return -1; // No converge en el límite de iteraciones
  }

  /**
   * Calcula un hash de la estructura para comparación
   */
  private static calcularHashEstructura(hipergrafo: Hipergrafo): string {
    const nodos = hipergrafo.cardinalV();
    const edges = hipergrafo.cardinalE();
    const gradoPromedio = hipergrafo.gradoPromedio();
    
    return `${nodos}-${edges}-${gradoPromedio.toFixed(3)}`;
  }
}
