import { Hipergrafo, Nodo } from '../core';

/**
 * Métricas de Centralidad para Hipergrafos
 * 
 * Miden la importancia de nodos en la red
 */
export class CentralidadHipergrafo {
  /**
   * Centralidad de Grado: cantidad de hiperedges que contienen al nodo
   * Normalizado entre 0 y 1
   */
  static centralidadGrado(hipergrafo: Hipergrafo, nodoId: string): number {
    const grado = hipergrafo.calcularGradoNodo(nodoId);
    const gradoMaximoPosible = hipergrafo.cardinalE();
    
    if (gradoMaximoPosible === 0) return 0;
    return grado / gradoMaximoPosible;
  }

  /**
   * Centralidad Ponderada: suma de pesos de hiperedges
   */
  static centralidadPonderada(hipergrafo: Hipergrafo, nodoId: string): number {
    const hiperedges = hipergrafo.obtenerHiperedgesDelNodo(nodoId);
    return hiperedges.reduce((sum, edge) => sum + edge.weight, 0);
  }

  /**
   * Centralidad de Betweenness para hipergrafos:
   * Mide cuántas rutas más cortas pasan a través de un nodo
   * (Aproximación: basada en pares de hiperedges conectadas)
   */
  static centralidadBetweenness(hipergrafo: Hipergrafo, nodoId: string): number {
    const nodo = hipergrafo.obtenerNodo(nodoId);
    if (!nodo) return 0;

    const hiperedgesDelNodo = hipergrafo.obtenerHiperedgesDelNodo(nodoId);
    let betweenness = 0;

    // Para cada par de hiperedges que contienen al nodo
    for (let i = 0; i < hiperedgesDelNodo.length; i++) {
      for (let j = i + 1; j < hiperedgesDelNodo.length; j++) {
        const edge1 = hiperedgesDelNodo[i];
        const edge2 = hiperedgesDelNodo[j];

        // Contar nodos que son reachables solo a través del nodo
        const nodosEdge1 = Array.from(edge1.nodos);
        const nodosEdge2 = Array.from(edge2.nodos);

        const nodosComunes = nodosEdge1.filter(n => nodosEdge2.includes(n));
        betweenness += Math.max(0, Math.max(nodosEdge1.length, nodosEdge2.length) - nodosComunes.length);
      }
    }

    return betweenness;
  }

  /**
   * Centralidad de Cercanía (Closeness):
   * Inversa de la distancia promedio a otros nodos
   * Para hipergrafos, usa distancia en el grafo inducido
   */
  static centralidadCercanía(hipergrafo: Hipergrafo, nodoId: string): number {
    const nodo = hipergrafo.obtenerNodo(nodoId);
    if (!nodo) return 0;

    const distancias = this.calcularDistanciasDesde(hipergrafo, nodoId);

    let sumaDistancias = 0;
    let nodosAlcanzables = 0;

    distancias.forEach((distancia, nodeId) => {
      if (nodeId !== nodoId && distancia !== Infinity) {
        sumaDistancias += distancia;
        nodosAlcanzables++;
      }
    });

    if (nodosAlcanzables === 0) return 0;
    return nodosAlcanzables / sumaDistancias;
  }

  /**
   * Calcula distancias desde un nodo a todos los demás (BFS)
   */
  private static calcularDistanciasDesde(hipergrafo: Hipergrafo, nodoInicio: string): Map<string, number> {
    const distancias = new Map<string, number>();
    const cola: { nodoId: string; distancia: number }[] = [];

    // Inicializar
    hipergrafo.obtenerNodos().forEach(n => {
      distancias.set(n.id, Infinity);
    });
    distancias.set(nodoInicio, 0);
    cola.push({ nodoId: nodoInicio, distancia: 0 });

    // BFS
    while (cola.length > 0) {
      const { nodoId, distancia } = cola.shift()!;
      const vecinos = hipergrafo.obtenerVecinos(nodoId);

      vecinos.forEach(vecino => {
        const distanciaVecino = distancias.get(vecino.id) || Infinity;
        if (distancia + 1 < distanciaVecino) {
          distancias.set(vecino.id, distancia + 1);
          cola.push({ nodoId: vecino.id, distancia: distancia + 1 });
        }
      });
    }

    return distancias;
  }

  /**
   * Centralidad de Eigenvector: recursivo, basado en la importancia de vecinos
   * (Aproximación iterativa)
   */
  static centralidadEigenvector(hipergrafo: Hipergrafo, maxIteraciones: number = 10): Map<string, number> {
    const nodos = hipergrafo.obtenerNodos();
    const centralidades = new Map<string, number>();

    // Inicializar uniformemente
    nodos.forEach(n => centralidades.set(n.id, 1 / nodos.length));

    // Iteración de potencia
    for (let iter = 0; iter < maxIteraciones; iter++) {
      const nuevasCentralidades = new Map<string, number>();

      nodos.forEach(nodo => {
        const vecinos = hipergrafo.obtenerVecinos(nodo.id);
        const suma = vecinos.reduce((sum, vecino) => {
          return sum + (centralidades.get(vecino.id) || 0);
        }, 0);

        nuevasCentralidades.set(nodo.id, suma);
      });

      // Normalizar
      const max = Math.max(...Array.from(nuevasCentralidades.values()));
      if (max > 0) {
        nuevasCentralidades.forEach((_, nodoId) => {
          nuevasCentralidades.set(nodoId, (nuevasCentralidades.get(nodoId) || 0) / max);
        });
      }

      Object.assign(centralidades, nuevasCentralidades);
    }

    return centralidades;
  }

  /**
   * Ranking de nodos por centralidad de grado
   */
  static rankingPorCentralidad(hipergrafo: Hipergrafo, tipo: 'grado' | 'ponderada' | 'betweenness' = 'grado'): Array<{ nodo: Nodo; centralidad: number }> {
    const nodos = hipergrafo.obtenerNodos();
    const medidas = nodos.map(nodo => {
      let centralidad: number;

      switch (tipo) {
        case 'ponderada':
          centralidad = this.centralidadPonderada(hipergrafo, nodo.id);
          break;
        case 'betweenness':
          centralidad = this.centralidadBetweenness(hipergrafo, nodo.id);
          break;
        case 'grado':
        default:
          centralidad = this.centralidadGrado(hipergrafo, nodo.id);
      }

      return { nodo, centralidad };
    });

    return medidas.sort((a, b) => b.centralidad - a.centralidad);
  }
}
