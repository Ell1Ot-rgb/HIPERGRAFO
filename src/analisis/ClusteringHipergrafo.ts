import { Hipergrafo } from '../core';

/**
 * Coeficiente de Clustering para Hipergrafos
 * 
 * En grafos tradicionales, mide la fracción de triángulos
 * En hipergrafos, generalizamos este concepto a hipersimplices
 */
export class ClusteringHipergrafo {
  /**
   * Coeficiente de Clustering Local para un nodo
   * Mide qué tan densamente conectados están los vecinos
   */
  static coeficienteClusteringLocal(hipergrafo: Hipergrafo, nodoId: string): number {
    const nodo = hipergrafo.obtenerNodo(nodoId);
    if (!nodo) return 0;

    const vecinos = hipergrafo.obtenerVecinos(nodoId);
    if (vecinos.length < 2) return 0;

    // Contar conexiones entre vecinos
    let conexionesEntre = 0;
    for (let i = 0; i < vecinos.length; i++) {
      for (let j = i + 1; j < vecinos.length; j++) {
        if (hipergrafo.estaConectados(vecinos[i].id, vecinos[j].id)) {
          conexionesEntre++;
        }
      }
    }

    // Máximo posible de conexiones
    const maxConexiones = (vecinos.length * (vecinos.length - 1)) / 2;
    
    if (maxConexiones === 0) return 0;
    return conexionesEntre / maxConexiones;
  }

  /**
   * Coeficiente de Clustering Global (Transitivity)
   * Proporción de triángulos cerrados vs. potenciales
   */
  static coeficienteClusteringGlobal(hipergrafo: Hipergrafo): number {
    const nodos = hipergrafo.obtenerNodos();
    
    if (nodos.length < 3) return 0;

    let triangulosCerrados = 0;
    let triangulosPotenciales = 0;

    // Para cada triple de nodos
    for (let i = 0; i < nodos.length; i++) {
      for (let j = i + 1; j < nodos.length; j++) {
        for (let k = j + 1; k < nodos.length; k++) {
          const n1 = nodos[i];
          const n2 = nodos[j];
          const n3 = nodos[k];

          const conectado12 = hipergrafo.estaConectados(n1.id, n2.id);
          const conectado23 = hipergrafo.estaConectados(n2.id, n3.id);
          const conectado31 = hipergrafo.estaConectados(n3.id, n1.id);

          // Si al menos dos pares están conectados
          if ((conectado12 && conectado23) || (conectado23 && conectado31) || (conectado31 && conectado12)) {
            triangulosPotenciales++;

            // Si todos tres están conectados
            if (conectado12 && conectado23 && conectado31) {
              triangulosCerrados++;
            }
          }
        }
      }
    }

    if (triangulosPotenciales === 0) return 0;
    return triangulosCerrados / triangulosPotenciales;
  }

  /**
   * Coeficiente de Clustering Promedio
   */
  static coeficienteClusteringPromedio(hipergrafo: Hipergrafo): number {
    const nodos = hipergrafo.obtenerNodos();
    
    if (nodos.length === 0) return 0;

    const coeficientes = nodos.map(nodo => 
      this.coeficienteClusteringLocal(hipergrafo, nodo.id)
    );

    const suma = coeficientes.reduce((a, b) => a + b, 0);
    return suma / nodos.length;
  }

  /**
   * Índice de Homofilia del Hipergrafo
   * Mide tendencia de nodos similares a conectarse (para hipergrafos con atributos)
   */
  static indiceHomofilia(
    hipergrafo: Hipergrafo,
    atributoKey: string
  ): number {
    const nodos = hipergrafo.obtenerNodos();
    let conexionesHomogéneas = 0;
    let conxexionesTotales = 0;

    const pares = new Set<string>();

    nodos.forEach(n1 => {
      hipergrafo.obtenerVecinos(n1.id).forEach(n2 => {
        const pareja = [n1.id, n2.id].sort().join('-');
        if (!pares.has(pareja)) {
          pares.add(pareja);
          conxexionesTotales++;

          const valor1 = n1.metadata[atributoKey];
          const valor2 = n2.metadata[atributoKey];

          if (valor1 === valor2) {
            conexionesHomogéneas++;
          }
        }
      });
    });

    if (conxexionesTotales === 0) return 0;
    return conexionesHomogéneas / conxexionesTotales;
  }

  /**
   * Modularidad del Hipergrafo
   * Mide la solidez de la estructura de comunidades
   */
  static calcularModularidad(
    hipergrafo: Hipergrafo,
    particion: Map<string, number>  // Mapeo de nodoId a comunidadId
  ): number {
    const nodos = hipergrafo.obtenerNodos();
    const m = hipergrafo.cardinalE(); // Número de hiperedges

    if (m === 0) return 0;

    let q = 0;

    nodos.forEach(nodo1 => {
      hipergrafo.obtenerVecinos(nodo1.id).forEach(nodo2 => {
        const comunidad1 = particion.get(nodo1.id) ?? -1;
        const comunidad2 = particion.get(nodo2.id) ?? -1;

        if (comunidad1 === comunidad2 && comunidad1 !== -1) {
          const grado1 = hipergrafo.calcularGradoNodo(nodo1.id);
          const grado2 = hipergrafo.calcularGradoNodo(nodo2.id);

          // Término observado - término esperado
          q += (1 - (grado1 * grado2) / (2 * m));
        }
      });
    });

    return q / (2 * nodos.length);
  }
}
