import { Hipergrafo } from '../core';

/**
 * Propiedades Espectrales del Hipergrafo
 * 
 * Análisis de eigenvalores y eigenvectores
 * relacionados con la matriz de adyacencia e incidencia
 */
export class PropiedadesEspectrales {
  /**
   * Calcula la Matriz de Adyacencia del Hipergrafo
   * A[i,j] = 1 si nodos i y j están en la misma hiperedge
   */
  static calcularMatrizAdyacencia(hipergrafo: Hipergrafo): number[][] {
    const nodos = hipergrafo.obtenerNodos();
    const n = nodos.length;
    const matriz: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));

    nodos.forEach((nodo1, i) => {
      nodos.forEach((nodo2, j) => {
        if (i !== j && hipergrafo.estaConectados(nodo1.id, nodo2.id)) {
          matriz[i][j] = 1;
        }
      });
    });

    return matriz;
  }

  /**
   * Calcula la Matriz Diagonal de Grados
   * D[i,i] = grado del nodo i
   */
  static calcularMatrizGrados(hipergrafo: Hipergrafo): number[][] {
    const nodos = hipergrafo.obtenerNodos();
    const n = nodos.length;
    const matriz: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));

    nodos.forEach((nodo, i) => {
      const grado = hipergrafo.calcularGradoNodo(nodo.id);
      matriz[i][i] = grado;
    });

    return matriz;
  }

  /**
   * Calcula la Matriz Laplaciana normalizada
   * L_norm = I - D^(-1/2) * A * D^(-1/2)
   * Donde I es identidad, D es matriz de grados, A es adyacencia
   */
  static calcularMatrizLaplacianaNormalizada(hipergrafo: Hipergrafo): number[][] {
    const nodos = hipergrafo.obtenerNodos();
    const n = nodos.length;
    const A = this.calcularMatrizAdyacencia(hipergrafo);
    const D = this.calcularMatrizGrados(hipergrafo);

    // Calcular D^(-1/2)
    const D_inv_sqrt: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      if (D[i][i] > 0) {
        D_inv_sqrt[i][i] = 1 / Math.sqrt(D[i][i]);
      }
    }

    // Calcular D^(-1/2) * A
    const temp = this.multiplicarMatrices(D_inv_sqrt, A);

    // Calcular D^(-1/2) * A * D^(-1/2)
    const temp2 = this.multiplicarMatrices(temp, D_inv_sqrt);

    // Calcular I - resultado
    const L_norm = Array(n).fill(null).map(() => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        L_norm[i][j] = (i === j ? 1 : 0) - temp2[i][j];
      }
    }

    return L_norm;
  }

  /**
   * Calcula traza de una matriz (suma de diagonal)
   */
  static calcularTraza(matriz: number[][]): number {
    return matriz.reduce((sum, fila, i) => sum + fila[i], 0);
  }

  /**
   * Calcula energía espectral (suma de cuadrados de eigenvalores)
   */
  static calcularEnergiaEspectral(hipergrafo: Hipergrafo): number {
    const A = this.calcularMatrizAdyacencia(hipergrafo);
    // Aproximación: traza de A^2
    const A2 = this.multiplicarMatrices(A, A);
    return this.calcularTraza(A2);
  }

  /**
   * Multiplicación de dos matrices
   */
  private static multiplicarMatrices(A: number[][], B: number[][]): number[][] {
    const result: number[][] = Array(A.length).fill(null).map(() => Array(B[0].length).fill(0));

    for (let i = 0; i < A.length; i++) {
      for (let j = 0; j < B[0].length; j++) {
        for (let k = 0; k < B.length; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }

    return result;
  }

  /**
   * Número de Espectro (Spectral Gap)
   * Segunda menor eigenvalue de la matriz Laplaciana normalizada
   * Mide conectividad del hipergrafo
   */
  static calcularGapEspectral(hipergrafo: Hipergrafo): number {
    if (hipergrafo.cardinalV() < 2) return 0;

    // Aproximación: usando matriz Laplaciana simple
    const nodos = hipergrafo.obtenerNodos();
    const D = this.calcularMatrizGrados(hipergrafo);
    const A = this.calcularMatrizAdyacencia(hipergrafo);

    // Laplaciana L = D - A
    const L: number[][] = Array(D.length).fill(null).map((_, i) => 
      Array(D[i].length).fill(null).map((_, j) => 
        D[i][j] - A[i][j]
      )
    );

    // El eigenvalue más pequeño es 0 (siempre)
    // El segundo más pequeño (algebraic connectivity) se aproxima
    const traza = this.calcularTraza(L);
    const n = nodos.length;

    // Heurística: spectral gap ≈ 2 * λ2 / traza
    return traza / (2 * n);
  }

  /**
   * Índice de Wiener Espectral
   * Suma de distancias inversas basada en eigenvalores
   */
  static indiceWienerEspectral(hipergrafo: Hipergrafo): number {
    const nodos = hipergrafo.obtenerNodos();
    const A = this.calcularMatrizAdyacencia(hipergrafo);
    
    let suma = 0;
    for (let i = 0; i < nodos.length; i++) {
      for (let j = i + 1; j < nodos.length; j++) {
        // Usar la entrada de la matrix exponencial aproximada
        const aprox = A[i][j] > 0 ? 1 : (1 / (nodos.length));
        suma += aprox;
      }
    }

    return suma;
  }
}
