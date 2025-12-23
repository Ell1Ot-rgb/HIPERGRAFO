import { Hipergrafo, Nodo, Hiperedge } from '../core';
import { DualidadHipergrafo, CentralidadHipergrafo, ClusteringHipergrafo, PropiedadesEspectrales } from '../analisis';

describe('Dualidad del Hipergrafo', () => {
  let hipergrafo: Hipergrafo;

  beforeEach(() => {
    hipergrafo = new Hipergrafo('Test Dualidad');

    // Crear estructura simple: 3 nodos, 2 hiperedges
    const n1 = new Nodo('N1');
    const n2 = new Nodo('N2');
    const n3 = new Nodo('N3');

    hipergrafo.agregarNodos([n1, n2, n3]);

    const e1 = new Hiperedge('E1', [n1, n2]);
    const e2 = new Hiperedge('E2', [n2, n3]);

    hipergrafo.agregarHiperedge(e1);
    hipergrafo.agregarHiperedge(e2);
  });

  test('Calcular dual básico', () => {
    const dual = DualidadHipergrafo.calcularDual(hipergrafo);

    expect(dual.cardinalV()).toBe(hipergrafo.cardinalE()); // |V*| = |E|
    expect(dual.cardinalE()).toBe(hipergrafo.cardinalV()); // |E*| = |V|
  });

  test('Doble dual retorna estructura similar', () => {
    const dual1 = DualidadHipergrafo.calcularDual(hipergrafo);
    const dual2 = DualidadHipergrafo.calcularDual(dual1);

    expect(dual2.cardinalV()).toBe(hipergrafo.cardinalV());
    expect(dual2.cardinalE()).toBe(hipergrafo.cardinalE());
  });

  test('Periodo de dualidad', () => {
    const periodo = DualidadHipergrafo.calcularPeriodoDualidad(hipergrafo);
    expect(periodo).toBeGreaterThan(0);
    expect(periodo).toBeLessThanOrEqual(10);
  });
});

describe('Centralidad del Hipergrafo', () => {
  let hipergrafo: Hipergrafo;

  beforeEach(() => {
    hipergrafo = new Hipergrafo('Test Centralidad');

    const nodos = Array.from({ length: 5 }, (_, i) => new Nodo(`N${i}`));
    hipergrafo.agregarNodos(nodos);

    // Crear hiperedges
    const e1 = new Hiperedge('E1', [nodos[0], nodos[1], nodos[2]], 1.0);
    const e2 = new Hiperedge('E2', [nodos[1], nodos[3]], 0.8);
    const e3 = new Hiperedge('E3', [nodos[2], nodos[3], nodos[4]], 1.2);

    hipergrafo.agregarHiperedge(e1);
    hipergrafo.agregarHiperedge(e2);
    hipergrafo.agregarHiperedge(e3);
  });

  test('Centralidad de grado', () => {
    const cent = CentralidadHipergrafo.centralidadGrado(hipergrafo, hipergrafo.obtenerNodos()[0].id);
    expect(cent).toBeGreaterThanOrEqual(0);
    expect(cent).toBeLessThanOrEqual(1);
  });

  test('Centralidad ponderada', () => {
    const cent = CentralidadHipergrafo.centralidadPonderada(hipergrafo, hipergrafo.obtenerNodos()[1].id);
    expect(cent).toBeGreaterThanOrEqual(0);
  });

  test('Centralidad de betweenness', () => {
    const cent = CentralidadHipergrafo.centralidadBetweenness(hipergrafo, hipergrafo.obtenerNodos()[1].id);
    expect(cent).toBeGreaterThanOrEqual(0);
  });

  test('Ranking de centralidad', () => {
    const ranking = CentralidadHipergrafo.rankingPorCentralidad(hipergrafo, 'grado');
    expect(ranking.length).toBe(5);
    expect(ranking[0].centralidad).toBeGreaterThanOrEqual(ranking[1].centralidad);
  });
});

describe('Clustering del Hipergrafo', () => {
  let hipergrafo: Hipergrafo;

  beforeEach(() => {
    hipergrafo = new Hipergrafo('Test Clustering');

    const nodos = Array.from({ length: 4 }, (_, i) => new Nodo(`N${i}`));
    hipergrafo.agregarNodos(nodos);

    // Crear triángulo + nodo aislado
    const e1 = new Hiperedge('E1', [nodos[0], nodos[1]]);
    const e2 = new Hiperedge('E2', [nodos[1], nodos[2]]);
    const e3 = new Hiperedge('E3', [nodos[0], nodos[2]]);

    hipergrafo.agregarHiperedge(e1);
    hipergrafo.agregarHiperedge(e2);
    hipergrafo.agregarHiperedge(e3);
  });

  test('Coeficiente de clustering local', () => {
    const coef = ClusteringHipergrafo.coeficienteClusteringLocal(hipergrafo, hipergrafo.obtenerNodos()[0].id);
    expect(coef).toBeGreaterThanOrEqual(0);
    expect(coef).toBeLessThanOrEqual(1);
  });

  test('Coeficiente de clustering global', () => {
    const coef = ClusteringHipergrafo.coeficienteClusteringGlobal(hipergrafo);
    expect(coef).toBeGreaterThanOrEqual(0);
    expect(coef).toBeLessThanOrEqual(1);
  });

  test('Coeficiente de clustering promedio', () => {
    const coef = ClusteringHipergrafo.coeficienteClusteringPromedio(hipergrafo);
    expect(coef).toBeGreaterThanOrEqual(0);
    expect(coef).toBeLessThanOrEqual(1);
  });

  test('Índice de homofilia', () => {
    const n1 = hipergrafo.obtenerNodos()[0];
    const n2 = hipergrafo.obtenerNodos()[1];
    
    n1.metadata.tipo = 'A';
    n2.metadata.tipo = 'A';

    const homofilia = ClusteringHipergrafo.indiceHomofilia(hipergrafo, 'tipo');
    expect(homofilia).toBeGreaterThanOrEqual(0);
    expect(homofilia).toBeLessThanOrEqual(1);
  });
});

describe('Propiedades Espectrales', () => {
  let hipergrafo: Hipergrafo;

  beforeEach(() => {
    hipergrafo = new Hipergrafo('Test Espectral');

    const nodos = Array.from({ length: 3 }, (_, i) => new Nodo(`N${i}`));
    hipergrafo.agregarNodos(nodos);

    const e1 = new Hiperedge('E1', [nodos[0], nodos[1]]);
    const e2 = new Hiperedge('E2', [nodos[1], nodos[2]]);

    hipergrafo.agregarHiperedge(e1);
    hipergrafo.agregarHiperedge(e2);
  });

  test('Matriz de adyacencia', () => {
    const A = PropiedadesEspectrales.calcularMatrizAdyacencia(hipergrafo);
    expect(A.length).toBe(3);
    expect(A[0].length).toBe(3);
    expect(A[0][0]).toBe(0); // Diagonal es 0
  });

  test('Matriz de grados', () => {
    const D = PropiedadesEspectrales.calcularMatrizGrados(hipergrafo);
    expect(D.length).toBe(3);
    expect(D[0][0]).toBeGreaterThan(0); // Nodos tienen grado > 0
  });

  test('Matriz Laplaciana normalizada', () => {
    const L = PropiedadesEspectrales.calcularMatrizLaplacianaNormalizada(hipergrafo);
    expect(L.length).toBe(3);
    expect(L[0].length).toBe(3);
  });

  test('Energía espectral', () => {
    const energia = PropiedadesEspectrales.calcularEnergiaEspectral(hipergrafo);
    expect(energia).toBeGreaterThanOrEqual(0);
  });

  test('Gap espectral', () => {
    const gap = PropiedadesEspectrales.calcularGapEspectral(hipergrafo);
    expect(gap).toBeGreaterThanOrEqual(0);
  });
});
