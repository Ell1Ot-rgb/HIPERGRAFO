import { Hipergrafo, Nodo, Hiperedge } from '../core';

describe('Hipergrafo - Pruebas Fundamentales', () => {
  let hipergrafo: Hipergrafo;

  beforeEach(() => {
    hipergrafo = new Hipergrafo('Test Hipergrafo');
  });

  test('Crear hipergrafo vacío', () => {
    expect(hipergrafo.cardinalV()).toBe(0);
    expect(hipergrafo.cardinalE()).toBe(0);
  });

  test('Agregar nodos al hipergrafo', () => {
    const nodo1 = new Nodo('Nodo1');
    const nodo2 = new Nodo('Nodo2');

    hipergrafo.agregarNodo(nodo1);
    hipergrafo.agregarNodo(nodo2);

    expect(hipergrafo.cardinalV()).toBe(2);
  });

  test('Obtener nodos del hipergrafo', () => {
    const nodo = new Nodo('TestNodo');
    hipergrafo.agregarNodo(nodo);

    const obtenido = hipergrafo.obtenerNodo(nodo.id);
    expect(obtenido).toBe(nodo);
  });

  test('Agregar hiperedge', () => {
    const n1 = new Nodo('N1');
    const n2 = new Nodo('N2');
    const n3 = new Nodo('N3');

    hipergrafo.agregarNodos([n1, n2, n3]);

    const edge = new Hiperedge('Edge1', [n1, n2, n3], 1.5);
    hipergrafo.agregarHiperedge(edge);

    expect(hipergrafo.cardinalE()).toBe(1);
  });

  test('Grado de un nodo', () => {
    const n1 = new Nodo('N1');
    const n2 = new Nodo('N2');
    const n3 = new Nodo('N3');

    hipergrafo.agregarNodos([n1, n2, n3]);

    const edge1 = new Hiperedge('E1', [n1, n2]);
    const edge2 = new Hiperedge('E2', [n1, n3]);

    hipergrafo.agregarHiperedge(edge1);
    hipergrafo.agregarHiperedge(edge2);

    expect(hipergrafo.calcularGradoNodo(n1.id)).toBe(2);
    expect(hipergrafo.calcularGradoNodo(n2.id)).toBe(1);
  });

  test('Vecinos de un nodo', () => {
    const n1 = new Nodo('N1');
    const n2 = new Nodo('N2');
    const n3 = new Nodo('N3');

    hipergrafo.agregarNodos([n1, n2, n3]);

    const edge = new Hiperedge('E1', [n1, n2, n3]);
    hipergrafo.agregarHiperedge(edge);

    const vecinos = hipergrafo.obtenerVecinos(n1.id);
    expect(vecinos.length).toBe(2);
  });

  test('Matriz de incidencia', () => {
    const n1 = new Nodo('N1');
    const n2 = new Nodo('N2');

    hipergrafo.agregarNodos([n1, n2]);

    const edge = new Hiperedge('E1', [n1, n2]);
    hipergrafo.agregarHiperedge(edge);

    const matriz = hipergrafo.calcularMatrizIncidencia();
    expect(matriz.length).toBe(2);
    expect(matriz[0][0]).toBe(1);
  });

  test('Clonar hipergrafo', () => {
    const n1 = new Nodo('N1', { test: 'valor' });
    hipergrafo.agregarNodo(n1);

    const edge = new Hiperedge('E1', [n1]);
    hipergrafo.agregarHiperedge(edge);

    const clon = hipergrafo.clone();
    expect(clon.cardinalV()).toBe(1);
    expect(clon.cardinalE()).toBe(1);
  });
});
