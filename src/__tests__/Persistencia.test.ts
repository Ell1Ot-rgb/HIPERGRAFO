import { Hipergrafo, Nodo, Hiperedge } from '../core';
import { ServicioPersistencia, GestorAlmacenamiento } from '../persistencia';
import * as fs from 'fs';

describe('Persistencia de Hipergrafos', () => {
  let hipergrafo: Hipergrafo;
  let servicio: ServicioPersistencia;
  let gestor: GestorAlmacenamiento;
  const directorioTest = './test_hipergrafos';

  beforeEach(() => {
    hipergrafo = new Hipergrafo('Test Persistencia');
    servicio = new ServicioPersistencia();
    gestor = new GestorAlmacenamiento(directorioTest);

    // Crear estructura de prueba
    const n1 = new Nodo('Nodo1', { tipo: 'entrada' });
    const n2 = new Nodo('Nodo2', { tipo: 'procesamiento' });
    const n3 = new Nodo('Nodo3', { tipo: 'salida' });

    hipergrafo.agregarNodos([n1, n2, n3]);

    const edge1 = new Hiperedge('Edge1', [n1, n2], 0.5);
    const edge2 = new Hiperedge('Edge2', [n2, n3], 0.8);

    hipergrafo.agregarHiperedge(edge1);
    hipergrafo.agregarHiperedge(edge2);
  });

  afterEach(() => {
    // Limpiar archivos de prueba
    if (fs.existsSync(directorioTest)) {
      fs.rmSync(directorioTest, { recursive: true });
    }
  });

  test('Serializar hipergrafo a JSON', () => {
    const json = servicio.serializarAJSON(hipergrafo);
    expect(typeof json).toBe('string');
    
    const datos = JSON.parse(json);
    expect(datos.label).toBe('Test Persistencia');
    expect(datos.nodos.length).toBe(3);
    expect(datos.hiperedges.length).toBe(2);
  });

  test('Deserializar hipergrafo desde JSON', () => {
    const json = servicio.serializarAJSON(hipergrafo);
    const hipergrafoCargado = servicio.deserializarDesdeJSON(json);

    expect(hipergrafoCargado.cardinalV()).toBe(3);
    expect(hipergrafoCargado.cardinalE()).toBe(2);
  });

  test('Guardar y cargar hipergrafo', () => {
    const rutaGuardada = gestor.guardarHipergrafo(hipergrafo, 'test_hg');
    expect(fs.existsSync(rutaGuardada)).toBe(true);

    const hipergrafoCargado = gestor.cargarHipergrafo('test_hg');
    expect(hipergrafoCargado.cardinalV()).toBe(3);
    expect(hipergrafoCargado.cardinalE()).toBe(2);
  });

  test('Listar hipergrafos guardados', () => {
    gestor.guardarHipergrafo(hipergrafo, 'hg1');
    gestor.guardarHipergrafo(hipergrafo, 'hg2');

    const lista = gestor.listarHipergrafos();
    expect(lista.includes('hg1')).toBe(true);
    expect(lista.includes('hg2')).toBe(true);
  });

  test('Calcular hash del hipergrafo', () => {
    const hash1 = servicio.calcularHash(hipergrafo);
    const hash2 = servicio.calcularHash(hipergrafo);
    
    expect(typeof hash1).toBe('string');
    expect(hash1).toBe(hash2); // Hashes idénticos para mismo hipergrafo
  });

  test('Generar reporte de estadísticas', () => {
    const reporte = servicio.generarReporte(hipergrafo);
    
    expect(reporte.label).toBe('Test Persistencia');
    expect(reporte.cardinalV).toBe(3);
    expect(reporte.cardinalE).toBe(2);
    expect(typeof reporte.gradoPromedio).toBe('number');
  });

  test('Exportar a CSV', () => {
    const ruta = gestor.exportarACSV(hipergrafo, 'test_export');
    expect(fs.existsSync(ruta)).toBe(true);

    const contenido = fs.readFileSync(ruta, 'utf-8');
    expect(contenido.includes('Tipo')).toBe(true);
    expect(contenido.includes('Nodo')).toBe(true);
  });
});
