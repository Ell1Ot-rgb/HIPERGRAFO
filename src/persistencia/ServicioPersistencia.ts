import { Hipergrafo, Nodo, Hiperedge } from '../core';

/**
 * Interfaz para serialización JSON del hipergrafo
 */
export interface HipergrafoDatos {
  label: string;
  nodos: Array<{
    id: string;
    label: string;
    metadata: Record<string, any>;
  }>;
  hiperedges: Array<{
    id: string;
    label: string;
    nodosIds: string[];
    weight: number;
    metadata: Record<string, any>;
  }>;
  version: string;
  fechaCreacion: string;
}

/**
 * Servicio de persistencia para hipergrafos
 * Soporta serialización/deserialización JSON
 */
export class ServicioPersistencia {
  /**
   * Serializa un hipergrafo a JSON
   */
  serializarAJSON(hipergrafo: Hipergrafo): string {
    const datos = this.convertirADatos(hipergrafo);
    return JSON.stringify(datos, null, 2);
  }

  /**
   * Deserializa un hipergrafo desde JSON
   */
  deserializarDesdeJSON(jsonString: string): Hipergrafo {
    const datos: HipergrafoDatos = JSON.parse(jsonString);
    return this.convertirDesdeDatos(datos);
  }

  /**
   * Convierte un hipergrafo a estructura de datos serializable
   */
  private convertirADatos(hipergrafo: Hipergrafo): HipergrafoDatos {
    const nodos = hipergrafo.obtenerNodos();
    const hiperedges = hipergrafo.obtenerHiperedges();

    return {
      label: hipergrafo.label,
      nodos: nodos.map(n => ({
        id: n.id,
        label: n.label,
        metadata: n.metadata
      })),
      hiperedges: hiperedges.map(e => ({
        id: e.id,
        label: e.label,
        nodosIds: Array.from(e.nodos),
        weight: e.weight,
        metadata: e.metadata
      })),
      version: '1.0.0',
      fechaCreacion: new Date().toISOString()
    };
  }

  /**
   * Convierte una estructura de datos a hipergrafo
   */
  private convertirDesdeDatos(datos: HipergrafoDatos): Hipergrafo {
    const hipergrafo = new Hipergrafo(datos.label);

    // Mapeo temporal de IDs antiguos a nuevos nodos
    const mapeoNodos = new Map<string, Nodo>();

    // Crear y agregar nodos
    datos.nodos.forEach(nodoData => {
      const nodo = new Nodo(nodoData.label, nodoData.metadata);
      // Preservar el ID original para mantener consistencia
      (nodo as any).id = nodoData.id;
      mapeoNodos.set(nodoData.id, nodo);
      hipergrafo.agregarNodo(nodo);
    });

    // Crear y agregar hiperedges
    datos.hiperedges.forEach(edgeData => {
      const nodosEnEdge = edgeData.nodosIds
        .map(id => mapeoNodos.get(id))
        .filter((n): n is Nodo => n !== undefined);

      if (nodosEnEdge.length > 0) {
        const hiperedge = new Hiperedge(
          edgeData.label,
          nodosEnEdge,
          edgeData.weight,
          edgeData.metadata
        );
        // Preservar el ID original
        (hiperedge as any).id = edgeData.id;
        hipergrafo.agregarHiperedge(hiperedge);
      }
    });

    return hipergrafo;
  }

  /**
   * Calcula un hash simple del hipergrafo para validación
   */
  calcularHash(hipergrafo: Hipergrafo): string {
    const datos = this.convertirADatos(hipergrafo);
    const json = JSON.stringify(datos);
    
    let hash = 0;
    for (let i = 0; i < json.length; i++) {
      const char = json.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convertir a 32-bit integer
    }
    return hash.toString(16);
  }

  /**
   * Genera un reporte de estadísticas del hipergrafo
   */
  generarReporte(hipergrafo: Hipergrafo): Record<string, any> {
    const nodos = hipergrafo.obtenerNodos();
    const hiperedges = hipergrafo.obtenerHiperedges();

    return {
      label: hipergrafo.label,
      cardinalV: hipergrafo.cardinalV(),
      cardinalE: hipergrafo.cardinalE(),
      gradoPromedio: hipergrafo.gradoPromedio(),
      densidad: hipergrafo.densidad(),
      nodos: nodos.map(n => ({
        id: n.id,
        label: n.label,
        grado: hipergrafo.obtenerHiperedgesDelNodo(n.id).length
      })),
      hiperedges: hiperedges.map(e => ({
        id: e.id,
        label: e.label,
        grado: e.grado(),
        weight: e.weight
      }))
    };
  }
}
