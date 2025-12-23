import { Nodo } from './Nodo';
import { Hiperedge } from './Hiperedge';

/**
 * Representa un hipergrafo: H = (V, E)
 * V = conjunto de nodos
 * E = conjunto de hiperedges (subconjuntos de V)
 *
 * Propiedades matemáticas:
 * - Cada hiperedge es un subconjunto de V
 * - Permite relaciones de orden superior (más allá de pares binarios)
 * - Fundamental para modelar redes complejas
 */
export class Hipergrafo {
  private nodos: Map<string, Nodo>;
  private hiperedges: Map<string, Hiperedge>;
  label: string;

  constructor(label: string = 'Hipergrafo') {
    this.nodos = new Map();
    this.hiperedges = new Map();
    this.label = label;
  }

  /**
   * Agrega un nodo al hipergrafo
   */
  agregarNodo(nodo: Nodo): void {
    this.nodos.set(nodo.id, nodo);
  }

  /**
   * Agrega múltiples nodos
   */
  agregarNodos(nodos: Nodo[]): void {
    nodos.forEach(nodo => this.agregarNodo(nodo));
  }

  /**
   * Obtiene un nodo por ID
   */
  obtenerNodo(nodoId: string): Nodo | undefined {
    return this.nodos.get(nodoId);
  }

  /**
   * Obtiene todos los nodos
   */
  obtenerNodos(): Nodo[] {
    return Array.from(this.nodos.values());
  }

  /**
   * Agrega una hiperedge al hipergrafo
   * Valida que todos los nodos existan en el hipergrafo
   */
  agregarHiperedge(hiperedge: Hiperedge): void {
    // Validar que todos los nodos de la hiperedge existan
    for (const nodoId of hiperedge.nodos) {
      if (!this.nodos.has(nodoId)) {
        throw new Error(`Nodo ${nodoId} no existe en el hipergrafo`);
      }
    }
    this.hiperedges.set(hiperedge.id, hiperedge);
  }

  /**
   * Obtiene una hiperedge por ID
   */
  obtenerHiperedge(edgeId: string): Hiperedge | undefined {
    return this.hiperedges.get(edgeId);
  }

  /**
   * Obtiene todas las hiperedges
   */
  obtenerHiperedges(): Hiperedge[] {
    return Array.from(this.hiperedges.values());
  }

  /**
   * Obtiene las hiperedges incidentes a un nodo
   */
  obtenerHiperedgesDelNodo(nodoId: string): Hiperedge[] {
    return Array.from(this.hiperedges.values()).filter(edge => edge.contiene(nodoId));
  }

  /**
   * Calcula el grado de un nodo (cantidad de hiperedges que lo contienen)
   */
  calcularGradoNodo(nodoId: string): number {
    return this.obtenerHiperedgesDelNodo(nodoId).length;
  }

  /**
   * Elimina un nodo del hipergrafo y de todas las hiperedges que lo contienen
   */
  eliminarNodo(nodoId: string): boolean {
    if (!this.nodos.has(nodoId)) return false;
    
    // Eliminar de todas las hiperedges
    this.hiperedges.forEach(edge => {
      edge.nodos.delete(nodoId);
    });

    return this.nodos.delete(nodoId);
  }

  /**
   * Elimina una hiperedge
   */
  eliminarHiperedge(edgeId: string): boolean {
    return this.hiperedges.delete(edgeId);
  }

  /**
   * Obtiene el número de nodos (|V|)
   */
  cardinalV(): number {
    return this.nodos.size;
  }

  /**
   * Obtiene el número de hiperedges (|E|)
   */
  cardinalE(): number {
    return this.hiperedges.size;
  }

  /**
   * Verifica si dos nodos están conectados (por al menos una hiperedge)
   */
  estaConectados(nodoId1: string, nodoId2: string): boolean {
    const hiperedgesNodo1 = this.obtenerHiperedgesDelNodo(nodoId1);
    return hiperedgesNodo1.some(edge => edge.contiene(nodoId2));
  }

  /**
   * Obtiene los vecinos de un nodo (nodos conectados por hiperedges)
   */
  obtenerVecinos(nodoId: string): Nodo[] {
    const hiperedges = this.obtenerHiperedgesDelNodo(nodoId);
    const vecinosIds = new Set<string>();

    hiperedges.forEach(edge => {
      edge.nodos.forEach(id => {
        if (id !== nodoId) {
          vecinosIds.add(id);
        }
      });
    });

    return Array.from(vecinosIds)
      .map(id => this.obtenerNodo(id))
      .filter((nodo): nodo is Nodo => nodo !== undefined);
  }

  /**
   * Matriz de incidencia: M donde M[i,j] = 1 si nodo i está en hiperedge j
   */
  calcularMatrizIncidencia(): number[][] {
    const nodos = this.obtenerNodos();
    const edges = this.obtenerHiperedges();
    const matriz: number[][] = [];

    nodos.forEach((nodo, i) => {
      matriz[i] = [];
      edges.forEach((edge, j) => {
        matriz[i][j] = edge.contiene(nodo.id) ? 1 : 0;
      });
    });

    return matriz;
  }

  /**
   * Grado promedio del hipergrafo
   */
  gradoPromedio(): number {
    if (this.cardinalV() === 0) return 0;
    const sumaGrados = this.obtenerNodos().reduce((sum, nodo) => sum + this.calcularGradoNodo(nodo.id), 0);
    return sumaGrados / this.cardinalV();
  }

  /**
   * Densidad del hipergrafo (qué tan conectado está)
   */
  densidad(): number {
    const n = this.cardinalV();
    if (n < 2) return 0;
    // Para hipergrafos, la densidad es más compleja que en grafos normales
    return this.cardinalE() / Math.pow(2, n);
  }

  /**
   * Clona el hipergrafo completo
   */
  clone(): Hipergrafo {
    const clon = new Hipergrafo(this.label);
    
    // Mapeo de IDs antiguos a nuevos nodos
    const mapeoNodos = new Map<string, Nodo>();
    
    this.obtenerNodos().forEach(nodo => {
      const nuevoNodo = nodo.clone();
      mapeoNodos.set(nodo.id, nuevoNodo);
      clon.agregarNodo(nuevoNodo);
    });

    this.obtenerHiperedges().forEach(edge => {
      const nodosEnEdge = Array.from(edge.nodos)
        .map(id => mapeoNodos.get(id))
        .filter((nodo): nodo is Nodo => nodo !== undefined);
      
      const nuevoEdge = new Hiperedge(edge.label, nodosEnEdge, edge.weight, { ...edge.metadata });
      clon.agregarHiperedge(nuevoEdge);
    });

    return clon;
  }

  /**
   * Limpia el hipergrafo
   */
  limpiar(): void {
    this.nodos.clear();
    this.hiperedges.clear();
  }
}
