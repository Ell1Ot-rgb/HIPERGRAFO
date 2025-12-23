import { Nodo } from './Nodo';

/**
 * Representa una hiperedge (arista generalizada)
 * Una hiperedge puede conectar un conjunto arbitrario de nodos (no solo 2)
 * Concepto clave: E ⊆ V (conjunto de nodos en la hiperedge E está contenido en todos los nodos V)
 */
export class Hiperedge {
  readonly id: string;
  readonly nodos: Set<string>; // IDs de los nodos
  readonly label: string;
  weight: number;
  metadata: Record<string, any>;

  constructor(label: string, nodos: Nodo[] = [], weight: number = 1, metadata: Record<string, any> = {}) {
    this.id = `edge_${Date.now()}_${Math.random()}`;
    this.nodos = new Set(nodos.map(n => n.id));
    this.label = label;
    this.weight = weight;
    this.metadata = metadata;
  }

  /**
   * Agrega un nodo a la hiperedge
   */
  agregarNodo(nodo: Nodo): void {
    this.nodos.add(nodo.id);
  }

  /**
   * Remueve un nodo de la hiperedge
   */
  removerNodo(nodoId: string): boolean {
    return this.nodos.delete(nodoId);
  }

  /**
   * Verifica si contiene un nodo específico
   */
  contiene(nodoId: string): boolean {
    return this.nodos.has(nodoId);
  }

  /**
   * Obtiene el grado de la hiperedge (cantidad de nodos que conecta)
   */
  grado(): number {
    return this.nodos.size;
  }

  /**
   * Clona la hiperedge con referencias a los mismos nodos
   */
  clone(): Hiperedge {
    const nodoArray = Array.from(this.nodos).map(id => new Nodo(`ref_${id}`));
    return new Hiperedge(this.label, nodoArray, this.weight, { ...this.metadata });
  }
}
