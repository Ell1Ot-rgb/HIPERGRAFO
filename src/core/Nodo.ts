import { v4 as uuidv4 } from 'uuid';

/**
 * Representa un nodo en el hipergrafo
 * Un nodo es una entidad que puede estar contenido en múltiples hiperedges
 */
export class Nodo {
  readonly id: string;
  readonly label: string;
  metadata: Record<string, any>;

  constructor(label: string, metadata: Record<string, any> = {}) {
    this.id = uuidv4();
    this.label = label;
    this.metadata = metadata;
  }

  /**
   * Clona el nodo preservando datos
   */
  clone(): Nodo {
    return new Nodo(this.label, { ...this.metadata });
  }

  /**
   * Establece un valor de metadato
   */
  setMetadato(key: string, value: any): void {
    this.metadata[key] = value;
  }

  /**
   * Obtiene un valor de metadato
   */
  getMetadato(key: string): any {
    return this.metadata[key];
  }
}
