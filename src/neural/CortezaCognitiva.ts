import { Hipergrafo, Nodo, Hiperedge } from '../core';

/**
 * CortezaCognitiva - Red Neuronal Convencional de Alto Nivel.
 * Su función es generar "Coherencia Mental" a partir de los Átomos Topológicos.
 */
export class CortezaCognitiva {
    private mapaMental: Hipergrafo;

    constructor() {
        this.mapaMental = new Hipergrafo('Imagen_Mental_Persistente');
    }

    public getMapaMental(): Hipergrafo {
        return this.mapaMental;
    }

    /**
     * Procesa las percepciones de todos los átomos para generar un estado de coherencia.
     * @param percepciones Atomos procesados con su física y neuronal
     */
    generarCoherencia(percepciones: any[]): Hipergrafo {
        // 1. Extraer el "Vector de Conciencia" (Concatenación de estados neuronales)
        const vectorConciencia = percepciones.flatMap(p => p.neuronal.ajustes_dendritas.slice(0, 10));
        
        // 2. Simular la Red Convencional (MLP/Transformer)
        // En un sistema real, aquí pasaríamos el vector por un modelo ONNX convencional
        const idCoherencia = this.calcularHashCoherencia(vectorConciencia);
        
        // 3. Crear o actualizar el Nodo de Coherencia (La "Imagen Mental")
        let nodoCoherencia = this.mapaMental.obtenerNodo(idCoherencia);
        
        if (!nodoCoherencia) {
            nodoCoherencia = new Nodo(idCoherencia, {
                tipo: 'COHERENCIA_MENTAL',
                estabilidad: this.calcularEstabilidad(percepciones),
                timestamp: Date.now(),
                energiaTotal: percepciones.reduce((acc, p) => acc + p.fisica.metricas.physicsLoss.energia, 0)
            });
            this.mapaMental.agregarNodo(nodoCoherencia);
            
            // Conectar con la estructura persistente anterior (Memoria)
            this.vincularConPasado(nodoCoherencia);
        }

        return this.mapaMental;
    }

    private calcularHashCoherencia(vector: number[]): string {
        // Representa la ubicación en el espacio latente
        const suma = vector.reduce((a, b) => a + b, 0);
        return `CONCEPT_${Math.floor(suma % 1000)}`;
    }

    private calcularEstabilidad(percepciones: any[]): number {
        const varianzaTension = percepciones.reduce((acc, p) => acc + p.fisica.metricas.tension, 0) / percepciones.length;
        return 100 - varianzaTension;
    }

    private vincularConPasado(nuevoNodo: Nodo) {
        const nodosAnteriores = this.mapaMental.obtenerNodos()
            .filter(n => n.id !== nuevoNodo.id)
            .slice(-3); // Conectar con los últimos 3 estados mentales

        if (nodosAnteriores.length > 0) {
            const edgePersistencia = new Hiperedge(
                `flujo_conciencia_${Date.now()}`,
                [nuevoNodo, ...nodosAnteriores],
                0.9 // Alta fuerza de persistencia
            );
            this.mapaMental.agregarHiperedge(edgePersistencia);
        }
    }
}
