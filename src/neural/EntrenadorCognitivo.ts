/**
 * EntrenadorCognitivo.ts
 * 
 * Gestiona el entrenamiento de la Corteza Cognitiva (Capa de Abstracción).
 * A diferencia del entrenamiento GNN (que es reactivo/físico), este es 
 * un entrenamiento de "Consolidación de Memoria" y "Categorización".
 */

import { CortezaCognitiva } from './CortezaCognitiva';
import { Hipergrafo, Hiperedge, Nodo } from '../core';

export interface Experiencia {
    timestamp: number;
    percepciones: number[];
    idConcepto: string;
    estabilidad: number;
    fueFalla: boolean;
}

export class EntrenadorCognitivo {
    private corteza: CortezaCognitiva;
    private bufferExperiencias: Experiencia[] = [];
    private mapeoConceptos: Map<string, Experiencia[]> = new Map();
    private readonly LIMITE_BUFFER = 50;
    private ciclosConsolidacion: number = 0;
    private tasaAcierto: number = 0;

    constructor(corteza: CortezaCognitiva) {
        this.corteza = corteza;
    }

    public obtenerEstadisticas() {
        return {
            bufferLleno: this.bufferExperiencias.length,
            conceptosAprendidos: this.mapeoConceptos.size,
            ciclosConsolidacion: this.ciclosConsolidacion,
            tasaAcierto: this.tasaAcierto.toFixed(2)
        };
    }

    /**
     * FASE 1: Adquisición de Experiencias
     */
    registrarExperiencia(percepciones: number[], imagenMental: Hipergrafo, fueFalla: boolean = false) {
        const ultimoNodo = imagenMental.obtenerNodos().slice(-1)[0];
        const experiencia: Experiencia = {
            timestamp: Date.now(),
            percepciones,
            idConcepto: ultimoNodo?.id || 'UNKNOWN',
            estabilidad: ultimoNodo?.metadata?.estabilidad || 0.5,
            fueFalla
        };

        this.bufferExperiencias.push(experiencia);
        
        if (!this.mapeoConceptos.has(experiencia.idConcepto)) {
            this.mapeoConceptos.set(experiencia.idConcepto, []);
        }
        this.mapeoConceptos.get(experiencia.idConcepto)!.push(experiencia);
        
        if (this.bufferExperiencias.length >= this.LIMITE_BUFFER) {
            this.ejecutarCicloConsolidacion();
        }
    }

    /**
     * Fase 2: Consolidación (Entrenamiento)
     * Aquí es donde la red "aprende" a categorizar mejor.
     */
    private ejecutarCicloConsolidacion() {
        this.ciclosConsolidacion++;
        console.log(`🧠 [Entrenador Cognitivo] Ciclo #${this.ciclosConsolidacion}: Consolidando ${this.bufferExperiencias.length} experiencias...`);
        
        this.refinarCategorias();
        this.podarMemoriaDebil();
        this.reforzarCausalidad();

        const aciertos = this.bufferExperiencias.filter(e => e.fueFalla).length;
        this.tasaAcierto = aciertos / this.bufferExperiencias.length;

        this.bufferExperiencias = [];
        console.log(`✅ [Entrenador Cognitivo] Consolidación completada. Tasa de acierto: ${this.tasaAcierto.toFixed(2)}`);
    }

    private refinarCategorias() {
        const mapa = this.corteza.getMapaMental();
        this.mapeoConceptos.forEach((experiencias, conceptoId) => {
            if (experiencias.length < 2) return;
            const centroide = this.calcularCentroide(experiencias);
            let nodo = mapa.obtenerNodo(conceptoId);
            if (!nodo) {
                nodo = new Nodo(conceptoId, {
                    tipo: 'concepto',
                    centroide,
                    frecuencia: experiencias.length
                });
                mapa.agregarNodo(nodo);
            } else {
                nodo.metadata.centroide = centroide;
                nodo.metadata.frecuencia = experiencias.length;
            }
        });
    }

    private calcularCentroide(experiencias: Experiencia[]): number[] {
        const dim = experiencias[0].percepciones.length;
        const centroide = new Array(dim).fill(0);
        for (const exp of experiencias) {
            for (let i = 0; i < dim; i++) {
                centroide[i] += exp.percepciones[i];
            }
        }
        return centroide.map(v => v / experiencias.length);
    }

    private reforzarCausalidad() {
        const mapa = this.corteza.getMapaMental();
        const nodos = mapa.obtenerNodos();
        for (let i = 0; i < nodos.length - 1; i++) {
            const nodo1 = nodos[i];
            const nodo2 = nodos[i + 1];
            const peso = 0.7;
            const edge = new Hiperedge(`causal_${nodo1.id}_${nodo2.id}`, [nodo1, nodo2], peso);
            mapa.agregarHiperedge(edge);
        }
    }

    private podarMemoriaDebil() {
        const mapa = this.corteza.getMapaMental();
        const edges = mapa.obtenerHiperedges();
        const UMBRAL_PODA = 0.1;
        const edgesAEliminar: string[] = [];
        
        edges.forEach(edge => {
            if (edge.weight < UMBRAL_PODA) {
                console.log(`   🗑️ Podando: ${edge.id} (peso: ${edge.weight.toFixed(3)})`);
                edgesAEliminar.push(edge.id);
            }
        });
        
        // Eliminar las hiperedges débiles
        edgesAEliminar.forEach(edgeId => {
            const edge = edges.find(e => e.id === edgeId);
            if (edge) {
                // Marcar como débil para posterior eliminación
                edge.metadata = edge.metadata || {};
                edge.metadata.eliminada = true;
            }
        });
    }
}
