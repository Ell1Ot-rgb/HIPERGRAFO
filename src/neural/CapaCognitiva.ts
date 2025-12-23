/**
 * CapaCognitiva.ts
 * 
 * Implementa la Capa 3 (Consenso y Ejecución) de la Corteza Cognitiva.
 * 
 * Responsabilidades:
 * 1. Integración de vectores contextuales de la Capa 2.
 * 2. Toma de decisiones basada en umbrales y políticas.
 * 3. Generación de señales de control o alertas.
 */

import { SalidaEspacioTemporal } from './CapaEspacioTemporal';

export interface DecisionCognitiva {
    tipo: 'MONITOREO' | 'ALERTA' | 'INTERVENCION' | 'APRENDIZAJE';
    descripcion: string;
    nivelUrgencia: number; // 0.0 a 1.0
    metadata: Record<string, any>;
}

export class CapaCognitiva {
    private historialDecisiones: DecisionCognitiva[] = [];
    private readonly UMBRAL_ALERTA = 0.8;

    constructor() {}

    public async procesar(entrada: SalidaEspacioTemporal): Promise<DecisionCognitiva> {
        // Lógica de decisión basada en la salida de la red híbrida
        
        let decision: DecisionCognitiva;

        if (entrada.anomaliaDetectada) {
            decision = {
                tipo: 'ALERTA',
                descripcion: 'Anomalía detectada en patrones espacio-temporales',
                nivelUrgencia: 0.9,
                metadata: {
                    confianzaModelo: entrada.confianza,
                    vector: entrada.vectorContextual.slice(0, 5) // Log parcial
                }
            };
        } else if (entrada.confianza < 0.3) {
            decision = {
                tipo: 'APRENDIZAJE',
                descripcion: 'Baja confianza en inferencia, solicitando re-entrenamiento o datos adicionales',
                nivelUrgencia: 0.4,
                metadata: { confianza: entrada.confianza }
            };
        } else {
            decision = {
                tipo: 'MONITOREO',
                descripcion: 'Operación nominal',
                nivelUrgencia: 0.1,
                metadata: {}
            };
        }

        this.registrarDecision(decision);
        return decision;
    }

    private registrarDecision(decision: DecisionCognitiva) {
        this.historialDecisiones.push(decision);
        if (this.historialDecisiones.length > 100) {
            this.historialDecisiones.shift();
        }
    }

    public obtenerUltimaDecision(): DecisionCognitiva | null {
        return this.historialDecisiones[this.historialDecisiones.length - 1] || null;
    }
}
