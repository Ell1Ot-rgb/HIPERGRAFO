import { ZXDiagram } from './ZXDiagram';
import { ReglaZX } from './Reglas';

/**
 * Motor de Inferencia y Reescritura ZX.
 * Orquesta la aplicación de reglas topológicas para transformar el diagrama.
 * 
 * En el contexto del Ruliad, este motor actúa como el ejecutor de la evolución
 * del sistema, buscando la forma normal (confluencia) del diagrama.
 */
export class MotorZX {
    private reglas: ReglaZX[];

    constructor() {
        this.reglas = [];
    }

    /**
     * Registra una nueva regla de reescritura en el motor.
     */
    public agregarRegla(regla: ReglaZX): void {
        this.reglas.push(regla);
    }

    /**
     * Ejecuta un único paso de reescritura.
     * Busca el primer match válido de cualquier regla y lo aplica.
     * @returns true si se aplicó alguna regla, false si no (estado estable).
     */
    public ejecutarPaso(diagrama: ZXDiagram): boolean {
        // Estrategia: Prioridad por orden de registro
        for (const regla of this.reglas) {
            const matches = regla.buscarMatches(diagrama);
            if (matches.length > 0) {
                // Aplicar al primer match encontrado (determinismo local)
                // En un sistema Multiway real, aquí bifurcaríamos el universo.
                regla.aplicar(diagrama, matches[0]);
                return true;
            }
        }
        return false;
    }

    /**
     * Ejecuta reglas hasta que el diagrama alcanza una forma normal (no más matches)
     * o se alcanza el límite de pasos.
     * @returns Número de pasos ejecutados.
     */
    public ejecutarHastaConvergencia(diagrama: ZXDiagram, maxPasos: number = 1000): number {
        let pasos = 0;
        while (pasos < maxPasos) {
            const cambio = this.ejecutarPaso(diagrama);
            if (!cambio) {
                break; // Forma normal alcanzada
            }
            pasos++;
        }
        return pasos;
    }
}
