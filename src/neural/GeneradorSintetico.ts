/**
 * GeneradorSintetico.ts
 * 
 * Generador de datos para la "Caja Fenomenológica".
 * Produce vectores 256D con patrones específicos para entrenar y probar
 * la jerarquía cognitiva (Capas 0-3).
 */

import { Vector256D } from './CapaSensorial';

export enum TipoPatron {
    NOMINAL = 'NOMINAL',
    ANOMALIA_SENSORIAL = 'ANOMALIA_SENSORIAL',
    ANOMALIA_TEMPORAL = 'ANOMALIA_TEMPORAL',
    CONFLICTO_MODAL = 'CONFLICTO_MODAL',
    RUIDO_BLANCO = 'RUIDO_BLANCO',
    DEGRADACION_LENTA = 'DEGRADACION_LENTA',
    RAFAGA_RUIDO = 'RAFAGA_RUIDO'
}

export class GeneradorSintetico {
    private timestep: number = 0;

    constructor() {}

    /**
     * Genera una secuencia de vectores 256D
     */
    public generarSecuencia(longitud: number, patron: TipoPatron): Vector256D[] {
        const secuencia: Vector256D[] = [];
        for (let i = 0; i < longitud; i++) {
            secuencia.push(this.generarVector(patron));
            this.timestep++;
        }
        return secuencia;
    }

    /**
     * Genera un único vector 256D basado en un patrón
     */
    public generarVector(patron: TipoPatron): Vector256D {
        const vector: Vector256D = {};
        
        // Inicializar con base nominal (ruido base)
        for (let i = 1; i <= 256; i++) {
            const key = `D${i.toString().padStart(3, '0')}`;
            vector[key] = Math.random() * 0.1;
        }

        switch (patron) {
            case TipoPatron.NOMINAL:
                this.aplicarPatronNominal(vector);
                break;
            case TipoPatron.ANOMALIA_SENSORIAL:
                this.aplicarAnomaliaSensorial(vector);
                break;
            case TipoPatron.ANOMALIA_TEMPORAL:
                this.aplicarAnomaliaTemporal(vector);
                break;
            case TipoPatron.CONFLICTO_MODAL:
                this.aplicarConflictoModal(vector);
                break;
            case TipoPatron.DEGRADACION_LENTA:
                this.aplicarDegradacionLenta(vector);
                break;
            case TipoPatron.RAFAGA_RUIDO:
                this.aplicarRafagaRuido(vector);
                break;
            case TipoPatron.RUIDO_BLANCO:
                // Ya inicializado con ruido
                break;
        }

        return vector;
    }

    private aplicarPatronNominal(v: Vector256D) {
        // S1-S3: Oscilaciones suaves (Cripto/Fenomenología)
        for (let i = 1; i <= 48; i++) {
            const key = `D${i.toString().padStart(3, '0')}`;
            v[key] = Math.sin(this.timestep * 0.1 + i) * 0.5 + 0.5;
        }
        // S9: Física simulada (Gravedad/Inercia)
        for (let i = 105; i <= 116; i++) {
            const key = `D${i.toString().padStart(3, '0')}`;
            v[key] = 0.8; // Constante estable
        }
    }

    private aplicarAnomaliaSensorial(v: Vector256D) {
        this.aplicarPatronNominal(v);
        // Pico masivo en S11 (Anomalías) y S5 (Seguridad)
        for (let i = 125; i <= 132; i++) {
            v[`D${i.toString().padStart(3, '0')}`] = 1.0;
        }
        for (let i = 57; i <= 72; i++) {
            v[`D${i.toString().padStart(3, '0')}`] = Math.random() > 0.5 ? 1.0 : 0.0;
        }
    }

    private aplicarAnomaliaTemporal(v: Vector256D) {
        // Cambio brusco de frecuencia en las oscilaciones
        for (let i = 1; i <= 48; i++) {
            const key = `D${i.toString().padStart(3, '0')}`;
            v[key] = Math.sin(this.timestep * 2.0 + i); // Frecuencia 20x
        }
    }

    private aplicarConflictoModal(v: Vector256D) {
        // S9 (Física) dice que todo está estable
        for (let i = 105; i <= 116; i++) {
            v[`D${i.toString().padStart(3, '0')}`] = 0.8;
        }
        // Pero S14 (Espectral) muestra caos total
        for (let i = 149; i <= 156; i++) {
            v[`D${i.toString().padStart(3, '0')}`] = Math.random() * 2.0 - 1.0;
        }
    }

    private aplicarDegradacionLenta(v: Vector256D) {
        this.aplicarPatronNominal(v);
        // El ruido base aumenta lentamente con el tiempo
        const factorDrift = Math.min(this.timestep / 1000, 1.0);
        for (let i = 1; i <= 256; i++) {
            const key = `D${i.toString().padStart(3, '0')}`;
            v[key] += (Math.random() - 0.5) * factorDrift;
        }
    }

    private aplicarRafagaRuido(v: Vector256D) {
        this.aplicarPatronNominal(v);
        // Ruido masivo en todos los canales por un instante
        if (Math.random() > 0.8) {
            for (let i = 1; i <= 256; i++) {
                const key = `D${i.toString().padStart(3, '0')}`;
                v[key] = Math.random() * 5.0; // Saturación
            }
        }
    }
}
