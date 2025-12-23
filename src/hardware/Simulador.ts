/**
 * Simulador de Telemetría Omega 21
 * 
 * Genera datos sintéticos que imitan el comportamiento de la red neuronal LIF 1024
 * y las 16 dendritas para permitir el desarrollo sin el hardware real.
 */

import { Omega21Telemetry } from '../omega21/Schema';

export class Omega21Simulador {
    private tick: number = 0;
    private parametrosDendriticos: any | null = null;

    /**
     * Configura los parámetros dendríticos que alterarán el comportamiento del átomo.
     * Estos vienen del Vector 256D (D001-D056).
     */
    configurarDendritas(parametros: any): void {
        this.parametrosDendriticos = parametros;
    }

    /**
     * Genera un paquete de telemetría.
     * Si hay parámetros dendríticos configurados, estos alteran la generación de datos,
     * simulando la estabilización del átomo por inputs externos.
     */
    generarMuestra(): Omega21Telemetry {
        this.tick++;
        
        const tieneSpike = Math.random() > 0.95;
        const novelty = Math.random() * 255;

        // Valores base (aleatorios si no hay input dendrítico)
        let d = {
            voltage: 20 + Math.random() * 50,
            current: 100 + Math.random() * 500,
            power: 1000 + Math.random() * 2000,
            altitude: 500,
            dew_temp: 2000,
            velocity: Math.random() * 100,
            phase: Math.floor(Math.random() * 255),
            freq: 50 + Math.random() * 100,
            soma_v: tieneSpike ? -5500 : -7000 + (Math.random() * 500),
            spike: tieneSpike ? 1 : 0,
            loss: Math.random() * 1000000,
            // Campos extendidos con valores por defecto
            delay: Math.random() * 100,
            memory: Math.random(),
            decay: Math.random(),
            michaelis: Math.random() * 100,
            hill: 1 + Math.random(),
            capacitor: Math.random() * 100,
            entropy: Math.random() * 5
        };

        // Si hay input dendrítico, este "altera" y estabiliza los valores
        if (this.parametrosDendriticos) {
            // Aplicamos los valores del vector 256D con una pequeña varianza (ruido térmico)
            // Esto simula que el átomo ha sido "forzado" a un estado por las dendritas
            d.voltage = this.mezclar(d.voltage, this.parametrosDendriticos.voltage, 0.9);
            d.current = this.mezclar(d.current, this.parametrosDendriticos.current, 0.9);
            d.power = this.mezclar(d.power, this.parametrosDendriticos.power, 0.9);
            d.altitude = this.parametrosDendriticos.altitude;
            d.dew_temp = this.parametrosDendriticos.dew_temp;
            d.velocity = this.mezclar(d.velocity, this.parametrosDendriticos.velocity, 0.8);
            d.phase = this.parametrosDendriticos.phase;
            d.freq = this.mezclar(d.freq, this.parametrosDendriticos.freq, 0.95);
            
            // El soma se ve fuertemente influenciado por el input dendrítico D016
            d.soma_v = this.mezclar(d.soma_v, this.parametrosDendriticos.soma_v * 100, 0.7); // Escala ajustada
            
            // Copiar resto de parámetros físicos
            if (this.parametrosDendriticos.delay) d.delay = this.parametrosDendriticos.delay;
            if (this.parametrosDendriticos.memory) d.memory = this.parametrosDendriticos.memory;
            if (this.parametrosDendriticos.decay) d.decay = this.parametrosDendriticos.decay;
            if (this.parametrosDendriticos.michaelis) d.michaelis = this.parametrosDendriticos.michaelis;
            if (this.parametrosDendriticos.hill) d.hill = this.parametrosDendriticos.hill;
            if (this.parametrosDendriticos.capacitor) d.capacitor = this.parametrosDendriticos.capacitor;
            if (this.parametrosDendriticos.entropy) d.entropy = this.parametrosDendriticos.entropy;
        }
        
        return {
            meta: {
                ts: Date.now(),
                blk: this.tick,
                sz: 256
            },
            logic: {
                h: 4000 + Math.random() * 2000,
                lz: Math.floor(Math.random() * 20),
                chi: 60000 + Math.random() * 5000,
                pad: [0, 0, 0]
            },
            neuro: {
                id: Math.floor(Math.random() * 1024),
                sim: 150 + Math.random() * 100,
                nov: novelty,
                cat: Math.floor(Math.random() * 8)
            },
            sig: {
                fp: Math.random().toString(16).substring(2, 18),
                lsh: Math.floor(Math.random() * 255),
                eq: Math.floor(Math.random() * 8),
                sc: Math.floor(Math.random() * 3)
            },
            dendrites: d,
            vector_72d: new Array(72).fill(0).map(() => Math.random())
        };
    }

    private mezclar(actual: number, objetivo: number, fuerza: number): number {
        return actual * (1 - fuerza) + objetivo * fuerza;
    }

    /**
     * Inicia un flujo de datos simulado
     */
    iniciarFlujo(callback: (t: Omega21Telemetry) => void, intervaloMs: number = 100): NodeJS.Timeout {
        return setInterval(() => {
            callback(this.generarMuestra());
        }, intervaloMs);
    }
}
