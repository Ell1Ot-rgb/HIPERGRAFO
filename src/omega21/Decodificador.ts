/**
 * Decodificador de Telemetría Omega 21
 */

import { Omega21Telemetry } from './Schema';

export class Omega21Decodificador {
    
    static decodificarJSON(jsonString: string): Omega21Telemetry {
        try {
            const raw = JSON.parse(jsonString);
            return {
                meta: {
                    ts: raw.meta?.ts || Date.now(),
                    blk: raw.meta?.blk || 0,
                    sz: raw.meta?.sz || 0
                },
                logic: {
                    h: raw.logic?.h || 0,
                    lz: raw.logic?.lz || 0,
                    chi: raw.logic?.chi || 0,
                    pad: raw.logic?.pad || [0, 0, 0]
                },
                neuro: {
                    id: raw.neuro?.id || 0,
                    sim: raw.neuro?.sim || 0,
                    nov: raw.neuro?.nov || 0,
                    cat: raw.neuro?.cat || 0
                },
                sig: {
                    fp: raw.sig?.fp || '',
                    lsh: raw.sig?.lsh || 0,
                    eq: raw.sig?.eq || 0,
                    sc: raw.sig?.sc || 0
                },
                dendrites: raw.dendrites || this.crearDendritasVacias()
            };
        } catch (error) {
            throw new Error(`Error decodificando JSON: ${error}`);
        }
    }

    private static crearDendritasVacias(): Omega21Telemetry['dendrites'] {
        return {
            voltage: 0, current: 0, power: 0, altitude: 0, dew_temp: 0,
            velocity: 0, phase: 0, freq: 0, soma_v: -7000, spike: 0, loss: 0
        };
    }

    static normalizarDendritas(dendrites: Omega21Telemetry['dendrites']): Record<string, number> {
        return {
            voltage_norm: dendrites.voltage / 100.0,
            current_norm: dendrites.current / 1000.0,
            power_norm: dendrites.power / 10000.0,
            altitude_norm: dendrites.altitude / 10000.0,
            dew_temp_norm: (dendrites.dew_temp + 4000) / 8000.0,
            velocity_norm: (dendrites.velocity + 10000) / 20000.0,
            phase_norm: dendrites.phase / 255.0,
            freq_norm: dendrites.freq / 1000.0,
            soma_v_norm: (dendrites.soma_v + 10000) / 20000.0,
            spike_norm: dendrites.spike,
            loss_norm: Math.log10(dendrites.loss + 1) / 10.0
        };
    }

    static extraerResumen(telemetry: Omega21Telemetry): {
        esAnomalia: boolean;
        tieneSpike: boolean;
        entropiaAlta: boolean;
        categoria: number;
    } {
        return {
            esAnomalia: telemetry.neuro.nov > 200,
            tieneSpike: telemetry.dendrites.spike === 1,
            entropiaAlta: (telemetry.logic?.chi || 0) > 7000,
            categoria: telemetry.neuro.cat
        };
    }
}
