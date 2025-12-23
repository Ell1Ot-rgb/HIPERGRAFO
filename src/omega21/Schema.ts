/**
 * Schema de Tipos para el Sistema Omega 21
 */

// =========================================================================
// Tipos Base de Telemetría
// =========================================================================

export interface Omega21Meta {
    ts: number;
    blk: number;
    sz: number;
}

export interface Omega21Logic {
    h: number;       // Entropía
    lz: number;      // Longest zero run
    chi: number;     // Chi-cuadrado
    pad: [number, number, number]; // Análisis de padding
}

export interface Omega21Neuro {
    id: number;      // Patron ID (0-1023)
    sim: number;     // Similitud
    nov: number;     // Novedad
    cat: number;     // Categoría
}

export interface Omega21Signature {
    fp: string;      // Fingerprint (64-bit hex)
    lsh: number;     // LSH bucket
    eq: number;      // Entropy quantile
    sc: number;      // Scale type
}

export interface Omega21Dendrites {
    voltage: number;
    current: number;
    power: number;
    altitude: number;
    dew_temp: number;
    velocity: number;
    phase: number;
    freq: number;
    
    // Campos extendidos para estabilización (D009-D016)
    delay?: number;
    memory?: number;
    decay?: number;
    michaelis?: number;
    hill?: number;
    capacitor?: number;
    entropy?: number;
    
    soma_v: number;
    spike: number;
    loss: number;
}

export interface Omega21Telemetry {
    meta: Omega21Meta;
    logic: Omega21Logic;
    neuro: Omega21Neuro;
    sig: Omega21Signature;
    dendrites: Omega21Dendrites;
    vector_72d?: number[]; // El vector generado por metrics.c
}

// =========================================================================
// Tipos de Dendritas
// =========================================================================

export type DendriteType = 
    | 'voltage' | 'current' | 'power' | 'altitude' | 'dew'
    | 'velocity' | 'phase' | 'frequency' | 'delay' | 'memory'
    | 'decay' | 'michaelis' | 'hill' | 'capacitor' | 'entropy' | 'soma';

export interface DendriteInfo {
    tipo: DendriteType;
    nombre: string;
    tipoFisico: string;
    modelo: string;
    unidad: string;
    rangoNormal: [number, number];
}

export const DENDRITE_DEFINITIONS: DendriteInfo[] = [
    { tipo: 'voltage', nombre: 'Voltage', tipoFisico: 'Eléctrico', modelo: 'Ohm', unidad: 'mV', rangoNormal: [0, 100] },
    { tipo: 'current', nombre: 'Current', tipoFisico: 'Eléctrico', modelo: 'Ohm', unidad: 'nA', rangoNormal: [0, 1000] },
    { tipo: 'power', nombre: 'Power', tipoFisico: 'Eléctrico', modelo: 'Power', unidad: 'mW', rangoNormal: [0, 10000] },
    { tipo: 'altitude', nombre: 'Altitude', tipoFisico: 'Ambiental', modelo: 'Barométrico', unidad: 'm', rangoNormal: [0, 10000] },
    { tipo: 'dew', nombre: 'Dew Point', tipoFisico: 'Ambiental', modelo: 'Magnus', unidad: '°C×100', rangoNormal: [-4000, 4000] },
    { tipo: 'velocity', nombre: 'Velocity', tipoFisico: 'Mecánico', modelo: 'Doppler', unidad: 'cm/s', rangoNormal: [-10000, 10000] },
    { tipo: 'phase', nombre: 'Phase', tipoFisico: 'Ondulatorio', modelo: 'Sinusoidal', unidad: 'rad×40', rangoNormal: [0, 255] },
    { tipo: 'frequency', nombre: 'Frequency', tipoFisico: 'Ondulatorio', modelo: 'Fourier', unidad: 'Hz', rangoNormal: [0, 1000] },
    { tipo: 'delay', nombre: 'Delay', tipoFisico: 'Temporal', modelo: 'Exponencial', unidad: 'ms', rangoNormal: [0, 1000] },
    { tipo: 'memory', nombre: 'Memory', tipoFisico: 'Cognitivo', modelo: 'Decay', unidad: 'idx', rangoNormal: [0, 255] },
    { tipo: 'decay', nombre: 'Decay', tipoFisico: 'Temporal', modelo: 'Exponencial', unidad: 'τ', rangoNormal: [0, 100] },
    { tipo: 'michaelis', nombre: 'Michaelis', tipoFisico: 'Bioquímico', modelo: 'Michaelis-Menten', unidad: 'Vmax%', rangoNormal: [0, 100] },
    { tipo: 'hill', nombre: 'Hill', tipoFisico: 'Bioquímico', modelo: 'Hill', unidad: 'coef', rangoNormal: [0, 100] },
    { tipo: 'capacitor', nombre: 'Capacitor', tipoFisico: 'Eléctrico', modelo: 'RC', unidad: 'µF×100', rangoNormal: [0, 10000] },
    { tipo: 'entropy', nombre: 'Entropy', tipoFisico: 'Informacional', modelo: 'Shannon', unidad: 'bits×100', rangoNormal: [0, 1000] },
    { tipo: 'soma', nombre: 'SOMA', tipoFisico: 'Neuronal', modelo: 'LIF', unidad: 'mV×100', rangoNormal: [-7000, -5500] }
];

// =========================================================================
// Constantes de Hardware
// =========================================================================

export const OMEGA21_MEMORY_MAP = {
    RAMDISK_CTRL: 0x52000000,
    RAMDISK_DATA: 0x52000100,
    PMU_BRIDGE: 0x53000000,
    NEURO_CTRL: 0x51000000,
    REG_CTRL: 0x51000000,
    REG_STATUS: 0x51000004,
    DENDRITES_BASE: 0x50000000,
    DENDRITE_STRIDE: 0x1000,
    SOMA_BASE: 0x50010000,
    STATUS_READY: 1,
    STATUS_BUSY: 2,
    STATUS_SPIKE: 4,
    STATUS_ERROR: 8
};

export const OMEGA21_PORTS = {
    UART_TELEMETRY: 4561,
    RENODE_MONITOR: 1234,
    UDP_CONTROL: 4560
};
