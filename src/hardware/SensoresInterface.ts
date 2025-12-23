/**
 * Interfaz de Abstracción de Hardware
 * Permite cambiar entre Sensores Reales (PC Local) y Sensores Simulados (Colab/Dev)
 */

export interface SensorData {
    timestamp: number;
    valores: Float32Array; // Array de 1024 valores
    metadata?: any;
}

export interface ISensorArray {
    inicializar(): Promise<void>;
    leerDatos(): Promise<SensorData>;
    detener(): Promise<void>;
}

/**
 * Implementación MOCK (Simulada) para Entornos de Desarrollo/Colab
 * Genera patrones sintéticos que imitan a los sensores reales
 */
export class MockSensorArray implements ISensorArray {
    private activo: boolean = false;
    private dimension: number = 1024;

    async inicializar(): Promise<void> {
        console.log("⚠️ MODO SIMULACIÓN: Iniciando Array de Sensores Virtuales 1024d");
        this.activo = true;
    }

    async leerDatos(): Promise<SensorData> {
        if (!this.activo) throw new Error("Sensores no inicializados");

        // Generar ruido simulado o patrones específicos
        // Aquí simulamos una señal LIF (picos esporádicos)
        const datos = new Float32Array(this.dimension);
        
        for (let i = 0; i < this.dimension; i++) {
            // Simulación: 10% de probabilidad de disparo (spike)
            datos[i] = Math.random() > 0.9 ? 1.0 : 0.0;
        }

        return {
            timestamp: Date.now(),
            valores: datos
        };
    }

    async detener(): Promise<void> {
        this.activo = false;
        console.log("Sensores virtuales detenidos");
    }
}
