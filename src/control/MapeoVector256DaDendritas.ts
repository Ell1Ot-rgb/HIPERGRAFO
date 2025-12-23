import { Vector256D } from '../neural/CapaSensorial';
import { Omega21Dendrites } from '../omega21/Schema';

/**
 * MapeoVector256DaDendritas
 * 
 * Responsable de extraer los primeros 56 subespacios del Vector 256D
 * y convertirlos en señales de control para las 16 dendritas del átomo.
 * 
 * Flujo: Vector 256D -> [D001...D056] -> Dendritas -> Alteración del Átomo
 */
export class MapeoVector256DaDendritas {
    
    /**
     * Extrae y transforma los campos del vector 256D a la estructura de dendritas.
     * @param vector256D El vector completo de entrada
     */
    extraerCamposDendriticos(vector256D: Vector256D): Partial<Omega21Dendrites> {
        // Mapeo directo de los primeros 16 campos a las propiedades físicas principales
        // y uso de campos adicionales (D017-D056) para modulación fina si fuera necesario.
        
        return {
            // S1: Criptografía / Base Eléctrica (D001-D003)
            voltage: this.getVal(vector256D, 'D001', 0),
            current: this.getVal(vector256D, 'D002', 0),
            power: this.getVal(vector256D, 'D003', 0),
            
            // S1: Ambiental (D004-D005)
            altitude: this.getVal(vector256D, 'D004', 500),
            dew_temp: this.getVal(vector256D, 'D005', 20),
            
            // S1: Mecánico y Onda (D006-D008)
            velocity: this.getVal(vector256D, 'D006', 0),
            phase: this.getVal(vector256D, 'D007', 0),
            freq: this.getVal(vector256D, 'D008', 60),
            
            // S1: Temporal y Cognitivo (D009-D011)
            delay: this.getVal(vector256D, 'D009', 0),
            memory: this.getVal(vector256D, 'D010', 0.5),
            decay: this.getVal(vector256D, 'D011', 0.1),
            
            // S1: Bioquímico (D012-D013)
            michaelis: this.getVal(vector256D, 'D012', 50),
            hill: this.getVal(vector256D, 'D013', 1),
            
            // S1: Componentes Finales (D014-D016)
            capacitor: this.getVal(vector256D, 'D014', 10),
            entropy: this.getVal(vector256D, 'D015', 0),
            soma_v: this.getVal(vector256D, 'D016', -70) // Potencial de membrana base
        };
    }

    private getVal(vec: Vector256D, key: string, defaultVal: number): number {
        return vec[key] !== undefined ? vec[key] : defaultVal;
    }
}
