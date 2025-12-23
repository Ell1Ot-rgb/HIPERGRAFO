/**
 * DatasetExporter.ts
 * 
 * Genera y envía un dataset masivo de entrenamiento a Colab.
 * Útil para el pre-entrenamiento (Warm-up) del modelo híbrido.
 */

import { GeneradorSintetico, TipoPatron } from './GeneradorSintetico';
import { ProcesadorSensorial } from './CapaSensorial';
import { StreamingBridge } from './StreamingBridge';

export class DatasetExporter {
    private generador: GeneradorSintetico;
    private sensorial: ProcesadorSensorial;
    private bridge: StreamingBridge;

    constructor(urlColab: string) {
        this.generador = new GeneradorSintetico();
        this.sensorial = new ProcesadorSensorial();
        this.bridge = new StreamingBridge(urlColab);
    }

    /**
     * Genera un dataset equilibrado y lo envía por streaming a máxima velocidad
     * @param numMuestras Cantidad total de vectores 1600D a generar
     */
    public async exportar(numMuestras: number = 5000) {
        console.log(`📦 Iniciando exportación de dataset: ${numMuestras} muestras...`);
        await this.sensorial.inicializar();

        const patrones = [
            { tipo: TipoPatron.NOMINAL, peso: 0.4 },
            { tipo: TipoPatron.ANOMALIA_SENSORIAL, peso: 0.15 },
            { tipo: TipoPatron.DEGRADACION_LENTA, peso: 0.15 },
            { tipo: TipoPatron.RAFAGA_RUIDO, peso: 0.15 },
            { tipo: TipoPatron.CONFLICTO_MODAL, peso: 0.15 }
        ];

        let enviadas = 0;
        
        for (const p of patrones) {
            const cantidad = Math.floor(numMuestras * p.peso);
            console.log(`   -> Generando ${cantidad} muestras de tipo: ${p.tipo}`);
            
            const secuencia = this.generador.generarSecuencia(cantidad, p.tipo);
            
            for (const vector of secuencia) {
                // 1. Pasar por Capa 0 y 1 (Normalización y Átomos)
                const salidaSensorial = await this.sensorial.procesar(vector);
                
                // 2. Aplanar a 1600D
                const vector1600d = Object.values(salidaSensorial).flat();
                
                // 3. Enviar al bridge (el bridge se encarga del batching de 64)
                const esAnomalia = p.tipo !== TipoPatron.NOMINAL;
                await this.bridge.enviarVector(vector1600d, esAnomalia);
                
                enviadas++;
                if (enviadas % 500 === 0) {
                    console.log(`   ✅ Progreso: ${enviadas}/${numMuestras} muestras procesadas.`);
                }
            }
        }

        console.log("🚀 Exportación completada. Esperando a que el buffer se vacíe...");
        
        // Esperar a que el bridge termine de enviar los últimos batches
        while (this.bridge.obtenerEstadoBuffer() > 0) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            console.log(`   ⏳ Batches restantes en cola: ${this.bridge.obtenerEstadoBuffer()}`);
        }

        console.log("✨ Dataset enviado con éxito a Colab.");
    }
}
