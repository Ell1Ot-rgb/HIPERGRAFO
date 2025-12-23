import { SistemaOmnisciente } from './SistemaOmnisciente';
import { StreamingBridge } from './neural/StreamingBridge';
import { MapeoVector256DaDendritas } from './control/MapeoVector256DaDendritas';
import { Vector256D } from './neural/CapaSensorial';
import { CONFIG_COLAB } from './neural/configColab';

/**
 * Script Maestro de Entrenamiento
 * 
 * Implementa el flujo completo:
 * 1. Generación/Recepción de Vector 256D
 * 2. Extracción de señales dendríticas (D001-D056)
 * 3. Alteración y estabilización de Átomos
 * 4. Generación de Telemetría Real
 * 5. Envío a Colab para entrenamiento de Capas 2-5
 */
class EntrenadorCompleto {
    private sistema: SistemaOmnisciente;
    private bridge: StreamingBridge;
    private mapeador: MapeoVector256DaDendritas;
    
    constructor() {
        console.log("🚀 Inicializando Entrenador Completo...");
        this.sistema = new SistemaOmnisciente(); 
        this.bridge = new StreamingBridge(CONFIG_COLAB.urlServidor);
        this.mapeador = new MapeoVector256DaDendritas();
    }

    async iniciar(ciclos: number = 1000) {
        console.log(`\n⚡ Iniciando ciclo de entrenamiento (${ciclos} iteraciones)`);
        
        // Crear los 25 átomos correspondientes a los subespacios S1-S25
        console.log("🧬 Creando 25 Átomos Topológicos (S1-S25)...");
        for (let i = 1; i <= 25; i++) {
            const id = `S${i}`;
            await this.sistema.crearAtomo(id);
        }
        console.log("✅ 25 Átomos creados y listos.");

        for (let i = 0; i < ciclos; i++) {
            // 1. Obtener Vector 256D (Simulado por ahora, vendría de sensores reales)
            const vector256D = this.generarVectorEntrada();
            
            // 2. Extraer configuración para dendritas
            const configDendritas = this.mapeador.extraerCamposDendriticos(vector256D);
            
            // 3. Procesar con cada átomo del sistema
            for (const [_id, atom] of this.sistema.atomos) {
                // A. Alterar el átomo con las dendritas
                // El simulador interno del átomo recibe la configuración
                atom.simulador.configurarDendritas(configDendritas);
                
                // B. El átomo se estabiliza (simulado por el paso del tiempo/ticks)
                // Generamos una muestra que ya estará "alterada" por las dendritas
                const telemetria = atom.simulador.generarMuestra();
                
                // C. El átomo percibe y procesa (Inferencia ONNX Local)
                const resultado = await atom.percibir(telemetria);
                
                // D. Preparar vector para Colab (1600D)
                // Usamos ajustes_dendritas (256D) como embedding latente
                const vectorOutput = this.prepararVectorSalida(resultado.neuronal.ajustes_dendritas, vector256D);
                
                // E. Enviar a Colab
                const esAnomalia = telemetria.neuro.nov > 200;
                await this.bridge.enviarVector(vectorOutput, esAnomalia);
            }
            
            // PROTOCOLO DE INFECCIÓN (cada 50 ciclos)
            if (i % 50 === 0 && i > 0) {
                console.log(`\n🦠 CICLO ${i}: Ejecutando Protocolo de Infección`);
                await this.sistema.propagarInfeccion();
            }
            
            if (i % 10 === 0) process.stdout.write('.');
            await new Promise(r => setTimeout(r, 100)); // Control de flujo
        }
        
        console.log("\n✅ Entrenamiento finalizado.");
    }

    private generarVectorEntrada(): Vector256D {
        const vec: Vector256D = {};
        // Generamos un vector con coherencia básica
        const baseVoltage = 20 + Math.sin(Date.now() / 1000) * 10;
        
        for (let i = 1; i <= 256; i++) {
            const key = `D${i.toString().padStart(3, '0')}`;
            if (i === 1) vec[key] = baseVoltage;
            else if (i === 16) vec[key] = -70; // Soma
            else vec[key] = Math.random() * 100;
        }
        return vec;
    }

    private prepararVectorSalida(embeddingAtom: number[], _input256: Vector256D): number[] {
        // Simulamos la expansión a 1600D que espera la Capa 2
        // En producción, esto sería la concatenación de los 25 subespacios
        const output: number[] = [];
        
        // El embeddingAtom tiene 256 valores (ajustes_dendritas)
        // Para llegar a 1600D, podemos concatenar el embedding con partes del input256
        // o simplemente repetir/expandir el embedding.
        // 1600 / 256 = 6.25. 
        
        // Usamos el embedding de 256D y lo repetimos 6 veces + 64 valores extra
        for(let i=0; i<6; i++) {
            output.push(...embeddingAtom);
        }
        output.push(...embeddingAtom.slice(0, 64));
        
        return output;
    }
}

// Ejecución
const entrenador = new EntrenadorCompleto();
entrenador.iniciar(500).catch(console.error);
