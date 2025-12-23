/**
 * Orquestador Principal
 */

import { EventEmitter } from 'events';
import { Hipergrafo } from '../core';
import { 
    Omega21Telemetry, 
    MapeoOmegaAHipergrafo
} from '../omega21';
import { Omega21Client, RenodeController } from '../hardware';
import { DendriteController, EstadoAnalisis, AjusteDendrita } from '../control';
import { InferenciaLocal } from '../neural';

export interface ConfiguracionOrquestador {
    hostOmega21: string;
    puertoTelemetria: number;
    puertoControl: number;
    puertoRenode: number;
    modoSimulacion: boolean;
    habilitarControl: boolean;
}

export interface ResultadoAnalisis {
    timestamp: Date;
    hipergrafo: Hipergrafo;
    estadoAnalisis: EstadoAnalisis;
    ajustesAplicados: AjusteDendrita[];
    telemetriaOriginal: Omega21Telemetry;
}

export class Orquestador extends EventEmitter {
    private config: ConfiguracionOrquestador;
    private omega21Client: Omega21Client;
    private renodeController: RenodeController;
    private mapeador: MapeoOmegaAHipergrafo;
    private controlador: DendriteController;
    private inferencia: InferenciaLocal;
    private activo: boolean = false;

    constructor(config: Partial<ConfiguracionOrquestador> = {}) {
        super();
        
        this.config = {
            hostOmega21: config.hostOmega21 || 'localhost',
            puertoTelemetria: config.puertoTelemetria || 4561,
            puertoControl: config.puertoControl || 4560,
            puertoRenode: config.puertoRenode || 1234,
            modoSimulacion: config.modoSimulacion ?? true,
            habilitarControl: config.habilitarControl ?? false
        };
        
        this.omega21Client = new Omega21Client({
            host: this.config.hostOmega21,
            puertoTelemetria: this.config.puertoTelemetria
        });
        
        this.renodeController = new RenodeController({
            host: this.config.hostOmega21,
            puerto: this.config.puertoRenode
        });
        
        this.mapeador = new MapeoOmegaAHipergrafo();
        this.controlador = new DendriteController({ autoAjuste: this.config.habilitarControl });
        this.inferencia = new InferenciaLocal();
        
        this.configurarEventos();
    }

    async iniciar(): Promise<boolean> {
        await this.inferencia.inicializar();
        if (!this.config.modoSimulacion) {
            await this.omega21Client.conectar();
            if (this.config.habilitarControl) {
                await this.renodeController.conectar();
                await this.controlador.conectarHardware(this.renodeController);
            }
        }
        this.activo = true;
        this.emit('iniciado');
        return true;
    }

    async detener(): Promise<void> {
        this.activo = false;
        this.omega21Client.desconectar();
        this.renodeController.desconectar();
        this.emit('detenido');
    }

    async procesar(telemetria: Omega21Telemetry): Promise<ResultadoAnalisis | null> {
        if (!this.activo) return null;
        
        const hipergrafo = this.mapeador.mapear(telemetria);
        
        // Inferencia Neuronal (Cerebro 1024)
        const { nodeFeatures, edgeIndex, globalVector } = this.mapeador.extraerTensores(hipergrafo);
        const prediccion = await this.inferencia.predecir(nodeFeatures, edgeIndex, globalVector, telemetria);

        const { estado, ajustes } = await this.controlador.ejecutarCiclo(hipergrafo, telemetria);
        
        // Integrar predicción en el estado
        if (prediccion) {
            estado.anomaliaDetectada = prediccion.prediccion_anomalia > 0.5;
            estado.confianza = prediccion.prediccion_estabilidad;
        }

        const resultado: ResultadoAnalisis = {
            timestamp: new Date(),
            hipergrafo,
            estadoAnalisis: estado,
            ajustesAplicados: ajustes,
            telemetriaOriginal: telemetria
        };
        
        this.emit('procesado', resultado);
        return resultado;
    }

    async aplicarAjustesIA(ajustes: number[]): Promise<void> {
        if (this.activo && this.controlador) {
            await this.controlador.aplicarAjustesVector(ajustes);
        }
    }

    private configurarEventos(): void {
        this.omega21Client.on('telemetria', (t) => this.procesar(t));
    }
}

export default Orquestador;
