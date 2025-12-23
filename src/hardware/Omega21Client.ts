/**
 * Cliente TCP para comunicación con Omega 21
 */

import * as net from 'net';
import { EventEmitter } from 'events';
import { OMEGA21_PORTS } from '../omega21/Schema';
import { Omega21Decodificador } from '../omega21/Decodificador';

export interface Omega21ClientConfig {
    host: string;
    puertoTelemetria: number;
    puertoControl: number;
    reconexionAutomatica: boolean;
    intentosReconexion: number;
    timeoutMs: number;
}

export interface EstadoConexion {
    telemetria: 'desconectado' | 'conectando' | 'conectado' | 'error';
    control: 'desconectado' | 'conectando' | 'conectado' | 'error';
    ultimoError?: string;
    mensajesRecibidos: number;
}

export class Omega21Client extends EventEmitter {
    private config: Omega21ClientConfig;
    private socketTelemetria: net.Socket | null = null;
    private buffer: string = '';
    private estado: EstadoConexion;

    constructor(config: Partial<Omega21ClientConfig> = {}) {
        super();
        
        this.config = {
            host: config.host || 'localhost',
            puertoTelemetria: config.puertoTelemetria || OMEGA21_PORTS.UART_TELEMETRY,
            puertoControl: config.puertoControl || OMEGA21_PORTS.UDP_CONTROL,
            reconexionAutomatica: config.reconexionAutomatica ?? true,
            intentosReconexion: config.intentosReconexion || 5,
            timeoutMs: config.timeoutMs || 5000
        };

        this.estado = {
            telemetria: 'desconectado',
            control: 'desconectado',
            mensajesRecibidos: 0
        };
    }

    async conectar(): Promise<boolean> {
        return new Promise((resolve, reject) => {
            this.estado.telemetria = 'conectando';
            
            this.socketTelemetria = new net.Socket();
            this.socketTelemetria.setEncoding('utf8');
            
            const timeout = setTimeout(() => {
                this.socketTelemetria?.destroy();
                reject(new Error('Timeout'));
            }, this.config.timeoutMs);

            this.socketTelemetria.connect(
                this.config.puertoTelemetria, 
                this.config.host, 
                () => {
                    clearTimeout(timeout);
                    this.estado.telemetria = 'conectado';
                    console.log(`[Omega21Client] Conectado a ${this.config.host}:${this.config.puertoTelemetria}`);
                    this.emit('conectado');
                    resolve(true);
                }
            );

            this.socketTelemetria.on('data', (data: string) => {
                this.procesarDatos(data);
            });

            this.socketTelemetria.on('error', (error) => {
                clearTimeout(timeout);
                this.estado.telemetria = 'error';
                this.estado.ultimoError = error.message;
                this.emit('error', error);
                reject(error);
            });

            this.socketTelemetria.on('close', () => {
                this.estado.telemetria = 'desconectado';
                this.emit('desconectado');
            });
        });
    }

    private procesarDatos(data: string): void {
        this.buffer += data;

        let newlineIndex: number;
        while ((newlineIndex = this.buffer.indexOf('\n')) !== -1) {
            const linea = this.buffer.slice(0, newlineIndex).trim();
            this.buffer = this.buffer.slice(newlineIndex + 1);

            if (linea.startsWith('{')) {
                try {
                    const telemetria = Omega21Decodificador.decodificarJSON(linea);
                    this.estado.mensajesRecibidos++;
                    this.emit('telemetria', telemetria);
                    
                    const resumen = Omega21Decodificador.extraerResumen(telemetria);
                    if (resumen.tieneSpike) this.emit('spike', telemetria);
                    if (resumen.esAnomalia) this.emit('anomalia', telemetria);
                } catch (e) {
                    this.emit('parseError', { linea, error: e });
                }
            }
        }
    }

    obtenerEstado(): EstadoConexion {
        return { ...this.estado };
    }

    estaConectado(): boolean {
        return this.estado.telemetria === 'conectado';
    }

    desconectar(): void {
        if (this.socketTelemetria) {
            this.socketTelemetria.destroy();
            this.socketTelemetria = null;
        }
        this.estado.telemetria = 'desconectado';
        this.buffer = '';
        this.emit('desconectado');
    }
}

export default Omega21Client;
