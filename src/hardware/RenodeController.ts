/**
 * Controlador Renode - Interface Telnet con monitor Renode
 */

import * as net from 'net';
import { EventEmitter } from 'events';
import { OMEGA21_PORTS, OMEGA21_MEMORY_MAP } from '../omega21/Schema';

export interface RenodeControllerConfig {
    host: string;
    puerto: number;
    timeoutMs: number;
}

export interface ComandoRenode {
    comando: string;
    descripcion: string;
}

export const COMANDOS_RENODE = {
    PAUSE: { comando: 'pause', descripcion: 'Pausar simulación' },
    START: { comando: 'start', descripcion: 'Iniciar simulación' },
    RESET: { comando: 'machine reset', descripcion: 'Reiniciar máquina' },
    READ_DWORD: (addr: number) => ({ 
        comando: `sysbus ReadDoubleWord ${addr}`,
        descripcion: `Leer en 0x${addr.toString(16)}`
    }),
    WRITE_DWORD: (addr: number, val: number) => ({ 
        comando: `sysbus WriteDoubleWord ${addr} ${val}`,
        descripcion: `Escribir ${val} en 0x${addr.toString(16)}`
    }),
};

export class RenodeController extends EventEmitter {
    private config: RenodeControllerConfig;
    private socket: net.Socket | null = null;
    private buffer: string = '';
    private comandoPendiente: {
        resolve: (value: string) => void;
        reject: (reason: Error) => void;
        timeout: NodeJS.Timeout;
    } | null = null;

    constructor(config: Partial<RenodeControllerConfig> = {}) {
        super();
        
        this.config = {
            host: config.host || 'localhost',
            puerto: config.puerto || OMEGA21_PORTS.RENODE_MONITOR,
            timeoutMs: config.timeoutMs || 10000
        };
    }

    async conectar(): Promise<boolean> {
        return new Promise((resolve, reject) => {
            this.socket = new net.Socket();

            const timeout = setTimeout(() => {
                this.socket?.destroy();
                reject(new Error('Timeout'));
            }, this.config.timeoutMs);

            this.socket.connect(this.config.puerto, this.config.host, () => {
                clearTimeout(timeout);
                console.log(`[RenodeController] Conectado`);
                this.emit('conectado');
                resolve(true);
            });

            this.socket.on('data', (data) => {
                this.procesarRespuesta(data.toString());
            });

            this.socket.on('error', (error) => {
                clearTimeout(timeout);
                this.emit('error', error);
                reject(error);
            });

            this.socket.on('close', () => {
                this.emit('desconectado');
            });
        });
    }

    private procesarRespuesta(data: string): void {
        this.buffer += data;

        if (/\(machine-\d+\)\s*$/.test(this.buffer) && this.comandoPendiente) {
            const respuesta = this.buffer;
            this.buffer = '';
            
            clearTimeout(this.comandoPendiente.timeout);
            this.comandoPendiente.resolve(respuesta);
            this.comandoPendiente = null;
        }
    }

    async ejecutar(cmd: ComandoRenode | string): Promise<string> {
        const comando = typeof cmd === 'string' ? cmd : cmd.comando;
        
        if (!this.socket) throw new Error('No conectado');
        if (this.comandoPendiente) throw new Error('Comando pendiente');

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.comandoPendiente = null;
                reject(new Error(`Timeout: ${comando}`));
            }, this.config.timeoutMs);

            this.comandoPendiente = { resolve, reject, timeout };
            this.socket!.write(comando + '\n');
        });
    }

    async leerMemoria(direccion: number): Promise<number> {
        const cmd = COMANDOS_RENODE.READ_DWORD(direccion);
        const respuesta = await this.ejecutar(cmd);
        const match = respuesta.match(/0x([0-9a-fA-F]+)/);
        return match ? parseInt(match[1], 16) : 0;
    }

    async escribirMemoria(direccion: number, valor: number): Promise<void> {
        const cmd = COMANDOS_RENODE.WRITE_DWORD(direccion, valor);
        await this.ejecutar(cmd);
    }

    async modificarDendrita(indice: number, parametro: string, valor: number): Promise<void> {
        const base = OMEGA21_MEMORY_MAP.DENDRITES_BASE + (indice * OMEGA21_MEMORY_MAP.DENDRITE_STRIDE);
        const offset = parametro === 'param_a' ? 0x04 : parametro === 'param_b' ? 0x08 : 0x00;
        await this.escribirMemoria(base + offset, valor);
    }

    async leerSOMA(): Promise<{ voltage: number; spikeCount: number }> {
        const base = OMEGA21_MEMORY_MAP.SOMA_BASE;
        return {
            voltage: await this.leerMemoria(base + 0x00),
            spikeCount: await this.leerMemoria(base + 0x08)
        };
    }

    desconectar(): void {
        if (this.socket) {
            this.socket.destroy();
            this.socket = null;
        }
        if (this.comandoPendiente) {
            clearTimeout(this.comandoPendiente.timeout);
            this.comandoPendiente = null;
        }
        this.buffer = '';
    }
}

export default RenodeController;
