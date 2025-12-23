import { Nodo } from '../core/Nodo';

export type SpiderType = 'Z' | 'X' | 'H' | 'INPUT' | 'OUTPUT';

/**
 * Representa una "Araña" (Spider) en el Cálculo ZX.
 * Extiende la clase Nodo base para añadir propiedades cuánticas/topológicas.
 * 
 * - Z-Spider (Verde): Copia en la base computacional |0>, |1>.
 * - X-Spider (Rojo): Copia en la base diagonal |+>, |->.
 * - Fase: Rotación en la esfera de Bloch (α * π).
 */
export class ZXSpider extends Nodo {
    public readonly tipoSpider: SpiderType;
    public fase: number; // Representa α donde el ángulo es α * π

    constructor(id: string, tipo: SpiderType, fase: number = 0, metadatos: any = {}) {
        // Persistimos las propiedades ZX también en los metadatos del nodo base
        // para compatibilidad con el sistema de persistencia existente.
        const meta = { 
            ...metadatos, 
            tipoClase: 'ZXSpider', 
            spiderType: tipo, 
            fase: fase 
        };
        super(id, meta);
        this.tipoSpider = tipo;
        this.fase = fase;
    }

    /**
     * Fusiona esta araña con otra (si son del mismo tipo).
     * Regla (f): Spider Fusion.
     * @param otra La otra araña a fusionar.
     */
    public fusionar(otra: ZXSpider): void {
        if (this.tipoSpider !== otra.tipoSpider) {
            throw new Error("Solo se pueden fusionar arañas del mismo tipo (Color).");
        }
        // La fase se suma (módulo 2π, pero aquí trabajamos con coeficientes de π)
        this.fase = (this.fase + otra.fase) % 2;
        
        // Actualizar metadatos
        this.setMetadato('fase', this.fase);
    }

    /**
     * Cambia el color de la araña (Color Change).
     * Convierte Z <-> X aplicando transformaciones de Hadamard.
     */
    public cambiarColor(): void {
        if (this.tipoSpider === 'Z') {
            // Hack de TypeScript para readonly, en un sistema real usaríamos un setter protegido
            (this as any).tipoSpider = 'X';
        } else if (this.tipoSpider === 'X') {
            (this as any).tipoSpider = 'Z';
        }
        this.setMetadato('spiderType', this.tipoSpider);
    }
}
