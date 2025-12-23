/**
 * Ejemplo de Integración: Hipergrafo ↔️ Google Colab
 * El flujo es:
 * 1. Crear un Hipergrafo localmente
 * 2. Extraer sus datos
 * 3. Enviar a Colab para procesamiento con IA
 * 4. Recibir resultados
 */

import { Hipergrafo, Nodo, Hiperedge } from '../core';
import { ColabBridge } from './ColabBridge';
import { CONFIG_COLAB } from './configColab';

export class IntegradorHipergrafoColo {
    private hipergrafo: Hipergrafo;
    private puente: ColabBridge;

    constructor(nombreHipergrafo: string = "HipergrafoColo") {
        this.hipergrafo = new Hipergrafo(nombreHipergrafo);
        this.puente = new ColabBridge(CONFIG_COLAB.urlServidor);
    }

    /**
     * Verifica conexión con Colab
     */
    async verificarPuente(): Promise<boolean> {
        console.log("🔗 Verificando conexión con Colab...");
        const conectado = await this.puente.verificarConexion();
        
        if (conectado) {
            console.log("✅ Puente con Colab ACTIVO");
        } else {
            console.error("❌ No se puede alcanzar Colab.");
        }
        
        return conectado;
    }

    /**
     * Obtiene la estructura del Hipergrafo como JSON
     */
    obtenerEstructura(): Record<string, any> {
        const nodos = this.hipergrafo.obtenerNodos();
        const hiperedges = this.hipergrafo.obtenerHiperedges();

        return {
            nombre: this.hipergrafo.label,
            nodos: nodos.map(n => ({
                id: n.id,
                label: n.label,
                metadata: n.metadata
            })),
            hiperedges: hiperedges.map(h => ({
                id: h.id,
                nodos: Array.from(h.nodos),
                label: h.label,
                weight: h.weight
            }))
        };
    }

    /**
     * Envía los datos del Hipergrafo a Colab
     */
    async procesarEnColab(): Promise<any> {
        try {
            console.log("\n📤 Enviando Hipergrafo a Colab...");
            
            const datosHipergrafo = this.obtenerEstructura();
            console.log(`   Enviando ${datosHipergrafo.nodos.length} nodos y ${datosHipergrafo.hiperedges.length} hiperedges`);
            
            const resultado = await this.puente.ejecutarModelo(datosHipergrafo);
            
            console.log("📥 Respuesta recibida de Colab:");
            console.log(JSON.stringify(resultado, null, 2));
            
            return resultado;
        } catch (error) {
            console.error("❌ Error durante el procesamiento en Colab:", error);
            throw error;
        }
    }

    /**
     * Crea un Hipergrafo de ejemplo
     */
    crearEjemplo(): void {
        console.log("🔨 Creando Hipergrafo de ejemplo...");
        
        const nodo1 = new Nodo("Entrada", { tipo: "input" });
        const nodo2 = new Nodo("Procesamiento", { tipo: "process" });
        const nodo3 = new Nodo("Salida", { tipo: "output" });
        
        this.hipergrafo.agregarNodo(nodo1);
        this.hipergrafo.agregarNodo(nodo2);
        this.hipergrafo.agregarNodo(nodo3);
        
        const edge1 = new Hiperedge("Flujo 1", [nodo1, nodo2], 0.8);
        const edge2 = new Hiperedge("Flujo 2", [nodo2, nodo3], 0.9);
        
        this.hipergrafo.agregarHiperedge(edge1);
        this.hipergrafo.agregarHiperedge(edge2);
        
        console.log("✅ Ejemplo creado");
    }

    /**
     * Flujo completo
     */
    async flujoCompleto(): Promise<void> {
        console.log("\n🚀 INICIANDO FLUJO COMPLETO DE INTEGRACIÓN\n");
        console.log("═".repeat(50));
        
        this.crearEjemplo();
        
        const conectado = await this.verificarPuente();
        if (!conectado) {
            console.error("\n❌ No se puede continuar sin conexión a Colab");
            return;
        }
        
        const resultados = await this.procesarEnColab();
        
        console.log("\n═".repeat(50));
        console.log("✅ FLUJO COMPLETADO EXITOSAMENTE");
        console.log(`Resultado del análisis de IA:`, resultados);
    }
}

export default IntegradorHipergrafoColo;
