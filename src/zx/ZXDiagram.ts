import { Hipergrafo } from '../core/Hipergrafo';
import { Hiperedge } from '../core/Hiperedge';
import { ZXSpider, SpiderType } from './ZXSpider';
import { v4 as uuidv4 } from 'uuid';

/**
 * Representa un Diagrama ZX completo.
 * Es un Hipergrafo especializado donde los nodos son Spiders (Z/X)
 * y las conexiones representan contracciones de índices tensoriales.
 */
export class ZXDiagram extends Hipergrafo {
    
    constructor(id: string = uuidv4(), _metadatos: any = {}) {
        super(id);
    }

    /**
     * Añade una araña al diagrama.
     */
    public agregarSpider(spider: ZXSpider): void {
        this.agregarNodo(spider);
    }

    /**
     * Crea y añade una nueva araña.
     */
    public crearSpider(tipo: SpiderType, fase: number = 0): ZXSpider {
        const spider = new ZXSpider(uuidv4(), tipo, fase);
        this.agregarSpider(spider);
        return spider;
    }

    /**
     * Conecta dos arañas con un "cable" (wire).
     * En términos de hipergrafo, es una arista de grado 2.
     * @param a Spider origen
     * @param b Spider destino
     * @param tipo Tipo de conexión (Hadamard o Simple)
     */
    public conectarSpiders(a: ZXSpider, b: ZXSpider, tipo: 'Simple' | 'Hadamard' = 'Simple'): Hiperedge {
        const edgeId = uuidv4();
        const edge = new Hiperedge(edgeId, [a, b], 1, { tipoConexion: tipo });
        this.agregarHiperedge(edge);
        return edge;
    }

    /**
     * Obtiene todas las arañas de un tipo específico.
     */
    public obtenerSpidersPorTipo(tipo: SpiderType): ZXSpider[] {
        return this.obtenerNodos()
            .filter(n => n instanceof ZXSpider && n.tipoSpider === tipo) as ZXSpider[];
    }

    /**
     * Aplica la regla de identidad (Identity Rule).
     * Elimina arañas de fase 0 con grado 2 (son cables identidad).
     * @returns número de simplificaciones realizadas
     */
    public simplificarIdentidad(): number {
        let cambios = 0;
        const spiders = this.obtenerNodos() as ZXSpider[];

        for (const spider of spiders) {
            // Solo aplica a Z o X spiders con fase 0
            if ((spider.tipoSpider === 'Z' || spider.tipoSpider === 'X') && spider.fase === 0) {
                const grado = this.calcularGradoNodo(spider.id);
                if (grado === 2) {
                    // Es un nodo identidad. Fusionar los dos cables.
                    // Nota: Esta es una implementación simplificada. 
                    // En un motor completo, necesitamos reconectar los vecinos.
                    this.eliminarNodoIdentidad(spider);
                    cambios++;
                }
            }
        }
        return cambios;
    }

    /**
     * Elimina un nodo identidad y conecta sus dos vecinos directamente.
     */
    private eliminarNodoIdentidad(spider: ZXSpider): void {
        const vecinos = this.obtenerVecinos(spider.id);
        if (vecinos.length !== 2) return; // Seguridad

        const [v1, v2] = vecinos;
        
        // Eliminar el nodo y sus aristas incidentes
        this.eliminarNodo(spider.id);

        // Crear nueva conexión directa entre v1 y v2
        // (Asumiendo conexión simple por ahora)
        const edge = new Hiperedge(uuidv4(), [v1, v2], 1, { tipoConexion: 'Simple' });
        this.agregarHiperedge(edge);
    }
}
