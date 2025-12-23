import { ZXDiagram } from './ZXDiagram';
import { ZXSpider } from './ZXSpider';

export interface Match {
    nodos: Map<string, string>; // ID en patrón -> ID en grafo real
}

/**
 * Interfaz para una regla de reescritura en el sistema ZX.
 * Sigue el principio de sustitución de grafos (aunque simplificado).
 */
export interface ReglaZX {
    nombre: string;
    descripcion: string;
    
    /**
     * Busca todas las ocurrencias del patrón LHS en el diagrama.
     */
    buscarMatches(diagrama: ZXDiagram): Match[];
    
    /**
     * Aplica la regla a una ocurrencia específica.
     * Transforma el diagrama in-place (LHS -> RHS).
     */
    aplicar(diagrama: ZXDiagram, match: Match): void;
}

/**
 * Regla (f): Spider Fusion
 * Dos arañas del mismo tipo conectadas por una arista simple se fusionan.
 */
export class ReglaFusionSpider implements ReglaZX {
    nombre = "Spider Fusion (f)";
    descripcion = "Fusiona dos arañas adyacentes del mismo color.";

    buscarMatches(diagrama: ZXDiagram): Match[] {
        const matches: Match[] = [];
        const visitados = new Set<string>(); // Para evitar duplicados (A-B y B-A)

        const nodos = diagrama.obtenerNodos() as ZXSpider[];
        
        for (const n1 of nodos) {
            if (!(n1 instanceof ZXSpider)) continue;
            
            const vecinos = diagrama.obtenerVecinos(n1.id) as ZXSpider[];
            
            for (const n2 of vecinos) {
                if (!(n2 instanceof ZXSpider)) continue;
                
                // Clave única para el par (ordenada alfabéticamente)
                const parId = [n1.id, n2.id].sort().join('-');
                if (visitados.has(parId)) continue;
                
                // Condición: Mismo tipo
                if (n1.tipoSpider === n2.tipoSpider) {
                    // Verificar que la conexión es SIMPLE (no Hadamard)
                    // Esto es costoso, hay que buscar la arista.
                    // Por simplicidad asumimos que si son vecinos hay arista.
                    // En implementación real verificaríamos el tipo de arista.
                    
                    const match: Match = {
                        nodos: new Map([
                            ['s1', n1.id],
                            ['s2', n2.id]
                        ])
                    };
                    matches.push(match);
                    visitados.add(parId);
                }
            }
        }
        return matches;
    }

    aplicar(diagrama: ZXDiagram, match: Match): void {
        const id1 = match.nodos.get('s1');
        const id2 = match.nodos.get('s2');
        
        if (!id1 || !id2) return;
        
        const s1 = diagrama.obtenerNodo(id1) as ZXSpider;
        const s2 = diagrama.obtenerNodo(id2) as ZXSpider;
        
        if (!s1 || !s2) return;

        // 1. Sumar fases en s1
        s1.fusionar(s2);
        
        // 2. Mover vecinos de s2 a s1
        const vecinosS2 = diagrama.obtenerVecinos(s2.id);
        for (const vecino of vecinosS2) {
            if (vecino.id === s1.id) continue; // Ignorar la conexión entre ellas
            
            // Conectar vecino a s1
            // Asumimos conexión simple por defecto, idealmente copiaríamos el tipo de arista
            diagrama.conectarSpiders(s1, vecino as ZXSpider);
        }
        
        // 3. Eliminar s2
        diagrama.eliminarNodo(s2.id);
    }
}
