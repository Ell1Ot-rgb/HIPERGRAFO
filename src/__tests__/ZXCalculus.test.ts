import { ZXDiagram } from '../zx/ZXDiagram';
import { ZXSpider } from '../zx/ZXSpider';

describe('Motor de Cálculo ZX', () => {
    let diagrama: ZXDiagram;

    beforeEach(() => {
        diagrama = new ZXDiagram();
    });

    test('Debe crear Spiders Z y X correctamente', () => {
        const zSpider = diagrama.crearSpider('Z', 0.5);
        const xSpider = diagrama.crearSpider('X', 0.25);

        expect(zSpider.tipoSpider).toBe('Z');
        expect(zSpider.fase).toBe(0.5);
        expect(xSpider.tipoSpider).toBe('X');
        expect(diagrama.obtenerNodos().length).toBe(2);
    });

    test('Debe conectar Spiders', () => {
        const s1 = diagrama.crearSpider('Z');
        const s2 = diagrama.crearSpider('X');
        
        const edge = diagrama.conectarSpiders(s1, s2);
        
        expect(diagrama.estaConectados(s1.id, s2.id)).toBe(true);
        expect(edge.contiene(s1.id)).toBe(true);
    });

    test('Debe fusionar Spiders del mismo color', () => {
        const s1 = new ZXSpider('id1', 'Z', 0.25);
        const s2 = new ZXSpider('id2', 'Z', 0.25);

        s1.fusionar(s2);

        expect(s1.fase).toBe(0.5); // 0.25 + 0.25
    });

    test('Debe lanzar error al fusionar Spiders de distinto color', () => {
        const s1 = new ZXSpider('id1', 'Z');
        const s2 = new ZXSpider('id2', 'X');

        expect(() => s1.fusionar(s2)).toThrow();
    });

    test('Debe simplificar identidad (Spider fase 0 con grado 2)', () => {
        // Configuración: A -- (Id) -- B
        const a = diagrama.crearSpider('Z', 0.5); // Nodo externo
        const idNode = diagrama.crearSpider('Z', 0); // Identidad
        const b = diagrama.crearSpider('X', 0.5); // Nodo externo

        diagrama.conectarSpiders(a, idNode);
        diagrama.conectarSpiders(idNode, b);

        expect(diagrama.obtenerNodos().length).toBe(3);
        expect(diagrama.estaConectados(a.id, b.id)).toBe(false);

        // Ejecutar simplificación
        const cambios = diagrama.simplificarIdentidad();

        expect(cambios).toBe(1);
        expect(diagrama.obtenerNodos().length).toBe(2); // idNode eliminado
        expect(diagrama.estaConectados(a.id, b.id)).toBe(true); // A conectado a B
    });
});
