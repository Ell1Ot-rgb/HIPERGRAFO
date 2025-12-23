import { ZXDiagram } from '../zx/ZXDiagram';
import { MotorZX } from '../zx/MotorZX';
import { ReglaFusionSpider } from '../zx/Reglas';

describe('Motor de Reescritura ZX', () => {
    let diagrama: ZXDiagram;
    let motor: MotorZX;

    beforeEach(() => {
        diagrama = new ZXDiagram();
        motor = new MotorZX();
        motor.agregarRegla(new ReglaFusionSpider());
    });

    test('Debe fusionar dos spiders Z conectados', () => {
        // Configuración: A(Z, 0.2) -- B(Z, 0.3)
        const a = diagrama.crearSpider('Z', 0.2);
        const b = diagrama.crearSpider('Z', 0.3);
        diagrama.conectarSpiders(a, b);

        expect(diagrama.obtenerNodos().length).toBe(2);

        // Ejecutar motor
        const pasos = motor.ejecutarHastaConvergencia(diagrama);

        expect(pasos).toBe(1);
        expect(diagrama.obtenerNodos().length).toBe(1);
        
        const nodoRestante = diagrama.obtenerNodos()[0] as any;
        expect(nodoRestante.tipoSpider).toBe('Z');
        expect(nodoRestante.fase).toBeCloseTo(0.5);
    });

    test('No debe fusionar spiders de distinto color', () => {
        // Configuración: A(Z) -- B(X)
        const a = diagrama.crearSpider('Z', 0.2);
        const b = diagrama.crearSpider('X', 0.3);
        diagrama.conectarSpiders(a, b);

        const pasos = motor.ejecutarHastaConvergencia(diagrama);

        expect(pasos).toBe(0);
        expect(diagrama.obtenerNodos().length).toBe(2);
    });

    test('Debe manejar cadenas de fusión', () => {
        // A(Z) -- B(Z) -- C(Z) -> ABC(Z)
        const a = diagrama.crearSpider('Z', 0.1);
        const b = diagrama.crearSpider('Z', 0.1);
        const c = diagrama.crearSpider('Z', 0.1);
        
        diagrama.conectarSpiders(a, b);
        diagrama.conectarSpiders(b, c);

        const pasos = motor.ejecutarHastaConvergencia(diagrama);

        // Puede tomar 2 pasos: (A+B)+C o A+(B+C)
        expect(pasos).toBeGreaterThanOrEqual(2);
        expect(diagrama.obtenerNodos().length).toBe(1);
        
        const final = diagrama.obtenerNodos()[0] as any;
        expect(final.fase).toBeCloseTo(0.3);
    });
});
