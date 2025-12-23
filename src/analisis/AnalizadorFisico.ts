/**
 * Analizador Físico de Hipergrafo
 * 
 * Extrae métricas inspiradas en la física de la información y la teoría de Wolfram
 * a partir del estado del hipergrafo y el feedback de la red neuronal.
 */

export interface MetricasFisicas {
    tension: number;          // Estrés topológico (0-100)
    curvatura: number;        // Desviación de la centralidad (Gravedad de info)
    dimensionFractal: number; // Complejidad estructural
    fatiga: number;           // Acumulación de inestabilidad
    tiempoRelajacion: number; // Ms para volver al equilibrio
    physicsLoss: {            // Pérdidas físicas reales (Omega 21)
        energia: number;
        termo: number;
        causal: number;
        entropia: number;
    };
}

export class AnalizadorFisico {
    private historialLoss: number[] = [];
    private ultimoSpike: number = 0;
    private fatigaAcumulada: number = 0;

    /**
     * Analiza el hipergrafo y la telemetría para extraer métricas físicas
     */
    analizar(hipergrafo: any, telemetria: any, feedbackIA: any = {}): MetricasFisicas {
        // Simular un EstadoAnalisis básico si no se proporciona
        const numNodos = hipergrafo.obtenerNodos().length;
        const numAristas = hipergrafo.obtenerHiperedges().length;
        const estadoSimulado: any = {
            numNodos: numNodos,
            numAristas: numAristas,
            densidad: numAristas / (numNodos || 1),
            centralidadMaxima: 1.0, // Simplificación
        };

        return this.calcular(estadoSimulado, feedbackIA, telemetria);
    }

    /**
     * Calcula las métricas físicas basadas en el estado actual y el feedback de la IA
     */
    calcular(estado: any, feedbackIA: any, telemetria?: any): MetricasFisicas {
        // 1. Tensión Topológica (Fuerza de estiramiento)
        const prediccion = feedbackIA.prediccion_estabilidad || 0;
        const tension = Math.abs(estado.densidad - prediccion) * 100;

        // 2. Curvatura (Gravedad de la información)
        const curvatura = estado.centralidadMaxima / (estado.densidad + 0.01);

        // 3. Dimensión Fractal (Complejidad)
        const dimensionFractal = estado.numNodos > 1 
            ? Math.log(estado.numAristas + 1) / Math.log(estado.numNodos)
            : 1;

        // 4. Fatiga (Histéresis)
        if (feedbackIA.loss !== undefined) {
            this.historialLoss.push(feedbackIA.loss);
            if (this.historialLoss.length > 20) this.historialLoss.shift();
            
            const promedioLoss = this.historialLoss.reduce((a, b) => a + b, 0) / this.historialLoss.length;
            this.fatigaAcumulada = (this.fatigaAcumulada * 0.95) + (promedioLoss * 5);
        }

        // 5. Pérdidas Físicas Reales (Omega 21)
        const physicsLoss = {
            energia: telemetria?.dendrites?.power > 5000 ? Math.pow(telemetria.dendrites.power - 5000, 2) : 0,
            termo: telemetria?.dendrites?.dew_temp > telemetria?.dendrites?.altitude ? 100 : 0,
            causal: telemetria?.dendrites?.velocity > 100 ? 500 : 0,
            entropia: telemetria?.logic?.h || 0
        };

        // 6. Tiempo de Relajación
        let tiempoRelajacion = 0;
        const ahora = Date.now();
        if (estado.ultimoSpike) {
            this.ultimoSpike = ahora;
        }
        
        if (this.ultimoSpike > 0) {
            if ((feedbackIA.loss || 0) < 0.01) {
                tiempoRelajacion = ahora - this.ultimoSpike;
            }
        }

        return {
            tension: Math.min(tension, 100),
            curvatura: Math.min(curvatura, 50),
            dimensionFractal: Number(dimensionFractal.toFixed(3)),
            fatiga: Math.min(this.fatigaAcumulada * 10, 100),
            tiempoRelajacion,
            physicsLoss
        };
    }

    /**
     * Genera un reporte textual de la "salud física" del sistema
     */
    generarDiagnostico(metricas: MetricasFisicas): string {
        if (metricas.tension > 70) return "⚠️ ALTA TENSIÓN: El sistema está siendo estirado al límite por el vector 256D.";
        if (metricas.fatiga > 50) return "💤 FATIGA DETECTADA: La red de 1024 neuronas necesita tiempo de relajación.";
        if (metricas.curvatura > 30) return "🕳️ CURVATURA CRÍTICA: Se ha formado un atractor masivo de información.";
        return "✅ ESTADO ÓPTIMO: La liga topológica mantiene su forma persistente.";
    }
}
