/**
 * CapaCognitivaV2.ts
 * 
 * Implementa la Capa 3 (Consenso y Ejecución) de la Corteza Cognitiva.
 * Versión mejorada basada en el informe de fusión de datos.
 * 
 * Mejoras:
 * 1. Sistema de votación ponderada para decisiones (Late Fusion).
 * 2. Historial con decaimiento exponencial (memoria a largo plazo).
 * 3. Detector de patrones recurrentes.
 * 4. Umbrales adaptativos basados en historial.
 * 5. Sistema de confianza multi-factor.
 */

import { SalidaEspacioTemporal } from './CapaEspacioTemporalV2';

/**
 * Tipos de decisión cognitiva
 */
export type TipoDecision = 
    | 'MONITOREO'      // Operación normal
    | 'ALERTA_LEVE'    // Anomalía detectada, monitoreo aumentado
    | 'ALERTA_GRAVE'   // Anomalía significativa, requiere atención
    | 'INTERVENCION'   // Acción correctiva necesaria
    | 'APRENDIZAJE'    // Datos nuevos/desconocidos, solicitar entrenamiento
    | 'ESTABILIZACION';// Transición de alerta a normal

/**
 * Decisión cognitiva con contexto enriquecido
 */
export interface DecisionCognitiva {
    tipo: TipoDecision;
    descripcion: string;
    nivelUrgencia: number;           // 0.0 a 1.0
    confianzaDecision: number;       // Confianza en la decisión tomada
    factoresContribuyentes: {
        nombre: string;
        peso: number;
        valor: number;
    }[];
    recomendaciones: string[];
    metadata: Record<string, any>;
    timestamp: Date;
}

/**
 * Configuración de la capa cognitiva
 */
export interface ConfiguracionCognitiva {
    umbralAlertaLeve: number;
    umbralAlertaGrave: number;
    umbralIntervencion: number;
    umbralAprendizaje: number;        // Umbral de confianza para solicitar aprendizaje
    ventanaHistorial: number;         // Tamaño del historial
    decaimientoMemoria: number;       // Factor de decaimiento (0-1)
    umbralPatronRecurrente: number;   // Veces que debe repetirse para ser patrón
}

const CONFIG_DEFAULT: ConfiguracionCognitiva = {
    umbralAlertaLeve: 0.5,
    umbralAlertaGrave: 0.75,
    umbralIntervencion: 0.9,
    umbralAprendizaje: 0.3,
    ventanaHistorial: 100,
    decaimientoMemoria: 0.95,
    umbralPatronRecurrente: 3
};

/**
 * Entrada histórica con peso decaído
 */
interface EntradaHistorial {
    decision: DecisionCognitiva;
    pesoActual: number;              // Decae con el tiempo
    timestep: number;
}

/**
 * Capa 3: Cognición y Consenso
 */
export class CapaCognitiva {
    private config: ConfiguracionCognitiva;
    private historial: EntradaHistorial[] = [];
    private contadorPatrones: Map<TipoDecision, number> = new Map();
    private ultimaDecision: DecisionCognitiva | null = null;
    private timestep: number = 0;
    
    // Umbrales adaptativos (se ajustan con el historial)
    private umbralesAdaptativos: {
        leve: number;
        grave: number;
        intervencion: number;
    };

    constructor(config: Partial<ConfiguracionCognitiva> = {}) {
        this.config = { ...CONFIG_DEFAULT, ...config };
        this.umbralesAdaptativos = {
            leve: this.config.umbralAlertaLeve,
            grave: this.config.umbralAlertaGrave,
            intervencion: this.config.umbralIntervencion
        };
    }

    /**
     * Procesa la salida de la Capa 2 y genera una decisión
     */
    public async procesar(entrada: SalidaEspacioTemporal): Promise<DecisionCognitiva> {
        this.timestep++;
        this.decaerHistorial();

        // 1. Calcular factores de decisión
        const factores = this.calcularFactores(entrada);
        
        // 2. Score compuesto ponderado
        const scoreCompuesto = this.calcularScoreCompuesto(factores);
        
        // 3. Determinar tipo de decisión
        const tipo = this.determinarTipo(scoreCompuesto, entrada);
        
        // 4. Calcular confianza de la decisión
        const confianza = this.calcularConfianzaDecision(entrada, factores);
        
        // 5. Generar recomendaciones
        const recomendaciones = this.generarRecomendaciones(tipo, entrada, factores);
        
        // 6. Construir decisión
        const decision: DecisionCognitiva = {
            tipo,
            descripcion: this.generarDescripcion(tipo, entrada),
            nivelUrgencia: this.calcularUrgencia(scoreCompuesto, tipo),
            confianzaDecision: confianza,
            factoresContribuyentes: factores,
            recomendaciones,
            metadata: {
                scoreCompuesto,
                modalidadDominante: entrada.modalidadDominante,
                metricas: entrada.metricas,
                timestep: this.timestep,
                umbralesActuales: { ...this.umbralesAdaptativos }
            },
            timestamp: new Date()
        };

        // 7. Registrar y actualizar patrones
        this.registrarDecision(decision);
        this.actualizarPatrones(tipo);
        this.ajustarUmbrales();

        this.ultimaDecision = decision;
        return decision;
    }

    /**
     * Calcula factores que contribuyen a la decisión
     */
    private calcularFactores(entrada: SalidaEspacioTemporal): {
        nombre: string;
        peso: number;
        valor: number;
    }[] {
        return [
            {
                nombre: 'score_anomalia',
                peso: 0.35,
                valor: entrada.scoreAnomalia
            },
            {
                nombre: 'confianza_modelo',
                peso: 0.25,
                valor: 1 - entrada.confianza // Invertido: baja confianza = más peso
            },
            {
                nombre: 'inestabilidad_secuencia',
                peso: 0.20,
                valor: 1 - entrada.metricas.estabilidadSecuencia
            },
            {
                nombre: 'divergencia_modalidades',
                peso: 0.10,
                valor: Math.abs(entrada.metricas.ratioFusion - 0.5) * 2 // Lejos de 0.5 = divergencia
            },
            {
                nombre: 'patron_recurrente',
                peso: 0.10,
                valor: this.calcularFactorPatronRecurrente()
            }
        ];
    }

    /**
     * Score compuesto ponderado
     */
    private calcularScoreCompuesto(factores: { peso: number; valor: number }[]): number {
        return factores.reduce((acc, f) => acc + f.peso * f.valor, 0);
    }

    /**
     * Determina el tipo de decisión basándose en umbrales adaptativos
     */
    private determinarTipo(score: number, entrada: SalidaEspacioTemporal): TipoDecision {
        // Caso especial: baja confianza sugiere necesidad de aprendizaje
        if (entrada.confianza < this.config.umbralAprendizaje) {
            return 'APRENDIZAJE';
        }

        // Transición: de alerta a normal (estabilización)
        if (this.ultimaDecision && 
            this.ultimaDecision.tipo !== 'MONITOREO' && 
            score < this.umbralesAdaptativos.leve * 0.8) {
            return 'ESTABILIZACION';
        }

        // Umbrales de severidad
        if (score >= this.umbralesAdaptativos.intervencion) {
            return 'INTERVENCION';
        }
        if (score >= this.umbralesAdaptativos.grave) {
            return 'ALERTA_GRAVE';
        }
        if (score >= this.umbralesAdaptativos.leve) {
            return 'ALERTA_LEVE';
        }

        return 'MONITOREO';
    }

    /**
     * Calcula confianza en la decisión tomada
     */
    private calcularConfianzaDecision(
        entrada: SalidaEspacioTemporal,
        factores: { valor: number }[]
    ): number {
        // Base: confianza del modelo
        let confianza = entrada.confianza;

        // Penalizar si hay mucha varianza entre factores
        const valores = factores.map(f => f.valor);
        const media = valores.reduce((a, b) => a + b, 0) / valores.length;
        const varianza = valores.reduce((a, v) => a + Math.pow(v - media, 2), 0) / valores.length;
        confianza *= (1 - varianza * 0.5);

        // Bonus si hay consistencia con historial
        if (this.historial.length > 5) {
            const consistencia = this.calcularConsistenciaHistorial();
            confianza = confianza * 0.7 + consistencia * 0.3;
        }

        return Math.max(0, Math.min(1, confianza));
    }

    /**
     * Calcula qué tan consistentes han sido las decisiones recientes
     */
    private calcularConsistenciaHistorial(): number {
        if (this.historial.length < 3) return 0.5;

        const ultimas = this.historial.slice(-10);
        const tipos = ultimas.map(h => h.decision.tipo);
        
        // Contar transiciones de tipo
        let transiciones = 0;
        for (let i = 1; i < tipos.length; i++) {
            if (tipos[i] !== tipos[i - 1]) transiciones++;
        }

        // Pocas transiciones = alta consistencia
        return 1 - (transiciones / (tipos.length - 1));
    }

    /**
     * Factor de patrón recurrente
     */
    private calcularFactorPatronRecurrente(): number {
        const maxContador = Math.max(...Array.from(this.contadorPatrones.values()), 0);
        return Math.min(maxContador / this.config.umbralPatronRecurrente, 1);
    }

    /**
     * Genera recomendaciones basadas en el estado
     */
    private generarRecomendaciones(
        tipo: TipoDecision,
        entrada: SalidaEspacioTemporal,
        factores: { nombre: string; peso: number; valor: number }[]
    ): string[] {
        const recomendaciones: string[] = [];

        switch (tipo) {
            case 'INTERVENCION':
                recomendaciones.push('Ejecutar protocolo de emergencia');
                recomendaciones.push('Notificar al operador humano');
                break;
            case 'ALERTA_GRAVE':
                recomendaciones.push('Aumentar frecuencia de monitoreo');
                recomendaciones.push('Preparar protocolos de contingencia');
                break;
            case 'ALERTA_LEVE':
                recomendaciones.push('Continuar observación');
                break;
            case 'APRENDIZAJE':
                recomendaciones.push('Almacenar datos para re-entrenamiento');
                recomendaciones.push('Solicitar etiquetado de datos');
                break;
            case 'ESTABILIZACION':
                recomendaciones.push('Reducir nivel de alerta gradualmente');
                break;
        }

        // Recomendaciones basadas en factores dominantes
        const factorDominante = factores.reduce((a, b) => 
            a.valor * a.peso > b.valor * b.peso ? a : b
        );

        if (factorDominante.nombre === 'inestabilidad_secuencia' && factorDominante.valor > 0.7) {
            recomendaciones.push('Alta variabilidad detectada: considerar filtrado de señal');
        }

        if (entrada.modalidadDominante !== 'equilibrado') {
            recomendaciones.push(
                `Modalidad ${entrada.modalidadDominante} dominante: ` +
                `verificar integridad de sensores ${entrada.modalidadDominante === 'temporal' ? 'espaciales' : 'temporales'}`
            );
        }

        return recomendaciones;
    }

    /**
     * Genera descripción legible del estado
     */
    private generarDescripcion(tipo: TipoDecision, entrada: SalidaEspacioTemporal): string {
        const descripciones: Record<TipoDecision, string> = {
            'MONITOREO': 'Sistema operando dentro de parámetros normales',
            'ALERTA_LEVE': 'Anomalía leve detectada, requiere observación',
            'ALERTA_GRAVE': 'Anomalía significativa detectada, evaluar intervención',
            'INTERVENCION': 'Estado crítico, intervención inmediata requerida',
            'APRENDIZAJE': 'Patrón desconocido detectado, modelo requiere actualización',
            'ESTABILIZACION': 'Sistema retornando a operación normal'
        };

        let desc = descripciones[tipo];
        
        if (entrada.anomaliaDetectada) {
            desc += ` (Score: ${(entrada.scoreAnomalia * 100).toFixed(1)}%)`;
        }

        return desc;
    }

    /**
     * Calcula nivel de urgencia
     */
    private calcularUrgencia(score: number, tipo: TipoDecision): number {
        const baseUrgencia: Record<TipoDecision, number> = {
            'MONITOREO': 0.1,
            'ALERTA_LEVE': 0.4,
            'ALERTA_GRAVE': 0.7,
            'INTERVENCION': 0.95,
            'APRENDIZAJE': 0.3,
            'ESTABILIZACION': 0.2
        };

        // Ajustar por score real
        return Math.min(1, baseUrgencia[tipo] * (0.5 + score * 0.5));
    }

    /**
     * Registra decisión en historial con decaimiento
     */
    private registrarDecision(decision: DecisionCognitiva): void {
        this.historial.push({
            decision,
            pesoActual: 1.0,
            timestep: this.timestep
        });

        if (this.historial.length > this.config.ventanaHistorial) {
            this.historial.shift();
        }
    }

    /**
     * Aplica decaimiento exponencial al historial
     */
    private decaerHistorial(): void {
        for (const entrada of this.historial) {
            entrada.pesoActual *= this.config.decaimientoMemoria;
        }
    }

    /**
     * Actualiza contadores de patrones
     */
    private actualizarPatrones(tipo: TipoDecision): void {
        const actual = this.contadorPatrones.get(tipo) || 0;
        this.contadorPatrones.set(tipo, actual + 1);

        // Decaer otros patrones
        for (const [key, value] of this.contadorPatrones.entries()) {
            if (key !== tipo) {
                this.contadorPatrones.set(key, Math.max(0, value - 0.5));
            }
        }
    }

    /**
     * Ajusta umbrales basándose en el historial (umbral adaptativo)
     */
    private ajustarUmbrales(): void {
        if (this.historial.length < 10) return;

        const ultimas = this.historial.slice(-20);
        const urgencias = ultimas.map(h => h.decision.nivelUrgencia);
        const mediaUrgencia = urgencias.reduce((a, b) => a + b, 0) / urgencias.length;

        // Si la urgencia promedio es muy alta, subir umbrales (evitar fatiga de alertas)
        if (mediaUrgencia > 0.7) {
            this.umbralesAdaptativos.leve = Math.min(0.6, this.config.umbralAlertaLeve * 1.1);
            this.umbralesAdaptativos.grave = Math.min(0.85, this.config.umbralAlertaGrave * 1.1);
        }
        // Si es muy baja, bajar umbrales para ser más sensible
        else if (mediaUrgencia < 0.2) {
            this.umbralesAdaptativos.leve = Math.max(0.4, this.config.umbralAlertaLeve * 0.9);
            this.umbralesAdaptativos.grave = Math.max(0.65, this.config.umbralAlertaGrave * 0.9);
        }
    }

    /**
     * Obtiene última decisión
     */
    public obtenerUltimaDecision(): DecisionCognitiva | null {
        return this.ultimaDecision;
    }

    /**
     * Obtiene estadísticas del sistema
     */
    public obtenerEstadisticas(): {
        totalDecisiones: number;
        distribucionTipos: Record<TipoDecision, number>;
        urgenciaPromedio: number;
        confianzaPromedio: number;
        umbralesActuales: { leve: number; grave: number; intervencion: number };
    } {
        const distribucion: Record<TipoDecision, number> = {
            'MONITOREO': 0,
            'ALERTA_LEVE': 0,
            'ALERTA_GRAVE': 0,
            'INTERVENCION': 0,
            'APRENDIZAJE': 0,
            'ESTABILIZACION': 0
        };

        let sumaUrgencia = 0;
        let sumaConfianza = 0;

        for (const entrada of this.historial) {
            distribucion[entrada.decision.tipo]++;
            sumaUrgencia += entrada.decision.nivelUrgencia;
            sumaConfianza += entrada.decision.confianzaDecision;
        }

        const total = this.historial.length || 1;

        return {
            totalDecisiones: this.historial.length,
            distribucionTipos: distribucion,
            urgenciaPromedio: sumaUrgencia / total,
            confianzaPromedio: sumaConfianza / total,
            umbralesActuales: { ...this.umbralesAdaptativos }
        };
    }

    /**
     * Resetea el estado cognitivo
     */
    public resetear(): void {
        this.historial = [];
        this.contadorPatrones.clear();
        this.ultimaDecision = null;
        this.timestep = 0;
        this.umbralesAdaptativos = {
            leve: this.config.umbralAlertaLeve,
            grave: this.config.umbralAlertaGrave,
            intervencion: this.config.umbralIntervencion
        };
    }
}
