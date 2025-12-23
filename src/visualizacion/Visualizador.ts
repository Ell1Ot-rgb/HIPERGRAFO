import express from 'express';
import * as http from 'http';
import * as path from 'path';
import { ResultadoAnalisis } from '../orquestador';

export class Visualizador {
    private app: express.Application;
    private server: http.Server | null = null;
    private ultimoEstado: any = null;
    private ultimoFeedback: any = null;
    private ultimaFisica: any = null;
    private ultimaNeuronal: any = null;
    private ultimaCoherencia: any = null;
    private puerto: number;

    constructor(puerto: number = 3000) {
        this.puerto = puerto;
        this.app = express();
        this.configurarRutas();
    }

    private configurarRutas() {
        // Servir archivos estáticos
        this.app.use(express.static(path.join(__dirname, 'public')));

        // Endpoint de datos
        this.app.get('/api/estado', (_req, res) => {
            res.json(this.ultimoEstado || { status: 'esperando_datos' });
        });
    }

    public actualizarEstado(resultado: ResultadoAnalisis, feedbackIA?: any) {
        if (feedbackIA) this.ultimoFeedback = feedbackIA;
        
        // Simplificar el grafo para visualización
        const nodos = resultado.hipergrafo.obtenerNodos().map(n => ({
            data: { 
                id: n.id, 
                label: n.label,
                tipo: n.metadata.tipo || 'GENERICO',
                valor: n.metadata.valor || 0
            }
        }));

        const edges: any[] = [];
        resultado.hipergrafo.obtenerHiperedges().forEach(edge => {
            // Representar hiperedge como un nodo central conectado a sus miembros
            const edgeId = edge.id;
            nodos.push({
                data: { id: edgeId, label: edge.label, tipo: 'HIPEREDGE', valor: 0 }
            });

            edge.nodos.forEach(nodoId => {
                edges.push({
                    data: { source: nodoId, target: edgeId }
                });
            });
        });

        this.ultimoEstado = {
            timestamp: resultado.timestamp,
            grafo: { nodes: nodos, edges: edges },
            metricas: resultado.estadoAnalisis,
            feedback: this.ultimoFeedback,
            fisica: this.ultimaFisica,
            neuronal: this.ultimaNeuronal,
            coherencia: this.ultimaCoherencia,
            memoria: (resultado.estadoAnalisis as any).memoria 
        };
    }

    public actualizarNeuronal(neuronal: any) {
        this.ultimaNeuronal = neuronal;
    }

    public actualizarCoherencia(coherencia: any) {
        this.ultimaCoherencia = coherencia;
    }

    public actualizarFeedback(feedback: any) {
        this.ultimoFeedback = feedback;
        if (this.ultimoEstado) {
            this.ultimoEstado.feedback = feedback;
        }
    }

    public actualizarFisica(fisica: any) {
        this.ultimaFisica = fisica;
        if (this.ultimoEstado) {
            this.ultimoEstado.fisica = fisica;
        }
    }

    /**
     * Actualiza el estado con los resultados de la jerarquía cognitiva (Capas 0-3)
     */
    public actualizarCognicion(resultado: any) {
        // Si no hay un estado base, creamos uno mínimo para que el frontend no falle
        if (!this.ultimoEstado) {
            this.ultimoEstado = {
                timestamp: Date.now(),
                grafo: { nodes: [], edges: [] },
                metricas: {},
                feedback: null,
                fisica: null,
                neuronal: null,
                coherencia: null
            };
        }

        // Integrar datos de cognición
        this.ultimoEstado.cognicion = {
            sensorial: resultado.sensorial, // 25 átomos
            contexto: resultado.contexto,   // Capa 2 (LSTM/Transformer)
            decision: resultado.decision,   // Capa 3 (Urgencia/Confianza)
            coherencia: resultado.coherencia // Imagen Mental
        };

        // Actualizar coherencia y neuronal para compatibilidad
        this.ultimaCoherencia = resultado.coherencia;
        this.ultimaNeuronal = resultado.contexto;
    }

    public iniciar() {
        // Escuchar en 0.0.0.0 para permitir acceso externo desde Codespaces
        this.server = this.app.listen(this.puerto, '0.0.0.0', () => {
            console.log(`📊 Visualizador activo en puerto ${this.puerto}`);
        });
    }

    public detener() {
        if (this.server) {
            this.server.close();
        }
    }
}
