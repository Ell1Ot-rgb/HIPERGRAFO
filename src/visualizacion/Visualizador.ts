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
        const { sensorial, contexto, decision, coherencia } = resultado;

        // Crear un estado base si no existe
        if (!this.ultimoEstado) {
            this.ultimoEstado = {
                timestamp: Date.now(),
                grafo: { nodes: [], edges: [] },
                metricas: {},
                feedback: null,
                fisica: null,
                neuronal: null,
                coherencia: null,
                cognicion: {}
            };
        }

        // 1. Mapear la Imagen Mental (Hipergrafo de Coherencia) a un formato visual
        const nodosGrafo: any[] = [];
        const edgesGrafo: any[] = [];
        if (coherencia && coherencia.imagenMental) {
            coherencia.imagenMental.obtenerNodos().forEach((n: any) => {
                nodosGrafo.push({
                    data: { 
                        id: n.id, 
                        label: n.label.substring(0, 20), // Acortar etiquetas largas
                        tipo: n.metadata.tipo || 'CONCEPTO',
                        valor: n.metadata.relevancia || n.metadata.valor || 0
                    }
                });
            });
            coherencia.imagenMental.obtenerHiperedges().forEach((edge: any) => {
                const edgeId = `edge_${edge.id}`;
                nodosGrafo.push({
                    data: { id: edgeId, label: '', tipo: 'HIPEREDGE', valor: 0 }
                });
                edge.nodos.forEach((nodoId: any) => {
                    edgesGrafo.push({
                        data: { source: nodoId, target: edgeId }
                    });
                });
            });
        }
        
        // 2. Fusionar el estado
        this.ultimoEstado = {
            ...this.ultimoEstado,
            timestamp: Date.now(),
            grafo: { nodes: nodosGrafo, edges: edgesGrafo },
            cognicion: {
                sensorial,
                contexto,
                decision
            },
            coherencia: {
                ...coherencia,
                imagenMental: undefined // No enviar el objeto completo, ya está en 'grafo'
            },
            neuronal: contexto, // Para compatibilidad con vistas antiguas
        };
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
