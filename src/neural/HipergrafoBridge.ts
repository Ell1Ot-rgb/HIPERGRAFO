import { AtomoTopologico } from '../core/AtomoTopologico';
import { Hipergrafo } from '../core/Hipergrafo';
import { Nodo } from '../core/Nodo';
import { Hiperedge } from '../core/Hiperedge';

export interface DecisionColab {
  anomaly_prob: number;
  dendrite_adjustments: number[];
  coherence_state: number[];
  timestamp: string;
  batch_size: number;
  loss?: number;
}

export interface ReporteHipergrafoBridge {
  totalDecisiones: number;
  anomaliasDetectadas: number;
  nodosModificados: number;
  hiperedgesCreadas: number;
  coherenciaPromedio: number;
  ultimaModificacion: string;
}

export class HipergrafoBridge {
  private totalDecisiones: number = 0;
  private anomaliasDetectadas: number = 0;
  private nodosModificados: number = 0;
  private hiperedgesCreadas: number = 0;
  private coherenciaPromedio: number = 0;
  private ultimaModificacion: string = '';

  /**
   * Procesa decisión del Colab y actualiza el Hipergrafo dinámicamente
   */
  procesarDecision(decision: DecisionColab, atomo: AtomoTopologico): void {
    this.totalDecisiones++;
    this.ultimaModificacion = new Date().toISOString();

    // Procesar anomalía si es significativa
    if (decision.anomaly_prob > 0.7) {
      this.marcarAnomalía(atomo.hipergrafo, decision);
    }

    // Aplicar ajustes dendríticos a los parámetros del ONNX
    this.aplicarAjustesDendríticos(atomo, decision.dendrite_adjustments);

    // Validar estructura con el estado de coherencia
    this.validarCoherencia(atomo.hipergrafo, decision.coherence_state);
  }

  /**
   * Marca nodos críticos como anomalía y crea hiperedges de alerta
   */
  private marcarAnomalía(hipergrafo: Hipergrafo, decision: DecisionColab): void {
    this.anomaliasDetectadas++;

    // Crear nodo ANOMALIA
    const nodoAnomaliaId = `ANOMALIA_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const nodoAnomalia = new Nodo(nodoAnomaliaId, {
      tipo: 'ANOMALIA',
      intensidad: decision.anomaly_prob,
      timestamp: decision.timestamp,
      peso: decision.anomaly_prob,
    });

    hipergrafo.agregarNodo(nodoAnomalia);
    this.nodosModificados++;

    // Encontrar nodos críticos (con peso > 0.7)
    const nodosCriticos = hipergrafo.nodos.filter(
      (n) => (n.metadata?.peso ?? 0) > 0.7 && n.id !== nodoAnomaliaId
    );

    // Crear hiperedge de ALERTA conectando anomalía con nodos críticos
    if (nodosCriticos.length > 0) {
      const nodosAlerta = [nodoAnomalia, ...nodosCriticos.slice(0, 5)]; // Limitar a 5 nodos
      const hiperedgeAlerta = new Hiperedge(`EDGE_ALERTA_${Date.now()}`, nodosAlerta, {
        tipo: 'ALERTA',
        peso: decision.anomaly_prob,
        razon: 'anomaly_detected',
        timestamp: decision.timestamp,
      });

      hipergrafo.agregarHiperedge(hiperedgeAlerta);
      this.hiperedgesCreadas++;
    }

    // Aumentar pesos de nodos afectados (multiplicar por (1 + anomaly_prob))
    nodosCriticos.forEach((nodo) => {
      const pesoActual = nodo.metadata?.peso ?? 1.0;
      nodo.metadata = {
        ...nodo.metadata,
        peso: pesoActual * (1 + decision.anomaly_prob),
        ultima_anomalia: decision.timestamp,
      };
      this.nodosModificados++;
    });
  }

  /**
   * Aplica ajustes dendríticos a los parámetros del ONNX del átomo
   * adjustments[0-4]:   LSTM parameters (forget_gate, input_gate, etc)
   * adjustments[5-9]:   Transformer weights (attention parameters)
   * adjustments[10-15]: MLP learning rates
   */
  private aplicarAjustesDendríticos(atomo: AtomoTopologico, adjustments: number[]): void {
    if (!atomo.cerebro || !atomo.cerebro.inputData) {
      return;
    }

    // Aplicar ajustes a los bias/weights del ONNX
    // Los ajustes se mapean a parámetros internos del modelo
    const ajustesLSTM = adjustments.slice(0, 5);
    const ajustesTransformer = adjustments.slice(5, 10);
    const ajustesMLP = adjustments.slice(10, 16);

    // Simulación de aplicación: en producción, esto modificaría
    // los bias del LSTM, pesos del Transformer, y learning rates del MLP
    atomo.metadata = {
      ...atomo.metadata,
      lstm_bias_adjustments: ajustesLSTM,
      transformer_adjustments: ajustesTransformer,
      mlp_adjustments: ajustesMLP,
      last_adjustment_timestamp: new Date().toISOString(),
    };
  }

  /**
   * Valida la estructura del Hipergrafo contra el estado de coherencia (64D)
   */
  private validarCoherencia(hipergrafo: Hipergrafo, coherence_state: number[]): void {
    // Calcular coherencia promedio
    const coherenciaActual = coherence_state.reduce((a, b) => a + b, 0) / coherence_state.length;
    this.coherenciaPromedio = (this.coherenciaPromedio + coherenciaActual) / 2;

    // Validaciones básicas de estructura
    const totalNodos = hipergrafo.nodos.length;
    const totalHiperedges = hipergrafo.hiperedges.length;

    // Ratio mínimo: 1 hiperedge por nodo (en promedio)
    const ratioEsperado = totalHiperedges / totalNodos;
    if (ratioEsperado < 0.5) {
      // Estructura poco conectada, potencial anomalía
      console.warn(
        `[HipergrafoBridge] Baja conectividad detectada: ratio ${ratioEsperado.toFixed(2)}`
      );
    }

    // Si coherencia baja, puede requerir rebalanceo (futuro trabajo)
    if (coherenciaActual < 0.3) {
      console.warn(`[HipergrafoBridge] Coherencia baja: ${coherenciaActual.toFixed(3)}`);
    }
  }

  /**
   * Genera reporte de estadísticas del puente
   */
  generarReporte(): ReporteHipergrafoBridge {
    return {
      totalDecisiones: this.totalDecisiones,
      anomaliasDetectadas: this.anomaliasDetectadas,
      nodosModificados: this.nodosModificados,
      hiperedgesCreadas: this.hiperedgesCreadas,
      coherenciaPromedio: Math.round(this.coherenciaPromedio * 1000) / 1000,
      ultimaModificacion: this.ultimaModificacion,
    };
  }

  /**
   * Reinicia estadísticas
   */
  reiniciarEstadisticas(): void {
    this.totalDecisiones = 0;
    this.anomaliasDetectadas = 0;
    this.nodosModificados = 0;
    this.hiperedgesCreadas = 0;
    this.coherenciaPromedio = 0;
    this.ultimaModificacion = '';
  }
}
