import { ConfiguracionMapeo } from '../neural/tipos';

/**
 * Configuración optimizada para el escenario Distribuido:
 * Servidor GPU (Entrenamiento) ↔️ Cliente CPU (Inferencia/Persistencia)
 */
export const CONFIG_ENTRENAMIENTO_DISTRIBUIDO = {
    // 🖥️ Configuración del Servidor (GPU)
    servidor: {
        dimensionEntrada: 1024,
        dimensionLatente: 128, // Compresión fuerte para transmisión rápida
        
        // Estrategia de Entrenamiento: "Proyección Dispersa"
        // Obliga a la red a generar grafos con pocas aristas pero muy significativas
        estrategia: "SPARSE_TOPOLOGICAL_PROJECTION",
        
        // Factor de penalización de densidad (L1 Regularization)
        // Cuanto más alto, más "limpio" es el grafo para el cliente
        lambdaSparsity: 0.05,
        
        // Maximizar el Spectral Gap asegura que el grafo no se rompa en islas
        maximizarSpectralGap: true
    },

    // 💻 Configuración del Cliente (App Low-Resource)
    cliente: {
        // Modo "Lazy": Solo analiza nodos cuando el usuario los consulta
        analisisPerezoso: true,
        
        // Límite de nodos en memoria RAM antes de forzar persistencia a disco
        maxNodosEnMemoria: 500,
        
        // Intervalo de sincronización con el servidor (ms)
        intervaloSync: 5000,
        
        // Métricas permitidas en CPU de bajos recursos
        metricasHabilitadas: [
            "GRADO",           // O(1)
            "DENSIDAD_LOCAL",  // O(k)
            "CLUSTERING_LOCAL" // O(k^2) - Solo si k es pequeño
        ],
        
        // Métricas prohibidas (requieren GPU o mucha CPU)
        metricasDeshabilitadas: [
            "EIGENVECTOR_CENTRALITY", // O(n^3)
            "BETWEENNESS_GLOBAL",     // O(n*m)
            "MATRIZ_LAPLACIANA"       // O(n^2)
        ]
    },

    // 🌉 Protocolo de Comunicación
    protocolo: {
        formato: "JSON_DELTA", // Solo enviar cambios, no todo el grafo
        compresion: "GZIP"
    }
};

/**
 * Genera la configuración de mapeo adaptada al cliente
 */
export function obtenerConfiguracionCliente(): ConfiguracionMapeo {
    return {
        umbralPeso: 0.3, // Umbral alto para reducir ruido y carga en CPU
        umbralActivacion: 0.6,
        agruparPorCapas: true,
        detectarPatrones: false, // Desactivado por costo computacional en cliente
        tamanoMinimoPatron: 0,
        incluirPesos: true
    };
}
