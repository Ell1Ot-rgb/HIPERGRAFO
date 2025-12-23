# ⚡ Arquitectura Distribuida: Proyección Topológica Dispersa
## Optimización para Cliente de Bajos Recursos

### 🎯 Objetivo
Entrenar una Red Neuronal (en GPU) para generar Hipergrafos Persistentes que sean consumidos eficientemente por una App Cliente (CPU Limitada).

### 🏗️ Diagrama de Componentes

```mermaid
graph TD
    subgraph "🖥️ SERVIDOR (GPU 3D)"
        RN[Red Neuronal 1024d]
        PROY[Capa de Proyección Topológica]
        OPT[Optimizador Espectral]
        SERIAL[Serializador Ligero]
        
        RN --> PROY
        PROY --> OPT
        OPT --> SERIAL
    end

    subgraph "🌐 RED"
        JSON[JSON Disperso (Payload < 50kb)]
    end

    subgraph "💻 CLIENTE (App Low-Resource)"
        GEN[Generador de Instancias]
        DB[(Persistencia Local)]
        VIS[Visualizador]
        
        SERIAL --> JSON --> GEN
        GEN --> DB
        GEN --> VIS
    end
```

### 🧠 Estrategia de Entrenamiento (Server-Side)

El entrenamiento ocurre **exclusivamente en el servidor**. La función de pérdida está diseñada para facilitar la vida del cliente.

$$ \mathcal{L}_{total} = \mathcal{L}_{topología} + \lambda_{sparsity} \cdot ||A||_1 $$

1.  **$\mathcal{L}_{topología}$**: Maximiza la conectividad útil (Spectral Gap).
2.  **$\lambda_{sparsity}$ (Penalización de Densidad)**: **CRUCIAL**. Castiga a la red si crea demasiadas conexiones. Obliga al modelo a elegir solo las aristas más importantes. Esto reduce drásticamente el uso de RAM en el cliente.

### 🚀 Flujo de la App Cliente

1.  **Conexión**: Solicita inferencia al servidor.
2.  **Recepción**: Recibe lista de adyacencia (no matriz densa).
3.  **Hidratación**: Convierte IDs en objetos `Nodo` y `Hiperedge`.
4.  **Persistencia**: Guarda en disco local (JSON/SQLite) solo la estructura topológica.
5.  **Análisis Ligero**:
    *   ❌ NO calcula Eigenvalores (muy caro para CPU).
    *   ✅ Calcula Grado y Centralidad Local (muy barato).

### 📦 Formato de Datos Optimizado

En lugar de enviar arrays de 1024 floats, enviamos "deltas":

```json
{
  "timestamp": 17000000,
  "nodos_activos": [12, 45, 89], // Solo los que cambiaron
  "nuevas_conexiones": [
    [12, 45, 0.95], // [origen, destino, peso]
    [45, 89, 0.88]
  ]
}
```
