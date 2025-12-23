
/**
 * Puente de comunicación entre el Hipergrafo (TypeScript) y Google Colab (Python).
 * Permite enviar datos a modelos de IA ejecutándose remotamente.
 */
export class ColabBridge {
    private baseUrl: string;
    private isConnected: boolean = false;

    constructor(baseUrl: string) {
        // Asegurar que la URL no tenga slash al final
        this.baseUrl = baseUrl.replace(/\/$/, "");
    }

    /**
     * Establece la URL del servidor Colab dinámicamente
     */
    setUrl(url: string) {
        this.baseUrl = url.replace(/\/$/, "");
    }

    /**
     * Verifica si el servidor en Colab está respondiendo
     */
    async verificarConexion(): Promise<boolean> {
        try {
            // Intentamos conectar a la documentación (siempre disponible en FastAPI)
            const response = await fetch(`${this.baseUrl}/docs`);
            this.isConnected = response.status === 200;
            return this.isConnected;
        } catch (error) {
            console.error("❌ No se pudo conectar con Colab. Verifica la URL y que el notebook esté corriendo.");
            this.isConnected = false;
            return false;
        }
    }

    /**
     * Envía una estructura de datos (ej. estado de nodos) al modelo en Colab
     */
    async ejecutarModelo(datos: any): Promise<any> {
        if (!this.baseUrl) {
            throw new Error("URL de Colab no configurada.");
        }

        try {
            // Endpoint actualizado a /entrenar
            const response = await fetch(`${this.baseUrl}/entrenar`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(datos)
            });

            if (!response.ok) {
                throw new Error(`Error en Colab: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            return result as Record<string, any>;
        } catch (error) {
            console.error("❌ Error ejecutando modelo remoto:", error);
            throw error;
        }
    }
}
