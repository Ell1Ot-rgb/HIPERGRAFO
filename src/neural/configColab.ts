/**
 * Configuración del puente de comunicación con Google Colab
 * Aquí se define la URL del servidor remoto que corre en Colab
 */

export const CONFIG_COLAB = {
    // 🔗 URL generada por ngrok en Colab
    // Actualiza esta URL cuando obtengas una nueva desde Colab
    urlServidor: "https://TU_URL_NGROK_AQUI.ngrok-free.dev",
    
    // Intentos de reconexión en caso de timeout
    maxReintentos: 3,
    
    // Timeout en milisegundos
    timeoutMs: 30000,
    
    // Enable logging para debugging
    debug: true
};

/**
 * Función para actualizar dinámicamente la URL si cambia
 */
export function actualizarUrlColab(nuevaUrl: string): void {
    CONFIG_COLAB.urlServidor = nuevaUrl;
    if (CONFIG_COLAB.debug) {
        console.log(`✅ URL de Colab actualizada a: ${nuevaUrl}`);
    }
}
