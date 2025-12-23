/**
 * Script de Verificación de Conexión con Colab
 */

import { ColabBridge } from './neural/ColabBridge';
import { CONFIG_COLAB } from './neural/configColab';

async function verificar() {
    console.log("🔍 Verificando conexión con Google Colab...");
    console.log(`🌐 URL: ${CONFIG_COLAB.urlServidor}`);

    const bridge = new ColabBridge(CONFIG_COLAB.urlServidor);

    try {
        const estaActivo = await bridge.verificarConexion();
        
        if (estaActivo) {
            console.log("✅ CONEXIÓN EXITOSA: El servidor en Colab está respondiendo.");
            
            // Probar un envío de prueba
            console.log("🧪 Enviando datos de prueba...");
            const respuesta = await bridge.ejecutarModelo({
                accion: "ping",
                datos: { mensaje: "Hola desde HIPERGRAFO Codespace" }
            });
            
            console.log("📩 Respuesta del servidor:", respuesta);
            console.log("\n🚀 EL SISTEMA ESTÁ LISTO PARA EL ENTRENAMIENTO.");
        } else {
            console.log("❌ ERROR: El servidor en Colab no responde.");
            console.log("💡 Asegúrate de que:");
            console.log("   1. El notebook de Colab esté ejecutándose.");
            console.log("   2. El servidor FastAPI esté activo.");
            console.log("   3. La URL de ngrok en 'src/neural/configColab.ts' sea la correcta.");
        }
    } catch (error) {
        console.error("💥 Error crítico durante la verificación:", error);
    }
}

verificar();
