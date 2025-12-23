/**
 * Script de Prueba: Conectar Hipergrafo con Colab
 * 
 * Ejecución: npx ts-node src/pruebas/prueba_colab.ts
 */

import IntegradorHipergrafoColo from '../neural/IntegradorHipergrafoColo';

async function main() {
    console.log("\n╔════════════════════════════════════════════════╗");
    console.log("║  HIPERGRAFO ↔️  GOOGLE COLAB  (IA ↔️ IA)      ║");
    console.log("╚════════════════════════════════════════════════╝\n");

    const integrador = new IntegradorHipergrafoColo("MiPrimerHipergrafo");
    
    try {
        await integrador.flujoCompleto();
    } catch (error) {
        console.error("\n❌ Error en la ejecución:", error);
        process.exit(1);
    }
}

main();
