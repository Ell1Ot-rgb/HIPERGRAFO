/**
 * enviar_dataset.ts
 * 
 * Script para enviar un dataset masivo de 10,000 muestras a Colab
 * para el entrenamiento inicial del modelo híbrido.
 */

import { DatasetExporter } from './neural/DatasetExporter';

async function main() {
    const urlColab = process.argv[2];
    
    if (!urlColab) {
        console.error("❌ Error: Debes proporcionar la URL de Colab.");
        console.log("Uso: npx ts-node src/enviar_dataset.ts https://tu-url.ngrok-free.app [numMuestras]");
        process.exit(1);
    }

    const numMuestras = parseInt(process.argv[3]) || 10000;
    const exporter = new DatasetExporter(urlColab);

    try {
        await exporter.exportar(numMuestras);
    } catch (error) {
        console.error("❌ Error durante la exportación:", error);
    }
}

main();
