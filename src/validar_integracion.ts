/**
 * validar_integracion.ts
 * 
 * Script de validación del sistema integrado
 * Prueba que EntrenadorCognitivo + SistemaOmnisciente funcionen juntos
 */

import { SistemaOmnisciente } from './SistemaOmnisciente';
import { Omega21Simulador } from './hardware/Simulador';
import { MapeoVector256DaDendritas } from './control/MapeoVector256DaDendritas';

async function validarIntegracion() {
    console.log('🔍 VALIDACIÓN DE INTEGRACIÓN\n');
    
    // 1. Inicializar Sistema Omnisciente
    console.log('✓ Inicializando SistemaOmnisciente...');
    const sistema = new SistemaOmnisciente();
    await sistema.inicializar();
    
    // 2. Crear 3 átomos de prueba
    console.log('✓ Creando átomos de prueba...');
    for (let i = 1; i <= 3; i++) {
        await sistema.crearAtomo(`S${i}`);
    }
    console.log(`  → ${sistema.atomos.size} átomos creados`);
    
    // 3. Verificar EntrenadorCognitivo
    console.log('✓ Verificando EntrenadorCognitivo...');
    const stats = sistema.entrenador.obtenerEstadisticas();
    console.log(`  → Buffer lleno: ${stats.bufferLleno}/${50}`);
    console.log(`  → Conceptos aprendidos: ${stats.conceptosAprendidos}`);
    console.log(`  → Ciclos consolidación: ${stats.ciclosConsolidacion}`);
    
    // 4. Simular procesamiento
    console.log('\n✓ Simulando procesamiento de flujo...');
    const simulador = new Omega21Simulador();
    const mapeador = new MapeoVector256DaDendritas();
    
    for (let ciclo = 1; ciclo <= 5; ciclo++) {
        const telemetria = simulador.generarMuestra();
        
        // Procesar con el primer átomo
        const resultado = await sistema.procesarFlujo('S1', telemetria);
        
        console.log(`  Ciclo ${ciclo}: Anomalía=${(resultado.neuronal.prediccion_anomalia * 100).toFixed(1)}% | Memoria=${resultado.memoria}`);
    }
    
    // 5. Verificar consolidación
    console.log('\n✓ Estadísticas finales:');
    const statsFinal = sistema.entrenador.obtenerEstadisticas();
    console.log(`  → Conceptos aprendidos: ${statsFinal.conceptosAprendidos}`);
    console.log(`  → Ciclos consolidación: ${statsFinal.ciclosConsolidacion}`);
    console.log(`  → Tasa acierto: ${statsFinal.tasaAcierto}`);
    
    console.log('\n✅ VALIDACIÓN COMPLETADA EXITOSAMENTE');
    process.exit(0);
}

validarIntegracion().catch(err => {
    console.error('❌ Error en validación:', err);
    process.exit(1);
});
