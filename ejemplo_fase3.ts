import { Hipergrafo, Nodo, Hiperedge, MapeoRedNeuronalAHipergrafo } from './src';
import { DualidadHipergrafo, CentralidadHipergrafo, ClusteringHipergrafo, PropiedadesEspectrales } from './src/analisis';

/**
 * Ejemplo avanzado: Análisis riguroso de Hipergrafo con Fase 3
 */

// ============================================
// CREAR RED NEURONAL Y MAPEAR A HIPERGRAFO
// ============================================

function crearRedNeuronalEjemplo() {
  const neuronas = Array.from({ length: 100 }, (_, i) => ({
    id: i,
    activacion: Math.random(),
    sesgo: Math.random() * 0.5
  }));

  // Matriz de pesos sparse
  const pesos: number[][][] = [];
  for (let capa = 0; capa < 2; capa++) {
    const capaConexiones: number[][] = [];
    for (let de = 0; de < 100; de++) {
      const conexiones = Array.from({ length: 100 }, () => 
        Math.random() < 0.1 ? Math.random() * 2 - 1 : 0
      );
      capaConexiones.push(conexiones);
    }
    pesos.push(capaConexiones);
  }

  return {
    neuronas,
    pesos,
    capas: [100, 50, 25],
    metadata: { arquitectura: 'MLP', neuromas: 100 }
  };
}

// ============================================
// 1. MAPEO Y ANÁLISIS BÁSICO
// ============================================

console.log('\n╔════════════════════════════════════════════════════════════╗');
console.log('║   Fase 3: ANÁLISIS RIGUROSO DE HIPERGRAFOS               ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

const redNeuronal = crearRedNeuronalEjemplo();
const mapeador = new MapeoRedNeuronalAHipergrafo({
  umbralPeso: 0.12,
  agruparPorCapas: true,
  detectarPatrones: true,
  tamanoMinimoPatron: 5
});

console.log('🔄 Mapeando 100 neuronas a hipergrafo...');
const hipergrafo = mapeador.mapear(redNeuronal);

console.log(`✅ Hipergrafo generado:`);
console.log(`   |V| = ${hipergrafo.cardinalV()} nodos`);
console.log(`   |E| = ${hipergrafo.cardinalE()} hiperedges`);
console.log(`   Densidad = ${hipergrafo.densidad().toFixed(6)}`);
console.log(`   Grado Promedio = ${hipergrafo.gradoPromedio().toFixed(3)}`);

// ============================================
// 2. DUALIDAD DEL HIPERGRAFO
// ============================================

console.log('\n📐 DUALIDAD DEL HIPERGRAFO\n');

const dual = DualidadHipergrafo.calcularDual(hipergrafo);
console.log(`Hipergrafo Original: |V| = ${hipergrafo.cardinalV()}, |E| = ${hipergrafo.cardinalE()}`);
console.log(`Hipergrafo Dual:     |V*| = ${dual.cardinalV()}, |E*| = ${dual.cardinalE()}`);

const esAutodual = DualidadHipergrafo.esAutodual(hipergrafo);
console.log(`¿Es autodual? ${esAutodual}`);

const periodo = DualidadHipergrafo.calcularPeriodoDualidad(hipergrafo, 5);
console.log(`Período de dualidad: ${periodo} iteraciones`);

// ============================================
// 3. CENTRALIDAD DE NODOS
// ============================================

console.log('\n⭐ CENTRALIDAD DE NODOS\n');

const rankingGrado = CentralidadHipergrafo.rankingPorCentralidad(hipergrafo, 'grado');
const rankingBetweenness = CentralidadHipergrafo.rankingPorCentralidad(hipergrafo, 'betweenness');

console.log('Top 5 nodos por Centralidad de Grado:');
rankingGrado.slice(0, 5).forEach((item, idx) => {
  console.log(`   ${idx + 1}. ${item.nodo.label}: ${(item.centralidad * 100).toFixed(2)}%`);
});

console.log('\nTop 5 nodos por Betweenness:');
rankingBetweenness.slice(0, 5).forEach((item, idx) => {
  console.log(`   ${idx + 1}. ${item.nodo.label}: ${item.centralidad.toFixed(2)}`);
});

// Eigenvector centrality
const eigencentralidades = CentralidadHipergrafo.centralidadEigenvector(hipergrafo, 15);
const topEigen = Array.from(eigencentralidades.entries())
  .sort(([, a], [, b]) => b - a)
  .slice(0, 5);

console.log('\nTop 5 nodos por Eigenvector Centrality:');
topEigen.forEach(([nodoId, centralidad], idx) => {
  const nodo = hipergrafo.obtenerNodo(nodoId);
  console.log(`   ${idx + 1}. ${nodo?.label}: ${centralidad.toFixed(4)}`);
});

// ============================================
// 4. CLUSTERING Y COHESIÓN
// ============================================

console.log('\n🔗 CLUSTERING Y COHESIÓN\n');

const coefGlobal = ClusteringHipergrafo.coeficienteClusteringGlobal(hipergrafo);
const coefPromedio = ClusteringHipergrafo.coeficienteClusteringPromedio(hipergrafo);

console.log(`Coeficiente de Clustering Global: ${coefGlobal.toFixed(4)}`);
console.log(`Coeficiente de Clustering Promedio: ${coefPromedio.toFixed(4)}`);

const nodos = hipergrafo.obtenerNodos();
const coefLocal = ClusteringHipergrafo.coeficienteClusteringLocal(hipergrafo, nodos[0].id);
console.log(`Coeficiente Local (${nodos[0].label}): ${coefLocal.toFixed(4)}`);

// ============================================
// 5. PROPIEDADES ESPECTRALES
// ============================================

console.log('\n📊 PROPIEDADES ESPECTRALES\n');

const energia = PropiedadesEspectrales.calcularEnergiaEspectral(hipergrafo);
const gap = PropiedadesEspectrales.calcularGapEspectral(hipergrafo);
const wiener = PropiedadesEspectrales.indiceWienerEspectral(hipergrafo);

console.log(`Energía Espectral: ${energia.toFixed(2)}`);
console.log(`Spectral Gap (Algebraic Connectivity): ${gap.toFixed(6)}`);
console.log(`Índice de Wiener Espectral: ${wiener.toFixed(2)}`);

// ============================================
// 6. ANÁLISIS COMPARATIVO DUAL
// ============================================

console.log('\n🔄 ANÁLISIS COMPARATIVO: HIPERGRAFO vs DUAL\n');

console.log(`Clustering Original: ${coefGlobal.toFixed(4)}`);
const coefDualGlobal = ClusteringHipergrafo.coeficienteClusteringGlobal(dual);
console.log(`Clustering Dual:     ${coefDualGlobal.toFixed(4)}`);

const centrOriginal = hipergrafo.gradoPromedio();
const centrDual = dual.gradoPromedio();
console.log(`\nGrado Promedio Original: ${centrOriginal.toFixed(3)}`);
console.log(`Grado Promedio Dual:     ${centrDual.toFixed(3)}`);

const densidadOriginal = hipergrafo.densidad();
const densidadDual = dual.densidad();
console.log(`\nDensidad Original: ${densidadOriginal.toFixed(6)}`);
console.log(`Densidad Dual:     ${densidadDual.toFixed(6)}`);

// ============================================
// 7. DISTRIBUCIÓN DE GRADOS
// ============================================

console.log('\n📈 DISTRIBUCIÓN DE GRADOS\n');

const distribucion = new Map<number, number>();
nodos.forEach(n => {
  const grado = hipergrafo.calcularGradoNodo(n.id);
  distribucion.set(grado, (distribucion.get(grado) || 0) + 1);
});

console.log('Grado | Cantidad | Porcentaje');
console.log('-----|----------|----------');
Array.from(distribucion.entries())
  .sort((a, b) => a[0] - b[0])
  .slice(0, 10)
  .forEach(([grado, cantidad]) => {
    const porcentaje = ((cantidad / nodos.length) * 100).toFixed(1);
    console.log(`  ${grado}   |   ${cantidad.toString().padStart(2)} (${porcentaje}%)  `);
  });

// ============================================
// RESUMEN ESTADÍSTICO
// ============================================

console.log('\n╔════════════════════════════════════════════════════════════╗');
console.log('║   📋 RESUMEN ESTADÍSTICO FINAL                            ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

console.log(`Estructura:`);
console.log(`  • Nodos: ${hipergrafo.cardinalV()}`);
console.log(`  • Hiperedges: ${hipergrafo.cardinalE()}`);
console.log(`  • Razón E/V: ${(hipergrafo.cardinalE() / hipergrafo.cardinalV()).toFixed(3)}`);

console.log(`\nConexidad:`);
console.log(`  • Grado Promedio: ${hipergrafo.gradoPromedio().toFixed(3)}`);
console.log(`  • Densidad: ${hipergrafo.densidad().toFixed(6)}`);
console.log(`  • Clustering Global: ${coefGlobal.toFixed(4)}`);

console.log(`\nPropiedades Espectrales:`);
console.log(`  • Energía: ${energia.toFixed(2)}`);
console.log(`  • Gap Espectral: ${gap.toFixed(6)}`);

console.log(`\nDualidad:`);
console.log(`  • Es Autodual: ${esAutodual}`);
console.log(`  • Período: ${periodo}`);

console.log('\n✅ Análisis completado exitosamente\n');
