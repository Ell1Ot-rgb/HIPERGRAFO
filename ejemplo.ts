import { Hipergrafo, Nodo, Hiperedge, MapeoRedNeuronalAHipergrafo, GestorAlmacenamiento } from './src';

/**
 * Ejemplo completo: Mapeo de red neuronal de 1024 neuronas a hipergrafo
 */

// ============================================
// 1. CREAR RED NEURONAL SIMULADA (1024 neuronas)
// ============================================

function crearRedNeuronal1024() {
  // Crear 1024 neuronas con activaciones aleatorias
  const neuronas = Array.from({ length: 1024 }, (_, i) => ({
    id: i,
    activacion: Math.random(),
    sesgo: Math.random() * 0.5,
    metadata: {
      tipo: i < 512 ? 'entrada' : i < 768 ? 'oculta' : 'salida'
    }
  }));

  // Crear matriz de pesos 3D
  // Simplificado: 3 capas de conexiones
  const pesos: number[][][] = [];
  
  for (let capa = 0; capa < 3; capa++) {
    const capaConexiones: number[][] = [];
    for (let de = 0; de < 1024; de++) {
      const conexiones: number[] = [];
      for (let a = 0; a < 1024; a++) {
        // 20% de densidad de conexión (sparse network)
        conexiones.push(Math.random() < 0.2 ? Math.random() * 2 - 1 : 0);
      }
      capaConexiones.push(conexiones);
    }
    pesos.push(capaConexiones);
  }

  return {
    neuronas,
    pesos,
    capas: [1024, 768, 512, 256],
    metadata: {
      arquitectura: 'MLP',
      nombre: 'RedNeuronal1024',
      epocas: 100,
      dataset: 'MNIST-Enhanced'
    }
  };
}

// ============================================
// 2. MAPEAR RED NEURONAL A HIPERGRAFO
// ============================================

function demosMapeoBasico() {
  console.log('\n=== DEMO 1: Mapeo Básico ===\n');

  const redNeuronal = crearRedNeuronal1024();

  // Crear mapeador con configuración estándar
  const mapeador = new MapeoRedNeuronalAHipergrafo({
    umbralPeso: 0.15,
    umbralActivacion: 0.5,
    agruparPorCapas: true,
    detectarPatrones: true,
    tamanoMinimoPatron: 10,
    incluirPesos: true
  });

  console.log('🔄 Mapeando 1024 neuronas a hipergrafo...');
  const hipergrafo = mapeador.mapear(redNeuronal);

  console.log(`✅ Hipergrafo creado:`);
  console.log(`   - Nodos (V): ${hipergrafo.cardinalV()}`);
  console.log(`   - Hiperedges (E): ${hipergrafo.cardinalE()}`);
  console.log(`   - Grado promedio: ${hipergrafo.gradoPromedio().toFixed(3)}`);
  console.log(`   - Densidad: ${hipergrafo.densidad().toFixed(6)}`);

  return hipergrafo;
}

// ============================================
// 3. ANÁLISIS DEL HIPERGRAFO
// ============================================

function demosAnalisis(hipergrafo: Hipergrafo) {
  console.log('\n=== DEMO 2: Análisis del Hipergrafo ===\n');

  const nodos = hipergrafo.obtenerNodos();
  const hiperedges = hipergrafo.obtenerHiperedges();

  // Top 5 nodos con mayor grado
  console.log('🔝 Top 5 nodos por grado:');
  const gradosNodos = nodos
    .map(n => ({ nodo: n, grado: hipergrafo.calcularGradoNodo(n.id) }))
    .sort((a, b) => b.grado - a.grado)
    .slice(0, 5);

  gradosNodos.forEach((item, idx) => {
    console.log(`   ${idx + 1}. ${item.nodo.label}: grado=${item.grado}`);
  });

  // Top 5 hiperedges por tamaño
  console.log('\n📊 Top 5 hiperedges por tamaño:');
  const hiperedgesOrdenadas = hiperedges
    .map(e => ({ edge: e, grado: e.grado() }))
    .sort((a, b) => b.grado - a.grado)
    .slice(0, 5);

  hiperedgesOrdenadas.forEach((item, idx) => {
    console.log(`   ${idx + 1}. ${item.edge.label}: conecta ${item.grado} nodos`);
  });

  // Estadísticas de distribución de grados
  const distribucionGrados = new Map<number, number>();
  nodos.forEach(n => {
    const grado = hipergrafo.calcularGradoNodo(n.id);
    distribucionGrados.set(grado, (distribucionGrados.get(grado) || 0) + 1);
  });

  console.log('\n📈 Distribución de grados:');
  Array.from(distribucionGrados.entries())
    .sort((a, b) => a[0] - b[0])
    .forEach(([grado, count]) => {
      const porcentaje = ((count / nodos.length) * 100).toFixed(1);
      console.log(`   Grado ${grado}: ${count} nodos (${porcentaje}%)`);
    });
}

// ============================================
// 4. PERSISTENCIA
// ============================================

function demosPersistencia(hipergrafo: Hipergrafo) {
  console.log('\n=== DEMO 3: Persistencia ===\n');

  const gestor = new GestorAlmacenamiento('./ejemplos_hipergrafos');

  // Guardar
  console.log('💾 Guardando hipergrafo...');
  const ruta = gestor.guardarHipergrafo(hipergrafo, 'red_1024_ejemplo');
  console.log(`   ✅ Guardado en: ${ruta}`);

  // Información del archivo
  const info = gestor.obtenerInfoArchivo('red_1024_ejemplo');
  console.log(`   📦 Tamaño: ${(info.tamanio / 1024).toFixed(2)} KB`);
  console.log(`   📅 Fecha: ${info.fechaCreacion}`);

  // Listar
  const lista = gestor.listarHipergrafos();
  console.log(`\n📂 Hipergrafos guardados: ${lista.length}`);
  lista.slice(0, 3).forEach(nombre => console.log(`   - ${nombre}`));

  // Cargar
  console.log(`\n📖 Cargando hipergrafo guardado...`);
  const hipergrafoCargado = gestor.cargarHipergrafo('red_1024_ejemplo');
  console.log(`   ✅ Cargado: ${hipergrafoCargado.cardinalV()} nodos, ${hipergrafoCargado.cardinalE()} hiperedges`);

  // Exportar CSV
  console.log(`\n📊 Exportando a CSV...`);
  const rutaCSV = gestor.exportarACSV(hipergrafo, 'red_1024_analisis');
  console.log(`   ✅ Exportado a: ${rutaCSV}`);

  return gestor;
}

// ============================================
// 5. VALIDACIÓN Y VERIFICACIÓN
// ============================================

function demosValidacion(hipergrafo: Hipergrafo) {
  console.log('\n=== DEMO 4: Validación Matemática ===\n');

  const nodos = hipergrafo.obtenerNodos();
  const hiperedges = hipergrafo.obtenerHiperedges();

  // Validación 1: Todos los nodos existen
  console.log('✓ Validación de consistencia:');
  let valido = true;

  hiperedges.forEach(edge => {
    edge.nodos.forEach(nodoId => {
      if (!hipergrafo.obtenerNodo(nodoId)) {
        console.log(`   ❌ Nodo ${nodoId} referenciado en hiperedge ${edge.id} no existe`);
        valido = false;
      }
    });
  });

  if (valido) {
    console.log('   ✅ Todos los nodos referenciados existen');
  }

  // Validación 2: Matriz de incidencia
  console.log('\n✓ Matriz de incidencia:');
  const matriz = hipergrafo.calcularMatrizIncidencia();
  console.log(`   Dimensiones: ${matriz.length} x ${matriz[0]?.length}`);
  
  let sumaCeros = 0;
  let sumaUnos = 0;
  matriz.forEach(fila => {
    fila.forEach(valor => {
      if (valor === 0) sumaCeros++;
      else sumaUnos++;
    });
  });
  console.log(`   Densidad de matriz: ${((sumaUnos / (sumaCeros + sumaUnos)) * 100).toFixed(2)}%`);

  // Validación 3: Propiedades teóricas
  console.log('\n✓ Propiedades matemáticas:');
  console.log(`   Número de nodos |V|: ${hipergrafo.cardinalV()}`);
  console.log(`   Número de aristas |E|: ${hipergrafo.cardinalE()}`);
  console.log(`   Grado máximo: ${Math.max(...nodos.map(n => hipergrafo.calcularGradoNodo(n.id)))}`);
  console.log(`   Grado mínimo: ${Math.min(...nodos.map(n => hipergrafo.calcularGradoNodo(n.id)))}`);
  console.log(`   Grado promedio: ${hipergrafo.gradoPromedio().toFixed(3)}`);
}

// ============================================
// EJECUTAR TODAS LAS DEMOS
// ============================================

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║   HIPERGRAFO - Red Neuronal de 1024 Neuronas Demo        ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  try {
    // Demo 1: Mapeo
    const hipergrafo = demosMapeoBasico();

    // Demo 2: Análisis
    demosAnalisis(hipergrafo);

    // Demo 3: Persistencia
    demosPersistencia(hipergrafo);

    // Demo 4: Validación
    demosValidacion(hipergrafo);

    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║   ✅ Demos completadas exitosamente                       ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

  } catch (error) {
    console.error('❌ Error durante la ejecución:', error);
    process.exit(1);
  }
}

main();
