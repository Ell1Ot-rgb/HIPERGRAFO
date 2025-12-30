/**
 * GeneradorDatosEntrenamiento.ts
 * 
 * Genera datos sintéticos de 1600D para entrenar sin dataset real
 * Simula patrones realistas de datos neuronales
 */

export interface ConfiguracionDatos {
    numMuestras: number;
    numCaracteristicas: number;
    porcentajeAnomalias: number;
    semilla?: number;
}

export class GeneradorDatosEntrenamiento {
    private semilla: number;

    constructor(semilla: number = 42) {
        this.semilla = semilla;
    }

    /**
     * Generar muestras sintéticas para entrenamiento
     */
    generarMuestras(config: ConfiguracionDatos): Array<{
        input_data: number[];
        anomaly_label: number;
    }> {
        const muestras: Array<{
            input_data: number[];
            anomaly_label: number;
        }> = [];

        const rng = this.seededRandom(this.semilla);

        for (let i = 0; i < config.numMuestras; i++) {
            const esAnomalia = Math.random() < config.porcentajeAnomalias / 100;
            
            // Generar datos base
            const datos = new Array(config.numCaracteristicas)
                .fill(0)
                .map(() => rng() * 2 - 1); // Rango [-1, 1]

            if (esAnomalia) {
                // Inyectar patrón anómalo: picos fuertes
                const indiceAleatorio = Math.floor(rng() * datos.length);
                datos[indiceAleatorio] *= 3;
                
                // Amplificar región local
                for (let j = 1; j <= 5; j++) {
                    if (indiceAleatorio + j < datos.length) {
                        datos[indiceAleatorio + j] += rng() * 0.5;
                    }
                    if (indiceAleatorio - j >= 0) {
                        datos[indiceAleatorio - j] += rng() * 0.5;
                    }
                }
            } else {
                // Datos normales: suavizar con media móvil
                const ventana = 5;
                for (let j = ventana; j < datos.length - ventana; j++) {
                    let suma = 0;
                    for (let k = -ventana; k <= ventana; k++) {
                        suma += datos[j + k];
                    }
                    datos[j] = suma / (ventana * 2 + 1);
                }
            }

            muestras.push({
                input_data: datos,
                anomaly_label: esAnomalia ? 1 : 0
            });
        }

        return muestras;
    }

    /**
     * Generar dataset con patrón temporal
     * Simula series temporales
     */
    generarSeriesTemporal(
        numMuestras: number,
        longitudSecuencia: number = 16
    ): Array<{
        input_data: number[];
        anomaly_label: number;
    }> {
        const muestras: Array<{
            input_data: number[];
            anomaly_label: number;
        }> = [];

        const rng = this.seededRandom(this.semilla);

        for (let muestra = 0; muestra < numMuestras; muestra++) {
            const esAnomalia = muestra % 20 === 0; // 5% anomalías
            const datos: number[] = [];

            // Generar serie temporal
            let tendencia = rng() * 2 - 1;
            
            for (let t = 0; t < longitudSecuencia; t++) {
                const ciclo = Math.sin((t / longitudSecuencia) * Math.PI * 2);
                let valor = tendencia + ciclo * 0.3;

                // Agregar ruido
                valor += rng() * 0.1;

                if (esAnomalia && t > longitudSecuencia * 0.5) {
                    // Insertar pico anómalo
                    valor += 1.5 * Math.sin((t / 3) * Math.PI);
                }

                tendencia += rng() * 0.05;

                // Expandir a 1600D usando la serie temporal como base
                const bloque = this.expandirABloques(valor, 1600 / longitudSecuencia);
                datos.push(...bloque);
            }

            muestras.push({
                input_data: datos.slice(0, 1600), // Asegurar exactamente 1600D
                anomaly_label: esAnomalia ? 1 : 0
            });
        }

        return muestras;
    }

    /**
     * Generar dataset realista con patrones neuronales
     */
    generarPatronesNeuronales(numMuestras: number): Array<{
        input_data: number[];
        anomaly_label: number;
    }> {
        const muestras: Array<{
            input_data: number[];
            anomaly_label: number;
        }> = [];

        const rng = this.seededRandom(this.semilla);

        for (let i = 0; i < numMuestras; i++) {
            const esAnomalia = Math.random() < 0.1; // 10% anomalías
            const datos = new Array(1600).fill(0);

            // Simular activaciones neuronales con estructura
            const numNeuronasActivas = esAnomalia ? 100 : 50;
            
            for (let j = 0; j < numNeuronasActivas; j++) {
                const indiceNeurona = Math.floor(rng() * 1600);
                const amplitud = esAnomalia ? 1.5 + rng() : 0.5 + rng() * 0.3;
                const ancho = esAnomalia ? 20 : 10;

                // Crear pico gaussiano
                for (let k = indiceNeurona; k < Math.min(indiceNeurona + ancho, 1600); k++) {
                    const distancia = k - indiceNeurona;
                    const gaussiana = Math.exp(-((distancia - ancho / 2) ** 2) / (2 * 5 ** 2));
                    datos[k] += amplitud * gaussiana;
                }
            }

            // Normalizar
            const max = Math.max(...datos);
            if (max > 0) {
                for (let j = 0; j < datos.length; j++) {
                    datos[j] /= max;
                }
            }

            muestras.push({
                input_data: datos,
                anomaly_label: esAnomalia ? 1 : 0
            });
        }

        return muestras;
    }

    /**
     * Expandir valor escalar a bloque 1D
     */
    private expandirABloques(valor: number, tamano: number): number[] {
        const bloque: number[] = [];
        for (let i = 0; i < tamano; i++) {
            const ruido = Math.sin(i * valor) * 0.1;
            bloque.push(valor + ruido);
        }
        return bloque;
    }

    /**
     * Random seeded para reproducibilidad
     */
    private seededRandom(seed: number): () => number {
        return () => {
            seed = (seed * 9301 + 49297) % 233280;
            return seed / 233280;
        };
    }

    /**
     * Mostrar estadísticas del dataset
     */
    static mostrarEstadisticas(muestras: Array<{
        input_data: number[];
        anomaly_label: number;
    }>): void {
        const anomalias = muestras.filter(m => m.anomaly_label === 1).length;
        const normales = muestras.length - anomalias;

        console.log('\n📊 ESTADÍSTICAS DEL DATASET:');
        console.log(`   Total muestras: ${muestras.length}`);
        console.log(`   Normales: ${normales} (${((normales / muestras.length) * 100).toFixed(2)}%)`);
        console.log(`   Anomalías: ${anomalias} (${((anomalias / muestras.length) * 100).toFixed(2)}%)`);
        console.log(`   Dimensión por muestra: ${muestras[0]?.input_data.length || 0}D`);
        
        // Estadísticas de valores
        let minVal = Infinity;
        let maxVal = -Infinity;
        let sumaVal = 0;
        let totalValores = 0;

        muestras.forEach(m => {
            m.input_data.forEach(v => {
                minVal = Math.min(minVal, v);
                maxVal = Math.max(maxVal, v);
                sumaVal += v;
                totalValores++;
            });
        });

        const promedio = sumaVal / totalValores;
        console.log(`   Rango de valores: [${minVal.toFixed(3)}, ${maxVal.toFixed(3)}]`);
        console.log(`   Promedio: ${promedio.toFixed(3)}`);
    }
}
