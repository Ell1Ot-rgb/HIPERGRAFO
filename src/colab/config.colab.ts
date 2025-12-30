/**
 * config.colab.ts
 * 
 * Configuración centralizada para conexión con Colab
 */

export interface ConfiguracionColab {
    urlServidor: string;
    timeout: number;
    reintentos: number;
    mostrarDebug: boolean;
    guardarHistorial: boolean;
}

export const CONFIGURACION_COLAB_DEFECTO: ConfiguracionColab = {
    // ⭐ REEMPLAZA ESTO CON TU URL DE COLAB
    urlServidor: process.env.COLAB_SERVER_URL || 'https://tu-id-unico.ngrok-free.app',
    
    timeout: 60000, // 60 segundos
    reintentos: 3,
    mostrarDebug: process.env.DEBUG === 'true',
    guardarHistorial: true
};

/**
 * Validar que la URL de Colab sea válida
 */
export function validarUrlColab(url: string): boolean {
    try {
        const urlObj = new URL(url);
        
        // Aceptar localhost o ngrok
        if (urlObj.hostname === 'localhost' || urlObj.hostname.includes('ngrok')) {
            return true;
        }
        
        console.warn(`⚠️ Advertencia: URL puede no ser un servidor Colab válido: ${url}`);
        return false;
    } catch {
        console.error(`❌ URL no válida: ${url}`);
        return false;
    }
}

/**
 * Obtener URL de Colab desde variable de entorno o stdin
 */
export async function obtenerUrlColab(): Promise<string> {
    const url = process.env.COLAB_SERVER_URL;
    
    if (url && validarUrlColab(url)) {
        console.log(`✅ URL de Colab desde variable de entorno: ${url}`);
        return url;
    }
    
    // Si viene por argumentos
    const args = process.argv.slice(2);
    if (args[0] && validarUrlColab(args[0])) {
        return args[0];
    }
    
    throw new Error(
        'No se encontró URL de Colab.\n\n' +
        'Opciones:\n' +
        '1. Pasar como argumento: npx ts-node script.ts https://tu-url.ngrok-free.app\n' +
        '2. Variable de entorno: export COLAB_SERVER_URL=https://tu-url.ngrok-free.app\n' +
        '3. Actualizar en config.colab.ts\n'
    );
}

/**
 * Tipos de datos para generar
 */
export enum TipoDatos {
    SIMPLE = 'simple',
    TEMPORAL = 'temporal',
    NEURONAL = 'neuronal'
}

/**
 * Configuración predefinida para diferentes casos de uso
 */
export const PRESETS = {
    /**
     * Prueba rápida (< 30 segundos)
     */
    prueba_rapida: {
        numMuestras: 100,
        tamanoLote: 32,
        tipo: TipoDatos.SIMPLE,
        porcentajeAnomalias: 10
    },
    
    /**
     * Entrenamiento estándar
     */
    entrenamiento_estandar: {
        numMuestras: 1000,
        tamanoLote: 64,
        tipo: TipoDatos.NEURONAL,
        porcentajeAnomalias: 10
    },
    
    /**
     * Detección de anomalías
     */
    deteccion_anomalias: {
        numMuestras: 2000,
        tamanoLote: 64,
        tipo: TipoDatos.TEMPORAL,
        porcentajeAnomalias: 20
    },
    
    /**
     * Entrenamiento pesado
     */
    entrenamiento_pesado: {
        numMuestras: 5000,
        tamanoLote: 128,
        tipo: TipoDatos.NEURONAL,
        porcentajeAnomalias: 15
    }
};

/**
 * Validación de GPU en Colab
 */
export const REQUISITOS_GPU = {
    memoria_minima_mb: 2048,
    cuda_recomendado: true,
    dispositivo: 'A100 o T4'
};

/**
 * Información útil
 */
export const INFO = {
    documentacion: 'https://github.com/Ell1Ot-rgb/HIPERGRAFO',
    colab_nuevo: 'https://colab.research.google.com/',
    ngrok_free: 'https://ngrok.com/download',
    pytorch: 'https://pytorch.org/get-started/locally/'
};
