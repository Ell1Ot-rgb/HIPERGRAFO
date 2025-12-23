import fs from 'fs';
import path from 'path';
import { Hipergrafo } from '../core';
import { ServicioPersistencia } from './ServicioPersistencia';

/**
 * Gestor de almacenamiento de hipergrafos en archivos
 */
export class GestorAlmacenamiento {
  private servicioPersistencia: ServicioPersistencia;
  private directorio: string;

  constructor(directorio: string = './hipergrafos') {
    this.servicioPersistencia = new ServicioPersistencia();
    this.directorio = directorio;

    // Crear directorio si no existe
    if (!fs.existsSync(this.directorio)) {
      fs.mkdirSync(this.directorio, { recursive: true });
    }
  }

  /**
   * Guarda un hipergrafo en archivo
   */
  guardarHipergrafo(hipergrafo: Hipergrafo, nombre: string): string {
    const rutaArchivo = path.join(this.directorio, `${nombre}.json`);
    const jsonContenido = this.servicioPersistencia.serializarAJSON(hipergrafo);
    
    fs.writeFileSync(rutaArchivo, jsonContenido, 'utf-8');
    return rutaArchivo;
  }

  /**
   * Carga un hipergrafo desde archivo
   */
  cargarHipergrafo(nombre: string): Hipergrafo {
    const rutaArchivo = path.join(this.directorio, `${nombre}.json`);
    
    if (!fs.existsSync(rutaArchivo)) {
      throw new Error(`Archivo no encontrado: ${rutaArchivo}`);
    }

    const contenido = fs.readFileSync(rutaArchivo, 'utf-8');
    return this.servicioPersistencia.deserializarDesdeJSON(contenido);
  }

  /**
   * Lista todos los hipergrafos guardados
   */
  listarHipergrafos(): string[] {
    const archivos = fs.readdirSync(this.directorio);
    return archivos
      .filter(f => f.endsWith('.json'))
      .map(f => f.replace('.json', ''));
  }

  /**
   * Elimina un hipergrafo guardado
   */
  eliminarHipergrafo(nombre: string): void {
    const rutaArchivo = path.join(this.directorio, `${nombre}.json`);
    
    if (fs.existsSync(rutaArchivo)) {
      fs.unlinkSync(rutaArchivo);
    }
  }

  /**
   * Exporta un hipergrafo a CSV para análisis
   */
  exportarACSV(hipergrafo: Hipergrafo, nombre: string): string {
    let csvContenido = 'Tipo,ID,Label,Grado/Weight,Metadata\n';

    // Exportar nodos
    hipergrafo.obtenerNodos().forEach(nodo => {
      const grado = hipergrafo.obtenerHiperedgesDelNodo(nodo.id).length;
      const metadata = JSON.stringify(nodo.metadata).replace(/"/g, '""');
      csvContenido += `Nodo,${nodo.id},${nodo.label},${grado},"${metadata}"\n`;
    });

    // Exportar hiperedges
    hipergrafo.obtenerHiperedges().forEach(edge => {
      const nodosStr = Array.from(edge.nodos).join(';');
      const metadata = JSON.stringify(edge.metadata).replace(/"/g, '""');
      csvContenido += `Hiperedge,${edge.id},${edge.label},${edge.weight},"${metadata}";[${nodosStr}]\n`;
    });

    const rutaArchivo = path.join(this.directorio, `${nombre}.csv`);
    fs.writeFileSync(rutaArchivo, csvContenido, 'utf-8');
    return rutaArchivo;
  }

  /**
   * Obtiene información del archivo
   */
  obtenerInfoArchivo(nombre: string): Record<string, any> {
    const rutaArchivo = path.join(this.directorio, `${nombre}.json`);
    
    if (!fs.existsSync(rutaArchivo)) {
      throw new Error(`Archivo no encontrado: ${rutaArchivo}`);
    }

    const stats = fs.statSync(rutaArchivo);
    return {
      nombre,
      ruta: rutaArchivo,
      tamanio: stats.size,
      fechaCreacion: stats.birthtime,
      fechaModificacion: stats.mtime
    };
  }
}
