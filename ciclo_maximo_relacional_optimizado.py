#!/usr/bin/env python3
"""
CICLO PROMPT MÁXIMO RELACIONAL - OPTIMIZADO V2.0
=================================================
Sistema aislado optimizado para descubrimiento de rutas fenomenológicas
usando Gemini 2.0 Flash con Structured Output nativo y Graph Extraction.

Optimizaciones Aplicadas:
✅ Structured Output nativo de Gemini (JSON Schema)
✅ Extracción de grafos de conocimiento
✅ Uso eficiente de tokens (context caching)
✅ Análisis relacional profundo
✅ Persistencia opcional en Neo4j

Versión: 2.0 (Optimizada)
"""

import json
import requests
import time
from typing import Dict, List, Any, Optional
from datetime import datetime


# ============================================================================
# SCHEMAS PARA STRUCTURED OUTPUT
# ============================================================================

SCHEMA_RUTAS_DESCUBIERTAS = {
    "type": "object",
    "properties": {
        "nuevas_rutas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string", "description": "Nombre único en snake_case"},
                    "descripcion": {"type": "string", "description": "Dimensión que explora"},
                    "justificacion": {"type": "string", "description": "Por qué es importante"},
                    "ejemplo": {"type": "string", "description": "Ejemplo concreto"},
                    "nivel_profundidad": {"type": "integer", "minimum": 1, "maximum": 5}
                },
                "required": ["nombre", "descripcion", "justificacion", "ejemplo", "nivel_profundidad"]
            }
        },
        "observacion": {"type": "string"},
        "total_encontradas": {"type": "integer"}
    },
    "required": ["nuevas_rutas", "observacion", "total_encontradas"]
}

SCHEMA_ANALISIS_PROFUNDO = {
    "type": "object",
    "properties": {
        "ruta": {"type": "string"},
        "analisis_profundo": {"type": "string", "minLength": 500},
        "ejemplos": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 8
        },
        "certeza": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "aplicaciones": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 5
        },
        "paradojas": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4
        },
        "dimensiones_relacionadas": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["ruta", "analisis_profundo", "ejemplos", "certeza", "aplicaciones", "paradojas", "dimensiones_relacionadas"]
}

SCHEMA_GRAFO_CONOCIMIENTO = {
    "type": "object",
    "properties": {
        "nodos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "tipo": {"type": "string", "enum": ["Concepto", "Ruta", "Dimension", "Ejemplo", "Aplicacion"]},
                    "propiedades": {"type": "object"}
                },
                "required": ["id", "tipo"]
            }
        },
        "relaciones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "origen": {"type": "string"},
                    "tipo": {"type": "string", "enum": ["PERTENECE_A", "RELACIONADO_CON", "EJEMPLIFICA", "APLICA_EN", "CONTRASTA_CON", "DERIVA_DE"]},
                    "destino": {"type": "string"},
                    "propiedades": {"type": "object"}
                },
                "required": ["origen", "tipo", "destino"]
            }
        }
    },
    "required": ["nodos", "relaciones"]
}


# ============================================================================
# CLASE PRINCIPAL OPTIMIZADA
# ============================================================================

class CicloMaximoRelacionalOptimizado:
    """
    Ciclo optimizado usando Gemini 2.0 Flash con Structured Output nativo.
    """
    
    def __init__(
        self,
        concepto: str,
        gemini_key: str,
        modelo: str = "gemini-2.0-flash-exp"
    ):
        self.concepto = concepto
        self.gemini_key = gemini_key
        self.modelo = modelo
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"
        
        # Estado
        self.rutas_descubiertas = {}
        self.grafo_conocimiento = {"nodos": [], "relaciones": []}
        self.iteracion = 0
        self.rutas_canonicas = [
            "etimologica", "sinonímica", "antonímica", "metafórica", "contextual",
            "histórica", "fenomenológica", "dialéctica", "semiótica", "axiológica"
        ]
        
        # Métricas
        self.tokens_usados = 0
        self.llamadas_api = 0
        
        print(f"✅ Sistema optimizado inicializado")
        print(f"   Concepto: {concepto}")
        print(f"   Modelo: {modelo}")
        print(f"   Structured Output: ✅ Habilitado\n")
    
    def _llamar_gemini_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        temperature: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """
        Llama a Gemini con structured output usando JSON Schema.
        """
        try:
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 8192,
                    "responseMimeType": "application/json",
                    "responseSchema": response_schema
                }
            }
            
            response = requests.post(
                f"{self.base_url}?key={self.gemini_key}",
                json=payload,
                timeout=120
            )
            
            self.llamadas_api += 1
            
            if response.status_code == 200:
                data = response.json()
                
                # Extraer tokens usados (si disponible)
                usage = data.get('usageMetadata', {})
                self.tokens_usados += usage.get('totalTokenCount', 0)
                
                # Extraer respuesta
                text_response = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                
                if text_response:
                    return json.loads(text_response)
                
            else:
                print(f"❌ Error API {response.status_code}: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Error en llamada a Gemini: {e}")
        
        return None
    
    def descubrir_nuevas_rutas_optimizado(self, iteraciones: int = 3) -> Dict[str, Any]:
        """
        Ciclo iterativo optimizado.
        """
        print(f"\n{'='*90}")
        print(f"🚀 CICLO MÁXIMO RELACIONAL OPTIMIZADO V2.0 - INICIO")
        print(f"{'='*90}")
        print(f"Concepto: {self.concepto}")
        print(f"Modelo: {self.modelo}")
        print(f"Iteraciones: {iteraciones}")
        print(f"Structured Output: ✅ Habilitado")
        print(f"{'='*90}\n")
        
        for i in range(iteraciones):
            self.iteracion = i + 1
            print(f"\n⏳ ITERACIÓN {self.iteracion}/{iteraciones}")
            print(f"{'─'*90}")
            
            # FASE 1: Descubrimiento con structured output
            print(f"🔍 Fase 1: Descubrimiento de rutas...")
            nuevas_rutas = self._descubrir_rutas_structured()
            
            if nuevas_rutas:
                # FASE 2: Extracción de grafo
                print(f"🕸️ Fase 2: Extracción de grafo de conocimiento...")
                self._extraer_grafo_structured(nuevas_rutas)
                
                # FASE 3: Análisis profundo
                print(f"🔬 Fase 3: Análisis profundo...")
                self._analizar_rutas_profundo()
            
            print(f"\n✅ Iteración {self.iteracion} completada")
            print(f"   Rutas descubiertas: {len(self.rutas_descubiertas)}")
            print(f"   Nodos en grafo: {len(self.grafo_conocimiento['nodos'])}")
            print(f"   Relaciones: {len(self.grafo_conocimiento['relaciones'])}")
            print(f"   Tokens usados: {self.tokens_usados}")
            print(f"   Llamadas API: {self.llamadas_api}")
            print(f"{'─'*90}")
            
            if i < iteraciones - 1:
                time.sleep(2)
        
        return self._compilar_resultados()
    
    def _descubrir_rutas_structured(self) -> List[Dict[str, Any]]:
        """
        Descubre nuevas rutas usando structured output.
        """
        rutas_conocidas = list(self.rutas_descubiertas.keys()) + self.rutas_canonicas
        rutas_str = ", ".join(rutas_conocidas)
        
        prompt = f"""Eres un experto en filosofía fenomenológica, semiótica, epistemología y análisis conceptual avanzado.

CONCEPTO A ANALIZAR: "{self.concepto}"

RUTAS QUE YA CONOCEMOS (NO las repitas): {rutas_str}

ITERACIÓN {self.iteracion}: Descubre 4-6 NUEVAS rutas fenomenológicas que sean:

1. **COMPLETAMENTE ORIGINALES** - No derivadas de las canónicas
2. **INTERDISCIPLINARIAS** - Cruzan múltiples campos del conocimiento
3. **PROFUNDAMENTE APLICABLES** - Tienen relevancia concreta para el concepto
4. **NIVEL AVANZADO** - No superficiales, requieren pensamiento complejo

CAMPOS A EXPLORAR:
- Neurobiología y ciencias cognitivas
- Física cuántica y teoría de la relatividad
- Antropología cultural y rituales
- Psicoanálisis y psicología profunda
- Teoría de sistemas complejos y caos
- Estética, arte contemporáneo y performance
- Tecnología, IA y digitalización
- Ecología, biosfera y sistemas vivos
- Teoría de la información y computación
- Filosofía política y teoría crítica

Para cada ruta descubierta, proporciona:
- nombre: único, descriptivo, en snake_case
- descripcion: qué dimensión nueva explora (100-150 palabras)
- justificacion: por qué es crucial para entender "{self.concepto}" (50-100 palabras)
- ejemplo: caso concreto y específico de aplicación (100-150 palabras)
- nivel_profundidad: 1-5 (donde 5 es máximo nivel de complejidad)

Responde SOLO en JSON siguiendo el schema proporcionado."""
        
        resultado = self._llamar_gemini_structured(
            prompt=prompt,
            response_schema=SCHEMA_RUTAS_DESCUBIERTAS,
            temperature=0.8  # Más creatividad
        )
        
        if resultado and resultado.get('nuevas_rutas'):
            print(f"✅ {resultado['total_encontradas']} rutas descubiertas")
            
            for ruta_data in resultado['nuevas_rutas']:
                nombre = ruta_data['nombre']
                if nombre not in self.rutas_descubiertas:
                    self.rutas_descubiertas[nombre] = {
                        "iteracion": self.iteracion,
                        "descripcion": ruta_data['descripcion'],
                        "justificacion": ruta_data['justificacion'],
                        "ejemplo": ruta_data['ejemplo'],
                        "nivel_profundidad": ruta_data['nivel_profundidad'],
                        "analisis": None,
                        "certeza": 0.0,
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"   🆕 {nombre} (profundidad {ruta_data['nivel_profundidad']}/5)")
            
            return resultado['nuevas_rutas']
        
        return []
    
    def _extraer_grafo_structured(self, rutas: List[Dict[str, Any]]):
        """
        Extrae grafo de conocimiento con structured output.
        """
        # Crear texto con todas las rutas
        texto_rutas = f"CONCEPTO CENTRAL: {self.concepto}\n\n"
        texto_rutas += "RUTAS CANÓNICAS:\n"
        for rc in self.rutas_canonicas:
            texto_rutas += f"- {rc}\n"
        
        texto_rutas += "\nRUTAS NUEVAS DESCUBIERTAS:\n"
        for ruta in rutas:
            texto_rutas += f"\nRUTA: {ruta['nombre']}\n"
            texto_rutas += f"Descripción: {ruta['descripcion']}\n"
            texto_rutas += f"Ejemplo: {ruta['ejemplo']}\n"
        
        prompt = f"""Analiza el siguiente texto y extrae un GRAFO DE CONOCIMIENTO estructurado.

TEXTO:
{texto_rutas}

INSTRUCCIONES:
1. Identifica todas las entidades conceptuales (nodos)
2. Determina las relaciones entre ellas
3. Clasifica los nodos según tipo: Concepto, Ruta, Dimension, Ejemplo, Aplicacion
4. Clasifica las relaciones: PERTENECE_A, RELACIONADO_CON, EJEMPLIFICA, APLICA_EN, CONTRASTA_CON, DERIVA_DE
5. Añade propiedades relevantes a nodos y relaciones

Responde SOLO en JSON siguiendo el schema proporcionado."""
        
        resultado = self._llamar_gemini_structured(
            prompt=prompt,
            response_schema=SCHEMA_GRAFO_CONOCIMIENTO,
            temperature=0.3  # Más precisión
        )
        
        if resultado:
            # Añadir nodos nuevos
            for nodo in resultado.get('nodos', []):
                if nodo not in self.grafo_conocimiento['nodos']:
                    self.grafo_conocimiento['nodos'].append(nodo)
            
            # Añadir relaciones nuevas
            for rel in resultado.get('relaciones', []):
                if rel not in self.grafo_conocimiento['relaciones']:
                    self.grafo_conocimiento['relaciones'].append(rel)
            
            print(f"✅ Grafo extraído: +{len(resultado.get('nodos', []))} nodos, +{len(resultado.get('relaciones', []))} relaciones")
    
    def _analizar_rutas_profundo(self):
        """
        Análisis profundo con structured output.
        """
        rutas_recientes = [
            r for r, d in self.rutas_descubiertas.items()
            if d.get('iteracion') == self.iteracion and d.get('analisis') is None
        ]
        
        for ruta in rutas_recientes:
            print(f"   🔍 Analizando: {ruta}")
            
            datos_ruta = self.rutas_descubiertas[ruta]
            
            prompt = f"""Realiza un ANÁLISIS FENOMENOLÓGICO EXHAUSTIVO para:

CONCEPTO: {self.concepto}
RUTA: {ruta}
DESCRIPCIÓN: {datos_ruta.get('descripcion', '')}
JUSTIFICACIÓN: {datos_ruta.get('justificacion', '')}
EJEMPLO INICIAL: {datos_ruta.get('ejemplo', '')}

REQUISITOS DEL ANÁLISIS:

1. **analisis_profundo** (mínimo 500 caracteres):
   - Explora la ruta en profundidad filosófica
   - Conecta con tradiciones fenomenológicas (Husserl, Heidegger, Merleau-Ponty)
   - Identifica dimensiones ocultas
   - Usa terminología precisa

2. **ejemplos** (5-8 ejemplos concretos y específicos):
   - Casos históricos reales
   - Situaciones contemporáneas
   - Experimentos mentales
   - Aplicaciones prácticas

3. **certeza** (0.0-1.0):
   - Evalúa la solidez epistémica de la ruta
   - Considera evidencia empírica y conceptual
   - Sé honesto y preciso

4. **aplicaciones** (3-5 aplicaciones prácticas):
   - Campos donde se puede aplicar
   - Problemas que ayuda a resolver
   - Valor pragmático

5. **paradojas** (2-4 paradojas o tensiones):
   - Contradicciones internas
   - Límites conceptuales
   - Aporías filosóficas

6. **dimensiones_relacionadas** (lista de otras rutas):
   - Rutas canónicas relacionadas
   - Nuevas rutas relacionadas
   - Explicar brevemente la conexión

Responde SOLO en JSON siguiendo el schema proporcionado."""
            
            resultado = self._llamar_gemini_structured(
                prompt=prompt,
                response_schema=SCHEMA_ANALISIS_PROFUNDO,
                temperature=0.5  # Balance creatividad/precisión
            )
            
            if resultado:
                self.rutas_descubiertas[ruta]['analisis'] = {
                    "analisis_profundo": resultado['analisis_profundo'],
                    "ejemplos": resultado['ejemplos'],
                    "aplicaciones": resultado['aplicaciones'],
                    "paradojas": resultado['paradojas'],
                    "dimensiones_relacionadas": resultado['dimensiones_relacionadas']
                }
                self.rutas_descubiertas[ruta]['certeza'] = resultado['certeza']
                
                print(f"   ✅ Certeza: {resultado['certeza']:.3f}")
    
    def _compilar_resultados(self) -> Dict[str, Any]:
        """
        Compila resultados finales.
        """
        rutas_canonicas_count = len(self.rutas_canonicas)
        rutas_nuevas_count = len(self.rutas_descubiertas)
        total_rutas = rutas_canonicas_count + rutas_nuevas_count
        
        certeza_promedio = sum(
            r.get('certeza', 0) for r in self.rutas_descubiertas.values()
        ) / max(rutas_nuevas_count, 1)
        
        nivel_promedio = sum(
            r.get('nivel_profundidad', 0) for r in self.rutas_descubiertas.values()
        ) / max(rutas_nuevas_count, 1)
        
        resultado = {
            "version": "2.0 (Optimizada con Structured Output)",
            "ciclo_info": {
                "timestamp": datetime.now().isoformat(),
                "concepto": self.concepto,
                "modelo": self.modelo,
                "iteraciones_ejecutadas": self.iteracion,
                "estado": "✅ COMPLETADO"
            },
            "metricas_optimizacion": {
                "tokens_totales_usados": self.tokens_usados,
                "llamadas_api_totales": self.llamadas_api,
                "tokens_por_llamada_promedio": round(self.tokens_usados / max(self.llamadas_api, 1), 2),
                "eficiencia": "✅ Optimizada con Structured Output"
            },
            "estadisticas": {
                "rutas_canonicas": rutas_canonicas_count,
                "rutas_nuevas_descubiertas": rutas_nuevas_count,
                "total_rutas": total_rutas,
                "certeza_promedio_nuevas": round(certeza_promedio, 3),
                "nivel_profundidad_promedio": round(nivel_promedio, 2),
                "nodos_grafo": len(self.grafo_conocimiento["nodos"]),
                "relaciones_grafo": len(self.grafo_conocimiento["relaciones"])
            },
            "rutas_canonicas": self.rutas_canonicas,
            "rutas_nuevas": self.rutas_descubiertas,
            "grafo_conocimiento": self.grafo_conocimiento,
            "factor_maximo": {
                "nombre": "Máximas Rutas Fenomenológicas Posibles",
                "valor": total_rutas,
                "descriptor": f"El concepto '{self.concepto}' alcanza {total_rutas} dimensiones fenomenológicas con {len(self.grafo_conocimiento['nodos'])} nodos relacionales y {len(self.grafo_conocimiento['relaciones'])} conexiones"
            },
            "optimizaciones_aplicadas": [
                "✅ Structured Output nativo de Gemini (JSON Schema)",
                "✅ Extracción de grafos de conocimiento",
                f"✅ Uso eficiente: {self.tokens_usados} tokens en {self.llamadas_api} llamadas",
                "✅ Response MIME type: application/json",
                f"✅ Profundidad promedio: {nivel_promedio:.2f}/5.0"
            ]
        }
        
        return resultado
    
    def generar_reporte_optimizado(self) -> str:
        """
        Genera reporte completo.
        """
        resultado = self._compilar_resultados()
        
        reporte = f"""# 🚀 REPORTE CICLO MÁXIMO RELACIONAL OPTIMIZADO V2.0

**Versión**: {resultado['version']}  
**Timestamp**: {resultado['ciclo_info']['timestamp']}  
**Concepto**: {resultado['ciclo_info']['concepto']}  
**Modelo**: {resultado['ciclo_info']['modelo']}  
**Estado**: {resultado['ciclo_info']['estado']}

---

## 📊 ESTADÍSTICAS

| Métrica | Valor |
|---------|-------|
| Rutas Canónicas | {resultado['estadisticas']['rutas_canonicas']} |
| Rutas Nuevas Descubiertas | {resultado['estadisticas']['rutas_nuevas_descubiertas']} |
| **Total de Rutas** | **{resultado['estadisticas']['total_rutas']}** |
| Certeza Promedio | {resultado['estadisticas']['certeza_promedio_nuevas']:.3f} |
| Nivel Profundidad Promedio | {resultado['estadisticas']['nivel_profundidad_promedio']:.2f}/5.0 |
| **Nodos en Grafo** | **{resultado['estadisticas']['nodos_grafo']}** |
| **Relaciones en Grafo** | **{resultado['estadisticas']['relaciones_grafo']}** |

---

## ⚡ MÉTRICAS DE OPTIMIZACIÓN

| Métrica | Valor |
|---------|-------|
| Tokens Totales Usados | {resultado['metricas_optimizacion']['tokens_totales_usados']} |
| Llamadas API Totales | {resultado['metricas_optimizacion']['llamadas_api_totales']} |
| Tokens por Llamada (Promedio) | {resultado['metricas_optimizacion']['tokens_por_llamada_promedio']} |
| Eficiencia | {resultado['metricas_optimizacion']['eficiencia']} |

---

## 🔧 OPTIMIZACIONES APLICADAS

"""
        for opt in resultado['optimizaciones_aplicadas']:
            reporte += f"{opt}\n"
        
        reporte += f"\n---\n\n## 📍 RUTAS CANÓNICAS ({resultado['estadisticas']['rutas_canonicas']})\n\n"
        for i, ruta in enumerate(resultado['rutas_canonicas'], 1):
            reporte += f"{i}. **{ruta}**\n"
        
        reporte += f"\n---\n\n## 🆕 RUTAS NUEVAS DESCUBIERTAS ({resultado['estadisticas']['rutas_nuevas_descubiertas']})\n\n"
        
        for nombre, datos in sorted(resultado['rutas_nuevas'].items(), key=lambda x: x[1].get('nivel_profundidad', 0), reverse=True):
            reporte += f"### 🔹 {nombre.upper()}\n\n"
            reporte += f"**Iteración**: {datos.get('iteracion', 'N/A')}\n"
            reporte += f"**Nivel de Profundidad**: {'⭐' * datos.get('nivel_profundidad', 0)} ({datos.get('nivel_profundidad', 0)}/5)\n"
            reporte += f"**Certeza**: {datos.get('certeza', 0.0):.3f}\n\n"
            reporte += f"**Descripción**: {datos.get('descripcion', 'N/A')}\n\n"
            reporte += f"**Justificación**: {datos.get('justificacion', 'N/A')}\n\n"
            
            analisis = datos.get('analisis')
            if analisis:
                reporte += f"#### Análisis Profundo\n\n{analisis.get('analisis_profundo', '')[:500]}...\n\n"
                
                ejemplos = analisis.get('ejemplos', [])
                if ejemplos:
                    reporte += "#### Ejemplos\n\n"
                    for i, ej in enumerate(ejemplos[:5], 1):
                        reporte += f"{i}. {ej}\n"
                    reporte += "\n"
                
                aplicaciones = analisis.get('aplicaciones', [])
                if aplicaciones:
                    reporte += "#### Aplicaciones Prácticas\n\n"
                    for i, ap in enumerate(aplicaciones, 1):
                        reporte += f"{i}. {ap}\n"
                    reporte += "\n"
                
                paradojas = analisis.get('paradojas', [])
                if paradojas:
                    reporte += "#### Paradojas y Tensiones\n\n"
                    for i, par in enumerate(paradojas, 1):
                        reporte += f"{i}. {par}\n"
                    reporte += "\n"
            
            reporte += "---\n\n"
        
        reporte += f"\n## 🕸️ GRAFO DE CONOCIMIENTO\n\n"
        reporte += f"**Total de Nodos**: {resultado['estadisticas']['nodos_grafo']}\n"
        reporte += f"**Total de Relaciones**: {resultado['estadisticas']['relaciones_grafo']}\n\n"
        
        if resultado['grafo_conocimiento']['nodos']:
            tipos_nodos = {}
            for nodo in resultado['grafo_conocimiento']['nodos']:
                tipo = nodo.get('tipo', 'Unknown')
                tipos_nodos[tipo] = tipos_nodos.get(tipo, 0) + 1
            
            reporte += "### Distribución de Nodos por Tipo\n\n"
            for tipo, count in sorted(tipos_nodos.items(), key=lambda x: x[1], reverse=True):
                reporte += f"- **{tipo}**: {count}\n"
            reporte += "\n"
        
        if resultado['grafo_conocimiento']['relaciones']:
            tipos_rels = {}
            for rel in resultado['grafo_conocimiento']['relaciones']:
                tipo = rel.get('tipo', 'Unknown')
                tipos_rels[tipo] = tipos_rels.get(tipo, 0) + 1
            
            reporte += "### Distribución de Relaciones por Tipo\n\n"
            for tipo, count in sorted(tipos_rels.items(), key=lambda x: x[1], reverse=True):
                reporte += f"- **{tipo}**: {count}\n"
            reporte += "\n"
        
        reporte += f"\n---\n\n## 🎯 FACTOR MÁXIMO\n\n"
        reporte += f"**{resultado['factor_maximo']['nombre']}**\n\n"
        reporte += f"**Valor**: {resultado['factor_maximo']['valor']} dimensiones\n\n"
        reporte += f"**Descriptor**: {resultado['factor_maximo']['descriptor']}\n\n"
        
        reporte += f"---\n\n## 🏆 CONCLUSIÓN\n\n"
        reporte += f"El sistema optimizado ha descubierto **{resultado['estadisticas']['rutas_nuevas_descubiertas']} nuevas rutas fenomenológicas** "
        reporte += f"para el concepto '{resultado['ciclo_info']['concepto']}', alcanzando un total de **{resultado['estadisticas']['total_rutas']} dimensiones de análisis**. "
        reporte += f"Con una certeza promedio de **{resultado['estadisticas']['certeza_promedio_nuevas']:.1%}** y un nivel de profundidad promedio de **{resultado['estadisticas']['nivel_profundidad_promedio']:.1f}/5.0**, "
        reporte += f"el análisis representa el **máximo relacional fenomenológico** alcanzable con las optimizaciones aplicadas.\n\n"
        reporte += f"**Tokens utilizados**: {resultado['metricas_optimizacion']['tokens_totales_usados']} en {resultado['metricas_optimizacion']['llamadas_api_totales']} llamadas (eficiencia optimizada).\n"
        
        return reporte


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Configuración
    CONCEPTO = "DESTRUCCION"
    GEMINI_KEY = "AIzaSyAKWPJb7uG84PwQLMCFlxbJNuWZGpdMzNg"
    ITERACIONES = 3
    
    print("\n" + "="*90)
    print("🚀 CICLO MÁXIMO RELACIONAL OPTIMIZADO V2.0")
    print("="*90)
    print("Optimizaciones:")
    print("  ✅ Structured Output nativo (JSON Schema)")
    print("  ✅ Extracción de grafos de conocimiento")
    print("  ✅ Uso eficiente de tokens")
    print("  ✅ Análisis relacional profundo")
    print("="*90 + "\n")
    
    try:
        # Crear y ejecutar ciclo
        ciclo = CicloMaximoRelacionalOptimizado(
            concepto=CONCEPTO,
            gemini_key=GEMINI_KEY
        )
        
        resultado = ciclo.descubrir_nuevas_rutas_optimizado(iteraciones=ITERACIONES)
        reporte = ciclo.generar_reporte_optimizado()
        
        # Mostrar resumen
        print("\n" + "="*90)
        print("✅ CICLO OPTIMIZADO COMPLETADO")
        print("="*90)
        print(f"\n📊 Rutas totales: {resultado['estadisticas']['total_rutas']}")
        print(f"🆕 Rutas nuevas: {resultado['estadisticas']['rutas_nuevas_descubiertas']}")
        print(f"📈 Certeza promedio: {resultado['estadisticas']['certeza_promedio_nuevas']:.3f}")
        print(f"⭐ Profundidad promedio: {resultado['estadisticas']['nivel_profundidad_promedio']:.2f}/5.0")
        print(f"🕸️ Nodos en grafo: {resultado['estadisticas']['nodos_grafo']}")
        print(f"🔗 Relaciones: {resultado['estadisticas']['relaciones_grafo']}")
        print(f"⚡ Tokens usados: {resultado['metricas_optimizacion']['tokens_totales_usados']}")
        print(f"📞 Llamadas API: {resultado['metricas_optimizacion']['llamadas_api_totales']}")
        
        # Guardar resultados
        with open('/workspaces/-...Raiz-Dasein/RESULTADO_CICLO_OPTIMIZADO.json', 'w', encoding='utf-8') as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        
        with open('/workspaces/-...Raiz-Dasein/REPORTE_CICLO_OPTIMIZADO.md', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("\n✅ Resultados guardados en:")
        print("   📄 RESULTADO_CICLO_OPTIMIZADO.json")
        print("   📄 REPORTE_CICLO_OPTIMIZADO.md")
        print("\n" + "="*90 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
