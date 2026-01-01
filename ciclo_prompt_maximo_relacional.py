#!/usr/bin/env python3
"""
CICLO PROMPT MÁXIMO RELACIONAL
================================
Sistema aislado para descubrimiento dinámico de rutas fenomenológicas máximas.
Objetivo: Encontrar y expandir rutas fenomenológicas más allá de las 10 canónicas.
Factor Clave: "Máximas Rutas Fenomenológicas Posibles"

Independiente del sistema YO Estructural v2.1
Versión: 1.0
"""

import requests
import json
import re
import time
from typing import Dict, List, Any
from datetime import datetime


class CicloPromptMaximoRelacional:
    """
    Ciclo de descubrimiento de rutas fenomenológicas máximas.
    Itera progresivamente para encontrar nuevas dimensiones de análisis.
    """
    
    def __init__(self, concepto: str, gemini_key: str):
        self.concepto = concepto
        self.gemini_key = gemini_key
        self.url_gemini = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'
        self.rutas_descubiertas = {}
        self.iteracion = 0
        self.rutas_canonicas = [
            "etimologica", "sinonímica", "antonímica", "metafórica", "contextual",
            "histórica", "fenomenológica", "dialéctica", "semiótica", "axiológica"
        ]
    
    def _enviar_prompt_a_gemini(self, prompt: str) -> str:
        """Envía prompt a Gemini y retorna la respuesta."""
        try:
            response = requests.post(
                f"{self.url_gemini}?key={self.gemini_key}",
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            else:
                print(f"❌ Error Gemini {response.status_code}: {response.text}")
                return ""
        except Exception as e:
            print(f"❌ Error en solicitud a Gemini: {e}")
            return ""
    
    def descubrir_nuevas_rutas(self, iteraciones: int = 3) -> Dict[str, Any]:
        """
        Ciclo iterativo para descubrir nuevas rutas fenomenológicas.
        Cada iteración busca rutas no exploradas previamente.
        """
        print(f"\n{'='*80}")
        print(f"🔄 CICLO PROMPT MÁXIMO RELACIONAL - INICIO")
        print(f"{'='*80}")
        print(f"Concepto: {self.concepto}")
        print(f"Iteraciones de descubrimiento: {iteraciones}")
        print(f"Rutas canónicas iniciales: {len(self.rutas_canonicas)}")
        print(f"{'='*80}\n")
        
        for i in range(iteraciones):
            self.iteracion = i + 1
            print(f"⏳ ITERACIÓN {self.iteracion}/{iteraciones}")
            print(f"{'─'*80}")
            
            # Paso 1: Generar prompt de descubrimiento
            prompt = self._generar_prompt_descubrimiento()
            
            # Paso 2: Enviar a Gemini
            print(f"📤 Enviando prompt de descubrimiento a Gemini 2.0 Flash...")
            respuesta = self._enviar_prompt_a_gemini(prompt)
            
            if not respuesta:
                print(f"⚠️ No se recibió respuesta en iteración {self.iteracion}")
                continue
            
            # Paso 3: Extraer rutas nuevas
            nuevas_rutas = self._extraer_rutas_nuevas(respuesta)
            
            # Paso 4: Analizar cada ruta nueva
            for nombre_ruta, descripcion in nuevas_rutas.items():
                if nombre_ruta not in self.rutas_descubiertas:
                    print(f"🆕 Ruta nueva descubierta: {nombre_ruta}")
                    self.rutas_descubiertas[nombre_ruta] = {
                        "iteracion_descubrimiento": self.iteracion,
                        "descripcion": descripcion,
                        "analisis": {},
                        "certeza": 0.0
                    }
            
            # Paso 5: Profundizar en rutas nuevas
            self._profundizar_rutas_nuevas()
            
            print(f"\n✅ Iteración {self.iteracion} completada")
            print(f"   Total de rutas descubiertas: {len(self.rutas_descubiertas)}")
            print(f"{'─'*80}\n")
            
            # Pequeña pausa entre iteraciones
            if i < iteraciones - 1:
                time.sleep(1)
        
        return self._compilar_resultados()
    
    def _generar_prompt_descubrimiento(self) -> str:
        """Genera prompt para descubrir nuevas rutas fenomenológicas."""
        rutas_conocidas = list(self.rutas_descubiertas.keys()) + self.rutas_canonicas
        rutas_str = ", ".join(rutas_conocidas)
        
        prompt = f"""Eres un experto en filosofía fenomenológica, semiótica, epistemología y análisis conceptual avanzado.

TAREA: Descubrir NUEVAS RUTAS DE ANÁLISIS FENOMENOLÓGICO para el concepto: "{self.concepto}"

RESTRICCIÓN IMPORTANTE:
- NO incluyas estas rutas que ya conocemos: {rutas_str}
- Busca dimensiones COMPLETAMENTE NUEVAS de análisis
- Debe ser ORIGINAL y no derivado de las rutas canónicas

ITERACIÓN {self.iteracion}: Busca rutas de NIVEL MÁS PROFUNDO

Responde en JSON con esta estructura:
{{
  "nuevas_rutas": [
    {{
      "nombre": "nombre_de_ruta_unica",
      "descripcion": "¿Qué dimensión nueva explora?",
      "justificacion": "¿Por qué es importante para entender '{self.concepto}'?",
      "ejemplo": "Ejemplo concreto de análisis en esta ruta"
    }},
    ...
  ],
  "observacion": "Observación sobre nuevas dimensiones descubiertas"
}}

Sé innovador. Piensa en:
- Perspectivas interdisciplinarias no exploradas
- Contextos culturales específicos
- Dimensiones temporales o espaciales únicas
- Relaciones con fenómenos naturales o artificiales
- Aspectos neurobiológicos o psicológicos profundos
- Conectiones con otros campos del conocimiento
"""
        return prompt
    
    def _extraer_rutas_nuevas(self, respuesta: str) -> Dict[str, str]:
        """Extrae nuevas rutas de la respuesta de Gemini."""
        nuevas_rutas = {}
        
        try:
            # Intentar extraer JSON
            match = re.search(r'\{[\s\S]*\}', respuesta)
            if match:
                datos = json.loads(match.group())
                for ruta in datos.get('nuevas_rutas', []):
                    nombre = ruta.get('nombre', '').lower().replace(' ', '_')
                    descripcion = ruta.get('descripcion', '')
                    if nombre and descripcion:
                        nuevas_rutas[nombre] = descripcion
        except Exception as e:
            print(f"⚠️ Error extrayendo rutas: {e}")
        
        return nuevas_rutas
    
    def _profundizar_rutas_nuevas(self):
        """Profundiza análisis de rutas recientemente descubiertas."""
        rutas_recientes = [r for r, d in self.rutas_descubiertas.items() 
                          if d.get('iteracion_descubrimiento') == self.iteracion]
        
        for ruta in rutas_recientes:
            print(f"   🔍 Profundizando en ruta: {ruta}")
            
            prompt_profundidad = f"""Para el concepto "{self.concepto}" en la ruta "{ruta}":

1. Proporciona un análisis profundo (mínimo 300 palabras)
2. Incluye 5-8 ejemplos específicos
3. Calcula grado de certeza (0.0-1.0)
4. Sugiere 3 aplicaciones prácticas
5. Identifica paradojas o tensiones internas

Responde en JSON:
{{
  "ruta": "{ruta}",
  "analisis_profundo": "...",
  "ejemplos": ["...", "...", ...],
  "certeza": 0.85,
  "aplicaciones": ["...", "...", "..."],
  "paradojas": ["...", "..."]
}}"""
            
            respuesta = self._enviar_prompt_a_gemini(prompt_profundidad)
            
            try:
                match = re.search(r'\{[\s\S]*\}', respuesta)
                if match:
                    analisis = json.loads(match.group())
                    self.rutas_descubiertas[ruta]['analisis'] = analisis
                    self.rutas_descubiertas[ruta]['certeza'] = analisis.get('certeza', 0.0)
            except Exception as e:
                print(f"   ⚠️ Error procesando profundidad: {e}")
    
    def _compilar_resultados(self) -> Dict[str, Any]:
        """Compila todos los resultados del ciclo."""
        rutas_canonicas_count = len(self.rutas_canonicas)
        rutas_nuevas_count = len(self.rutas_descubiertas)
        total_rutas = rutas_canonicas_count + rutas_nuevas_count
        
        certeza_promedio = sum(r.get('certeza', 0) for r in self.rutas_descubiertas.values()) / max(rutas_nuevas_count, 1)
        
        resultado = {
            "ciclo_info": {
                "timestamp": datetime.now().isoformat(),
                "concepto": self.concepto,
                "iteraciones_ejecutadas": self.iteracion,
                "estado": "✅ COMPLETADO"
            },
            "estadisticas": {
                "rutas_canonicas": rutas_canonicas_count,
                "rutas_nuevas_descubiertas": rutas_nuevas_count,
                "total_rutas": total_rutas,
                "certeza_promedio_nuevas": round(certeza_promedio, 3)
            },
            "rutas_canonicas": self.rutas_canonicas,
            "rutas_nuevas": self.rutas_descubiertas,
            "factor_maximo": {
                "nombre": "Máximas Rutas Fenomenológicas Posibles",
                "valor": total_rutas,
                "descriptor": f"El concepto '{self.concepto}' alcanza {total_rutas} dimensiones de análisis fenomenológico"
            }
        }
        
        return resultado
    
    def generar_reporte_completo(self) -> str:
        """Genera reporte completo en markdown."""
        resultado = self._compilar_resultados()
        
        reporte = f"""# 🔄 REPORTE CICLO PROMPT MÁXIMO RELACIONAL

**Timestamp**: {resultado['ciclo_info']['timestamp']}  
**Concepto Analizado**: {resultado['ciclo_info']['concepto']}  
**Estado**: {resultado['ciclo_info']['estado']}

---

## 📊 ESTADÍSTICAS

| Métrica | Valor |
|---------|-------|
| Rutas Canónicas | {resultado['estadisticas']['rutas_canonicas']} |
| Rutas Nuevas Descubiertas | {resultado['estadisticas']['rutas_nuevas_descubiertas']} |
| **Total de Rutas** | **{resultado['estadisticas']['total_rutas']}** |
| Certeza Promedio (Nuevas) | {resultado['estadisticas']['certeza_promedio_nuevas']} |

---

## 📍 RUTAS CANÓNICAS ({len(resultado['rutas_canonicas'])})

"""
        for i, ruta in enumerate(resultado['rutas_canonicas'], 1):
            reporte += f"{i}. {ruta}\n"
        
        reporte += f"\n---\n\n## 🆕 RUTAS NUEVAS DESCUBIERTAS ({resultado['estadisticas']['rutas_nuevas_descubiertas']})\n\n"
        
        for nombre, datos in resultado['rutas_nuevas'].items():
            reporte += f"### {nombre.upper()}\n\n"
            reporte += f"**Iteración de Descubrimiento**: {datos.get('iteracion_descubrimiento', 'N/A')}\n"
            reporte += f"**Descripción**: {datos.get('descripcion', 'N/A')}\n"
            reporte += f"**Certeza**: {datos.get('certeza', 0.0)}\n\n"
            
            analisis = datos.get('analisis', {})
            if analisis:
                reporte += f"**Análisis Profundo**: {analisis.get('analisis_profundo', '')[:300]}...\n\n"
                ejemplos = analisis.get('ejemplos', [])
                if ejemplos:
                    reporte += "**Ejemplos**:\n"
                    for ej in ejemplos[:3]:
                        reporte += f"- {ej}\n"
                    reporte += "\n"
            
            reporte += "---\n\n"
        
        reporte += f"\n## 🎯 FACTOR MÁXIMO\n\n"
        reporte += f"**Nombre**: {resultado['factor_maximo']['nombre']}\n"
        reporte += f"**Valor**: {resultado['factor_maximo']['valor']}\n"
        reporte += f"**Descripción**: {resultado['factor_maximo']['descriptor']}\n"
        
        return reporte


def ejecutar_ciclo_completo(concepto: str, gemini_key: str, iteraciones: int = 3):
    """Función principal para ejecutar el ciclo completo."""
    ciclo = CicloPromptMaximoRelacional(concepto, gemini_key)
    
    # Ejecutar ciclo
    resultado = ciclo.descubrir_nuevas_rutas(iteraciones=iteraciones)
    
    # Generar reporte
    reporte = ciclo.generar_reporte_completo()
    
    return resultado, reporte


if __name__ == "__main__":
    import sys
    
    # Configuración
    CONCEPTO = "DESTRUCCION"
    GEMINI_KEY = "AIzaSyB3cpQ-nVNn8qeC6fUhwozpgYxEFoB_Jdk"
    ITERACIONES = 3
    
    print("\n" + "="*80)
    print("🚀 CICLO PROMPT MÁXIMO RELACIONAL - SISTEMA AISLADO")
    print("="*80)
    
    # Ejecutar ciclo
    resultado, reporte = ejecutar_ciclo_completo(CONCEPTO, GEMINI_KEY, ITERACIONES)
    
    # Mostrar resultados
    print("\n" + "="*80)
    print("✅ CICLO COMPLETADO")
    print("="*80)
    
    print("\n📊 RESULTADO JSON:")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    
    print("\n\n📄 REPORTE MARKDOWN:")
    print(reporte)
    
    # Guardar resultados
    with open('/workspaces/-...Raiz-Dasein/RESULTADO_CICLO_MAXIMO_RELACIONAL.json', 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    
    with open('/workspaces/-...Raiz-Dasein/REPORTE_CICLO_MAXIMO_RELACIONAL.md', 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    print("\n✅ Resultados guardados en:")
    print("   - RESULTADO_CICLO_MAXIMO_RELACIONAL.json")
    print("   - REPORTE_CICLO_MAXIMO_RELACIONAL.md")
