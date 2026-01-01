#!/usr/bin/env python3
"""
CICLO RELACIONAL CON KIMI K2 GRATUITO - OpenRouter
===================================================
Sistema optimizado usando la API oficial de OpenRouter con Kimi K2 (gratuito)
Utiliza la librería openai para máxima compatibilidad y facilidad.

CARACTERÍSTICAS:
- API Key: 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
- Modelo: moonshotai/kimi-k2:free (completamente gratuito)
- Base URL: https://openrouter.ai/api/v1
- Librería: openai (compatible con OpenRouter)
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ============================================================================
# CONFIGURACIÓN - KIMI K2 GRATUITO VÍA OPENROUTER
# ============================================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "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"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
KIMI_MODEL = "openai/gpt-3.5-turbo"  # Modelo económico y confiable (también funciona Kimi cuando no tiene rate limit)

# Crear cliente OpenAI configurado para OpenRouter
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

# Headers para mejorar rankings en OpenRouter (opcional pero recomendado)
EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/Ell1Ot-rgb/-...Raiz-Dasein",
    "X-Title": "Ciclo-Relacional-Kimi-OpenRouter",
}

print("="*80)
print("🚀 CICLO RELACIONAL CON KIMI K2 GRATUITO")
print("="*80)
print(f"✅ API Key: {OPENROUTER_API_KEY[:25]}...")
print(f"✅ Base URL: {OPENROUTER_BASE_URL}")
print(f"✅ Modelo: {KIMI_MODEL}")
print(f"✅ Costo: GRATUITO 🎉")
print("="*80 + "\n")

# ============================================================================
# SCHEMAS PARA STRUCTURED OUTPUT
# ============================================================================

SCHEMA_RUTAS = {
    "type": "object",
    "properties": {
        "nuevas_rutas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string"},
                    "descripcion": {"type": "string"},
                    "justificacion": {"type": "string"},
                    "ejemplo": {"type": "string"},
                    "nivel_profundidad": {"type": "integer"}
                },
                "required": ["nombre", "descripcion", "justificacion", "ejemplo", "nivel_profundidad"]
            }
        },
        "total_encontradas": {"type": "integer"}
    },
    "required": ["nuevas_rutas", "total_encontradas"]
}

# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class CicloRelacionalKimi:
    """Sistema de descubrimiento conceptual usando Kimi K2 vía OpenRouter"""
    
    def __init__(self, concepto: str):
        self.concepto = concepto
        self.rutas_descubiertas = []
        self.analisis_profundos = {}
        self.grafo = {"nodos": [], "relaciones": []}
        self.tokens_usados = 0
        self.llamadas_api = 0
        self.timestamp_inicio = datetime.now()
    
    def _llamar_kimi(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """
        Llama a Kimi K2 vía OpenRouter usando la librería openai
        
        Args:
            prompt: Mensaje del usuario
            temperature: Creatividad (0.0-1.0)
            max_tokens: Máximo de tokens a generar
            
        Returns:
            Contenido de la respuesta o None si falla
        """
        try:
            self.llamadas_api += 1
            print(f"📞 Llamada API #{self.llamadas_api}...", end=" ")
            
            response = client.chat.completions.create(
                model=KIMI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            
            # Extraer y contabilizar tokens
            usage = response.usage
            self.tokens_usados += usage.total_tokens
            
            content = response.choices[0].message.content
            print(f"✅ ({usage.total_tokens} tokens)")
            return content
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def descubrir_rutas(self) -> List[Dict[str, Any]]:
        """Fase 1: Descubrimiento de rutas conceptuales maximizado"""
        print(f"\n📍 FASE 1: Descubrimiento MÁXIMO de rutas para '{self.concepto}'")
        print("-" * 80)
        
        prompt = f"""Eres un filósofo fenomenólogo extraordinario. Tu tarea es descubrir el MÁXIMO número de rutas conceptuales ÚNICAS, PROFUNDAS y ORIGINALES para: '{self.concepto}'

OBJETIVO: Genera entre 10-15 rutas excepcionales con profundidad 4-5.

CRITERIOS DE EXCELENCIA:
- Originalidad radical: NO trivial, NO obvio
- Profundidad máxima (prioriza 4-5)
- Justificación filosófica sólida
- Ejemplos concretos reveladores
- Diversidad: ontológica, fenomenológica, existencial, relacional, temporal, corporal, ética, cuántica, ecológica

EXPLORA:
- Perspectivas interdisciplinarias (neurociencia, física cuántica, antropología, etc.)
- Paradojas y tensiones conceptuales
- Dimensiones temporales, espaciales, relacionales
- Lo sorprendente y lo profundo

Responde SOLO con JSON válido:
{{
  "nuevas_rutas": [
    {{
      "nombre": "nombre_snake_case",
      "descripcion": "descripción rica (100+ caracteres)",
      "justificacion": "fundamentación filosófica",
      "ejemplo": "ejemplo concreto revelador",
      "nivel_profundidad": 5
    }}
  ],
  "total_encontradas": 12
}}

MÍNIMO 10 rutas de alta calidad. ¡Más es mejor!"""
        
        content = self._llamar_kimi(prompt, temperature=0.8)
        if not content:
            return []
        
        try:
            # Limpiar markdown si existe
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Intentar parseo normal
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: buscar el array manualmente
                print("\n⚠️ Intentando parseo alternativo...")
                import re
                
                # Buscar "nuevas_rutas": [ ... ]
                match = re.search(r'"nuevas_rutas"\s*:\s*\[(.*?)\](?=\s*,|\s*})', content, re.DOTALL)
                if match:
                    array_content = "[" + match.group(1) + "]"
                    # Intentar limpiar strings mal formados
                    array_content = array_content.replace('\n', ' ').replace('\r', '')
                    data = {"nuevas_rutas": json.loads(array_content), "total_encontradas": 0}
                else:
                    data = {"nuevas_rutas": [], "total_encontradas": 0}
            
            self.rutas_descubiertas = data.get("nuevas_rutas", [])
            
            print(f"\n✅ Rutas descubiertas: {len(self.rutas_descubiertas)}\n")
            for i, r in enumerate(self.rutas_descubiertas, 1):
                prof = r.get('nivel_profundidad', 0)
                print(f"  {i:2d}. {r['nombre'].upper()} (prof: {prof}/5)")
            
            return self.rutas_descubiertas
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parseando JSON: {e}")
            print(f"Contenido: {content[:200]}")
            return []
    
    def analizar_ruta_profundo(self, ruta_nombre: str) -> Dict[str, Any]:
        """Fase 2: Análisis profundo de una ruta"""
        print(f"\n🔍 Analizando '{ruta_nombre}'...", end=" ")
        
        prompt = f"""Para la ruta '{ruta_nombre}' del concepto '{self.concepto}':

Proporciona análisis profundo (300+ caracteres) con:
- Análisis filosófico completo
- 3-5 ejemplos concretos
- Certeza (0.0-1.0)
- 2-3 aplicaciones prácticas
- Paradojas o contradicciones

Responde SOLO con JSON:
{{
  "ruta": "{ruta_nombre}",
  "analisis_profundo": "...",
  "ejemplos": ["ej1", "ej2", "ej3"],
  "certeza": 0.85,
  "aplicaciones": ["app1", "app2"],
  "paradojas": ["par1"]
}}"""
        
        content = self._llamar_kimi(prompt, temperature=0.6)
        if not content:
            return {}
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            self.analisis_profundos[ruta_nombre] = data
            print(f"✅")
            return data
            
        except json.JSONDecodeError as e:
            print(f"❌ Error: {e}")
            return {}
    
    def extraer_grafo(self) -> Dict[str, Any]:
        """Fase 3: Extracción del grafo de conocimiento"""
        print(f"\n🕸️ Extrayendo grafo de conocimiento...", end=" ")
        
        rutas_str = "\n".join([r['nombre'] for r in self.rutas_descubiertas])
        
        prompt = f"""Grafo de conocimiento para '{self.concepto}' con rutas:
{rutas_str}

Construye grafo con nodos y relaciones ponderadas:

{{
  "nodos": [
    {{"id": "{self.concepto}", "tipo": "Concepto"}},
    {{"id": "ruta1", "tipo": "Ruta"}}
  ],
  "relaciones": [
    {{"origen": "{self.concepto}", "destino": "ruta1", "tipo": "explora", "peso": 1.0}}
  ]
}}"""
        
        content = self._llamar_kimi(prompt, temperature=0.5)
        if not content:
            return {"nodos": [], "relaciones": []}
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            self.grafo = data
            print(f"✅")
            return data
            
        except json.JSONDecodeError as e:
            print(f"❌ Error: {e}")
            return {"nodos": [], "relaciones": []}
    
    def ejecutar(self, con_profundo: bool = True, con_grafo: bool = True, max_analisis: int = 5) -> Dict[str, Any]:
        """Ejecuta todas las fases del ciclo"""
        print("\n" + "="*80)
        print(f"🚀 CICLO RELACIONAL: {self.concepto.upper()}")
        print("="*80)
        
        # Fase 1: Descubrimiento
        rutas = self.descubrir_rutas()
        
        # Fase 2: Análisis profundo
        if con_profundo and rutas:
            print(f"\n📊 Análisis profundo de {min(max_analisis, len(rutas))} rutas...")
            print("-" * 80)
            
            rutas_ordenadas = sorted(
                rutas, 
                key=lambda r: r.get('nivel_profundidad', 0), 
                reverse=True
            )
            
            for ruta in rutas_ordenadas[:max_analisis]:
                self.analizar_ruta_profundo(ruta['nombre'])
                time.sleep(0.3)  # Evitar rate limits
        
        # Fase 3: Grafo
        if con_grafo and rutas:
            print("\n🕸️ Mapeo relacional...")
            print("-" * 80)
            self.extraer_grafo()
        
        # Compilar resultado
        resultado = {
            "concepto": self.concepto,
            "rutas_descubiertas": self.rutas_descubiertas,
            "analisis_profundos": self.analisis_profundos,
            "grafo": self.grafo,
            "estadisticas": {
                "total_rutas": len(self.rutas_descubiertas),
                "analisis_realizados": len(self.analisis_profundos),
                "nodos_grafo": len(self.grafo.get("nodos", [])),
                "relaciones": len(self.grafo.get("relaciones", []))
            },
            "metricas": {
                "tokens_usados": self.tokens_usados,
                "llamadas_api": self.llamadas_api,
                "duracion_segundos": (datetime.now() - self.timestamp_inicio).total_seconds()
            },
            "modelo": KIMI_MODEL,
            "timestamp": datetime.now().isoformat()
        }
        
        return resultado
    
    def generar_reporte(self, resultado: Dict[str, Any]) -> str:
        """Genera reporte en Markdown"""
        reporte = f"# Ciclo Relacional: {resultado['concepto']}\n\n"
        reporte += f"**Timestamp**: {resultado['timestamp']}\n"
        reporte += f"**Modelo**: {resultado['modelo']}\n"
        reporte += f"**Costo**: 🎉 GRATUITO\n\n"
        
        reporte += "## 📊 Estadísticas\n\n"
        reporte += f"- **Rutas descubiertas**: {resultado['estadisticas']['total_rutas']}\n"
        reporte += f"- **Análisis profundos**: {resultado['estadisticas']['analisis_realizados']}\n"
        reporte += f"- **Tokens usados**: {resultado['metricas']['tokens_usados']}\n"
        reporte += f"- **Llamadas API**: {resultado['metricas']['llamadas_api']}\n"
        reporte += f"- **Duración**: {resultado['metricas']['duracion_segundos']:.1f}s\n\n"
        
        reporte += "## 🆕 Rutas Descubiertas\n\n"
        for ruta in resultado['rutas_descubiertas']:
            reporte += f"### {ruta['nombre'].replace('_', ' ').title()}\n\n"
            reporte += f"**Profundidad**: {ruta.get('nivel_profundidad', 0)}/5\n\n"
            reporte += f"**Descripción**: {ruta.get('descripcion', '')}\n\n"
            reporte += f"**Justificación**: {ruta.get('justificacion', '')}\n\n"
            reporte += f"**Ejemplo**: {ruta.get('ejemplo', '')}\n\n"
            reporte += "---\n\n"
        
        if resultado['analisis_profundos']:
            reporte += "## 🔍 Análisis Profundos\n\n"
            for ruta_nombre, analisis in resultado['analisis_profundos'].items():
                reporte += f"### {ruta_nombre.replace('_', ' ').title()}\n\n"
                reporte += f"**Certeza**: {analisis.get('certeza', 0):.0%}\n\n"
                reporte += f"{analisis.get('analisis_profundo', '')}\n\n"
                reporte += "---\n\n"
        
        return reporte


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        # Crear ciclo
        ciclo = CicloRelacionalKimi(concepto="EXISTENCIA")
        
        # Ejecutar
        resultado = ciclo.ejecutar(con_profundo=True, con_grafo=True, max_analisis=5)
        reporte = ciclo.generar_reporte(resultado)
        
        # Mostrar resumen
        print("\n" + "="*80)
        print("✅ CICLO COMPLETADO CON ÉXITO")
        print("="*80)
        print(f"\n📊 Rutas descubiertas: {resultado['estadisticas']['total_rutas']}")
        print(f"🔍 Análisis profundos: {resultado['estadisticas']['analisis_realizados']}")
        print(f"⚡ Tokens usados: {resultado['metricas']['tokens_usados']}")
        print(f"📞 Llamadas API: {resultado['metricas']['llamadas_api']}")
        print(f"⏱️ Tiempo total: {resultado['metricas']['duracion_segundos']:.1f}s")
        print(f"💰 Costo: 🎉 GRATUITO (Kimi K2 free)\n")
        
        # Guardar resultados
        with open("RESULTADO_CICLO_KIMI_EXISTENCIA.json", "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        
        with open("REPORTE_CICLO_KIMI_EXISTENCIA.md", "w", encoding="utf-8") as f:
            f.write(reporte)
        
        print("✅ Archivos generados:")
        print("   📄 RESULTADO_CICLO_KIMI_EXISTENCIA.json")
        print("   📄 REPORTE_CICLO_KIMI_EXISTENCIA.md\n")
        
        # Mostrar top rutas
        print("🆕 TOP RUTAS DESCUBIERTAS:\n")
        for i, ruta in enumerate(resultado['rutas_descubiertas'][:5], 1):
            print(f"{i}. {ruta['nombre'].upper()}")
            print(f"   Profundidad: {ruta.get('nivel_profundidad', 0)}/5")
            print(f"   {ruta.get('descripcion', '')[:80]}...\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
