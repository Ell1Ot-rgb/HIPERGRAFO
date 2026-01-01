#!/usr/bin/env python3
"""
CICLO RELACIONAL - VERSIÓN OPENROUTER
======================================
Versión adaptada del ciclo optimizado v2.0 que usa OpenRouter API en lugar de Gemini.
Entrada: EXISTENCIA
Salida: JSON estructurado + Markdown report
"""

import os
import json
import requests
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN - API de OpenRouter para maximizar rutas descubiertas
# ============================================================================

# API proporcionada por el usuario - ÚLTIMA API
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Usar GPT-3.5-turbo (económico y confiable, funciona con ambas APIs)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://github.com/Ell1Ot-rgb/-...Raiz-Dasein",
    "X-Title": "Ciclo-Relacional-Maximo",
    "Content-Type": "application/json",
}

print(f"✅ API configurada: {OPENROUTER_API_KEY[:20]}...")
print(f"✅ Modelo: {OPENROUTER_MODEL}\n")

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
                    "nombre": {"type": "string"},
                    "descripcion": {"type": "string"},
                    "justificacion": {"type": "string"},
                    "ejemplo": {"type": "string"},
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
        "analisis_profundo": {"type": "string"},
        "ejemplos": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3
        },
        "certeza": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "aplicaciones": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2
        },
        "paradojas": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
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
                    "tipo": {"type": "string"},
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
                    "destino": {"type": "string"},
                    "tipo": {"type": "string"},
                    "peso": {"type": "number"}
                },
                "required": ["origen", "destino", "tipo"]
            }
        }
    },
    "required": ["nodos", "relaciones"]
}

# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class CicloRelacional:
    def __init__(self, concepto: str):
        self.concepto = concepto
        self.rutas_descubiertas = []
        self.analisis_profundos = {}
        self.grafo = {"nodos": [], "relaciones": []}
        self.tokens_usados = 0
        self.llamadas_api = 0
        self.timestamp_inicio = datetime.now()
        
        if not OPENROUTER_API_KEY:
            raise ValueError("❌ OPENROUTER_API_KEY no configurada")
    
    @staticmethod
    def _escape_newlines_in_json_like(s: str) -> str:
        """Escapa saltos de línea que aparecen dentro de cadenas entre comillas dobles."""
        out = []
        in_string = False
        esc = False
        for ch in s:
            if ch == '"' and not esc:
                in_string = not in_string
                out.append(ch)
                continue
            if ch == '\\' and not esc:
                esc = True
                out.append(ch)
                continue
            if ch in '\r\n' and in_string and not esc:
                out.append('\\n')
                continue
            out.append(ch)
            esc = False
        return ''.join(out)
    
    @staticmethod
    def _extract_objects_from_array(text: str, array_key: str) -> List[str]:
        """Extrae objetos JSON individuales contenidos en el array identificado por array_key."""
        start_idx = text.find(f'"{array_key}"')
        if start_idx == -1:
            return []
        arr_start = text.find('[', start_idx)
        if arr_start == -1:
            return []
        
        objs = []
        i = arr_start + 1
        n = len(text)
        brace = 0
        in_string = False
        esc = False
        current = []
        while i < n:
            ch = text[i]
            if ch == '"' and not esc:
                in_string = not in_string
                current.append(ch)
                i += 1
                continue
            if ch == '\\' and not esc:
                esc = True
                current.append(ch)
                i += 1
                continue
            if ch == '{' and not in_string:
                brace += 1
                current.append(ch)
                i += 1
                continue
            if ch == '}' and not in_string:
                brace -= 1
                current.append(ch)
                if brace == 0:
                    objs.append(''.join(current))
                    current = []
                i += 1
                continue
            current.append(ch)
            esc = False
            i += 1
        return objs
    
    def _llamar_openrouter(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """Llama a OpenRouter y retorna el contenido como string."""
        body = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": 2000,  # Limitar tokens para evitar errores 402
        }
        
        try:
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=body, timeout=120)
            self.llamadas_api += 1
            
            if resp.status_code == 429:
                print("⏳ Rate limit 429 detected. Esperando 30s...")
                time.sleep(30)
                return self._llamar_openrouter(prompt, temperature)
            
            if resp.status_code != 200:
                print(f"❌ Error OpenRouter {resp.status_code}: {resp.text[:200]}")
                return None
            
            data = resp.json()
            usage = data.get("usage", {})
            self.tokens_usados += usage.get("total_tokens", 0)
            
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        except Exception as e:
            print(f"❌ Excepción OpenRouter: {e}")
            return None
    
    def descubrir_rutas(self) -> List[Dict[str, Any]]:
        """Fase 1: Descubrimiento de rutas - MAXIMIZADO."""
        print(f"\n📍 Fase 1: Descubrimiento MÁXIMO de rutas para '{self.concepto}'")
        
        prompt = f"""Eres un filósofo fenomenólogo de elite. Tu tarea es descubrir el MÁXIMO número posible de rutas conceptuales únicas, profundas y originales para el concepto: '{self.concepto}'

OBJETIVO: Genera entre 10-15 rutas conceptuales excepcionales que exploren todas las dimensiones posibles.

CRITERIOS DE EXCELENCIA:
- Cada ruta debe ser radicalmente original y no trivial
- Profundidad intelectual máxima (priorizar nivel 4-5)
- Justificación filosófica sólida
- Ejemplos concretos y reveladores
- Diversidad de perspectivas: ontológica, fenomenológica, existencial, relacional, temporal, corporal, ética, etc.

DIRECTRICES:
- Explora perspectivas interdisciplinarias (neurociencia, física cuántica, antropología, psicología, etc.)
- Incluye paradojas y tensiones conceptuales
- Considera dimensiones temporales, espaciales, relacionales
- No te limites a lo obvio: busca lo sorprendente y lo profundo

Responde SOLO con un objeto JSON válido siguiendo EXACTAMENTE esta estructura:
{{
  "nuevas_rutas": [
    {{
      "nombre": "nombre_snake_case_descriptivo",
      "descripcion": "descripción rica y detallada (mínimo 100 caracteres)",
      "justificacion": "justificación filosófica profunda",
      "ejemplo": "ejemplo concreto y revelador",
      "nivel_profundidad": 5
    }}
  ],
  "observacion": "reflexión meta-filosófica sobre las rutas descubiertas",
  "total_encontradas": 12
}}

IMPORTANTE: Genera al menos 10 rutas de alta calidad. Más es mejor."""
        
        content = self._llamar_openrouter(prompt, temperature=0.8)
        if not content:
            return []
        
        try:
            # Intentar parseo robusto: extraer objetos del array 'nuevas_rutas'
            try:
                objs = self._extract_objects_from_array(content, 'nuevas_rutas')
                rutas = []
                for o in objs:
                    # Recortar comas o espacios iniciales que pueden preceder al objeto
                    o_clean = o.lstrip(', \n\r\t')
                    # escapar newlines dentro de strings y parsear cada objeto
                    cleaned_o = self._escape_newlines_in_json_like(o_clean)
                    try:
                        parsed = json.loads(cleaned_o)
                        rutas.append(parsed)
                    except Exception as e:
                        # mostrar el fragmento problemático para diagnóstico
                        print('❌ Falló parseo de objeto individual:', e)
                        print(cleaned_o[:500])
                data = {'nuevas_rutas': rutas, 'observacion': '', 'total_encontradas': len(rutas)}
            except Exception as e:
                # Fallback: parseo simple
                print('⚠️ Usando parseo simple...', e)
                # Limpiar si tiene markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                data = json.loads(content)
            self.rutas_descubiertas = data.get("nuevas_rutas", [])
            print(f"✅ Rutas descubiertas: {len(self.rutas_descubiertas)}")
            for r in self.rutas_descubiertas:
                print(f"   🆕 {r['nombre']} (profundidad {r['nivel_profundidad']}/5)")
            return self.rutas_descubiertas
        except Exception as e:
            print(f"❌ Error parsing rutas: {e}")
            return []
    
    def analizar_ruta_profundo(self, ruta_nombre: str) -> Dict[str, Any]:
        """Fase 2: Análisis profundo de una ruta."""
        print(f"\n🔍 Analizando '{ruta_nombre}'...")
        
        prompt = f"""Para la ruta conceptual '{ruta_nombre}' del concepto '{self.concepto}':

Proporciona un análisis profundo que incluya:
- Análisis filosófico completo (mínimo 300 caracteres)
- 3-5 ejemplos concretos
- Certeza del análisis (0.0-1.0)
- 2-3 aplicaciones prácticas
- 1-2 paradojas o contradicciones
- 2-4 dimensiones relacionadas

Responde SOLO con JSON:
{{
  "ruta": "{ruta_nombre}",
  "analisis_profundo": "...",
  "ejemplos": ["ej1", "ej2", "ej3"],
  "certeza": 0.85,
  "aplicaciones": ["app1", "app2"],
  "paradojas": ["par1"],
  "dimensiones_relacionadas": ["dim1", "dim2"]
}}"""
        
        content = self._llamar_openrouter(prompt, temperature=0.6)
        if not content:
            return {}
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            self.analisis_profundos[ruta_nombre] = data
            print(f"✅ Análisis completado (certeza: {data.get('certeza', 0):.2f})")
            return data
        except Exception as e:
            print(f"❌ Error parsing análisis: {e}")
            return {}
    
    def extraer_grafo(self) -> Dict[str, Any]:
        """Fase 3: Extracción de grafo de conocimiento."""
        print(f"\n🕸️ Extrayendo grafo de conocimiento...")
        
        rutas_str = "\n".join([r['nombre'] for r in self.rutas_descubiertas])
        
        prompt = f"""Basándote en estas rutas del concepto '{self.concepto}':
{rutas_str}

Construye un grafo de conocimiento que capture las relaciones entre conceptos.

Responde SOLO con JSON:
{{
  "nodos": [
    {{"id": "{self.concepto}", "tipo": "Concepto", "propiedades": {{}}}},
    {{"id": "ruta1", "tipo": "Ruta", "propiedades": {{}}}}
  ],
  "relaciones": [
    {{"origen": "{self.concepto}", "destino": "ruta1", "tipo": "explora", "peso": 1.0}}
  ]
}}"""
        
        content = self._llamar_openrouter(prompt, temperature=0.5)
        if not content:
            return {"nodos": [], "relaciones": []}
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            self.grafo = data
            print(f"✅ Grafo extraído: {len(data.get('nodos', []))} nodos, {len(data.get('relaciones', []))} relaciones")
            return data
        except Exception as e:
            print(f"❌ Error parsing grafo: {e}")
            return {"nodos": [], "relaciones": []}
    
    def ejecutar(self, con_profundo: bool = True, con_grafo: bool = True, max_analisis: int = 5) -> Dict[str, Any]:
        """Ejecuta todas las fases del ciclo.
        
        Args:
            con_profundo: Si True, realiza análisis profundo de rutas
            con_grafo: Si True, extrae grafo de conocimiento
            max_analisis: Número máximo de rutas a analizar en profundidad (default: 5)
        """
        print("\n" + "="*90)
        print(f"🚀 CICLO RELACIONAL MAXIMIZADO: {self.concepto.upper()}")
        print("="*90)
        print(f"Modelo: {OPENROUTER_MODEL}")
        print(f"Objetivo: Maximizar rutas descubiertas")
        print(f"Timestamp: {self.timestamp_inicio.isoformat()}\n")
        
        # Fase 1 - Maximizar descubrimiento
        rutas = self.descubrir_rutas()
        
        # Fase 2 (análisis profundo de las mejores rutas)
        if con_profundo and rutas:
            # Analizar hasta max_analisis rutas, priorizando las de mayor profundidad
            rutas_ordenadas = sorted(rutas, key=lambda r: r.get('nivel_profundidad', 0), reverse=True)
            num_analizar = min(max_analisis, len(rutas_ordenadas))
            print(f"\n🔍 Analizando las {num_analizar} rutas de mayor profundidad...")
            for ruta in rutas_ordenadas[:num_analizar]:
                self.analizar_ruta_profundo(ruta['nombre'])
                time.sleep(0.5)  # Pequeño delay entre llamadas
        
        # Fase 3 (opcional)
        if con_grafo and rutas:
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
            "modelo": OPENROUTER_MODEL,
            "timestamp": datetime.now().isoformat()
        }
        
        return resultado
    
    def generar_reporte(self, resultado: Dict[str, Any]) -> str:
        """Genera reporte en Markdown."""
        reporte = f"# Ciclo Relacional: {resultado['concepto']}\n\n"
        reporte += f"**Timestamp**: {resultado['timestamp']}\n"
        reporte += f"**Modelo**: {resultado['modelo']}\n\n"
        
        reporte += "## 📊 Estadísticas\n\n"
        reporte += f"- **Rutas descubiertas**: {resultado['estadisticas']['total_rutas']}\n"
        reporte += f"- **Análisis profundos**: {resultado['estadisticas']['analisis_realizados']}\n"
        reporte += f"- **Nodos en grafo**: {resultado['estadisticas']['nodos_grafo']}\n"
        reporte += f"- **Relaciones**: {resultado['estadisticas']['relaciones']}\n"
        reporte += f"- **Tokens usados**: {resultado['metricas']['tokens_usados']}\n"
        reporte += f"- **Llamadas API**: {resultado['metricas']['llamadas_api']}\n"
        reporte += f"- **Duración**: {resultado['metricas']['duracion_segundos']:.1f}s\n\n"
        
        reporte += "## 🆕 Rutas Descubiertas\n\n"
        for ruta in resultado['rutas_descubiertas']:
            reporte += f"### {ruta['nombre'].replace('_', ' ').title()}\n\n"
            reporte += f"**Profundidad**: {ruta['nivel_profundidad']}/5\n\n"
            reporte += f"**Descripción**: {ruta['descripcion']}\n\n"
            reporte += f"**Justificación**: {ruta['justificacion']}\n\n"
            reporte += f"**Ejemplo**: {ruta['ejemplo']}\n\n"
            reporte += "---\n\n"
        
        if resultado['analisis_profundos']:
            reporte += "## 🔍 Análisis Profundos\n\n"
            for ruta_nombre, analisis in resultado['analisis_profundos'].items():
                reporte += f"### {ruta_nombre.replace('_', ' ').title()}\n\n"
                reporte += f"**Certeza**: {analisis.get('certeza', 0):.2%}\n\n"
                reporte += f"{analisis.get('analisis_profundo', '')}\n\n"
                if analisis.get('ejemplos'):
                    reporte += "**Ejemplos**:\n"
                    for ej in analisis['ejemplos'][:3]:
                        reporte += f"- {ej}\n"
                    reporte += "\n"
                reporte += "---\n\n"
        
        return reporte


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        ciclo = CicloRelacional(concepto="EXISTENCIA")
        
        # Ejecutar MAXIMIZANDO rutas y análisis profundos
        print("🎯 CONFIGURACIÓN MAXIMIZADA:")
        print("   - Rutas objetivo: 10-15")
        print("   - Análisis profundos: Top 5 rutas")
        print(f"   - Modelo: GPT-3.5-turbo (muy económico ~$0.003 total)")
        print("   - Max tokens: 2000\n")
        
        resultado = ciclo.ejecutar(con_profundo=True, con_grafo=True, max_analisis=5)
        reporte = ciclo.generar_reporte(resultado)
        
        # Mostrar resumen
        print("\n" + "="*90)
        print("✅ CICLO MAXIMIZADO COMPLETADO")
        print("="*90)
        print(f"\n📊 Rutas descubiertas: {resultado['estadisticas']['total_rutas']}")
        print(f"🔍 Análisis profundos: {resultado['estadisticas']['analisis_realizados']}")
        print(f"🕸️ Nodos en grafo: {resultado['estadisticas']['nodos_grafo']}")
        print(f"⚡ Tokens usados: {resultado['metricas']['tokens_usados']}")
        print(f"📞 Llamadas API: {resultado['metricas']['llamadas_api']}")
        print(f"⏱️ Tiempo total: {resultado['metricas']['duracion_segundos']:.1f}s")
        
        # Guardar resultados
        with open("RESULTADO_CICLO_RELACIONAL_EXISTENCIA_MAXIMO.json", "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        
        with open("REPORTE_CICLO_RELACIONAL_EXISTENCIA_MAXIMO.md", "w", encoding="utf-8") as f:
            f.write(reporte)
        
        print("\n✅ Archivos generados:")
        print("   📄 RESULTADO_CICLO_RELACIONAL_EXISTENCIA_MAXIMO.json")
        print("   📄 REPORTE_CICLO_RELACIONAL_EXISTENCIA_MAXIMO.md")
        print("\n" + "="*90 + "\n")
        
        # Mostrar estadísticas detalladas
        print("📈 ESTADÍSTICAS DETALLADAS:\n")
        profundidades = [r.get('nivel_profundidad', 0) for r in resultado['rutas_descubiertas']]
        if profundidades:
            print(f"   Profundidad promedio: {sum(profundidades)/len(profundidades):.2f}/5")
            print(f"   Profundidad máxima: {max(profundidades)}/5")
            print(f"   Rutas nivel 5: {profundidades.count(5)}")
            print(f"   Rutas nivel 4: {profundidades.count(4)}\n")
        
        # Mostrar primeras rutas en pantalla
        print("🆕 TOP 10 RUTAS DESCUBIERTAS:\n")
        rutas_mostrar = sorted(resultado['rutas_descubiertas'], 
                               key=lambda r: r.get('nivel_profundidad', 0), 
                               reverse=True)[:10]
        for i, ruta in enumerate(rutas_mostrar, 1):
            print(f"{i}. {ruta['nombre'].upper()}")
            print(f"   Profundidad: {ruta['nivel_profundidad']}/5")
            print(f"   {ruta['descripcion']}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
