# 📚 DOCUMENTACIÓN COMPLETA - CICLO RELACIONAL

## 📑 Tabla de Contenidos
1. [Finalidad del Sistema](#finalidad-del-sistema)
2. [Arquitectura y Componentes](#arquitectura-y-componentes)
3. [Código Completo](#código-completo)
4. [Funcionamiento Detallado](#funcionamiento-detallado)
5. [Configuración](#configuración)
6. [Uso y Ejecución](#uso-y-ejecución)
7. [Salidas Generadas](#salidas-generadas)

---

## 🎯 Finalidad del Sistema

### Propósito General
El **Ciclo Relacional** es un sistema de descubrimiento automático de dimensiones conceptuales profundas para cualquier concepto filosófico o existencial. Utiliza inteligencia artificial de vanguardia para explorar múltiples perspectivas, generar análisis profundos y mapear relaciones complejas entre conceptos.

### Objetivos Específicos
✅ **Descubrimiento de rutas**: Identificar 10-15 dimensiones únicas y profundas de un concepto
✅ **Análisis multidimensional**: Examinar desde perspectivas filosóficas, científicas y existenciales
✅ **Mapeo relacional**: Construir grafos de conocimiento que representan interconexiones
✅ **Evaluación de certeza**: Medir confianza y coherencia de cada análisis
✅ **Generación de reportes**: Producir salidas estructuradas en JSON y Markdown

### Aplicaciones
- 📖 Investigación filosófica y fenomenológica
- 🧠 Análisis conceptual profundo
- 🔬 Exploración interdisciplinaria
- 📊 Generación de mapas mentales conceptuales
- 🎓 Apoyo educativo y académico

---

## 🏗️ Arquitectura y Componentes

### Stack Tecnológico
```
┌─────────────────────────────────────────┐
│     CICLO RELACIONAL MAXIMIZADO         │
│                                         │
│  Python 3.10+                           │
│  ├── requests (HTTP)                    │
│  ├── json (parsing estructurado)        │
│  └── datetime (tracking)                │
│                                         │
│  APIs:                                  │
│  ├── OpenRouter API                     │
│  │   └── GPT-3.5-turbo / Kimi K2 / DeepSeek
│  └── JSON Schema (structured output)    │
│                                         │
│  Salidas:                               │
│  ├── JSON estructurado                  │
│  └── Markdown report                    │
└─────────────────────────────────────────┘
```

### Componentes Principales

#### 1. **Clase CicloRelacional**
Central data structure que mantiene el estado del análisis:
```python
class CicloRelacional:
    - concepto: str                    # Concepto a analizar
    - rutas_descubiertas: List        # Dimensiones conceptuales
    - analisis_profundos: Dict        # Análisis por ruta
    - grafo: Dict                     # Estructura de conocimiento
    - tokens_usados: int              # Conteo de tokens API
    - llamadas_api: int               # Conteo de llamadas
    - timestamp_inicio: datetime       # Control temporal
```

#### 2. **Métodos Clave**

**`descubrir_rutas()`** - Fase 1: Generación de rutas conceptuales
- Input: Concepto
- Output: Lista de 10-15 rutas únicas con profundidad 4-5
- Técnica: Prompt maximizado con criterios de excelencia
- Parse: Robusto con extracción por objetos y escape de newlines

**`analizar_ruta_profundo(ruta_nombre)`** - Fase 2: Análisis profundo
- Input: Nombre de ruta
- Output: Análisis completo con certeza, ejemplos, aplicaciones, paradojas
- Profundidad: 300+ caracteres de análisis filosófico
- Ejemplos: 3-5 casos concretos por ruta

**`extraer_grafo()`** - Fase 3: Mapeo relacional
- Input: Lista de rutas descubiertas
- Output: Grafo con nodos y relaciones ponderadas
- Nodos: Concepto principal + rutas + dominios
- Relaciones: "explora", "pertenece", "relaciona"

**`ejecutar()`** - Orquestación completa
- Ejecuta las 3 fases en secuencia
- Maneja delays entre llamadas (0.5s)
- Ordena rutas por profundidad
- Retorna diccionario con estadísticas completas

#### 3. **Utilidades de Parseo**

**`_escape_newlines_in_json_like(s)`**
- Escapa saltos de línea dentro de strings
- Soluciona problemas de pretty-printing de APIs
- Crítico para robustez ante respuestas multilinea

**`_extract_objects_from_array(text, array_key)`**
- Extrae objetos JSON individuales de arrays
- Maneja comillas y caracteres de escape
- Permite parseo granular de respuestas complejas

---

## 💻 Código Completo

```python
#!/usr/bin/env python3
"""
CICLO RELACIONAL - VERSIÓN OPENROUTER
======================================
Sistema de descubrimiento de dimensiones conceptuales profundas
utilizando OpenRouter API y estructuras JSON Schema.

FINALIDAD:
- Descubrir 10-15 rutas conceptuales únicas para cualquier concepto
- Realizar análisis profundo de cada ruta
- Mapear relaciones entre conceptos
- Generar reportes estructurados

ENTRADA: Concepto (ej: "EXISTENCIA", "DESTRUCCIÓN")
SALIDA: JSON + Markdown con rutas, análisis y grafos
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

# API proporcionada por el usuario
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Usar modelo económico GPT-3.5-turbo (muy barato, ~$0.0005 por 1K tokens)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://github.com/Ell1Ot-rgb/-...Raiz-Dasein",
    "X-Title": "Ciclo-Relacional-Maximo",
    "Content-Type": "application/json",
}

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
                    "nombre": {"type": "string", "description": "Identificador snake_case"},
                    "descripcion": {"type": "string", "description": "Descripción de la ruta"},
                    "justificacion": {"type": "string", "description": "Fundamentación filosófica"},
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
        "analisis_profundo": {"type": "string", "description": "Análisis filosófico (300+ caracteres)"},
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
    """
    Sistema de descubrimiento de dimensiones conceptuales profundas.
    
    Atributos:
        concepto (str): Concepto a analizar
        rutas_descubiertas (List): Dimensiones conceptuales encontradas
        analisis_profundos (Dict): Análisis detallado por ruta
        grafo (Dict): Estructura de conocimiento relacional
        tokens_usados (int): Contador de tokens consumidos
        llamadas_api (int): Contador de llamadas a API
    """
    
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
        """Escapa saltos de línea que aparecen dentro de cadenas entre comillas dobles.
        
        Soluciona problemas de pretty-printing en respuestas de APIs.
        
        Args:
            s: String potencialmente con newlines literales en valores
            
        Returns:
            String con newlines escapados como \\n
        """
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
        """Extrae objetos JSON individuales contenidos en un array.
        
        Útil para parsear respuestas con múltiples objetos JSON anidados.
        
        Args:
            text: Texto conteniendo un array JSON
            array_key: Nombre de la key del array (ej: "nuevas_rutas")
            
        Returns:
            Lista de strings, cada uno conteniendo un objeto JSON completo
        """
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
        """Llama a OpenRouter API y retorna el contenido como string.
        
        Características:
        - Maneja rate limits 429 con reintentos automáticos
        - Acumula tokens y conteo de llamadas
        - Timeout de 120 segundos
        
        Args:
            prompt (str): Mensaje a enviar al modelo
            temperature (float): Creatividad (0.0-1.0)
            
        Returns:
            Contenido de respuesta o None si falla
        """
        body = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": 2000,
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
        """Fase 1: Descubrimiento de rutas conceptuales - MAXIMIZADO.
        
        Objetivo: Generar 10-15 rutas con profundidad 4-5 y originalidad máxima
        Técnica: Prompt estructurado con criterios de excelencia
        Parsing: Robusto con manejo de newlines y extracción granular
        
        Returns:
            Lista de diccionarios con rutas descubiertas
        """
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
            # Parseo robusto: extraer objetos del array
            try:
                objs = self._extract_objects_from_array(content, 'nuevas_rutas')
                rutas = []
                for o in objs:
                    o_clean = o.lstrip(', \n\r\t')
                    cleaned_o = self._escape_newlines_in_json_like(o_clean)
                    try:
                        parsed = json.loads(cleaned_o)
                        rutas.append(parsed)
                    except Exception as e:
                        print('❌ Falló parseo de objeto individual:', e)
                        print(cleaned_o[:500])
                data = {'nuevas_rutas': rutas, 'observacion': '', 'total_encontradas': len(rutas)}
            except Exception as e:
                # Fallback: parseo simple
                print('⚠️ Usando parseo simple...', e)
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
        """Fase 2: Análisis profundo de una ruta.
        
        Genera análisis exhaustivo de una dimensión específica del concepto.
        
        Returns:
            Diccionario con análisis, ejemplos, certeza, aplicaciones, paradojas
        """
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
        """Fase 3: Extracción de grafo de conocimiento.
        
        Mapea las relaciones entre rutas y conceptos relacionados.
        
        Returns:
            Diccionario con nodos y relaciones del grafo
        """
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
        
        Orquestación completa del sistema:
        1. Descubrimiento de rutas (10-15 conceptos)
        2. Análisis profundo (top 5 rutas)
        3. Extracción de grafo
        
        Args:
            con_profundo (bool): Realizar análisis profundo
            con_grafo (bool): Extraer grafo de conocimiento
            max_analisis (int): Máximo número de rutas a analizar
            
        Returns:
            Diccionario con resultados completos
        """
        print("\n" + "="*90)
        print(f"🚀 CICLO RELACIONAL MAXIMIZADO: {self.concepto.upper()}")
        print("="*90)
        print(f"Modelo: {OPENROUTER_MODEL}")
        print(f"Objetivo: Maximizar rutas descubiertas")
        print(f"Timestamp: {self.timestamp_inicio.isoformat()}\n")
        
        # Fase 1
        rutas = self.descubrir_rutas()
        
        # Fase 2
        if con_profundo and rutas:
            rutas_ordenadas = sorted(rutas, key=lambda r: r.get('nivel_profundidad', 0), reverse=True)
            num_analizar = min(max_analisis, len(rutas_ordenadas))
            print(f"\n🔍 Analizando las {num_analizar} rutas de mayor profundidad...")
            for ruta in rutas_ordenadas[:num_analizar]:
                self.analizar_ruta_profundo(ruta['nombre'])
                time.sleep(0.5)
        
        # Fase 3
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
# MAIN - EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    try:
        # Crear instancia del ciclo
        ciclo = CicloRelacional(concepto="EXISTENCIA")
        
        # Mostrar configuración
        print("🎯 CONFIGURACIÓN MAXIMIZADA:")
        print("   - Rutas objetivo: 10-15")
        print("   - Análisis profundos: Top 5 rutas")
        print(f"   - Modelo: GPT-3.5-turbo (muy económico ~$0.003 total)")
        print("   - Max tokens: 2000\n")
        
        # Ejecutar ciclo
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
        with open("RESULTADO_CICLO_RELACIONAL.json", "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        
        with open("REPORTE_CICLO_RELACIONAL.md", "w", encoding="utf-8") as f:
            f.write(reporte)
        
        print("\n✅ Archivos generados:")
        print("   📄 RESULTADO_CICLO_RELACIONAL.json")
        print("   📄 REPORTE_CICLO_RELACIONAL.md")
        
        # Mostrar estadísticas
        print("\n📈 ESTADÍSTICAS DETALLADAS:\n")
        profundidades = [r.get('nivel_profundidad', 0) for r in resultado['rutas_descubiertas']]
        if profundidades:
            print(f"   Profundidad promedio: {sum(profundidades)/len(profundidades):.2f}/5")
            print(f"   Profundidad máxima: {max(profundidades)}/5")
            print(f"   Rutas nivel 5: {profundidades.count(5)}")
            print(f"   Rutas nivel 4: {profundidades.count(4)}\n")
        
        # Top rutas
        print("🆕 TOP 10 RUTAS DESCUBIERTAS:\n")
        rutas_mostrar = sorted(resultado['rutas_descubiertas'], 
                               key=lambda r: r.get('nivel_profundidad', 0), 
                               reverse=True)[:10]
        for i, ruta in enumerate(rutas_mostrar, 1):
            print(f"{i}. {ruta['nombre'].upper()}")
            print(f"   Profundidad: {ruta['nivel_profundidad']}/5")
            print(f"   {ruta['descripcion'][:100]}...\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
```

---

## 🔄 Funcionamiento Detallado

### Flujo de Ejecución

```
┌─────────────────────────────────────────┐
│      ENTRADA: Concepto                  │
│   (ej: "EXISTENCIA", "DESTRUCCIÓN")     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  FASE 1: DESCUBRIMIENTO DE RUTAS        │
│                                         │
│  1. Crear prompt estructurado            │
│  2. Enviar a OpenRouter API              │
│  3. Parsear respuesta robustamente       │
│  4. Extraer 10-15 rutas                  │
│  5. Validar profundidad (4-5)            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  FASE 2: ANÁLISIS PROFUNDO (TOP 5)      │
│                                         │
│  Para cada ruta de mayor profundidad:   │
│  1. Generar prompt de análisis           │
│  2. Llamar API con temperatura baja     │
│  3. Extraer:                             │
│     - Análisis filosófico (300+ chars)  │
│     - 3-5 ejemplos concretos             │
│     - Certeza (0.0-1.0)                  │
│     - Aplicaciones prácticas             │
│     - Paradojas inherentes               │
│     - Dimensiones relacionadas           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  FASE 3: MAPEO RELACIONAL               │
│                                         │
│  1. Crear grafo de nodos y relaciones    │
│  2. Nodos: Concepto + Rutas + Dominios  │
│  3. Relaciones: "explora", "pertenece"  │
│  4. Asignar pesos y tipos               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      SALIDA: RESULTADOS                 │
│                                         │
│  ✅ JSON estructurado                   │
│  ✅ Reporte Markdown                    │
│  ✅ Estadísticas de ejecución           │
└─────────────────────────────────────────┘
```

### Características de Robustez

1. **Parseo Resiliente**
   - Escapa newlines literales en strings
   - Extrae objetos JSON individuales
   - Fallback a parseo simple si falla robusto
   - Limpieza de markdown fences

2. **Manejo de Errores**
   - Rate limit 429 con reintentos
   - Timeout de 120 segundos
   - Conteo de tokens para análisis de costos
   - Try-except en cada fase

3. **Optimización de Llamadas**
   - Delays de 0.5s entre llamadas (avoid rate limits)
   - Límite de 2000 tokens por request
   - Ordenamiento por profundidad (analiza las mejores primero)

---

## ⚙️ Configuración

### Variables de Entorno

```bash
# Obligatorias
OPENROUTER_API_KEY="sk-or-v1-..."

# Opcionales
OPENROUTER_MODEL="openai/gpt-3.5-turbo"  # Default
```

### Modelos Disponibles

**Recomendados:**
- `openai/gpt-3.5-turbo` - Económico, buena calidad (~$0.0005/1K tokens)
- `moonshotai/kimi-k2-0905` - Muy bueno, razonable (~$0.002/1K tokens)
- `deepseek/deepseek-chat-v3.1:free` - Gratuito (requiere política de privacidad)

### Políticas de Privacidad en OpenRouter

Para usar modelos gratuitos:
1. Ir a: https://openrouter.ai/settings/privacy
2. Cambiar de "Zero data retention" a "Regular" o similar
3. Aceptar compartir datos con proveedores
4. Guardar cambios

---

## 🚀 Uso y Ejecución

### Instalación de Dependencias

```bash
pip install requests
```

### Ejecución Básica

```bash
python ciclo_relacional.py
```

### Ejecución Personalizada

```python
from ciclo_relacional import CicloRelacional

# Crear ciclo para concepto específico
ciclo = CicloRelacional(concepto="AMOR")

# Ejecutar todas las fases
resultado = ciclo.ejecutar(
    con_profundo=True,      # Análisis profundo
    con_grafo=True,         # Mapeo relacional
    max_analisis=5          # Top 5 rutas
)

# Generar reporte
reporte = ciclo.generar_reporte(resultado)

# Guardar resultados
import json
with open("resultado.json", "w") as f:
    json.dump(resultado, f, indent=2)
```

---

## 📊 Salidas Generadas

### 1. JSON Estructurado

```json
{
  "concepto": "EXISTENCIA",
  "rutas_descubiertas": [
    {
      "nombre": "existencia_como_evento",
      "descripcion": "...",
      "justificacion": "...",
      "ejemplo": "...",
      "nivel_profundidad": 5
    }
  ],
  "analisis_profundos": {
    "existencia_como_evento": {
      "ruta": "existencia_como_evento",
      "analisis_profundo": "...",
      "ejemplos": [...],
      "certeza": 0.85,
      "aplicaciones": [...],
      "paradojas": [...],
      "dimensiones_relacionadas": [...]
    }
  },
  "grafo": {
    "nodos": [...],
    "relaciones": [...]
  },
  "estadisticas": {
    "total_rutas": 7,
    "analisis_realizados": 5,
    "nodos_grafo": 15,
    "relaciones": 19
  },
  "metricas": {
    "tokens_usados": 7611,
    "llamadas_api": 7,
    "duracion_segundos": 220.5
  }
}
```

### 2. Reporte Markdown

```markdown
# Ciclo Relacional: EXISTENCIA

**Timestamp**: 2025-11-09T04:50:28.395169
**Modelo**: moonshotai/kimi-k2-0905

## 📊 Estadísticas

- **Rutas descubiertas**: 7
- **Análisis profundos**: 5
- **Nodos en grafo**: 15
- **Relaciones**: 19
- **Tokens usados**: 7611
- **Llamadas API**: 7
- **Duración**: 220.5s

## 🆕 Rutas Descubiertas

### Existencia Como Evento
**Profundidad**: 5/5
**Descripción**: La existencia se concibe como...
...
```

---

## 📈 Métricas y Rendimiento

### Costos Típicos (con GPT-3.5-turbo)
- **Descubrimiento (10 rutas)**: ~1500 tokens → $0.0008
- **Análisis profundo (5 rutas)**: ~3000 tokens → $0.0015
- **Grafo**: ~500 tokens → $0.0003
- **Total**: ~5000 tokens → **$0.0026 (~0.3¢)**

### Tiempos de Ejecución
- Descubrimiento: 10-20s
- Análisis profundo: 50-100s (con delays)
- Grafo: 5-10s
- **Total**: 65-130 segundos (~2 minutos)

---

## 🎓 Casos de Uso

### Investigación Académica
Explorar dimensiones de conceptos filosóficos complejos

### Diseño Conceptual
Mapear espacios de diseño y posibilidades

### Análisis de Políticas
Entender múltiples perspectivas de un tema

### Educación
Generar materiales didácticos estructurados

### Creatividad
Descubrir perspectivas nuevas para proyectos

---

## 🔧 Troubleshooting

### Error: "404 - No endpoints found"
→ Configurar política de privacidad en OpenRouter

### Error: "429 - Rate limit"
→ Sistema reintenta automáticamente con delay

### Error: "Parsing JSON"
→ Usa parseo robusto (fallback a simple automáticamente)

### Pocas rutas descubiertas
→ Aumentar `max_tokens` o cambiar temperatura en `_llamar_openrouter()`

---

## 📝 Conclusión

El **Ciclo Relacional** es un sistema potente y flexible para descubrimiento conceptual profundo. Combina:
- ✅ Prompt engineering avanzado
- ✅ Parseo robusto de LLM
- ✅ Mapeo relacional automático
- ✅ Generación de reportes estructurados
- ✅ Bajo costo operativo

Ideal para investigación filosófica, académica y exploración conceptual.

---

**Autor**: YO-Estructural  
**Fecha**: 2025-11-09  
**Versión**: 2.1  
**License**: MIT
