#!/usr/bin/env python3
"""
CICLO RELACIONAL - OPENAI SDK CON KIMI K2 0711 GRATUITO
=======================================================
Sistema fenomenológico usando OpenAI SDK con OpenRouter y Kimi K2 gratuito
"""

import os
import json
import time
from datetime import datetime
from openai import OpenAI

# ============================================================================
# CONFIGURACIÓN - API GRATUITA DE KIMI K2 0711
# ============================================================================

# Tu API Key de OpenRouter
OPENROUTER_API_KEY = "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"

# Cliente OpenAI apuntando a OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Headers opcionales para statistics
HEADERS = {
    "HTTP-Referer": "https://github.com/Ell1Ot-rgb/-...Raiz-Dasein",
    "X-Title": "Ciclo-Kimi-Gratuito",
}

# ============================================================================
# CICLO RELACIONAL CON KIMI K2 0711 FREE
# ============================================================================

class CicloKimiGratuito:
    """Sistema de descubrimiento de rutas conceptuales con Kimi K2 gratuito."""
    
    def __init__(self, concepto: str):
        self.concepto = concepto
        self.rutas = []
        self.analisis = {}
        self.tokens_usados = 0
        self.llamadas_api = 0
        self.timestamp_inicio = datetime.now()
        
    def llamar_kimi(self, prompt: str, temperature: float = 0.7) -> str:
        """Llama a Kimi K2 0711 gratuito vía OpenRouter con OpenAI SDK."""
        try:
            print(f"   📡 Llamando a API (temperatura: {temperature})...")
            
            response = client.chat.completions.create(
                model="moonshotai/kimi-k2:free",  # Modelo gratuito
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1500,
                extra_headers=HEADERS,
            )
            
            self.llamadas_api += 1
            self.tokens_usados += response.usage.total_tokens
            
            content = response.choices[0].message.content
            print(f"   ✅ Respuesta recibida ({response.usage.total_tokens} tokens)")
            return content
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def descubrir_rutas(self) -> list:
        """Fase 1: Descubre rutas conceptuales."""
        print(f"\n📍 Fase 1: Descubrimiento de rutas para '{self.concepto}'")
        
        prompt = f"""Eres un filósofo fenomenólogo. Descubre 8-12 rutas conceptuales ÚNICAS y profundas para: '{self.concepto}'

FORMATO JSON REQUERIDO:
{{
  "rutas": [
    {{
      "nombre": "nombre_snake_case",
      "descripcion": "descripción breve",
      "profundidad": 5,
      "ejemplo": "ejemplo concreto"
    }}
  ],
  "total": 10
}}

Genera exactamente en JSON válido, sin markdown."""
        
        content = self.llamar_kimi(prompt, temperature=0.8)
        if not content:
            return []
        
        try:
            # Limpiar respuesta
            if "```" in content:
                content = content.split("```json")[1].split("```")[0] if "```json" in content else content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            self.rutas = data.get("rutas", [])
            print(f"✅ Rutas descubiertas: {len(self.rutas)}")
            
            for r in self.rutas[:5]:
                print(f"   🆕 {r['nombre']} (profundidad: {r.get('profundidad', '?')}/5)")
            
            return self.rutas
            
        except json.JSONDecodeError as e:
            print(f"❌ Error JSON: {e}")
            print(f"Contenido: {content[:200]}")
            return []
    
    def analizar_profundo(self, ruta_nombre: str) -> dict:
        """Fase 2: Análisis profundo de una ruta."""
        print(f"\n🔍 Analizando: {ruta_nombre}")
        
        prompt = f"""Analiza profundamente la ruta '{ruta_nombre}' del concepto '{self.concepto}'.

FORMATO JSON:
{{
  "ruta": "{ruta_nombre}",
  "analisis": "análisis filosófico (150+ caracteres)",
  "certeza": 0.85,
  "ejemplos": ["ej1", "ej2"],
  "aplicaciones": ["app1", "app2"],
  "paradojas": ["paradoja1"]
}}"""
        
        content = self.llamar_kimi(prompt, temperature=0.6)
        if not content:
            return {}
        
        try:
            if "```" in content:
                content = content.split("```json")[1].split("```")[0] if "```json" in content else content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            self.analisis[ruta_nombre] = data
            print(f"✅ Análisis: certeza {data.get('certeza', 0):.2f}")
            return data
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing: {e}")
            return {}
    
    def ejecutar(self) -> dict:
        """Ejecuta el ciclo completo."""
        print("\n" + "="*80)
        print(f"🚀 CICLO RELACIONAL CON KIMI K2 0711 GRATUITO")
        print("="*80)
        print(f"Concepto: {self.concepto}")
        print(f"API: OpenRouter (moonshotai/kimi-k2:free)")
        print(f"Timestamp: {self.timestamp_inicio.isoformat()}\n")
        
        # Fase 1
        rutas = self.descubrir_rutas()
        
        # Fase 2 - Analizar top 3
        if rutas:
            top_rutas = sorted(rutas, key=lambda r: r.get('profundidad', 0), reverse=True)[:3]
            print(f"\n🔍 Analizando top 3 rutas...")
            for ruta in top_rutas:
                self.analizar_profundo(ruta['nombre'])
                time.sleep(0.5)
        
        # Compilar resultado
        resultado = {
            "concepto": self.concepto,
            "rutas_descubiertas": self.rutas,
            "analisis_profundos": self.analisis,
            "estadisticas": {
                "total_rutas": len(self.rutas),
                "analisis_realizados": len(self.analisis),
            },
            "metricas": {
                "tokens_usados": self.tokens_usados,
                "llamadas_api": self.llamadas_api,
                "duracion_segundos": (datetime.now() - self.timestamp_inicio).total_seconds(),
            },
            "modelo": "moonshotai/kimi-k2:free",
            "timestamp": datetime.now().isoformat(),
        }
        
        return resultado


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("✅ Usando API gratuita de Kimi K2 0711")
    print(f"API Key: {OPENROUTER_API_KEY[:30]}...")
    print()
    
    # Crear ciclo
    ciclo = CicloKimiGratuito(concepto="EXISTENCIA")
    
    try:
        # Ejecutar
        resultado = ciclo.ejecutar()
        
        # Mostrar resultados
        print("\n" + "="*80)
        print("✅ CICLO COMPLETADO")
        print("="*80)
        print(f"\n📊 Rutas descubiertas: {resultado['estadisticas']['total_rutas']}")
        print(f"🔍 Análisis realizados: {resultado['estadisticas']['analisis_realizados']}")
        print(f"⚡ Tokens usados: {resultado['metricas']['tokens_usados']}")
        print(f"📞 Llamadas API: {resultado['metricas']['llamadas_api']}")
        print(f"⏱️ Duración: {resultado['metricas']['duracion_segundos']:.1f}s")
        
        # Guardar resultado
        filename = f"RESULTADO_CICLO_KIMI_FREE_{ciclo.concepto}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Resultado guardado: {filename}")
        
        # Mostrar TOP rutas
        print(f"\n🆕 TOP {min(5, len(resultado['rutas_descubiertas']))} RUTAS:\n")
        for i, ruta in enumerate(resultado['rutas_descubiertas'][:5], 1):
            print(f"{i}. {ruta['nombre'].upper()}")
            print(f"   Profundidad: {ruta.get('profundidad', '?')}/5")
            print(f"   {ruta.get('descripcion', '')[:80]}...\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
