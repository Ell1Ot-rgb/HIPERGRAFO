#!/usr/bin/env python3
"""
Runner del ciclo (fase de descubrimiento) usando OpenRouter API como backend.
No modifica archivos existentes. Usa el prompt y schema del ciclo v2.0,
pero realiza la llamada vía OpenRouter Chat Completions (OpenAI compatible).
"""

import os
import json
import time
from datetime import datetime
import requests
from typing import Dict, Any, Optional, List

from ciclo_maximo_relacional_optimizado import (
    CicloMaximoRelacionalOptimizado,
    SCHEMA_RUTAS_DESCUBIERTAS,
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://github.com/Ell1Ot-rgb/-...Raiz-Dasein",
    "X-Title": "YO-Estructural-Ciclo-OpenRouter",
    "Content-Type": "application/json",
}

class CicloOpenRouter(CicloMaximoRelacionalOptimizado):
    """Subclase que reusa el ciclo v2.0 pero llama OpenRouter en lugar de Gemini REST."""

    def _llamar_gemini_structured(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        temperature: float = 0.7,
    ) -> Optional[Dict[str, Any]]:
        if not OPENROUTER_API_KEY:
            print("❌ OPENROUTER_API_KEY no configurada")
            return None

        # Instruimos al modelo a responder SOLO JSON y seguir el schema.
        system_msg = (
            "Eres un asistente que responde exclusivamente en JSON válido. "
            "Tu salida debe cumplir estrictamente el siguiente JSON Schema."
        )

        schema_str = json.dumps(response_schema, ensure_ascii=False)
        user_prompt = (
            f"{prompt}\n\n"
            f"REQUISITO CRÍTICO: Responde SOLO con un objeto JSON válido que cumpla el JSON Schema siguiente.\n"
            f"JSON_SCHEMA={schema_str}"
        )

        body = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            # Forzamos JSON si el proveedor lo soporta (OpenAI style)
            "response_format": {"type": "json_object"},
            "temperature": temperature,
        }

        try:
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=body, timeout=120)
            self.llamadas_api += 1
            if resp.status_code != 200:
                print(f"❌ Error OpenRouter {resp.status_code}: {resp.text[:200]}")
                return None

            data = resp.json()
            # tokens si está disponible
            usage = data.get("usage", {})
            self.tokens_usados += usage.get("total_tokens", 0)

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return None

            # Intentar parseo directo
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Intentar limpiar code fences si vinieran
                content_clean = content.strip()
                if content_clean.startswith("```json"):
                    content_clean = content_clean[7:]
                if content_clean.startswith("```"):
                    content_clean = content_clean[3:]
                if content_clean.endswith("```"):
                    content_clean = content_clean[:-3]
                return json.loads(content_clean)
        except Exception as e:
            print(f"❌ Excepción al llamar OpenRouter: {e}")
            return None


def ejecutar_ciclo_reducido(concepto: str, max_iter: int = 1) -> Dict[str, Any]:
    """Ejecuta solo la FASE 1 (descubrimiento) para limitar llamadas y evitar rate limits."""
    ciclo = CicloOpenRouter(concepto=concepto, gemini_key="OPENROUTER", modelo="openrouter")

    print("\n" + "=" * 90)
    print("🚀 CICLO (OpenRouter) - FASE 1: Descubrimiento de rutas")
    print("=" * 90)
    print(f"Concepto: {concepto}")
    print(f"Modelo: {OPENROUTER_MODEL}")
    print("Nota: Fase 2 (grafo) y Fase 3 (análisis profundo) deshabilitadas para evitar 429.")

    todas_rutas: List[Dict[str, Any]] = []
    for i in range(max_iter):
        ciclo.iteracion = i + 1
        rutas = ciclo._descubrir_rutas_structured()
        if rutas:
            todas_rutas.extend(rutas)
        time.sleep(1)

    resultado = {
        "concepto": concepto,
        "rutas_descubiertas": todas_rutas,
        "total_encontradas": len(todas_rutas),
        "tokens_usados": ciclo.tokens_usados,
        "llamadas_api": ciclo.llamadas_api,
        "timestamp": datetime.now().isoformat(),
        "modelo": OPENROUTER_MODEL,
        "nota": "Ejecución reducida por OpenRouter (solo Fase 1)"
    }
    return resultado


if __name__ == "__main__":
    concepto = os.getenv("CONCEPTO", "EXISTENCIA")
    iteraciones = int(os.getenv("ITERACIONES", "1"))
    out_name = f"RESULTADO_CICLO_OPENROUTER_{concepto.lower()}.json"

    res = ejecutar_ciclo_reducido(concepto, max_iter=iteraciones)

    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Resultado guardado en {out_name}")
    print(f"   Rutas encontradas: {res['total_encontradas']}")
    print(f"   Llamadas API: {res['llamadas_api']}, Tokens: {res['tokens_usados']}")
