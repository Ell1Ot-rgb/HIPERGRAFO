#!/usr/bin/env python3
"""
Script de prueba para OpenRouter API
Resuelve operaciones matemáticas y consultas usando diferentes modelos
"""

import os
import requests
import json

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

def consultar_openrouter(prompt, modelo="openai/gpt-3.5-turbo"):
    """
    Consulta a OpenRouter API con el modelo especificado
    
    Modelos disponibles gratuitos:
    - openai/gpt-3.5-turbo
    - meta-llama/llama-3.2-3b-instruct:free
    - google/gemini-2.0-flash-exp:free
    - moonshot/moonshot-v1-8k (Kimi)
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/Ell1Ot-rgb/-...Raiz-Dasein",
        "X-Title": "YO-Estructural",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "status_code": response.status_code if hasattr(response, 'status_code') else None}

if __name__ == "__main__":
    print("🧮 Probando OpenRouter API...")
    print("=" * 50)
    
    # Prueba 1: Matemáticas básicas
    print("\n1️⃣ Pregunta: ¿Cuánto es 2 + 2?")
    resultado = consultar_openrouter("¿Cuánto es 2 + 2? Responde solo con el número.")
    
    if "error" not in resultado:
        respuesta = resultado.get("choices", [{}])[0].get("message", {}).get("content", "Sin respuesta")
        print(f"✅ Respuesta: {respuesta}")
        print(f"📊 Tokens usados: {resultado.get('usage', {}).get('total_tokens', 'N/A')}")
    else:
        print(f"❌ Error: {resultado['error']}")
    
    # Prueba 2: Consulta filosófica
    print("\n2️⃣ Pregunta filosófica: ¿Qué es la fenomenología?")
    resultado2 = consultar_openrouter("Define fenomenología en una frase corta.")
    
    if "error" not in resultado2:
        respuesta2 = resultado2.get("choices", [{}])[0].get("message", {}).get("content", "Sin respuesta")
        print(f"✅ Respuesta: {respuesta2}")
    else:
        print(f"❌ Error: {resultado2['error']}")
    
    print("\n" + "=" * 50)
    print("✨ Prueba completada")
