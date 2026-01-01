#!/usr/bin/env python3
"""
Chat Interactivo con OpenRouter
Permite conversar con modelos de OpenRouter desde la terminal
"""

import os
import sys
import requests
import json
from datetime import datetime

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

if not OPENROUTER_API_KEY:
    print("❌ Error: OPENROUTER_API_KEY no está configurada")
    print("   Ejecuta: export OPENROUTER_API_KEY='tu-clave'")
    sys.exit(1)

# Modelos disponibles
MODELOS = {
    "1": {"name": "openai/gpt-3.5-turbo", "desc": "GPT-3.5 Turbo (Rápido y económico)"},
    "2": {"name": "openai/gpt-4-turbo", "desc": "GPT-4 Turbo (Más inteligente)"},
    "3": {"name": "anthropic/claude-3.5-sonnet", "desc": "Claude 3.5 Sonnet (Excelente razonamiento)"},
    "4": {"name": "google/gemini-2.0-flash-exp:free", "desc": "Gemini 2.0 Flash (Gratis)"},
    "5": {"name": "meta-llama/llama-3.2-3b-instruct:free", "desc": "Llama 3.2 3B (Gratis)"},
    "6": {"name": "deepseek/deepseek-chat", "desc": "DeepSeek Chat (Económico)"},
    "7": {"name": "mistralai/mistral-7b-instruct:free", "desc": "Mistral 7B (Gratis)"},
}

def llamar_openrouter(mensajes, modelo):
    """Llamar a OpenRouter con historial de mensajes"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/Ell1Ot-rgb/-...Raiz-Dasein",
        "X-Title": "YO-Estructural-Chat",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": modelo,
        "messages": mensajes
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        resultado = response.json()
        
        if "error" in resultado:
            return {"error": resultado["error"]["message"]}
        
        return resultado
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def mostrar_menu():
    """Mostrar menú de modelos"""
    print("\n" + "="*60)
    print("🤖 CHAT INTERACTIVO CON OPENROUTER")
    print("="*60)
    print("\nModelos disponibles:")
    for key, modelo in MODELOS.items():
        print(f"  {key}. {modelo['desc']}")
    print("\n  0. Salir")
    print("="*60)

def chat_interactivo():
    """Iniciar chat interactivo"""
    mostrar_menu()
    
    # Seleccionar modelo
    while True:
        seleccion = input("\n👉 Selecciona un modelo (1-7): ").strip()
        if seleccion == "0":
            print("👋 ¡Hasta luego!")
            sys.exit(0)
        if seleccion in MODELOS:
            modelo_elegido = MODELOS[seleccion]["name"]
            print(f"\n✅ Modelo seleccionado: {MODELOS[seleccion]['desc']}")
            break
        print("❌ Opción inválida. Intenta de nuevo.")
    
    # Iniciar conversación
    historial = []
    print("\n" + "="*60)
    print("💬 Conversación iniciada")
    print("   Escribe 'salir' para terminar")
    print("   Escribe 'limpiar' para reiniciar la conversación")
    print("   Escribe 'modelo' para cambiar de modelo")
    print("="*60 + "\n")
    
    while True:
        # Obtener input del usuario
        try:
            usuario_input = input("🧑 Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Chat finalizado.")
            break
        
        if not usuario_input:
            continue
        
        # Comandos especiales
        if usuario_input.lower() == "salir":
            print("👋 ¡Hasta luego!")
            break
        
        if usuario_input.lower() == "limpiar":
            historial = []
            print("🧹 Historial limpiado. Conversación reiniciada.\n")
            continue
        
        if usuario_input.lower() == "modelo":
            chat_interactivo()  # Reiniciar con nuevo modelo
            return
        
        # Añadir mensaje del usuario al historial
        historial.append({"role": "user", "content": usuario_input})
        
        # Llamar a la API
        print("🤖 ", end="", flush=True)
        resultado = llamar_openrouter(historial, modelo_elegido)
        
        if "error" in resultado:
            print(f"❌ Error: {resultado['error']}\n")
            historial.pop()  # Remover último mensaje si hubo error
            continue
        
        # Extraer respuesta
        respuesta = resultado.get("choices", [{}])[0].get("message", {}).get("content", "Sin respuesta")
        tokens = resultado.get("usage", {}).get("total_tokens", "N/A")
        
        # Mostrar respuesta
        print(f"{respuesta}")
        print(f"   [Tokens: {tokens}]\n")
        
        # Añadir respuesta del asistente al historial
        historial.append({"role": "assistant", "content": respuesta})
    
    # Guardar conversación
    if historial:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversacion_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "modelo": modelo_elegido,
                "timestamp": timestamp,
                "mensajes": historial
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 Conversación guardada en: {filename}")

if __name__ == "__main__":
    try:
        chat_interactivo()
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrumpido. ¡Hasta luego!")
