#!/usr/bin/env python3
"""
CICLO MAXIMO RELACIONAL - MÓDULO n8n
=====================================
Versión integrable en workflows de n8n.
Genera payload compatible con n8n HTTP Request nodes.
"""

import requests
import json
import re
import time
from typing import Dict, List, Any
from datetime import datetime


class CicloMaximoRelacionalN8n:
    """Versión optimizada para n8n workflows."""
    
    def __init__(self, concepto: str, gemini_key: str):
        self.concepto = concepto
        self.gemini_key = gemini_key
        self.url_gemini = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'
        self.rutas_descubiertas = {}
        self.rutas_canonicas = [
            "etimologica", "sinonímica", "antonímica", "metafórica", "contextual",
            "histórica", "fenomenológica", "dialéctica", "semiótica", "axiológica"
        ]
    
    def _enviar_a_gemini(self, prompt: str) -> str:
        """Envía prompt a Gemini con reintentos."""
        reintentos = 3
        for intento in range(reintentos):
            try:
                response = requests.post(
                    f"{self.url_gemini}?key={self.gemini_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data['candidates'][0]['content']['parts'][0]['text']
                elif intento < reintentos - 1:
                    time.sleep(2 ** intento)
            except Exception as e:
                if intento < reintentos - 1:
                    time.sleep(2 ** intento)
                continue
        
        return ""
    
    def ejecutar_ciclo(self, iteraciones: int = 2) -> Dict[str, Any]:
        """Ejecuta ciclo y retorna payload para n8n."""
        
        for iteracion in range(1, iteraciones + 1):
            # Generar prompt
            rutas_conocidas = list(self.rutas_descubiertas.keys()) + self.rutas_canonicas
            rutas_str = ", ".join(rutas_conocidas)
            
            prompt = f"""DESCUBRE NUEVAS RUTAS FENOMENOLÓGICAS para "{self.concepto}"

NO incluyas: {rutas_str}

Responde SOLO en JSON válido:
{{
  "nuevas_rutas": [
    {{"nombre": "nombre_unico", "descripcion": "...", "certeza": 0.85}},
    ...
  ]
}}"""
            
            # Llamar Gemini
            respuesta = self._enviar_a_gemini(prompt)
            
            # Extraer rutas
            try:
                match = re.search(r'\{[\s\S]*\}', respuesta)
                if match:
                    datos = json.loads(match.group())
                    for ruta in datos.get('nuevas_rutas', []):
                        nombre = ruta.get('nombre', '').lower().replace(' ', '_')
                        if nombre and nombre not in self.rutas_descubiertas:
                            self.rutas_descubiertas[nombre] = {
                                "iteracion": iteracion,
                                "descripcion": ruta.get('descripcion', ''),
                                "certeza": ruta.get('certeza', 0.8)
                            }
            except:
                pass
        
        # Compilar resultado
        total_rutas = len(self.rutas_canonicas) + len(self.rutas_descubiertas)
        
        return {
            "concepto": self.concepto,
            "timestamp": datetime.now().isoformat(),
            "ciclo_completado": True,
            "rutas_canonicas": len(self.rutas_canonicas),
            "rutas_nuevas": len(self.rutas_descubiertas),
            "total_rutas": total_rutas,
            "rutas_descubiertas": self.rutas_descubiertas,
            "factor_maximo": {
                "nombre": "Máximas Rutas Fenomenológicas Posibles",
                "valor": total_rutas
            }
        }


def procesar_ciclo_n8n(concepto: str, gemini_key: str) -> str:
    """Función para usar directamente en n8n HTTP node."""
    ciclo = CicloMaximoRelacionalN8n(concepto, gemini_key)
    resultado = ciclo.ejecutar_ciclo(iteraciones=2)
    return json.dumps(resultado, ensure_ascii=False)


if __name__ == "__main__":
    CONCEPTO = "DESTRUCCION"
    GEMINI_KEY = "AIzaSyB3cpQ-nVNn8qeC6fUhwozpgYxEFoB_Jdk"
    
    ciclo = CicloMaximoRelacionalN8n(CONCEPTO, GEMINI_KEY)
    resultado = ciclo.ejecutar_ciclo(iteraciones=2)
    
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
