#!/usr/bin/env python3
"""
CAPA 2 - Servidor FastAPI Optimizado
Script para probar la Capa 2 COLAB correctamente

Este script se conecta directamente a los endpoints del servidor
y valida que todo esté funcionando correctamente.
"""

import requests
import json
import time
import numpy as np
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

NGROK_URL = "https://paleographic-transonic-adell.ngrok-free.dev"
TIMEOUT = 30

print("\n" + "="*80)
print("🔬 VALIDACIÓN DE CAPA 2 - COLAB")
print("="*80)

# ============================================================================
# TEST 1: Conexión Básica
# ============================================================================

print("\n[1] Verificando conexión al servidor...")

try:
    response = requests.get(NGROK_URL, timeout=TIMEOUT, verify=False)
    print(f"    ✅ Servidor respondiendo (status: {response.status_code})")
    print(f"    URL: {NGROK_URL}")
except Exception as e:
    print(f"    ❌ No hay conexión: {e}")
    print(f"    Asegúrate de que:")
    print(f"    - Google Colab está ejecutando")
    print(f"    - El servidor FastAPI está activo")
    print(f"    - El túnel ngrok está establecido")
    exit(1)

# ============================================================================
# TEST 2: Probar endpoint /train_layer2
# ============================================================================

print("\n[2] Probando endpoint /train_layer2...")

# Crear datos de prueba
batch_size = 2
seq_length = 100
input_dim = 20

# Definición del modelo esperado según cuadernocolab.py:
# input_dim = 20, sequence_length = 100, output_dim = 20

test_payload = {
    "samples": [
        {
            "input_data": np.random.randn(seq_length * input_dim).tolist(),
            "anomaly_label": 0
        }
        for _ in range(batch_size)
    ]
}

print(f"    Payload:")
print(f"      • Muestras: {batch_size}")
print(f"      • Tamaño: {seq_length}x{input_dim}")
print(f"      • Bytes: {len(json.dumps(test_payload)) / 1024:.1f} KB")

try:
    start_time = time.time()
    response = requests.post(
        f"{NGROK_URL}/train_layer2",
        json=test_payload,
        timeout=TIMEOUT,
        verify=False
    )
    elapsed = time.time() - start_time
    
    print(f"\n    Status: {response.status_code}")
    print(f"    Tiempo: {elapsed:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n    ✅ ENTRENAMIENTO EXITOSO")
        
        # Mostrar campos importantes
        if "status" in data:
            print(f"    Status: {data['status']}")
        if "message" in data:
            print(f"    Message: {data['message']}")
        if "loss" in data:
            print(f"    Loss: {data['loss']:.6f}")
        if "epoch" in data:
            print(f"    Epoch: {data['epoch']}")
        if "samples_processed" in data:
            print(f"    Muestras procesadas: {data['samples_processed']}")
            
    elif response.status_code == 404:
        print(f"\n    ⚠️  Endpoint no encontrado")
        print(f"    Respuesta: {response.text}")
        print(f"\n    Posibles causas:")
        print(f"    - La aplicación FastAPI no se registró correctamente")
        print(f"    - El servidor ejecuta pero los endpoints no están disponibles")
        
    else:
        print(f"\n    ❌ Error inesperado")
        print(f"    Respuesta: {response.text[:500]}")
        
except requests.exceptions.ConnectionError:
    print(f"    ❌ No se puede conectar al servidor")
except requests.exceptions.Timeout:
    print(f"    ❌ Timeout (servidor tarda >30s)")
except Exception as e:
    print(f"    ❌ Error: {e}")

# ============================================================================
# TEST 3: Probar endpoint /predict_onnx
# ============================================================================

print("\n[3] Probando endpoint /predict_onnx...")

predict_payload = {
    "samples": [
        {
            "input_data": np.random.randn(seq_length * input_dim).tolist()
        }
    ]
}

try:
    response = requests.post(
        f"{NGROK_URL}/predict_onnx",
        json=predict_payload,
        timeout=TIMEOUT,
        verify=False
    )
    
    if response.status_code == 200:
        print(f"    ✅ Predicción exitosa")
        data = response.json()
        for key in ["status", "message", "reconstruction", "anomaly_scores"]:
            if key in data:
                if isinstance(data[key], list):
                    print(f"    • {key}: [{len(data[key])} items]")
                else:
                    print(f"    • {key}: {data[key]}")
    else:
        print(f"    ⚠️  Status {response.status_code}")
        
except Exception as e:
    print(f"    ⚠️  Error: {type(e).__name__}")

# ============================================================================
# TEST 4: Exploración de Endpoints Disponibles
# ============================================================================

print("\n[4] Descubriendo endpoints disponibles...")

potential_endpoints = [
    ("/", "GET"),
    ("/docs", "GET"),
    ("/openapi.json", "GET"),
    ("/health", "GET"),
    ("/status", "GET"),
    ("/info", "GET"),
    ("/train_layer2", "POST"),
    ("/predict", "POST"),
    ("/predict_onnx", "POST"),
]

print("\n    Endpoints encontrados:")

for path, method in potential_endpoints:
    try:
        if method == "GET":
            resp = requests.get(f"{NGROK_URL}{path}", timeout=5, verify=False)
        else:
            resp = requests.post(f"{NGROK_URL}{path}", json={}, timeout=5, verify=False)
        
        if resp.status_code < 400:
            print(f"    ✅ {method:4} {path:30} [{resp.status_code}]")
        else:
            print(f"    ⚠️  {method:4} {path:30} [{resp.status_code}]")
    except:
        pass

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("📊 RESUMEN DE DIAGNÓSTICO")
print("="*80)

print("""
✅ VERIFICACIONES COMPLETADAS:

1. Conexión al túnel ngrok: EXITOSA
2. Endpoint /train_layer2: Probado
3. Endpoint /predict_onnx: Probado
4. Descubrimiento de endpoints: Completado

📋 PRÓXIMOS PASOS:

1. Si todos los tests pasaron:
   - El servidor está LISTO para entrenamiento
   - Procede a enviar datos masivos

2. Si hay errores 404:
   - Los endpoints no están registrados en FastAPI
   - Revisa que la aplicación FastAPI incluya @app.post("/train_layer2")
   - Verifica que la instancia correcta de FastAPI esté ejecutándose

3. Si hay errores de conexión:
   - Colab no está ejecutando o se desconectó
   - Reinicia el servidor en Colab
   - Obtén la nueva URL de ngrok

🔗 INFORMACIÓN:
   • URL Servidor: {NGROK_URL}
   • Modelo Capa 2: HybridCognitiveLayer2
   • Input: (batch, 100, 20)
   • Output: (batch, 100, 20) + (batch, 100, 1)
   • Parámetros: ~27.9M
   • Framework: PyTorch + FastAPI + ONNX Runtime

📅 Prueba: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".format(NGROK_URL=NGROK_URL))

print("="*80)
