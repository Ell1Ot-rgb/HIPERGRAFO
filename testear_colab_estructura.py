#!/usr/bin/env python3
"""
Script para testear la estructura del servidor Colab
y generar un informe profundo de su arquitectura
"""

import json
from datetime import datetime
from typing import Dict, List, Any
import time
import urllib.request
import urllib.error

class TestadorColabServidor:
    def __init__(self, url: str = "https://paleographic-transonic-adell.ngrok-free.dev"):
        self.url = url
        self.resultados = {}
        self.errores = []
        self.timestamp = datetime.now()
    
    def hacer_request(self, metodo: str, endpoint: str, payload: Dict = None) -> Dict[str, Any]:
        """Realiza request HTTP sin librerías externas"""
        try:
            url_completa = f"{self.url}{endpoint}"
            
            if metodo == "GET":
                start = time.time()
                with urllib.request.urlopen(url_completa, timeout=10) as response:
                    latencia = (time.time() - start) * 1000
                    data = json.loads(response.read().decode())
                    return {"status": 200, "data": data, "latencia": latencia}
            
            elif metodo == "POST":
                headers = {"Content-Type": "application/json"}
                body = json.dumps(payload).encode('utf-8') if payload else b''
                req = urllib.request.Request(url_completa, data=body, headers=headers, method="POST")
                
                start = time.time()
                with urllib.request.urlopen(req, timeout=15) as response:
                    latencia = (time.time() - start) * 1000
                    data = json.loads(response.read().decode())
                    return {"status": 200, "data": data, "latencia": latencia}
        
        except urllib.error.HTTPError as e:
            return {"status": e.code, "error": str(e), "latencia": 0}
        except urllib.error.URLError as e:
            return {"status": None, "error": str(e), "latencia": 0}
        except Exception as e:
            return {"status": None, "error": str(e), "latencia": 0}
        
    def test_conectividad(self) -> Dict[str, Any]:
        """Prueba conectividad básica"""
        print("\n" + "="*80)
        print("1️⃣ PRUEBA DE CONECTIVIDAD")
        print("="*80)
        
        resultado = {
            "servidor_activo": False,
            "latencia_ms": None,
            "version": None
        }
        
        resp = self.hacer_request("GET", "/health")
        
        if resp["status"] == 200:
            resultado["servidor_activo"] = True
            resultado["latencia_ms"] = resp.get("latencia", 0)
            print(f"✅ Servidor activo (latencia: {resp.get('latencia', 0):.2f}ms)")
            print(f"✅ Respuesta válida: {resp.get('data', {})}")
        else:
            print(f"❌ Error: {resp.get('error', 'Unknown error')}")
            self.errores.append(f"Health check falló: {resp.get('error')}")
        
        self.resultados["conectividad"] = resultado
        return resultado
    
    def test_endpoints(self) -> Dict[str, Any]:
        """Prueba todos los endpoints disponibles"""
        print("\n" + "="*80)
        print("2️⃣ PRUEBA DE ENDPOINTS")
        print("="*80)
        
        endpoints_get = [
            ("/health", "Health check"),
            ("/status", "Estado del servidor"),
            ("/info", "Información arquitectónica")
        ]
        
        endpoints_post = [
            ("/diagnostico", "Diagnóstico del sistema", {}),
            ("/train_layer2", "Entrenamiento de Capas 2-5", {
                "samples": [
                    {"input_data": [0.1] * 1600, "anomaly_label": 0},
                    {"input_data": [0.5] * 1600, "anomaly_label": 1}
                ],
                "epochs": 1
            })
        ]
        
        resultado = {"get": {}, "post": {}}
        
        # Probar GET
        print("\n📍 GET ENDPOINTS:")
        for endpoint, descripcion in endpoints_get:
            resp = self.hacer_request("GET", endpoint)
            if resp["status"] == 200:
                print(f"✅ {endpoint:20} ({resp['status']}) - {descripcion}")
                resultado["get"][endpoint] = resp.get('data', {})
            else:
                print(f"⚠️ {endpoint:20} ({resp['status']}) - {descripcion}")
                resultado["get"][endpoint] = {"error": resp.get('error', 'Unknown')}
                self.errores.append(f"{endpoint}: {resp.get('error')}")
        
        # Probar POST
        print("\n📍 POST ENDPOINTS:")
        for endpoint, descripcion, payload in endpoints_post:
            resp = self.hacer_request("POST", endpoint, payload)
            if resp["status"] == 200:
                print(f"✅ {endpoint:20} ({resp['status']}) - {descripcion}")
                resultado["post"][endpoint] = resp.get('data', {})
            else:
                print(f"⚠️ {endpoint:20} ({resp['status']}) - {descripcion}")
                resultado["post"][endpoint] = {"error": resp.get('error', 'Unknown')}
                self.errores.append(f"{endpoint}: {resp.get('error')}")
        
        self.resultados["endpoints"] = resultado
        return resultado
    
    def test_arquitectura(self) -> Dict[str, Any]:
        """Analiza la arquitectura del modelo"""
        print("\n" + "="*80)
        print("3️⃣ ANÁLISIS DE ARQUITECTURA")
        print("="*80)
        
        resultado = {
            "modelo": None,
            "capas": {},
            "entrenamiento": {},
            "flujo_datos": {}
        }
        
        # Obtener info del servidor
        resp = self.hacer_request("GET", "/info")
        if resp["status"] == 200:
            info = resp.get("data", {})
            
            # Extraer información de arquitectura
            if "arquitectura" in info:
                arch = info["arquitectura"]
                
                print("\n🏗️ CAPAS DEL MODELO:")
                if "capas" in arch:
                    for capa_name, capa_config in arch["capas"].items():
                        print(f"\n  {capa_name.upper()}:")
                        print(f"    • Nombre: {capa_config.get('nombre', 'N/A')}")
                        print(f"    • Tipo: {capa_config.get('tipo', 'N/A')}")
                        if 'input_dim' in capa_config:
                            print(f"    • Input: {capa_config['input_dim']}D")
                        if 'output_dim' in capa_config:
                            print(f"    • Output: {capa_config['output_dim']}D")
                        
                        resultado["capas"][capa_name] = capa_config
                
                print("\n⚙️ FUSIÓN (GMU):")
                if "fusion" in arch:
                    fusion = arch["fusion"]
                    print(f"    • Nombre: {fusion.get('nombre', 'N/A')}")
                    print(f"    • Tipo: {fusion.get('tipo', 'N/A')}")
                    print(f"    • Inputs: {fusion.get('fusion_inputs', [])}")
                    resultado["fusion"] = fusion
                
                print("\n📚 ENTRENAMIENTO:")
                if "entrenamiento" in arch:
                    training = arch["entrenamiento"]
                    print(f"    • Optimizador: {training.get('optimizador', 'N/A')}")
                    print(f"    • Learning Rate: {training.get('lr', 'N/A')}")
                    print(f"    • Criterio: {training.get('criterio_perdida', 'N/A')}")
                    print(f"    • Dispositivo: {training.get('dispositivo', 'N/A')}")
                    resultado["entrenamiento"] = training
                
                print("\n📊 FLUJO DE DATOS:")
                if "flujo_datos" in arch:
                    flujo = arch["flujo_datos"]
                    print(f"    • Entrada: {flujo.get('entrada', 'N/A')}")
                    print(f"    • Procesamiento: {flujo.get('procesamiento', 'N/A')}")
                    if "salida" in flujo:
                        print(f"    • Salida:")
                        for output_name, output_type in flujo["salida"].items():
                            print(f"      - {output_name}: {output_type}")
                    resultado["flujo_datos"] = flujo
        else:
            print(f"❌ Error obteniendo arquitectura: {resp.get('error')}")
            self.errores.append(f"Error en análisis arquitectónico: {resp.get('error')}")
        
        self.resultados["arquitectura"] = resultado
        return resultado
    
    def test_estadisticas(self) -> Dict[str, Any]:
        """Obtiene estadísticas del servidor"""
        print("\n" + "="*80)
        print("4️⃣ ESTADÍSTICAS DEL SERVIDOR")
        print("="*80)
        
        resultado = {}
        
        resp = self.hacer_request("GET", "/status")
        if resp["status"] == 200:
            status = resp.get("data", {})
            
            if "estadisticas" in status:
                stats = status["estadisticas"]
                print(f"\n📈 MÉTRICAS DE ENTRENAMIENTO:")
                print(f"    • Total muestras: {stats.get('total_muestras', 0):,}")
                print(f"    • Total batches: {stats.get('total_batches', 0)}")
                print(f"    • Loss promedio: {stats.get('loss_promedio', 0):.6f}")
                print(f"    • Tiempo transcurrido: {stats.get('tiempo_transcurrido_segundos', 0):.1f}s")
                print(f"    • Dispositivo: {stats.get('dispositivo', 'N/A')}")
                print(f"    • PyTorch version: {stats.get('version_pytorch', 'N/A')}")
                
                resultado = stats
            
            if "capacidad" in status:
                cap = status["capacidad"]
                print(f"\n⚡ CAPACIDAD DEL MODELO:")
                print(f"    • Capas: {cap.get('capas', 'N/A')}")
                print(f"    • Input dim: {cap.get('input_dim', 'N/A')}")
                print(f"    • Hidden dim: {cap.get('hidden_dim', 'N/A')}")
                print(f"    • Output anomalía: {cap.get('output_anomaly', 'N/A')}")
                print(f"    • Output dendritas: {cap.get('output_dendrites', 'N/A')}")
                print(f"    • Parámetros entrenables: {cap.get('parametros_entrenables', 0):,}")
        else:
            print(f"❌ Error obteniendo estadísticas: {resp.get('error')}")
            self.errores.append(f"Error en estadísticas: {resp.get('error')}")
        
        self.resultados["estadisticas"] = resultado
        return resultado
    
    def generar_informe(self) -> str:
        """Genera informe completo en markdown"""
        print("\n" + "="*80)
        print("📋 GENERANDO INFORME COMPLETO")
        print("="*80 + "\n")
        
        informe = f"""# 📊 INFORME DETALLADO DEL SERVIDOR COLAB
## OMEGA 21 - Corteza Cognitiva Distribuida

**Fecha de análisis:** {self.timestamp.isoformat()}
**URL del servidor:** {self.url}

---

## 1️⃣ ESTADO DE CONECTIVIDAD

"""
        
        # Conectividad
        if "conectividad" in self.resultados:
            conn = self.resultados["conectividad"]
            informe += f"""
### Estado General
- **Servidor activo:** {'✅ SÍ' if conn.get('servidor_activo') else '❌ NO'}
- **Latencia:** {conn.get('latencia_ms', 'N/A'):.2f}ms

"""
        
        # Endpoints
        if "endpoints" in self.resultados:
            endpoints = self.resultados["endpoints"]
            informe += """### Endpoints Disponibles

#### GET Endpoints
"""
            for endpoint, response in endpoints.get("get", {}).items():
                status = "✅ Funcional" if "error" not in response else "❌ Error"
                informe += f"- `{endpoint}` - {status}\n"
            
            informe += "\n#### POST Endpoints\n"
            for endpoint, response in endpoints.get("post", {}).items():
                status = "✅ Funcional" if "error" not in response else "❌ Error"
                informe += f"- `{endpoint}` - {status}\n"
        
        # Arquitectura
        if "arquitectura" in self.resultados:
            arch = self.resultados["arquitectura"]
            informe += """

---

## 2️⃣ ARQUITECTURA DEL MODELO

### Estructura de Capas

"""
            if "capas" in arch:
                for capa_name, capa_config in arch["capas"].items():
                    informe += f"""
#### {capa_config.get('nombre', capa_name)}
- **Identificador:** `{capa_name}`
- **Tipo:** {capa_config.get('tipo', 'N/A')}
"""
                    if 'input_dim' in capa_config:
                        informe += f"- **Input dimension:** {capa_config['input_dim']}D\n"
                    if 'hidden_dim' in capa_config:
                        informe += f"- **Hidden dimension:** {capa_config['hidden_dim']}D\n"
                    if 'output_dim' in capa_config:
                        informe += f"- **Output dimension:** {capa_config['output_dim']}D\n"
                    if 'num_heads' in capa_config:
                        informe += f"- **Attention heads:** {capa_config['num_heads']}\n"
                    if 'num_layers' in capa_config:
                        informe += f"- **Num layers:** {capa_config['num_layers']}\n"
            
            # Fusión
            if "fusion" in arch:
                fusion = arch["fusion"]
                informe += f"""

### Mecanismo de Fusión (GMU)
- **Nombre:** {fusion.get('nombre', 'N/A')}
- **Tipo:** {fusion.get('tipo', 'N/A')}
- **Inputs fusionados:** {', '.join(fusion.get('fusion_inputs', []))}
- **Activación:** {fusion.get('activation', 'N/A')}

"""
        
        # Estadísticas
        if "estadisticas" in self.resultados:
            stats = self.resultados["estadisticas"]
            informe += f"""

---

## 3️⃣ ESTADÍSTICAS DE ENTRENAMIENTO

- **Total muestras entrenadas:** {stats.get('total_muestras', 0):,}
- **Total batches procesados:** {stats.get('total_batches', 0)}
- **Loss promedio:** {stats.get('loss_promedio', 0):.6f}
- **Tiempo de ejecución:** {stats.get('tiempo_transcurrido_segundos', 0):.1f} segundos
- **Dispositivo:** {stats.get('dispositivo', 'N/A')}
- **PyTorch version:** {stats.get('version_pytorch', 'N/A')}

"""
        
        # Resumen de validación
        informe += f"""

---

## 4️⃣ RESUMEN DE VALIDACIÓN

### ✅ Componentes Operacionales
"""
        
        if self.resultados.get("endpoints", {}).get("get", {}).get("/status", {}).get("error") is None:
            informe += "- ✅ Estado del servidor\n"
        if self.resultados.get("endpoints", {}).get("get", {}).get("/info", {}).get("error") is None:
            informe += "- ✅ Información arquitectónica\n"
        if self.resultados.get("endpoints", {}).get("post", {}).get("/diagnostico", {}).get("error") is None:
            informe += "- ✅ Diagnóstico del sistema\n"
        if self.resultados.get("endpoints", {}).get("post", {}).get("/train_layer2", {}).get("error") is None:
            informe += "- ✅ Entrenamiento de Capas 2-5\n"
        
        if self.errores:
            informe += """

### ⚠️ Problemas Detectados
"""
            for error in self.errores:
                informe += f"- ⚠️ {error}\n"
        
        informe += f"""

---

## 5️⃣ RECOMENDACIONES

### Para Producción
1. **Verificar conectividad permanente** de ngrok
2. **Implementar logging** de todas las transacciones de entrenamiento
3. **Monitorear GPU** en tiempo real durante entrenamiento
4. **Establecer checkpoints** periódicos del modelo

### Para Optimización
1. **Batch size:** Considerar aumentar a 128 o 256
2. **Learning rate:** Ajustar según convergencia observada
3. **Capas adicionales:** Considerar expansión si es necesario

---

**Informe generado por:** Analizador de Estructura Colab
**Versión del análisis:** 1.0
"""
        
        return informe
    
    def ejecutar_pruebas_completas(self):
        """Ejecuta todas las pruebas"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + "  🔍 ANÁLISIS COMPLETO DEL SERVIDOR COLAB".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        self.test_conectividad()
        self.test_endpoints()
        self.test_arquitectura()
        self.test_estadisticas()
        
        informe = self.generar_informe()
        
        print(informe)
        
        # Guardar informe
        with open("/workspaces/HIPERGRAFO/INFORME_COLAB_ESTRUCTURA.md", "w") as f:
            f.write(informe)
        
        print("\n" + "="*80)
        print("✅ Informe guardado en: INFORME_COLAB_ESTRUCTURA.md")
        print("="*80)
        
        return informe


if __name__ == "__main__":
    testador = TestadorColabServidor()
    testador.ejecutar_pruebas_completas()
