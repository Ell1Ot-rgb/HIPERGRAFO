import sys
import requests
import json

# Configuración
LIGHTRAG_HOST = "http://192.168.1.37:9621"

def query_rag(text):
    """
    Consulta el servicio dasein_api usando Cypher.
    Para consultas de texto natural, el texto se puede usar como parte de una consulta Cypher.
    """
    # Nueva API: GET /query?cypher=...
    import urllib.parse
    
    # Construir consulta Cypher - buscar nodos que contengan el texto
    # Puedes personalizar esta consulta según tus necesidades
    cypher_query = f"MATCH (n) WHERE n.text CONTAINS '{text}' OR n.name CONTAINS '{text}' RETURN n LIMIT 10"
    
    url = f"{LIGHTRAG_HOST}/query?cypher={urllib.parse.quote(cypher_query)}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except requests.exceptions.RequestException as e:
        error_msg = f"Error al consultar: {e}"
        print(error_msg)
        return {"error": error_msg}

# --- Función original (LightRAG puro) - comentada para referencia ---
# def query_rag_original(text):
#     url = f"{LIGHTRAG_HOST}/query"
#     payload = {
#         "query": text,
#         "param": {
#             "mode": "hybrid"  # Puedes cambiar a 'local', 'global', 'naive'
#         }
#     }
#     try:
#         response = requests.post(url, json=payload)
#         response.raise_for_status()
#         print(response.text) 
#     except requests.exceptions.RequestException as e:
#         print(f"Error al consultar LightRAG: {e}")

def index_text(text):
    url = f"{LIGHTRAG_HOST}/insert"
    # Ajustar payload según la API específica de LightRAG (a veces es {"text": ...} o una lista)
    payload = {
        "text": text
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Texto indexado correctamente.")
        print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error al indexar en LightRAG: {e}")

def main():
    if len(sys.argv) < 3:
        print("Uso: python lightrag_api_client.py [query|index] [texto]")
        return

    command = sys.argv[1]
    text = sys.argv[2]

    if command == "query":
        query_rag(text)
    elif command == "index":
        index_text(text)
    else:
        print(f"Comando desconocido: {command}")

if __name__ == "__main__":
    main()
