import os
import sys
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete

# Configuración básica (puedes cambiar el modelo o usar variables de entorno)
WORKING_DIR = "./lightrag_index"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

def initialize_rag():
    # Inicializa LightRAG con GPT-4o-mini por defecto (más barato/rápido)
    # Asegúrate de tener OPENAI_API_KEY en tus variables de entorno
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete
    )
    return rag

def main():
    if len(sys.argv) < 2:
        print("Uso: python lightrag_wrapper.py [index|query] [texto]")
        return

    command = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else ""

    rag = initialize_rag()

    if command == "index":
        print(f"Indexando texto: {text[:50]}...")
        rag.insert(text)
        print("Indexado completado.")
    
    elif command == "query":
        print(f"Consultando: {text}")
        # Usamos búsqueda híbrida por defecto
        response = rag.query(text, param=QueryParam(mode="hybrid"))
        print("Respuesta:")
        print(response)

if __name__ == "__main__":
    main()
