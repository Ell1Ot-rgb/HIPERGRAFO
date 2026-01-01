import sys
import json
import logging
import os

# Configure logging to file
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'simple_server.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s'
)

def log(msg):
    logging.debug(msg)

def send_json(data):
    try:
        json_str = json.dumps(data)
        log(f"Sending: {json_str}")
        sys.stdout.write(json_str + "\n")
        sys.stdout.flush()
    except Exception as e:
        log(f"Error sending JSON: {e}")

def main():
    log("Server started")
    
    # Set stdin/stdout to binary mode to avoid encoding issues, then wrap in UTF-8
    # This is the most robust way on Windows
    if sys.platform == "win32":
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', newline='\n')

    while True:
        try:
            line = stdin.readline()
            if not line:
                log("Stdin closed, exiting")
                break
                
            log(f"Received: {line.strip()}")
            
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                log("Invalid JSON received")
                continue

            method = request.get("method")
            msg_id = request.get("id")

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "LightRAG-Simple",
                            "version": "1.0"
                        }
                    }
                }
                send_json(response)

            elif method == "notifications/initialized":
                log("Initialized notification")

            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "tools": [{
                            "name": "ask_lightrag",
                            "description": "Query LightRAG",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }]
                    }
                }
                send_json(response)

            elif method == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name")
                args = params.get("arguments", {})
                
                if tool_name == "ask_lightrag":
                    query = args.get("query")
                    log(f"Executing query: {query}")
                    
                    try:
                        # Call the external client script
                        client_script = os.path.join(os.path.dirname(__file__), "lightrag_api_client.py")
                        import subprocess
                        
                        # Run the client script and capture output
                        # Force UTF-8 encoding for the subprocess
                        result = subprocess.run(
                            [sys.executable, client_script, "query", query],
                            capture_output=True,
                            text=True,
                            encoding='utf-8',
                            errors='replace'
                        )
                        
                        if result.returncode == 0:
                            output_text = result.stdout.strip()
                            log("Query successful")
                        else:
                            output_text = f"Error: {result.stderr}"
                            log(f"Query failed: {result.stderr}")
                            
                        response = {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": {
                                "content": [{
                                    "type": "text",
                                    "text": output_text
                                }]
                            }
                        }
                    except Exception as e:
                        log(f"Execution error: {e}")
                        response = {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "error": {
                                "code": -32603,
                                "message": str(e)
                            }
                        }
                    send_json(response)
                else:
                    # Unknown tool
                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32601,
                            "message": f"Tool not found: {tool_name}"
                        }
                    }
                    send_json(response)
            
            elif method == "ping":
                response = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
                send_json(response)

        except Exception as e:
            log(f"Loop error: {e}")
            break

import io
if __name__ == "__main__":
    main()
