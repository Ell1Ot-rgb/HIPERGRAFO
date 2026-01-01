import sys
import json
import logging

# Configure logging to stderr so it doesn't corrupt stdout
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

def main():
    logging.info("Starting Manual MCP Server")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            logging.info(f"Received: {line.strip()}")
            req = json.loads(line)
            
            if req.get("method") == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {"name": "Manual", "version": "1.0"}
                    }
                }
                print(json.dumps(resp))
                sys.stdout.flush()
                logging.info("Sent initialize response")
                
            elif req.get("method") == "notifications/initialized":
                logging.info("Initialized notification received")
                
            elif req.get("method") == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "test_tool",
                                "description": "A test tool",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {}
                                }
                            }
                        ]
                    }
                }
                print(json.dumps(resp))
                sys.stdout.flush()
                logging.info("Sent tools/list response")
                
            elif req.get("method") == "ping":
                 resp = {
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": {}
                }
                 print(json.dumps(resp))
                 sys.stdout.flush()

        except Exception as e:
            logging.error(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
