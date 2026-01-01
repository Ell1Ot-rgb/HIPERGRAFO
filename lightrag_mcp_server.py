import sys
import os
import asyncio
import subprocess
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

# Configure logging to file to avoid stdout pollution
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'server_debug.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lightrag-server")

CLIENT_SCRIPT = os.path.join(os.path.dirname(__file__), "lightrag_api_client.py")

async def run():
    logger.info("Starting LightRAG MCP Server")
    
    server = Server("lightrag-server")

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        return [
            Tool(
                name="ask_lightrag",
                description="Consulta a la base de conocimiento de LightRAG (RAG Agent).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "La pregunta o consulta para el agente."
                        }
                    },
                    "required": ["query"]
                }
            )
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent | ImageContent | EmbeddedResource]:
        if name != "ask_lightrag":
            raise ValueError(f"Unknown tool: {name}")

        if not arguments or "query" not in arguments:
            raise ValueError("Missing 'query' argument")

        query = arguments["query"]
        logger.info(f"Executing query: {query}")

        try:
            # Execute the client script
            cmd = [sys.executable, CLIENT_SCRIPT, "query", query]
            # Use specific encoding for subprocess to avoid issues
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            output = stdout.decode('utf-8', errors='replace').strip()
            error_out = stderr.decode('utf-8', errors='replace').strip()

            if process.returncode != 0:
                logger.error(f"Script failed: {error_out}")
                return [TextContent(type="text", text=f"Error executing LightRAG: {error_out}")]

            logger.info("Query successful")
            return [TextContent(type="text", text=output)]

        except Exception as e:
            logger.exception("Internal error during tool execution")
            return [TextContent(type="text", text=f"Error interno: {str(e)}")]

    async with stdio_server() as (read_stream, write_stream):
        logger.info("STDIO server initialized, waiting for requests")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        logger.critical(f"Fatal error: {e}")

