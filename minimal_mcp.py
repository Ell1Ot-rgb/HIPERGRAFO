from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Minimal")

@mcp.tool()
def hello() -> str:
    return "Hello from Minimal MCP"

if __name__ == "__main__":
    mcp.run()
