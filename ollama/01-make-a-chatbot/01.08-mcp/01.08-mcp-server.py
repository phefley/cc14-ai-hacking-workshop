"""
MCP Server demonstrating all three MCP primitives:
  - Tool:     get_system_time
  - Resource: info://about
  - Prompt:   summarize_for_executive
"""

from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

# ============ Tool ============
@mcp.tool()
def get_system_time() -> str:
    """Return the current system date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============ Resource ============
@mcp.resource("info://about")
def about() -> str:
    """Describes what this MCP server does."""
    return (
        "This is a demo MCP server that exposes three primitives:\n"
        "  1. Tool  – get_system_time: returns the current system date/time.\n"
        "  2. Resource – info://about: this description of the server.\n"
        "  3. Prompt – summarize_for_executive: generates an executive-summary\n"
        "     prompt for a given topic."
    )


# ============ Prompt ============
@mcp.prompt()
def summarize_for_executive(topic: str) -> str:
    """Create an executive summary prompt for a given topic."""
    return (
        f"Summarize the following topic for a C-level executive.\n"
        f"Be concise (3-5 bullet points max), focus on business impact,\n"
        f"risks, and actionable recommendations.\n\n"
        f"Topic: {topic}"
    )


if __name__ == "__main__":
    mcp.run()
