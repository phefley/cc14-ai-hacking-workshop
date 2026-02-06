"""
MCP Host using Ollama's native tool-calling API.

This is a simplified alternative to 01.08-mcp-host.py that:
  - Uses Ollama's native tool-calling instead of text-parsing (CALL_TOOL:/NO_TOOL)
  - Removes all RAG/LangGraph/LangChain/ChromaDB dependencies
  - Keeps the same MCPClientManager for MCP server connections
  - Keeps the same FastAPI REST API and web UI compatibility

Inspired by the patterns in https://github.com/jonigl/mcp-client-for-ollama (ollmcp)
"""

import ollama

# MCP Client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack, asynccontextmanager
import json
import os
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

MODEL_NAME = 'llama3.1:8b'
MAX_TOOL_LOOPS = 5
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to external tools via the Model Context Protocol (MCP). "
    "You can call tools when the user's question requires real-time or external data. "
    "Only call a tool when it is genuinely needed to answer the question. "
    "For general questions, just respond normally using your own knowledge. "
    "Never paste raw tool names or JSON in your response text."
)

# ============ MCP Client Manager ============
class MCPClientManager:
    """Manages connections to multiple MCP servers configured via JSON."""

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}      # server_name -> session
        self.tools: dict[str, dict] = {}                   # tool_key -> {server_name, tool, ...}
        self.resources: dict[str, dict] = {}               # resource_key -> {server_name, uri, ...}
        self.prompts: dict[str, dict] = {}                 # prompt_key -> {server_name, prompt, ...}

    async def load_config(self, config_path: str = "mcp_config.json"):
        """Load MCP server configurations from a JSON config file."""
        if not os.path.exists(config_path):
            print(f"[!] MCP config file not found: {config_path}")
            return {}
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("mcpServers", {})

    async def connect_to_server(self, server_name: str, server_config: dict):
        """Connect to a single MCP server via STDIO transport."""
        try:
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env", None)
            )

            # Enter the stdio_client context manager via the exit stack
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport

            # Create and initialize the session
            session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()

            self.sessions[server_name] = session
            print(f"[+] Connected to MCP server: {server_name}")

            # Discover tools from this server
            response = await session.list_tools()
            for tool in response.tools:
                tool_key = f"{server_name}.{tool.name}"
                self.tools[tool_key] = {
                    "server_name": server_name,
                    "tool": tool,
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }
                print(f"    [+] Discovered tool: {tool_key} - {tool.description}")

            # Discover resources from this server
            try:
                res_response = await session.list_resources()
                for resource in res_response.resources:
                    res_key = f"{server_name}.{resource.name}"
                    self.resources[res_key] = {
                        "server_name": server_name,
                        "name": resource.name,
                        "uri": str(resource.uri),
                        "description": resource.description or "",
                        "mime_type": getattr(resource, "mimeType", "text/plain"),
                    }
                    print(f"    [+] Discovered resource: {res_key} ({resource.uri})")
            except Exception as e:
                print(f"    [!] Could not list resources from '{server_name}': {e}")

            # Discover prompts from this server
            try:
                prompt_response = await session.list_prompts()
                for prompt in prompt_response.prompts:
                    prompt_key = f"{server_name}.{prompt.name}"
                    self.prompts[prompt_key] = {
                        "server_name": server_name,
                        "name": prompt.name,
                        "description": prompt.description or "",
                        "arguments": [
                            {"name": a.name, "description": a.description or "", "required": a.required}
                            for a in (prompt.arguments or [])
                        ],
                    }
                    print(f"    [+] Discovered prompt: {prompt_key} - {prompt.description}")
            except Exception as e:
                print(f"    [!] Could not list prompts from '{server_name}': {e}")

        except Exception as e:
            print(f"[!] Failed to connect to MCP server '{server_name}': {e}")

    async def connect_all(self, config_path: str = "mcp_config.json"):
        """Load config and connect to all configured MCP servers."""
        servers = await self.load_config(config_path)
        for server_name, server_config in servers.items():
            await self.connect_to_server(server_name, server_config)
        print(f"[+] MCP setup complete. {len(self.tools)} tool(s), "
              f"{len(self.resources)} resource(s), {len(self.prompts)} prompt(s) available.")

    async def call_tool(self, tool_key: str, arguments: dict) -> str:
        """Call an MCP tool by its qualified name (server_name.tool_name)."""
        if tool_key not in self.tools:
            return f"Error: Tool '{tool_key}' not found."

        tool_info = self.tools[tool_key]
        server_name = tool_info["server_name"]
        session = self.sessions[server_name]

        try:
            result = await session.call_tool(tool_info["name"], arguments)
            # Extract text from content blocks
            output_parts = []
            for content_block in result.content:
                if hasattr(content_block, 'text'):
                    output_parts.append(content_block.text)
                else:
                    output_parts.append(str(content_block))
            return "\n".join(output_parts)
        except Exception as e:
            return f"Error calling tool '{tool_key}': {e}"

    def _safe_name(self, tool_key: str) -> str:
        """Convert a qualified tool key to a safe function name for Ollama.
        e.g. 'local-tools.get_system_time' -> 'local_tools__get_system_time'
        """
        return tool_key.replace("-", "_").replace(".", "__")

    def get_tools_for_ollama(self) -> tuple[list[dict], dict[str, str]]:
        """Convert MCP tools to Ollama's native tool-calling format.
        Returns (ollama_tools, name_map) where name_map maps safe_name -> tool_key.
        """
        ollama_tools = []
        name_map = {}  # safe_name -> original tool_key
        for tool_key, info in self.tools.items():
            safe_name = self._safe_name(tool_key)
            name_map[safe_name] = tool_key
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": safe_name,
                    "description": info["description"],
                    "parameters": info["input_schema"],
                }
            })
        return ollama_tools, name_map

    def get_tools_list(self) -> list[dict]:
        """Return tool metadata as a list of dicts for the API."""
        return [
            {
                "tool_key": key,
                "server_name": info["server_name"],
                "name": info["name"],
                "description": info["description"],
                "input_schema": info["input_schema"],
            }
            for key, info in self.tools.items()
        ]

    # ---------- Resource helpers ----------

    async def read_resource(self, resource_key: str) -> str:
        """Read an MCP resource by its qualified name (server_name.resource_name)."""
        if resource_key not in self.resources:
            return f"Error: Resource '{resource_key}' not found."

        info = self.resources[resource_key]
        session = self.sessions[info["server_name"]]

        try:
            result = await session.read_resource(info["uri"])
            parts = []
            for block in result.contents:
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        except Exception as e:
            return f"Error reading resource '{resource_key}': {e}"

    def get_resources_list(self) -> list[dict]:
        """Return resource metadata as a list of dicts for the API."""
        return [
            {
                "resource_key": key,
                "server_name": info["server_name"],
                "name": info["name"],
                "uri": info["uri"],
                "description": info["description"],
                "mime_type": info["mime_type"],
            }
            for key, info in self.resources.items()
        ]

    # ---------- Prompt helpers ----------

    async def get_prompt(self, prompt_key: str, arguments: dict[str, str] | None = None) -> str:
        """Render an MCP prompt by its qualified name (server_name.prompt_name)."""
        if prompt_key not in self.prompts:
            return f"Error: Prompt '{prompt_key}' not found."

        info = self.prompts[prompt_key]
        session = self.sessions[info["server_name"]]

        try:
            result = await session.get_prompt(info["name"], arguments)
            parts = []
            for msg in result.messages:
                role = msg.role
                text = msg.content.text if hasattr(msg.content, "text") else str(msg.content)
                parts.append(f"[{role}] {text}")
            return "\n".join(parts)
        except Exception as e:
            return f"Error rendering prompt '{prompt_key}': {e}"

    def get_prompts_list(self) -> list[dict]:
        """Return prompt metadata as a list of dicts for the API."""
        return [
            {
                "prompt_key": key,
                "server_name": info["server_name"],
                "name": info["name"],
                "description": info["description"],
                "arguments": info["arguments"],
            }
            for key, info in self.prompts.items()
        ]

    async def cleanup(self):
        """Disconnect from all servers."""
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.tools.clear()
        self.resources.clear()
        self.prompts.clear()
        print("[+] MCP connections closed.")


# Global instances
mcp_manager = MCPClientManager()
conversation_memory: dict[str, list[dict]] = {}
ollama_client: ollama.AsyncClient = None


# ============ Chat with Native Tool Calling ============

async def chat_with_tools(session_id: str, user_message: str) -> tuple[str, str]:
    """
    Process a chat message using Ollama's native tool-calling API.

    Returns (response_text, tool_info_string).
    """
    # Get or create session history
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    history = conversation_memory[session_id]

    # Build messages: system prompt + history + new user message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # Get available tools in Ollama format (with safe names)
    ollama_tools, tool_name_map = mcp_manager.get_tools_for_ollama()

    # Track tool usage for the response
    tools_used = []

    # Agentic tool-calling loop
    response = None
    for loop_idx in range(MAX_TOOL_LOOPS):
        try:
            response = await ollama_client.chat(
                model=MODEL_NAME,
                messages=messages,
                tools=ollama_tools if ollama_tools else None,
            )
        except Exception as e:
            error_msg = f"Error communicating with Ollama: {e}"
            print(f"[!] {error_msg}")
            return error_msg, ""

        assistant_message = response.message

        # Check if the model wants to call tools
        if not assistant_message.tool_calls:
            # No tool calls -- we have the final response
            break

        # Append the assistant's message (with tool_calls) to messages
        messages.append(assistant_message)

        # Execute each tool call
        for tool_call in assistant_message.tool_calls:
            safe_name = tool_call.function.name
            tool_args = tool_call.function.arguments or {}

            # Map safe name back to qualified MCP tool key
            tool_key = tool_name_map.get(safe_name, safe_name)
            print(f"[+] Model requested tool: {safe_name} -> {tool_key} with args: {tool_args}")

            result = await mcp_manager.call_tool(tool_key, tool_args)
            tools_used.append(f"{tool_key}({json.dumps(tool_args)}) -> {result[:200]}")

            # Append tool result to messages
            messages.append({
                "role": "tool",
                "content": str(result),
            })

        print(f"[+] Tool loop iteration {loop_idx + 1}/{MAX_TOOL_LOOPS} complete")
    else:
        print(f"[!] Reached max tool loop iterations ({MAX_TOOL_LOOPS})")

    # Extract final response
    final_text = response.message.content if response else "No response from model."

    # Update session memory (just the user message and final assistant response)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": final_text})

    # Return
    tool_info = " | ".join(tools_used) if tools_used else ""
    return final_text, tool_info


# ============ FastAPI Integration ============

@asynccontextmanager
async def lifespan(app):
    """Initialize Ollama client and MCP connections on startup; clean up on shutdown."""
    global ollama_client

    ollama_client = ollama.AsyncClient()

    # Connect to MCP servers
    await mcp_manager.connect_all("mcp_config.json")

    yield  # ---- app runs here ----

    # Shutdown
    await mcp_manager.cleanup()

api = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    context_used: str
    tool_used: str = ""

@api.post("/chatbot/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with native Ollama tool-calling and MCP integration."""

    response_text, tool_info = await chat_with_tools(
        request.session_id, request.message
    )

    return ChatResponse(
        response=response_text,
        context_used=tool_info if tool_info else "No tools used",
        tool_used=tool_info,
    )

@api.delete("/clear-session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in conversation_memory:
        del conversation_memory[session_id]
    return {"status": f"Session {session_id} cleared"}


# ============ MCP Tool Endpoints ============

@api.get("/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools across connected servers."""
    return {
        "tools": mcp_manager.get_tools_list(),
        "server_count": len(mcp_manager.sessions),
    }

class ToolCallRequest(BaseModel):
    tool_key: str       # e.g., "local-tools.my_function"
    arguments: dict = {}

class ToolCallResponse(BaseModel):
    tool_key: str
    result: str
    is_error: bool = False

@api.post("/mcp/call-tool", response_model=ToolCallResponse)
async def call_mcp_tool(request: ToolCallRequest):
    """Directly call an MCP tool by its qualified name."""
    result = await mcp_manager.call_tool(request.tool_key, request.arguments)
    is_error = result.startswith("Error")
    return ToolCallResponse(
        tool_key=request.tool_key,
        result=result,
        is_error=is_error,
    )

# ============ MCP Resource Endpoints ============

@api.get("/mcp/resources")
async def list_mcp_resources():
    """List all available MCP resources across connected servers."""
    return {
        "resources": mcp_manager.get_resources_list(),
        "server_count": len(mcp_manager.sessions),
    }

class ResourceReadRequest(BaseModel):
    resource_key: str   # e.g., "local-tools.about"

class ResourceReadResponse(BaseModel):
    resource_key: str
    content: str
    is_error: bool = False

@api.post("/mcp/read-resource", response_model=ResourceReadResponse)
async def read_mcp_resource(request: ResourceReadRequest):
    """Read an MCP resource by its qualified name."""
    content = await mcp_manager.read_resource(request.resource_key)
    is_error = content.startswith("Error")
    return ResourceReadResponse(
        resource_key=request.resource_key,
        content=content,
        is_error=is_error,
    )

# ============ MCP Prompt Endpoints ============

@api.get("/mcp/prompts")
async def list_mcp_prompts():
    """List all available MCP prompts across connected servers."""
    return {
        "prompts": mcp_manager.get_prompts_list(),
        "server_count": len(mcp_manager.sessions),
    }

class PromptGetRequest(BaseModel):
    prompt_key: str               # e.g., "local-tools.summarize_for_executive"
    arguments: dict[str, str] = {}

class PromptGetResponse(BaseModel):
    prompt_key: str
    rendered: str
    is_error: bool = False

@api.post("/mcp/get-prompt", response_model=PromptGetResponse)
async def get_mcp_prompt(request: PromptGetRequest):
    """Render an MCP prompt with the given arguments."""
    rendered = await mcp_manager.get_prompt(request.prompt_key, request.arguments)
    is_error = rendered.startswith("Error")
    return PromptGetResponse(
        prompt_key=request.prompt_key,
        rendered=rendered,
        is_error=is_error,
    )

# ============ MCP Reconnect ============

@api.post("/mcp/reconnect")
async def reconnect_mcp():
    """Reconnect to all MCP servers (useful after config changes)."""
    await mcp_manager.cleanup()
    await mcp_manager.connect_all("mcp_config.json")
    return {
        "status": "Reconnected",
        "servers": list(mcp_manager.sessions.keys()),
        "tools": len(mcp_manager.tools),
        "resources": len(mcp_manager.resources),
        "prompts": len(mcp_manager.prompts),
    }

# ============ Main ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="127.0.0.1", port=8000, timeout_keep_alive=300)
