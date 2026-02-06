"""
MCP Host v3 — Clean rewrite focusing on correctness and simplicity.

Addresses issues found in both prior implementations:
  - Proper async throughout (no nest_asyncio, no sync-in-async hacks)
  - Uses Ollama native tool calling (no fragile text parsing)
  - Proper logging instead of print()
  - Errors raise exceptions / return proper HTTP status codes
  - load_config is sync (it does no async I/O)
  - Session memory has a max history length
  - for/else replaced with explicit flag for readability

Compatible with: mcp_config.json, 01.08-mcp-server.py, mcpchat-*.html, 01.08-tester.py
"""

import json
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
MAX_TOOL_LOOPS = 5
MAX_HISTORY = 50  # max message pairs kept per session

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to external tools via the "
    "Model Context Protocol (MCP). Call a tool only when the user's question "
    "requires real-time or external data. For general questions, respond using "
    "your own knowledge. Never paste raw tool names or JSON in your answers."
)

log = logging.getLogger("mcp-host")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


# ── MCP Client Manager ─────────────────────────────────────────────

class MCPManager:
    """Connects to MCP servers declared in a JSON config and discovers their primitives."""

    def __init__(self):
        self._stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, dict] = {}       # qualified_name -> metadata
        self.resources: dict[str, dict] = {}
        self.prompts: dict[str, dict] = {}

    # ── connection ──

    def load_config(self, path: str = "mcp_config.json") -> dict:
        """Load server definitions from a JSON file (pure I/O, not async)."""
        if not os.path.exists(path):
            log.warning("Config not found: %s", path)
            return {}
        with open(path) as f:
            return json.load(f).get("mcpServers", {})

    async def connect(self, name: str, cfg: dict):
        """Connect to one MCP server and discover its primitives."""
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=cfg.get("env", None),
        )
        transport = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(
            ClientSession(transport[0], transport[1])
        )
        await session.initialize()
        self.sessions[name] = session
        log.info("Connected to %s", name)

        # Discover tools
        for t in (await session.list_tools()).tools:
            key = f"{name}.{t.name}"
            self.tools[key] = {
                "server": name,
                "name": t.name,
                "description": t.description or "",
                "schema": t.inputSchema,
            }
            log.info("  tool: %s", key)

        # Discover resources (optional capability)
        try:
            for r in (await session.list_resources()).resources:
                key = f"{name}.{r.name}"
                self.resources[key] = {
                    "server": name,
                    "name": r.name,
                    "uri": str(r.uri),
                    "description": r.description or "",
                    "mime_type": getattr(r, "mimeType", "text/plain"),
                }
                log.info("  resource: %s", key)
        except Exception:
            log.debug("Server %s does not support resources", name)

        # Discover prompts (optional capability)
        try:
            for p in (await session.list_prompts()).prompts:
                key = f"{name}.{p.name}"
                self.prompts[key] = {
                    "server": name,
                    "name": p.name,
                    "description": p.description or "",
                    "arguments": [
                        {"name": a.name, "description": a.description or "", "required": a.required}
                        for a in (p.arguments or [])
                    ],
                }
                log.info("  prompt: %s", key)
        except Exception:
            log.debug("Server %s does not support prompts", name)

    async def connect_all(self, config_path: str = "mcp_config.json"):
        for name, cfg in self.load_config(config_path).items():
            try:
                await self.connect(name, cfg)
            except Exception as e:
                log.error("Failed to connect to %s: %s", name, e)
        log.info(
            "MCP ready: %d tool(s), %d resource(s), %d prompt(s)",
            len(self.tools), len(self.resources), len(self.prompts),
        )

    async def close(self):
        await self._stack.aclose()
        self.sessions.clear()
        self.tools.clear()
        self.resources.clear()
        self.prompts.clear()
        log.info("MCP connections closed")

    # ── tool calling ──

    async def call_tool(self, qualified_name: str, arguments: dict) -> str:
        """Call a tool and return its text output. Raises KeyError if not found."""
        info = self.tools.get(qualified_name)
        if info is None:
            raise KeyError(f"Unknown tool: {qualified_name}")
        session = self.sessions[info["server"]]
        result = await session.call_tool(info["name"], arguments)
        return "\n".join(
            block.text if hasattr(block, "text") else str(block)
            for block in result.content
        )

    async def read_resource(self, qualified_name: str) -> str:
        info = self.resources.get(qualified_name)
        if info is None:
            raise KeyError(f"Unknown resource: {qualified_name}")
        session = self.sessions[info["server"]]
        result = await session.read_resource(info["uri"])
        return "\n".join(
            block.text if hasattr(block, "text") else str(block)
            for block in result.contents
        )

    async def get_prompt(self, qualified_name: str, arguments: dict | None = None) -> str:
        info = self.prompts.get(qualified_name)
        if info is None:
            raise KeyError(f"Unknown prompt: {qualified_name}")
        session = self.sessions[info["server"]]
        result = await session.get_prompt(info["name"], arguments)
        return "\n".join(
            f"[{m.role}] {m.content.text if hasattr(m.content, 'text') else m.content}"
            for m in result.messages
        )

    # ── Ollama integration ──

    def ollama_tools(self) -> tuple[list[dict], dict[str, str]]:
        """Return (tools_for_ollama, safe_name_to_qualified_name_map)."""
        tools, mapping = [], {}
        for qname, info in self.tools.items():
            safe = qname.replace("-", "_").replace(".", "__")
            mapping[safe] = qname
            tools.append({
                "type": "function",
                "function": {
                    "name": safe,
                    "description": info["description"],
                    "parameters": info["schema"],
                },
            })
        return tools, mapping


# ── Session memory ──────────────────────────────────────────────────

sessions: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]


def trim_history(history: list[dict]):
    """Keep the most recent MAX_HISTORY messages."""
    over = len(history) - MAX_HISTORY
    if over > 0:
        del history[:over]


# ── Chat with tool calling ─────────────────────────────────────────

async def chat(
    client: ollama.AsyncClient,
    mcp: MCPManager,
    session_id: str,
    user_message: str,
) -> tuple[str, list[str]]:
    """
    Send a message through Ollama with MCP tool support.
    Returns (assistant_response, list_of_tool_descriptions).
    """
    history = get_history(session_id)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    tools_spec, name_map = mcp.ollama_tools()
    tool_log: list[str] = []

    # Agentic loop: model may request tools multiple times
    exhausted = False
    for _ in range(MAX_TOOL_LOOPS):
        response = await client.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=tools_spec or None,
        )
        msg = response.message

        if not msg.tool_calls:
            break  # model produced a final text answer

        messages.append(msg)

        for tc in msg.tool_calls:
            safe_name = tc.function.name
            args = tc.function.arguments or {}
            qname = name_map.get(safe_name, safe_name)
            log.info("Tool call: %s %s", qname, args)

            try:
                result = await mcp.call_tool(qname, args)
            except Exception as e:
                result = f"Tool error: {e}"
                log.error("Tool %s failed: %s", qname, e)

            tool_log.append(f"{qname}({json.dumps(args)}) -> {result[:200]}")
            messages.append({"role": "tool", "content": result})
    else:
        exhausted = True
        log.warning("Tool loop hit MAX_TOOL_LOOPS (%d)", MAX_TOOL_LOOPS)

    answer = response.message.content or ("" if not exhausted else "I was unable to complete the request within the allowed number of tool calls.")

    # Persist only the user message and final answer
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    trim_history(history)

    return answer, tool_log


# ── FastAPI ─────────────────────────────────────────────────────────

mcp_mgr = MCPManager()
ollama_client: ollama.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global ollama_client
    ollama_client = ollama.AsyncClient()
    await mcp_mgr.connect_all("mcp_config.json")
    yield
    await mcp_mgr.close()


app = FastAPI(title="MCP Host v3", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Chat endpoint ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    context_used: str   # kept for UI compatibility
    tool_used: str = ""

@app.post("/chatbot/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        answer, tool_log = await chat(ollama_client, mcp_mgr, req.session_id, req.message)
    except Exception as e:
        log.exception("Chat failed")
        raise HTTPException(status_code=502, detail=str(e))

    tool_summary = " | ".join(tool_log)
    return ChatResponse(
        response=answer,
        context_used=tool_summary or "No tools used",
        tool_used=tool_summary,
    )


# ── Session management ──────────────────────────────────────────────

@app.delete("/clear-session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": f"Session {session_id} cleared"}


# ── MCP introspection endpoints ─────────────────────────────────────

@app.get("/mcp/tools")
async def list_tools():
    return {"tools": [
        {"tool_key": k, "server_name": v["server"], "name": v["name"],
         "description": v["description"], "input_schema": v["schema"]}
        for k, v in mcp_mgr.tools.items()
    ], "server_count": len(mcp_mgr.sessions)}


class ToolCallRequest(BaseModel):
    tool_key: str
    arguments: dict = {}

@app.post("/mcp/call-tool")
async def call_tool_endpoint(req: ToolCallRequest):
    try:
        result = await mcp_mgr.call_tool(req.tool_key, req.arguments)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"tool_key": req.tool_key, "result": result}


@app.get("/mcp/resources")
async def list_resources():
    return {"resources": [
        {"resource_key": k, "server_name": v["server"], "name": v["name"],
         "uri": v["uri"], "description": v["description"], "mime_type": v["mime_type"]}
        for k, v in mcp_mgr.resources.items()
    ], "server_count": len(mcp_mgr.sessions)}


class ResourceReadRequest(BaseModel):
    resource_key: str

@app.post("/mcp/read-resource")
async def read_resource_endpoint(req: ResourceReadRequest):
    try:
        content = await mcp_mgr.read_resource(req.resource_key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"resource_key": req.resource_key, "content": content}


@app.get("/mcp/prompts")
async def list_prompts():
    return {"prompts": [
        {"prompt_key": k, "server_name": v["server"], "name": v["name"],
         "description": v["description"], "arguments": v["arguments"]}
        for k, v in mcp_mgr.prompts.items()
    ], "server_count": len(mcp_mgr.sessions)}


class PromptGetRequest(BaseModel):
    prompt_key: str
    arguments: dict[str, str] = {}

@app.post("/mcp/get-prompt")
async def get_prompt_endpoint(req: PromptGetRequest):
    try:
        rendered = await mcp_mgr.get_prompt(req.prompt_key, req.arguments)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"prompt_key": req.prompt_key, "rendered": rendered}


@app.post("/mcp/reconnect")
async def reconnect():
    await mcp_mgr.close()
    await mcp_mgr.connect_all("mcp_config.json")
    return {
        "status": "Reconnected",
        "servers": list(mcp_mgr.sessions.keys()),
        "tools": len(mcp_mgr.tools),
        "resources": len(mcp_mgr.resources),
        "prompts": len(mcp_mgr.prompts),
    }


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, timeout_keep_alive=300)
