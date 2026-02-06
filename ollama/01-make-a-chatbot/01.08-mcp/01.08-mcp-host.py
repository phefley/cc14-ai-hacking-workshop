from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

# MCP Client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import json
import os
import asyncio
import nest_asyncio

nest_asyncio.apply()

BIG_MODEL_NAME = 'llama3:8b'
SMALL_MODEL_NAME = 'llama3.2:1b'

VECTORSTORE_DB_PATH = "./chroma_db"

MODEL_NAME = BIG_MODEL_NAME

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

    def get_tools_description(self) -> str:
        """Return a formatted string describing all available tools for the LLM prompt."""
        if not self.tools:
            return "No external tools are available."

        descriptions = []
        for tool_key, info in self.tools.items():
            params = json.dumps(info["input_schema"].get("properties", {}), indent=2)
            required = info["input_schema"].get("required", [])
            descriptions.append(
                f"- Tool: {tool_key}\n"
                f"  Description: {info['description']}\n"
                f"  Parameters: {params}\n"
                f"  Required: {required}"
            )
        return "\n".join(descriptions)

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

    def get_resources_description(self) -> str:
        """Return a formatted string describing all available resources."""
        if not self.resources:
            return "No external resources are available."
        lines = []
        for key, info in self.resources.items():
            lines.append(
                f"- Resource: {key}\n"
                f"  URI: {info['uri']}\n"
                f"  Description: {info['description']}"
            )
        return "\n".join(lines)

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

    def get_prompts_description(self) -> str:
        """Return a formatted string describing all available prompts."""
        if not self.prompts:
            return "No external prompts are available."
        lines = []
        for key, info in self.prompts.items():
            args_str = ", ".join(
                f"{a['name']} ({'required' if a.get('required') else 'optional'})"
                for a in info["arguments"]
            ) or "none"
            lines.append(
                f"- Prompt: {key}\n"
                f"  Description: {info['description']}\n"
                f"  Arguments: {args_str}"
            )
        return "\n".join(lines)

    async def cleanup(self):
        """Disconnect from all servers."""
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.tools.clear()
        self.resources.clear()
        self.prompts.clear()
        print("[+] MCP connections closed.")

# Global MCP client manager instance
mcp_manager = MCPClientManager()

# ============ RAG Setup (one-time) ============
def setup_vectorstore():
    """Run this once to create your vector database"""
    print("[+] Setting up vector store...")
    loader = DirectoryLoader('./docs', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    for doc in documents:
        print(f"\t* Loaded document: {doc.metadata['source']} (length: {len(doc.page_content)})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DB_PATH
    )
    print(f"[+] Persisting vector store to {VECTORSTORE_DB_PATH} ...")
    return vectorstore

# ============ LangGraph RAG with Memory ============

class AgentState(TypedDict):
    """State schema for our RAG agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str
    tool_output: str

# Initialize components (do this at module level or in FastAPI startup)
#embeddings = OllamaEmbeddings(model="nomic-embed-text")
#vectorstore = Chroma(
#    persist_directory=VECTORSTORE_DB_PATH,
#    embedding_function=embeddings
#)
#llm = OllamaLLM(model=MODEL_NAME)

def retrieve_context(state: AgentState):
    """Retrieve relevant documents based on the latest user message"""
    # Get the last user message
    last_message = state["messages"][-1].content
    
    # Search vector store
    docs = vectorstore.similarity_search(last_message, k=3)
    
    # Combine documents into context
    context = "\n\n".join([
        f"Document {i+1}:\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])
    
    return {"context": context}

def check_and_call_tools(state: AgentState):
    """Check if any MCP tools, resources, or prompts should be used based on the user's question."""
    tool_output = ""

    has_anything = mcp_manager.tools or mcp_manager.resources or mcp_manager.prompts
    if not has_anything:
        return {"tool_output": tool_output}

    last_message = state["messages"][-1].content
    tools_description = mcp_manager.get_tools_description()
    resources_description = mcp_manager.get_resources_description()
    prompts_description = mcp_manager.get_prompts_description()

    # Ask the LLM which MCP primitive (if any) should be used
    decision_prompt = f"""You are a decision maker. Based on the user's question, decide if any of the available MCP tools, resources, or prompts should be used.

Available tools (actions you can call):
{tools_description}

Available resources (static data you can read):
{resources_description}

Available prompts (prompt templates you can render):
{prompts_description}

User's question: {last_message}

If a tool should be called, respond with EXACTLY this format (no other text):
CALL_TOOL: <tool_key>
ARGUMENTS: {{"param_name": "value"}}

If a resource should be read, respond with EXACTLY:
READ_RESOURCE: <resource_key>

If a prompt should be rendered, respond with EXACTLY:
GET_PROMPT: <prompt_key>
ARGUMENTS: {{"param_name": "value"}}

If nothing is needed, respond with EXACTLY:
NO_TOOL

Your decision:"""

    decision = llm.invoke(decision_prompt).strip()

    loop = asyncio.get_event_loop()

    if decision.startswith("CALL_TOOL:"):
        try:
            lines = decision.split("\n")
            tool_key = lines[0].replace("CALL_TOOL:", "").strip()
            args_line = lines[1].replace("ARGUMENTS:", "").strip() if len(lines) > 1 else "{}"
            arguments = json.loads(args_line)

            tool_output = loop.run_until_complete(
                mcp_manager.call_tool(tool_key, arguments)
            )
            print(f"[+] Tool '{tool_key}' returned: {tool_output[:200]}...")
        except Exception as e:
            tool_output = f"Tool call failed: {e}"
            print(f"[!] Tool call error: {e}")

    elif decision.startswith("READ_RESOURCE:"):
        try:
            resource_key = decision.replace("READ_RESOURCE:", "").strip()
            tool_output = loop.run_until_complete(
                mcp_manager.read_resource(resource_key)
            )
            print(f"[+] Resource '{resource_key}' returned: {tool_output[:200]}...")
        except Exception as e:
            tool_output = f"Resource read failed: {e}"
            print(f"[!] Resource read error: {e}")

    elif decision.startswith("GET_PROMPT:"):
        try:
            lines = decision.split("\n")
            prompt_key = lines[0].replace("GET_PROMPT:", "").strip()
            args_line = lines[1].replace("ARGUMENTS:", "").strip() if len(lines) > 1 else "{}"
            arguments = json.loads(args_line)

            tool_output = loop.run_until_complete(
                mcp_manager.get_prompt(prompt_key, arguments)
            )
            print(f"[+] Prompt '{prompt_key}' returned: {tool_output[:200]}...")
        except Exception as e:
            tool_output = f"Prompt render failed: {e}"
            print(f"[!] Prompt render error: {e}")

    return {"tool_output": tool_output}

def generate_response(state: AgentState):
    """Generate response using LLM with retrieved context and conversation history"""
    context = state.get("context", "")
    tool_output = state.get("tool_output", "")
    messages = state["messages"]

    # Build conversation history
    conversation = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in messages[:-1]  # All except the last message
    ])

    # Get the current question
    current_question = messages[-1].content

    # Build prompt — prioritize tool output when present
    if tool_output:
        prompt = f"""You are a helpful assistant with access to external tools.
A tool was just called on your behalf and returned the following result.
This result is accurate and current. Do not question it or say it is from the future.
Use it directly to answer the user's question in a friendly, concise way.

Tool returned: {tool_output}

User's question: {current_question}

Answer:"""
    else:
        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question. If the answer cannot be found in the context, say so.

Context from knowledge base:
{context}

Previous conversation:
{conversation}

Current question: {current_question}

Answer:"""

    # Generate response
    response = llm.invoke(prompt)

    # Return as AIMessage
    return {"messages": [AIMessage(content=response)]}

# Conditional edge: skip RAG retrieval if a tool/resource/prompt returned data
def route_after_tools(state: AgentState):
    """If a tool produced output, skip retrieval and go straight to generate."""
    if state.get("tool_output"):
        return "generate"
    return "retrieve"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("check_tools", check_and_call_tools)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_response)

# Define flow: check_tools -> (retrieve OR generate) -> END
workflow.set_entry_point("check_tools")
workflow.add_conditional_edges("check_tools", route_after_tools, {
    "retrieve": "retrieve",
    "generate": "generate",
})
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ============ FastAPI Integration ============
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app):
    """Initialize vector store, LLM, and MCP connections on startup; clean up on shutdown."""
    global vectorstore, embeddings, llm

    # Load existing vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # Initialize the LLM
    llm = OllamaLLM(model=MODEL_NAME)

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
    """Chat endpoint with RAG, memory, and MCP tool support"""

    # Configure with session ID for memory persistence
    config = {
        "configurable": {
            "thread_id": request.session_id
        }
    }

    # Invoke the graph
    result = app.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config
    )

    # Extract response, context, and tool output
    response_text = result["messages"][-1].content
    context_used = result.get("context", "No context retrieved")
    tool_used = result.get("tool_output", "")

    return ChatResponse(
        response=response_text,
        context_used=context_used,
        tool_used=tool_used
    )

@api.post("/rebuild-index")
async def rebuild_index():
    """Endpoint to rebuild the vector store when documents change"""
    global vectorstore
    vectorstore = setup_vectorstore()
    return {"status": "Vector store rebuilt successfully"}

@api.delete("/clear-session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    # Note: MemorySaver keeps data in memory
    # For persistent storage, you'd use SqliteSaver or similar
    return {"status": f"Session {session_id} cleared (restart app to fully clear)"}

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
    
    # First time setup - uncomment to create vector store or comment out to speed it up after it's created
    setup_vectorstore()
    
    uvicorn.run(api, host="127.0.0.1", port=8000, timeout_keep_alive=300)