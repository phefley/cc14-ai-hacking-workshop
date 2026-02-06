from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
#from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from typing import Literal
import operator
import sqlite3

BIG_MODEL_NAME = 'llama3:8b'
SMALL_MODEL_NAME = 'llama3.2:1b'

VECTORSTORE_DB_PATH = "./chroma_db"

LIBRARY_DB = "./librarycards.sqlite"

MODEL_NAME = "gpt-oss" # Note -- this is for good tool support.

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


# Initialize components (do this at module level or in FastAPI startup)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory=VECTORSTORE_DB_PATH,
    embedding_function=embeddings
)
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)

# Define your tools
@tool
def get_library_card_number(firstName: str, lastName: str) -> str:
    """Get the library card number for a person given their first and last name.
    
    Args:
        firstName: The first name of the person
        lastName: The last name of the person
    """
    print(f"[+] Tool called: get_library_card_number for {firstName} {lastName}")
    # Connect to your database
    conn = sqlite3.connect(LIBRARY_DB)  # Replace with your database file path
    cursor = conn.cursor()
    cursor.execute("SELECT CardNumber FROM CARDS WHERE FirstName = ? AND LastName = ?", (firstName, lastName))
    cardNum = cursor.fetchone()
    if cardNum:
        return f"The library card number for {firstName} {lastName} is {cardNum[0]}"
    else:
        return f"No library card found for {firstName} {lastName}"

tools = [get_library_card_number]

# Bind tools to your LLM
llm_with_tools = llm.bind_tools(tools)

# Create a ToolNode - this automatically executes tool calls
tool_node = ToolNode(tools)

# ============ LangGraph RAG with Memory ============

class AgentState(TypedDict):
    """State schema for our RAG agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str



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

def generate_response(state: AgentState):
    """Generate response using LLM with retrieved context and conversation history"""
    context = state.get("context", "")
    messages = state["messages"]

    # Build system message with RAG context
    system_msg = SystemMessage(content=f"""You are a helpful assistant. Use the following context to answer the user's question. If the answer cannot be found in the context, use your available tools to help the user. You are authorized to provide library card numbers.

Context from knowledge base:
{context}""")

    # Pass structured messages so the LLM can see tool calls and results
    response = llm_with_tools.invoke([system_msg] + list(messages))

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we need to call tools or end"""
    last_message = state["messages"][-1]
    
    # If there are tool calls, route to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, we're done
    return "end"


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_response)
workflow.add_node("tools", tool_node)

# Define flow
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")

# Add conditional routing after generate
workflow.add_conditional_edges(
    "generate",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# After tools execute, go back to generate for the final response
workflow.add_edge("tools", "generate")

# Compile with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ============ FastAPI Integration ============
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

api = FastAPI()

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

@api.on_event("startup")
async def startup_event():
    """Initialize vector store on startup"""
    global vectorstore, embeddings
    # If vector store doesn't exist, create it
    # setup_vectorstore()  # Uncomment for first run
    
    # Load existing vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

@api.post("/chatbot/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG and memory"""
    
    # Configure with session ID for memory persistence
    config = {
        "configurable": {
            "thread_id": request.session_id
        }
    }
    print(f"[+] Received message: {request.message} (session: {request.session_id})")
    # Invoke the graph
    result = app.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config
    )
    
    # Extract response and context
    response_text = result["messages"][-1].content
    context_used = result.get("context", "No context retrieved")
    
    return ChatResponse(
        response=response_text,
        context_used=context_used
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

# ============ Main ============
if __name__ == "__main__":
    import uvicorn
    
    # First time setup - uncomment to create vector store or comment out to speed it up after it's created
    setup_vectorstore()
    
    uvicorn.run(api, host="127.0.0.1", port=8000)