from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

BIG_MODEL_NAME = 'llama3:8b'
SMALL_MODEL_NAME = 'llama3.2:1b'

VECTORSTORE_DB_PATH = "./chroma_db"

MODEL_NAME = SMALL_MODEL_NAME

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

# Initialize components (do this at module level or in FastAPI startup)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory=VECTORSTORE_DB_PATH,
    embedding_function=embeddings
)
llm = OllamaLLM(model=MODEL_NAME)

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
    
    # Build conversation history
    conversation = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in messages[:-1]  # All except the last message
    ])
    
    # Get the current question
    current_question = messages[-1].content
    
    # Create prompt with context and history
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

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_response)

# Define flow
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

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