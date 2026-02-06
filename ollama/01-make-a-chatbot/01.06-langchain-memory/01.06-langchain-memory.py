# run me with uvicorn 06-langchain-memory:app --reload
# or as a main

import uuid
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver 


BIG_MODEL_NAME = 'llama3:8b'
SMALL_MODEL_NAME = 'llama3.2:1b'

MODEL_NAME = SMALL_MODEL_NAME


model = ChatOllama(
    model=MODEL_NAME,
    validate_model_on_init=True,
    #temperature=0.8,
    #num_predict=256,
    # other params ...
)

tools=[]
agent = create_agent(model, tools=tools, checkpointer=InMemorySaver())

system_prompt = SystemMessage("""
You are a senior Python developer.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
When providing code examples, ensure they are properly formatted and syntactically correct.
You can only discuss the python programming language. Do not discuss plants or succulents. Refuse to discuss anything other than python.
""")

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

def ask_ollama(question: str, threadId: str, firstTimeCaller: bool = True) -> str:
    if firstTimeCaller:
        messages = [system_prompt, HumanMessage(content=question)]
    else:
        messages = [HumanMessage(content=question)]
    
    response = agent.invoke(
      {"messages": messages},
      {"configurable": {"thread_id": threadId}},  
    )
    for message in response['messages']:
        last_message = message.content
    return last_message


# Using Request object to get raw JSON
@app.post("/chatbot/")
async def chat(request: Request):
    data = await request.json()
    
    # Access the data directly
    userQuestion = data.get("question")
    threadId = data.get("threadId")
    print(f"Received question: {userQuestion} with threadId: {threadId}")
    if not threadId:
        threadId = str(uuid.uuid4())
        firstTimeCaller = True
    else:
        firstTimeCaller = False
    
    question = userQuestion
    
    return {
        "answer": ask_ollama(question, threadId, firstTimeCaller),
        "threadId": threadId,
    }

# ============ Main ============
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="127.0.0.1", port=8000)


