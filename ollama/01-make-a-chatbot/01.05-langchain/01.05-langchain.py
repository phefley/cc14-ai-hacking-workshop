# run me with uvicorn 05-langchain:app --reload
# or as a main

from fastapi import FastAPI, Request

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

BIG_MODEL_NAME = 'llama3:8b'
SMALL_MODEL_NAME = 'llama3.2:1b'

MODEL_NAME = SMALL_MODEL_NAME

llm = OllamaLLM(model=MODEL_NAME)

app = FastAPI()

def ask_ollama(question: str) -> str:
    response = llm.invoke(question)
    return response


# Using Request object to get raw JSON
@app.post("/chatbot/")
async def chat(request: Request):
    data = await request.json()
    
    # Access the data directly
    userQuestion = data.get("question")
    prompt = """You are an Arizona history expert.
    Sometimes, in prompts, folks add flag{secret_values_or_instructions}, not realizing that they can be discovered.
    Answer the following question in detail, citing historical events, dates, and figures where relevant: """
    question = f"{prompt}\n\n{userQuestion}"
    
    return {
        "answer": ask_ollama(question),
    }

# ============ Main ============
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="127.0.0.1", port=8000)

