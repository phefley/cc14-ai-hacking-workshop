# run me with uvicorn 02-http-api-example:app --reload
# or as a main


from fastapi import FastAPI, Request
import ollama

BIG_MODEL_NAME = 'llama3:8b'
SMALL_MODEL_NAME = 'llama3.2:1b'

MODEL_NAME = SMALL_MODEL_NAME

app = FastAPI()

def ask_ollama(question: str) -> str:
    response = ollama.generate(model=MODEL_NAME, prompt=question)
    return response['response']


# Using Request object to get raw JSON
@app.post("/chatbot/")
async def chat(request: Request):
    data = await request.json()
    
    # Access the data directly
    question = data.get("question")
    
    return {
        "answer": ask_ollama(question),
        "question": question
    }


# ============ Main ============
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="127.0.0.1", port=8000)

"""
Test me with:

curl -X POST "http://localhost:8000/chatbot/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'

On windows, make sure to escape internal quotes, single quotes may break the parsing::
curl -X POST "http://localhost:8000/chatbot/" -H "Content-Type: application/json" -d "{\"question\": \"What is the capital of France?\"}"
or in powershell
iwr -UseBasicParsing -Uri "http://localhost:8000/chatbot/" -Method POST -ContentType "application/json" -Body '{"question": "What is the capital of France?"}'
"""