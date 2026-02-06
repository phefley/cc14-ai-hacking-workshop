# Reference: https://github.com/ollama/ollama-python
import ollama

BIG_MODEL_NAME = 'llama3:8b'
SMALL_MODEL_NAME = 'llama3.2:1b'

MODEL_NAME = SMALL_MODEL_NAME

# Simple chat completion
response = ollama.chat(model=MODEL_NAME, messages=[
  {
    'role': 'user',
    'content': 'Why did the cat sit on the computer?',
  },
])
print(response['message']['content'])

# Streaming responses
stream = ollama.chat(
    model=MODEL_NAME,
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)