# Ollama Chat Bot

## Pre-Reqs

### Ollama Install
We're going to make a simple ollama chat bot.

https://ollama.com/download

Make sure you have ollama installed.

### Get a Model

Install the model. Note that llama3b:8b is about 5GB. This will take a while to download!!
If you want to go smaller, try llama3.2:1b

```bash
ollama pull llama3:8b
```
Reference:
https://ollama.com/library/llama3
llama3:8b

```bash
ollama list
```

```bash
ollama show llama3:8b
```

You will also need a text embedder for the RAG portion:

```bash
ollama pull nomic-embed-text
```

### Interact with the model

```bash
ollama run llama3:8b
```

### Python Environment

```bash
python3 -m venv ./ollama-example-venv
source ./ollama-example-venv/bin/activate
pip install -r requirements.txt
```
