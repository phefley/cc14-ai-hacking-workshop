# Ollama Chat Bot

## Pre-Reqs

### Ollama Install
We're going to make a simple ollama chat bot.

https://ollama.com/download

Make sure you have ollama installed.

### Get a Model

Install the model. Note that llama3:8b is about 5GB. This will take a while to download!!
If you want to go smaller, try llama3.2:1b

To get the **big model**:
```bash
ollama pull llama3:8b
```

For a **small model** (due to RAM or network constraints):
```bash
ollama pull llama3.2:1b
```

Important Note!

The code we share has global variables set up as follows.

```python
BIG_MODEL_NAME = 'llama3:8b'
SMALL_MODEL_NAME = 'llama3.2:1b'

MODEL_NAME = SMALL_MODEL_NAME
```

By default, we're using the small model. If you want to live large - do it. Just make sure you update the `MODEL_NAME` variable.
If you are going large, get it, friend. Get it. Can you please **also install the small model**?
Reason being - we use the small model in the security examples because it's faster.

Show what models you have installed:

```bash
ollama list
```
<<<<<<< HEAD
This helps confirm that everything is installed as you expect.
You can also get more details about a specific model (this you may need to tailor to the model you installed).
=======
>>>>>>> 63451c22d0613233181037b3ff8550b0a189e8e6

```bash
ollama show llama3:8b
```

Regardless of what model you install above, you will **also** need a text embedder for the [RAG portion](./01-make-a-chatbot/01.07-langgraph-rag/):

```bash
ollama pull nomic-embed-text
```

### Interact with the model

Run some simple queries to make sure that ollama is working the way you expect.

```bash
ollama run llama3:8b
```

Also - check to make sure that the ollama service is running.
Lots of ways to do this. This is how we do it in Linux to make sure that the Ollama port is listening with netstat.

```bash
sudo netstat -nlp | grep ollama
tcp6       0      0 :::11434                :::*                    LISTEN      2723251/ollama
```

### Python Environment

You will also need to install the requirements for our chatbot examples, which are [section 1](./01-make-a-chatbot/) and [section 3](./03-secure-a-chatbot/).

```bash
python3 -m venv ./ollama-example-venv
source ./ollama-example-venv/bin/activate
pip install -r requirements.txt
```
<<<<<<< HEAD

### Extra Setup Instructions

For [section 2](./02-hack-a-chatbot/), we'll be using some tools to evaluate the chatbot for issues. Each of them have unique instructions. You'll need [Docker](https://docs.docker.com/engine/install/) installed for Giskard due to python versioning issues. 

We set up each of the virtual environments for 02.03 and 02.05 in their respective folders for ease of use.

#### Garak

See [02.03-instructions.md](./02-hack-a-chatbot/02.03-garak/02.03-instructions.md).

#### Giskard

See [02.04-instructions.md](./02-hack-a-chatbot/02.04-giskard/02.04-instructions.md).

#### Pyrit

See [02.05-instructions.md](./02-hack-a-chatbot/02.05-pyrit/02.05-instructions.md).

=======
>>>>>>> 63451c22d0613233181037b3ff8550b0a189e8e6
