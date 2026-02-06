# Setup Guide

Please do all of this at home before the workshop. Some downloads are large.
It may not be reliable to download the required packages over conference wifi on the day.

## Prerequisites

Install the following:

- **uv** — https://docs.astral.sh/uv/getting-started/installation/ -if you don't have python already, installing uv will install it.
- **Git** — https://git-scm.com/downloads
- **Ollama** — https://ollama.com/download

## Folder Structure

```
~/workshop/
├── cc14-ai-hacking-workshop/        # this repo
├── venvs/
│   ├── ollama-venv/            # sections 1, 2.01–2.02, & 3
│   ├── garak-venv/             # section 2.03
│   ├── giskard-venv/           # section 2.04
│   └── pyrit-venv/             # section 2.05
└── tools/
    └── garak-repo/             # cloned NVIDIA garak
```

## 1 — Clone this Repo

```bash
mkdir -p ~/workshop && cd ~/workshop
git clone https://github.com/phefley/cc14-ai-hacking-workshop
```

## 2 — Ollama Models

```bash
ollama pull llama3:8b
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Verify: `ollama list` should show all three.

## 3 — Sections 1 & 3 Environment

```bash
cd ~/workshop
uv venv venvs/ollama-venv
source venvs/ollama-venv/bin/activate
uv pip install -r cc14-ai-hacking-workshop/ollama/requirements.txt
```

> Windows: use `venvs\ollama-venv\Scripts\activate` instead of `source ...`

## 4 — Garak (section 2.03)

Large download.

```bash
cd ~/workshop
uv venv venvs/garak-venv
source venvs/garak-venv/bin/activate
mkdir -p tools
git clone https://github.com/NVIDIA/garak tools/garak-repo
cd tools/garak-repo
uv pip install -e .
cd ~/workshop
cp cc14-ai-hacking-workshop/ollama/02-hack-a-chatbot/02.03-garak/custom-chatbot.py \
   tools/garak-repo/garak/generators/
cd tools/garak-repo
uv pip install -e .
cd ~/workshop
```

## 5 — Giskard (section 2.04)

Requires Python 3.11. `uv` will fetch it automatically.

```bash
cd ~/workshop
uv venv --python 3.11 venvs/giskard-venv
source venvs/giskard-venv/bin/activate
uv pip install "giskard[llm]"
uv pip install litellm==1.71.1
```

## 6 — PyRIT (section 2.05)

```bash
cd ~/workshop
uv venv venvs/pyrit-venv
source venvs/pyrit-venv/bin/activate
uv pip install -r cc14-ai-hacking-workshop/ollama/02-hack-a-chatbot/02.05-pyrit/frozen-requirements.txt
```

## Checklist

- [ ] `ollama list` shows `llama3:8b`, `llama3.2:1b`, and `nomic-embed-text`
- [ ] `ollama run llama3.2:1b` responds to prompts
- [ ] Workshop repo cloned
- [ ] `ollama-venv` created, `requirements.txt` installed
- [ ] `garak-venv` created, garak cloned & installed, custom generator copied
- [ ] `giskard-venv` created (Python 3.11), giskard & litellm installed
- [ ] `pyrit-venv` created, frozen requirements installed
