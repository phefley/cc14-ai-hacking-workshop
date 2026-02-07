# Setup Guide

Please do all of this at home before the workshop. Some downloads are large.
It may not be reliable to download the required packages over conference wifi on the day.

## Prerequisites

Have at least 100GB available on your harddrive before starting this process. Some of the files are large.

Additionally, laptops should have at least 16Gb of ram, with 32Gb+ ram preferred, to run the exercises. 

Install the following:

- **uv** — https://docs.astral.sh/uv/getting-started/installation/ -if you don't have python already, installing uv will install it.
- **Git** — https://git-scm.com/downloads
- **Ollama** — https://ollama.com/download
- **Docker** — https://docs.docker.com/engine/install/
  - Linux: also install the compose plugin — `sudo apt-get install docker-compose-plugin`
  - Docker Desktop (Windows/macOS) includes compose already.
    - On windows, you may also need to update Windows Subsystem for Linux > `wsl --update`
    - Open docker desktop and make sure Docker Engine is running (or launch Docker Desktop) before running docker compose later in this guide.

## Folder Structure

```
~/workshop/
├── cc14-ai-hacking-workshop/        # this repo
├── AI-Red-Teaming-Playground-Labs/  # Microsoft AI red teaming lab
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
mkdir ~/workshop && cd ~/workshop
git clone https://github.com/phefley/cc14-ai-hacking-workshop
git clone https://github.com/microsoft/AI-Red-Teaming-Playground-Labs
```
To check for and receive updates run:
```bash
git status
git pull --rebase origin main
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
mkdir tools
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
> Windows: use `venvs\garak-venv\Scripts\activate` instead of `source...` and
> `copy cc14-ai-hacking-workshop\ollama\02-hack-a-chatbot\02.03-garak\custom-chatbot.py tools\garak-repo\garak\generators\` instead of `cp ...`

## 5 — Giskard (section 2.04)

Requires Python 3.11. `uv` will fetch it automatically.

```bash
cd ~/workshop
uv venv --python 3.11 venvs/giskard-venv
source venvs/giskard-venv/bin/activate
uv pip install "giskard[llm]"
uv pip install litellm==1.71.1
```
> Windows: use `venvs\giskard-venv\Scripts\activate` instead of `source ...`

## 6 — PyRIT (section 2.05)

```bash
cd ~/workshop
uv venv venvs/pyrit-venv
source venvs/pyrit-venv/bin/activate
uv pip install -r cc14-ai-hacking-workshop/ollama/02-hack-a-chatbot/02.05-pyrit/frozen-requirements.txt
```
> Windows: use `venvs\pyrit-venv\Scripts\activate` instead of `source ...`

## 7 — Microsoft AI Red Teaming Playground Labs

Pre-build the Docker containers so they're ready for the workshop.

```bash
cd ~/workshop/AI-Red-Teaming-Playground-Labs
docker compose up
```

If you see error text like 'unable to get image', make sure to launch Docker Engine (or Docker Desktop) first.

This will build the containers but won't fully work yet — during the workshop we'll provide a `.env` file with the Azure endpoint, then run `docker compose up` again.
After composing, you can quit docker and feel free to restart your laptop ahead of the workshop. We're go for launch!

## Checklist

- [ ] `ollama list` shows `llama3:8b`, `llama3.2:1b`, and `nomic-embed-text`
- [ ] `ollama run llama3.2:1b` responds to prompts
- [ ] Workshop repo cloned
- [ ] `AI-Red-Teaming-Playground-Labs` repo cloned, `docker compose up` ran once
- [ ] `ollama-venv` created, `requirements.txt` installed
- [ ] `garak-venv` created, garak cloned & installed, custom generator copied
- [ ] `giskard-venv` created (Python 3.11), giskard & litellm installed
- [ ] `pyrit-venv` created, frozen requirements installed
