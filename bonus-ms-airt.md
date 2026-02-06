
# Microsoft AI Red Teaming Lab

For fun and practice!! This is a walk through of how you would stand that up.

## Clone the repo

git clone https://github.com/microsoft/AI-Red-Teaming-Playground-Labs

## Build the containers

Do this once from home to get the layers you need.

```bash
docker compose build
```

It will look like this...
```bash
> docker compose build
[+] Building 14.6s (14/35)
 => [challenge-home internal] load metadata for mcr.microsoft.com/cbl-mariner/base/python:3                                                                      0.5s
 => [chat-copilot-8 internal] load metadata for mcr.microsoft.com/devcontainers/javascript-node:20-bookworm                                                      1.0s
 => [chat-copilot-11 internal] load metadata for mcr.microsoft.com/dotnet/aspnet:7.0                                                                             0.1s
 => [chat-copilot-4 internal] load metadata for mcr.microsoft.com/dotnet/sdk:7.0                                                                                 0.1s
 => [chat-copilot-8 internal] load .dockerignore                                                                                                                 0.0s
                                                            6.9s
 ...
```

It will take some time.


## Edit your .env file

Normally, you would copy the `.env.example` file in to `.env` and make your edits.

```bash
> cat .env.example
# This an example of the .env file. You should create a .env file in the root directory of the project and fill in the values
# with your own values.

# Used for cookie security.
# You can generate one with python by running this python command `python -c 'import secrets; print(secrets.token_hex(16))'`
SECRET_KEY=YOUR_SECRET_KEY

# Used for the URL authentication. This is the key that needs to be included in the URL for autentication.
# You can generate one with python by running this python command `python -c 'import secrets; print(secrets.token_hex(16))'`
# You will access the labs at http://localhost:5000/login?auth=YOUR_AUTH_KEY
AUTH_KEY=YOUR_AUTH_KEY

# Azure OpenAI Information
AOAI_ENDPOINT=https://YOUR_AOAI_ENDPOINT
AOAI_API_KEY=YOUR_AOAI_API_KEY
AOAI_MODEL_NAME=gpt-4o

# OpenAI Standard Configuration
# OPENAI_API_KEY=sk-your_openai_api_key_here
# OPENAI_ORG_ID=  # Optional and can cause errors if mismatched
# OPENAI_TEXT_MODEL=gpt-3.5-turbo
# OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

> cp .env.example .env
```

What we're going to have you do is use our OpenAI endpoint. Be gentle.

```bash
wget -O .env http://range.peterhefley.net/FILE
```

## Rebuild and Run

```bash
docker compose build
docker compose up
```