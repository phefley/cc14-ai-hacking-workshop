# CactusCon 14 - Hacking AI Chatbots

[DRAFT - To be completed shortly]
This is the the set of prereqs you'll want to have in place to get started.

Microsoft AI RedTeaming Playground Lab: Spin up a docker container and connect it to our Azure endpoint during the workshop.
  1. Make sure you have **Docker** installed.
  2. Create your project folder and navigate there. > mkdir project; cd project
  3. Clone the MSFT AI Redteaming Playground Labs repo > git clone https://github.com/microsoft/AI-Red-Teaming-Playground-Labs
  4. Navigate into the repo and run **docker-compose up**. This will build the containers, but it will not work yet.
  5. During the workshop, we will set up the .env file and run **docker-compose up** again to rebuild it and use it. 

Much of this workshop will focus on running and testing models locally with ollama.
Follow the instructions in [setup.md](ollama/setup.md) before Cactuscon on Saturday.

If you want to do automated testing, you will want to follow instructions in the ollama 02 section [GARAK install](ollama/02-hack-a-chatbot/notes.md) to install Garak. The files are big, and you'll want to do this from the comfort of your own high-speed internet.

