FROM python:3.11

RUN pip install "giskard[llm]"
RUN pip install litellm==1.71.1
RUN mkdir -p /opt/chatbot-tester/reports
WORKDIR /opt/chatbot-tester

COPY *.py /opt/chatbot-tester

ENTRYPOINT [ "python" , "02.04-giskard-scanner.py" ]