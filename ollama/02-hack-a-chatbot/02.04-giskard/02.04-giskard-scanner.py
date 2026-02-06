import giskard
import pandas as pd
from requests import post
import litellm
import os

from datetime import datetime

# Get the current date and time as a datetime object
scanTime = datetime.now()
scanTimeStr = scanTime.strftime("%Y%m%d-%H%M")

# Using 01.06 as an example:

YOUR_IP_ADDRESS = "127.0.0.1" # You need to update this to your actual IP address
URL = f"http://{YOUR_IP_ADDRESS}:8000/chatbot/"
OLLAMA_BASE = f"http://{YOUR_IP_ADDRESS}:11434"

litellm._turn_on_debug()
giskard.llm.set_llm_model("ollama/llama3", base_url=OLLAMA_BASE)
giskard.llm.set_embedding_model("ollama/nomic-embed-text", base_url=OLLAMA_BASE)

def askChatbot(question: str) -> str:
    jsonData = {"question":question}
    print(f"[askChatbot] asking: {question}")
    response = post(URL, json=jsonData)
    answer = response.json()['answer']
    print(f"[askChatbot] got answer: {answer}")
    return answer

def model_predict(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and must return a list of the outputs (one for each row).
    """
    return [askChatbot(question) for question in df["question"]]

# Don’t forget to fill the `name` and `description`: they are used by Giskard
# to generate domain-specific tests.
giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="Demo Chatbot",
    description="This is a demo chatbot that we wrote. It's supposed to help with writing Python code.",
    feature_names=["question"],
)

scan_results = giskard.scan(giskard_model,only="jailbreak")
#scan_results = giskard.scan(giskard_model)

if os.path.exists("./reports") == False:
    os.mkdir("./reports")

reportFileName = f"./reports/giskard_scan_report_{scanTimeStr}.html"
print(f"Writing report to {reportFileName}")
scan_results.to_html(reportFileName)
print("[++] Giskard scan complete.")


