from requests import post
from sys import argv

URL = "http://localhost:8000/chatbot/"

questionStr = argv[1]

## Let's wrap this up as a function so we can reuse it later
def askChatbot(question: str) -> str:
    jsonData = {"question":question}
    response = post(URL, json=jsonData)
    return response.json()['answer']


print(askChatbot(questionStr))
