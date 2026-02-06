from requests import post
from sys import argv

URL = "http://localhost:8000/chatbot/"

questionStr = argv[1]
if len(argv) > 2:
    threadIdStr = argv[2]
else:
    threadIdStr = None

## Let's wrap this up as a function so we can reuse it later
def askChatbot(question: str) -> str:
    jsonData = {"message":question}
    if threadIdStr:
        jsonData['session_id'] = threadIdStr
    else:
        jsonData['session_id'] = "default_session"
    response = post(URL, json=jsonData)
    return str(response.json())


print(askChatbot(questionStr))
