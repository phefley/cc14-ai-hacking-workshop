from requests import post
from sys import argv

URL = "http://localhost:8000/chatbot/"

questionStr = argv[1]
jsonData = {"question":questionStr}

response = post(URL, json=jsonData)

print(response.json())

