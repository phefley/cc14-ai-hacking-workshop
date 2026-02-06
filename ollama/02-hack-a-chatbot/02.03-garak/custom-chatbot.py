# Reference: https://reference.garak.ai/en/latest/extending.generator.html

import logging
import random
import re
from typing import List, Union

from colorama import Fore, Style
import tqdm

from garak import _config
from garak.attempt import Message, Conversation
from garak.configurable import Configurable
from garak.exception import GarakException
import garak.resources.theme

import requests

from garak.generators.base import Generator

DEFAULT_CLASS = "CustomChatbotGenerator"
TARGET_API_URL = "http://127.0.0.1:8000/chatbot/"
DEBUG = False

class CustomChatbotGenerator(Generator):
    """Interface for our custom chatbot."""

    generator_family_name = "CustomChatbot"
    name = "CustomChatbotGenerator"
    description = "A custom chatbot generator that interacts with a FastAPI chatbot service."
    supports_multiple_generations = False

    def _call_api(self, prompt: str, debug: bool = False) -> str:
        """ 
        Call the chatbot API and yield the response.
        This isn't a mandatory method, but it's cleaner to separate it out.
        """

        # Set up the data to send. This will be our JSON payload.
        # Note: I cheated. I hardcoded session_id to "1" for simplicity.
        dataToSend = {"message": prompt, "session_id": "1"}
        
        if debug:
            print(f"{Fore.YELLOW} [DEBUG] Sending to API: {dataToSend} {Style.RESET_ALL}")

        # Send the data along to the chatbot API
        r = requests.post(TARGET_API_URL,json=dataToSend)
        
        if debug:
            print(f"{Fore.YELLOW} [DEBUG] Received from API: {r.json()} {Style.RESET_ALL}")
        
        return r.json()['response']

    def _call_model(self, prompt: Conversation, generations_this_call: int = 1):
        """
        Docstring for _call_model
        This is a mandatory method that all Generators must implement.
        
        :param self: Description
        :param prompt: Description
        :type prompt: Conversation
        :param generations_this_call: Description
        :type generations_this_call: int
        """
        prompt_text = prompt.last_message().text
        response = self._call_api(prompt_text, debug=DEBUG)
        m = Message(text=response)
        return [m]
