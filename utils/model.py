import os
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI

load_dotenv()

def get_client():
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2025-01-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_model_name():
    return os.getenv("MODEL_NAME","gpt-4o-mini")
