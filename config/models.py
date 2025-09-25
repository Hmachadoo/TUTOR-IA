import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

load_dotenv()

_llm = None  
_tavily_client =None
def get_llm():
    global _llm
    if _llm is None:
        print("Inicializando o modelo de linguagem (LLM)...")
        _llm = init_chat_model(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_provider="groq"
        )
    return _llm

def get_tavily_client():
    global _tavily_client 
    if _tavily_client is None:
        print("Inicializando o cliente Tavily...")
        _tavily_client = TavilySearch(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=5,
            include_answer=False
        )
    
    return _tavily_client