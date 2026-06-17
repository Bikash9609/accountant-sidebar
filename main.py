from pathlib import Path
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

from config import langchain, env
from tools import tools

# Logging and verbose
langchain.initialize()


DOCS_DIR = "./docs"
# Silently create and move on
Path(DOCS_DIR).mkdir(exist_ok=True)


embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    collection_name="docs",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


llm = ChatOpenAI(
    api_key=env.openai_api_key, model=env.openai_default_model, temperature=1
)
agent = create_agent(
    model=llm,
    tools=tools,
)
