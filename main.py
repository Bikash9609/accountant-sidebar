from pathlib import Path
from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
import hashlib

from config import langchain, env
from system_prompt import SYSTEM_PROMPT
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


loader = DirectoryLoader(
    glob="**/*.pdf",
    path=DOCS_DIR,
    loader_cls=PyPDFLoader,  # type:ignore
)

# Enrich Metadta
docs = loader.load()
for doc in docs:
    doc.metadata["filename"] = Path(doc.metadata["source"]).name or ""


def get_chunk_id(doc: Document):
    source = doc.metadata["source"]
    page = doc.metadata["page"]
    return hashlib.md5(f"{source}:{page}:{doc.page_content}".encode()).hexdigest()


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks: list[Document] = splitter.split_documents(docs)
ids = [get_chunk_id(chunk) for chunk in chunks]
vectorstore.add_documents(chunks, ids=ids)


llm = ChatOpenAI(
    api_key=env.openai_api_key, model=env.openai_default_model, temperature=0
)


retriever_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    description="""
Use this tool whenever information may exist in uploaded documents,
bank statements, salary slips, payroll records, invoices, or financial
documents.

Before saying information is unavailable, always search the documents
using this tool.
""",
    name="search_documents",
)
all_tools = [*tools, retriever_tool]


agent = create_agent(
    model=llm,
    tools=all_tools,
    system_prompt=SYSTEM_PROMPT,
)


query = str(input("Input your query: "))

intial_context = "\n".join([doc for doc in retriever_tool.invoke(query)])
res = agent.invoke(
    {
        "messages": [
            ToolMessage(
                content=intial_context,
            ),
            HumanMessage(content=query),
        ]
    }
)
print(res)
