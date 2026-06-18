import hashlib
import logging
from pathlib import Path
from typing import Any, List
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents.base import Document

# from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
from sentence_transformers import CrossEncoder


from pydantic import BaseModel, Field

from config import langchain, env
from system_prompt import REWRITER_SYSTEM_PROMPT, SYSTEM_PROMPT
from tools import get_retriever_tool, tools


# Logging and verbose
langchain.initialize()


DOCS_DIR = "./docs"
# Silently create and move on
Path(DOCS_DIR).mkdir(exist_ok=True)


# reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", cache_folder="./encoder")

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    collection_name="docs",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

retriever_tool = get_retriever_tool(vectorstore)
all_tools = [*tools, retriever_tool]


loader = DirectoryLoader(
    glob="**/*.pdf",
    path=DOCS_DIR,
    loader_cls=PyPDFLoader,  # type:ignore
)


def load_docs():
    # Enrich Metadta
    docs = loader.load()
    for doc in docs:
        doc.metadata["filename"] = Path(doc.metadata["source"]).name or ""
    return docs


def get_chunk_id(doc: Document):
    source = doc.metadata["source"]
    page = doc.metadata["page"]
    return hashlib.md5(f"{source}:{page}:{doc.page_content}".encode()).hexdigest()


def get_chunks(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def build_store():
    logging.info("Building store...")
    docs = load_docs()
    chunks = get_chunks(docs)
    ids = [get_chunk_id(chunk) for chunk in chunks]

    new_chunks = []
    new_chunk_ids = []
    existing = set(vectorstore.get(ids=ids, include=[])["ids"])
    for chunk_id, chunk in zip(ids, chunks):
        if chunk_id not in existing:
            new_chunks.append(chunk)
            new_chunk_ids.append(chunk_id)

    if not new_chunks or not new_chunk_ids:
        logging.info("No new chunks to append!")
        return
    vectorstore.add_documents(new_chunks, ids=new_chunk_ids)
    logging.info("Building store done...")


build_store()


# llm = ChatOpenAI(
#     api_key=env.openai_api_key,
#     model=env.openai_default_model,
#     temperature=0.5,
#     reasoning_effort="medium",
# )


llm = ChatOllama(
    model="qwen3.5:2b",
    temperature=0.0,
)


class RewriterResponse(BaseModel):
    """
    Rewrite user query into multiple similar queries for vector space search
    """

    queries: List[str] = Field(
        default_factory=List,
        description="List of queries for vector space search relating to the user query",
    )


def get_user_input():
    return str(input("Input your query: "))


def get_rewriter_agent():
    return create_agent(
        model=llm,
        response_format=RewriterResponse,
        name="rewriter_agent",
        system_prompt=REWRITER_SYSTEM_PROMPT,
    )


def query_rewriter(query: str):
    rewriter: dict[str, Any] | Any = get_rewriter_agent().invoke(
        {"messages": [HumanMessage(content=query)]}
    )
    response: RewriterResponse = rewriter["structured_response"]
    return response.queries


def fetch_cached_docs(queries: List[str]):
    fetched_docs: List[Document] = []
    for q in queries:
        temp_docs = retriever_tool.invoke(q)
        # pairs = [(query, doc) for doc in temp_docs]
        # scores = reranker.predict(pairs)
        # ranked_docs = [doc for _, doc in sorted(zip(scores, temp_docs), reverse=True)]
        fetched_docs.append(temp_docs)
    return fetched_docs


def build_initial_context(docs: List) -> str:
    return "\n\n".join(docs)


def get_main_agent_system_promopt(initial_context: str):
    return SYSTEM_PROMPT.format(initial_context=initial_context).content


def get_main_agent():
    return create_agent(model=llm, tools=all_tools, name="main_agent")


def execute_main_agent(input: str):
    logging.info(f"Executing main agent with input {input}")

    logging.info("Rewriting query...")
    queries = query_rewriter(input)
    logging.info(f"Fetching related docs with rewritten queries...{queries}")
    docs = fetch_cached_docs(queries)

    logging.info("Calling main agent thread now...")
    agent = get_main_agent()
    initial_context = build_initial_context(docs)
    res = agent.invoke(
        {
            "messages": [
                SystemMessage(get_main_agent_system_promopt(initial_context)),
                HumanMessage(content=input),
            ]
        }
    )
    print(res)


execute_main_agent(get_user_input())
