import logging

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
import sympy as sp


@tool
def math_solver(expr: str):
    """
    Use this tool to perform mathematical computations using SymPy.
    Accepts a math expression as a string and returns the evaluated, simplified, or solved result.
    Supports algebra, calculus (derivatives, integrals, limits), and equation solving.

    Args:
      expr: maths expression string

    Returns:
      Maths result output as string
    """
    logging.info(f"Solving math probelm... {expr}")
    return str(sp.sympify(expr).doit())


def get_retriever_tool(vs: Chroma):
    retriever_tool = create_retriever_tool(
        retriever=vs.as_retriever(search_kwargs={"k": 5}),
        description="""
                    Use this tool whenever information may exist in uploaded documents,
                    bank statements, salary slips, payroll records, invoices, or financial
                    documents.

                    Before saying information is unavailable, always search the documents
                    using this tool.
                    """,
        name="search_documents",
    )
    return retriever_tool


tools = [math_solver]
