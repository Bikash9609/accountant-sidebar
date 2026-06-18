import logging

from langchain.tools import tool
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


tools = [math_solver]
