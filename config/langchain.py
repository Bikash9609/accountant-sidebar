from langchain_core.globals import set_verbose, set_debug
import logging


def initialize():
    logging.basicConfig(level=logging.INFO)

    # Enabled by default
    set_verbose(True)
    set_debug(True)
