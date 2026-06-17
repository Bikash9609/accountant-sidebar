from dotenv import load_dotenv
import os

from pydantic import SecretStr

load_dotenv()


openai_api_key = SecretStr(os.environ["OPENAI_API_KEY"])
openai_default_model = os.environ["OPENAI_MODEL"]
tavily_api_key = os.environ["TAVILY_API_KEY"]
