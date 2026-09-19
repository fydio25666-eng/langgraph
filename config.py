"""Environment loading and model client configuration."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def load_model() -> ChatOpenAI | None:
    """Load configuration; missing settings select the fallback path."""
    load_dotenv()
    if not all(os.getenv(key) for key in ("BASE_URL", "API_KEY", "MODEL")):
        return None
    return ChatOpenAI(
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["API_KEY"],
        # Bound request time and retries so one question cannot block the process.
        model=os.environ["MODEL"], temperature=0, timeout=20, max_retries=1,
    )


model = load_model()  # Client initialization does not make an API request.
