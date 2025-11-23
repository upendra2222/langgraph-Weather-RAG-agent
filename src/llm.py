from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
import os

from .config import settings


def get_llm() -> BaseChatModel:
    """Return a Groq-backed open-source LLM (e.g. Llama 3.1).

    This checks the environment at call-time first so tests can manipulate
    `GROQ_API_KEY` using `os.environ`.
    """
    # Prefer an explicit environment variable presence. Tests may modify
    # `os.environ` at runtime; check `os.environ` directly so changes take effect.
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    # Model is open-source, hosted by Groq.
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=api_key,
    )


