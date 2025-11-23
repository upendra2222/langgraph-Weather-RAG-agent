from __future__ import annotations

from src.llm import get_llm


def test_get_llm_missing_api_key():
    # Ensure that get_llm raises a RuntimeError when GROQ_API_KEY is not set
    import os

    prev = os.environ.pop("GROQ_API_KEY", None)
    try:
        try:
            get_llm()
            raised = False
        except RuntimeError as e:
            raised = True
            assert "GROQ_API_KEY is not set" in str(e)
        assert raised
    finally:
        if prev is not None:
            os.environ["GROQ_API_KEY"] = prev
