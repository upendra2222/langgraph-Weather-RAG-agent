from __future__ import annotations

from unittest.mock import MagicMock

from src.graph import weather_node, rag_node, AppResources
from langchain_core.documents import Document


def test_weather_node_calls_llm_and_sets_answer():
    mock_llm = MagicMock()
    # simulate response object with content attribute
    mock_resp = MagicMock()
    mock_resp.content = "Sunny and 20°C"
    mock_llm.invoke.return_value = mock_resp

    mock_weather = MagicMock()
    mock_weather.get_weather.return_value = {"weather": "sunny"}

    resources = AppResources(llm=mock_llm, weather_client=mock_weather, pdf_vectorstore=None)

    node = weather_node(resources)
    state = {"query": "weather in London"}
    final = node(state)

    assert final["weather_raw"] == {"weather": "sunny"}
    assert final["answer"] == "Sunny and 20°C"


def test_rag_node_calls_retriever_and_llm_and_sets_answer():
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = "Answer from PDF"
    mock_llm.invoke.return_value = mock_resp

    # mock retriever with get_relevant_documents
    class MockRetriever:
        def get_relevant_documents(self, query):
            return [Document(page_content="Doc text 1"), Document(page_content="Doc text 2")]

    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MockRetriever()

    resources = AppResources(llm=mock_llm, weather_client=MagicMock(), pdf_vectorstore=mock_vectorstore)

    node = rag_node(resources)
    state = {"query": "What is this document about?"}
    final = node(state)

    assert isinstance(final.get("context_docs"), list)
    assert final["answer"] == "Answer from PDF"
