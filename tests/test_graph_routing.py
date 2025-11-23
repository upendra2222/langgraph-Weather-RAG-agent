from __future__ import annotations

from src.graph import router_node


def test_router_weather_route():
    state = {"query": "What is the weather in Paris today?"}
    new_state = router_node(state)
    assert new_state["route"] == "weather"


def test_router_rag_route():
    # RAG route should only be selected when a PDF has been provided.
    state = {"query": "Explain the main idea of the PDF.", "has_pdf": True}
    new_state = router_node(state)
    assert new_state["route"] == "rag"


