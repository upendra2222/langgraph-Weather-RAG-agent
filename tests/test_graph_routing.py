from __future__ import annotations

from src.graph import router_node


def test_router_weather_route():
    # Weather keywords take priority
    state = {"query": "What is the weather in Paris today?", "has_pdf": True}
    new_state = router_node(state)
    assert new_state["route"] == "weather"


def test_router_rag_route():
    # RAG route should be selected when a PDF is present and no weather keywords are used.
    state = {"query": "Explain the main idea of the PDF.", "has_pdf": True}
    new_state = router_node(state)
    assert new_state["route"] == "rag"


def test_router_default_route_no_rag():
    # 🌟 NEW TEST: When RAG is disabled, all queries default to weather.
    state = {"query": "What is the main idea?", "has_pdf": False}
    new_state = router_node(state)
    assert new_state["route"] == "weather"