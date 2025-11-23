from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from .llm import get_llm
from .rag import HFEmbeddings, build_qdrant_vectorstore_from_pdf, QdrantVectorStore
from .config import settings
import logging
from .weather import WeatherClient
import re


class AgentState(TypedDict, total=False):
    """State shared across LangGraph nodes."""

    query: str
    route: Literal["weather", "rag"]
    location: Optional[str]
    weather_raw: Optional[dict]
    context_docs: Optional[list[Document]]
    answer: Optional[str]
    # Required to pass RAG enablement status from Streamlit
    has_pdf: bool


@dataclass
class AppResources:
    """Shared resources (LLM, tools, vector store)."""

    llm: BaseChatModel
    weather_client: WeatherClient
    pdf_vectorstore: object | None


def router_node(state: AgentState) -> AgentState:
    """Decide whether to call the weather API or RAG based on the query."""
    query = state["query"].lower()
    has_pdf = bool(state.get("has_pdf"))

    # 1. ALWAYS prioritize weather if keywords are present, regardless of RAG state.
    if "weather" in query or "temperature" in query:
        state["route"] = "weather"
        return state
    
    # 2. If it's NOT a weather query, check if RAG is available.
    if has_pdf:
        state["route"] = "rag"
    else:
        # 3. If RAG is NOT available, default to the weather tool (as the general LLM fallback).
        state["route"] = "weather"
        
    return state


def weather_node(resources: AppResources):
    def _node(state: AgentState) -> AgentState:
        # Prefer explicit location, otherwise attempt to parse from the query.
        location = state.get("location")
        if not location:
            # Try to extract a location phrase like "in Paris" from the query.
            def _parse_location(q: str) -> str | None:
                m = re.search(r"\bin\s+([A-Za-z0-9\s,\-\.]+)", q, re.I)
                if not m:
                    return None
                loc = m.group(1).strip()
                # Remove trailing words that are not part of the location
                loc = re.sub(r"\b(today|tomorrow|now)\b", "", loc, flags=re.I).strip()
                # Strip trailing punctuation
                loc = loc.rstrip(".,!?;:\\/")
                return loc or None

            location = _parse_location(state["query"]) or state["query"]
        try:
            raw = resources.weather_client.get_weather(location)
        except Exception as e:  # catch WeatherAPIError and other request issues
            # Return a helpful message to the user instead of raising.
            state["weather_raw"] = None
            state["answer"] = f"Weather lookup failed: {e}. Please refine the location and try again."
            return state

        prompt = [
            SystemMessage(
                content=(
                    "You are a helpful weather assistant. Summarize the current weather "
                    "for a non-technical user in 2–3 sentences."
                )
            ),
            HumanMessage(content=str(raw)),
        ]
        response = resources.llm.invoke(prompt)
        state["weather_raw"] = raw
        state["answer"] = response.content  # type: ignore[attr-defined]
        return state

    return _node


def rag_node(resources: AppResources):
    def _node(state: AgentState) -> AgentState:
        # Use k=8 for higher context retrieval confidence
        retriever = resources.pdf_vectorstore.as_retriever(search_kwargs={"k": 8})
        docs = retriever.get_relevant_documents(state["query"])

        state["context_docs"] = docs
        
        # Explicitly handle cases where no documents are retrieved
        if not docs:
            state["answer"] = (
                "I could not find any relevant information in the indexed PDF documents "
                f"for the query: '{state['query']}'. Please try rephrasing or confirm that the PDF was indexed successfully."
            )
            return state

        system = SystemMessage(
            content=(
                "You are a helpful assistant that answers questions using only the "
                "provided PDF context. If the answer is not in the context, say that "
                "you do not know."
            )
        )
        context_str = "\n\n".join(d.page_content for d in docs)
        human = HumanMessage(
            content=f"Question: {state['query']}\n\nContext:\n{context_str}"
        )
        response = resources.llm.invoke([system, human])
        state["answer"] = response.content  # type: ignore[attr-defined]
        return state

    return _node


def build_graph(pdf_path: str) -> RunnableLambda:
    """Build and return the LangGraph workflow as a runnable."""
    resources = AppResources(llm=get_llm(), weather_client=WeatherClient(), pdf_vectorstore=None)

    # 1. Handle Weather-Only Mode
    if not pdf_path:
        workflow = StateGraph(AgentState)
        workflow.add_node("router", router_node)
        workflow.add_node("weather", weather_node(resources))
        workflow.set_entry_point("router")

        def route_decision(state: AgentState) -> str:
            return state["route"]

        workflow.add_conditional_edges("router", route_decision, {"weather": "weather"})
        workflow.add_edge("weather", END)

        app = workflow.compile()
        return app

    # 2. Handle Weather + Optional RAG Mode
    # Attempt to build Qdrant Vector Store
    try:
        qdrant_store = build_qdrant_vectorstore_from_pdf(pdf_path)
        resources.pdf_vectorstore = qdrant_store
    except Exception as e:
        resources.pdf_vectorstore = None
        logging.error(f"Qdrant vectorstore creation failed: {e}")

    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("weather", weather_node(resources))
    
    # Conditionally add the RAG node
    if resources.pdf_vectorstore is not None:
        workflow.add_node("rag", rag_node(resources))

    workflow.set_entry_point("router")

    def route_decision(state: AgentState) -> str:
        return state["route"]

    # Conditional edges
    edges = {"weather": "weather"}
    if resources.pdf_vectorstore is not None:
        edges["rag"] = "rag"

    workflow.add_conditional_edges("router", route_decision, edges)
    workflow.add_edge("weather", END)
    if resources.pdf_vectorstore is not None:
        workflow.add_edge("rag", END)

    app = workflow.compile()
    
    # Attach metadata and run_direct function
    try:
        setattr(app, "supports_rag", bool(resources.pdf_vectorstore))
        
        try:
            setattr(app, "_resources", resources)

            def run_direct(state: dict) -> dict:
                """Run the requested route directly using the underlying resources."""
                if state.get("route") == "rag" and resources.pdf_vectorstore is not None:
                    return rag_node(resources)(state) 
                return weather_node(resources)(state)

            setattr(app, "run_direct", run_direct)
        except Exception:
            pass
    except Exception:
        pass
        
    return app