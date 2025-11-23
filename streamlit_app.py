from __future__ import annotations

from pathlib import Path
import tempfile
import streamlit as st

# Note: assuming src folder is on the Python path
from src.graph import build_graph
from src.config import settings


if settings.langsmith_api_key and settings.langsmith_project:
    import os
    # Set the required environment variables for LangChain V2 tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    print("✅ LangSmith Tracing Enabled.")
elif settings.langsmith_api_key:
    # Optional: warn if key is present but project name is missing
    print("⚠️ LangSmith key found, but LANGCHAIN_PROJECT is missing. Tracing may not save correctly.")
def main() -> None:
    st.set_page_config(page_title="LangGraph Weather + PDF RAG ", page_icon="☁️")
    st.title("LangGraph Weather + PDF RAG Demo")

    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF for RAG (optional)", type=["pdf"])

    # If a PDF is uploaded, save it to a temporary path but do NOT index automatically.
    pdf_path = None
    if uploaded_file is not None:
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / "uploaded_sample.pdf"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        pdf_path = str(tmp_path)

    # If a PDF is present, show an explicit button to index it (chunk/embed/store).
    if pdf_path is not None:
        st.sidebar.write(f"Uploaded: {Path(pdf_path).name}")
        if st.sidebar.button("Index PDF and enable RAG"):
            with st.spinner("Indexing PDF (chunking, embedding, storing)..."):
                # Compute chunk count for user feedback
                try:
                    from src.rag import load_pdf
                    from langchain_text_splitters import RecursiveCharacterTextSplitter

                    docs = load_pdf(pdf_path)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                    chunks = splitter.split_documents(docs)
                    chunk_count = len(chunks)
                except Exception as e:
                    chunk_count = None
                    st.error(f"PDF chunking failed: {e}")

                # Always rebuild the graph after indexing
                graph = build_graph(pdf_path)
                st.session_state["graph"] = graph
                st.session_state["graph_pdf_path"] = pdf_path
                st.session_state["has_pdf"] = True
                st.session_state["chunk_count"] = chunk_count
                supports_rag = bool(getattr(graph, "supports_rag", False))
                if not supports_rag:
                    st.error("Graph was rebuilt but RAG is NOT enabled. This means vectorstore creation failed. Check PDF validity and dependencies (faiss-cpu, qdrant-client). Try a different PDF or check logs.")
                elif chunk_count is None:
                    st.success("PDF indexed and RAG enabled.")
                else:
                    st.success(f"PDF indexed and RAG enabled. Indexed {chunk_count} chunks.")
                # Quick Qdrant count check for diagnostics (best-effort)
                if supports_rag:
                    try:
                        from qdrant_client import QdrantClient

                        qc = QdrantClient(url="http://localhost:6333")
                        try:
                            cnt = qc.count(collection_name="pdf_collection")
                            points_count = getattr(cnt, "count", None) or (cnt.get("count") if isinstance(cnt, dict) else None)
                        except Exception as e:
                            points_count = f"count check failed: {e}"
                        st.info(f"Qdrant collection point count: {points_count}")
                    except Exception as e:
                        st.info(f"qdrant-client unavailable or count failed: {e}")
    else:
        # If no PDF provided, ensure there's a weather-only graph available.
        if "graph" not in st.session_state:
            st.session_state["graph"] = build_graph(None)
            st.session_state["graph_pdf_path"] = None
            st.session_state["has_pdf"] = False

    query = st.text_input("Ask a question (about weather or the PDF):")
    # location = st.text_input(
    #     "Location (optional, used for weather queries):",
    #     help="If empty, the query text will be used as the location.",
    # )

    if st.button("Submit") and query:
        app = st.session_state.get("graph")
        if app is None:
            st.error("Graph is not initialized. Please index a PDF or retry.")
            return

        supports_rag = bool(getattr(app, "supports_rag", False))
        is_rag_enabled_in_session = st.session_state.get("has_pdf", False)
        
        # 💡 IMPORTANT: Pass the RAG status into the state. DO NOT pre-set the route here.
        state = {
            "query": query,
            "has_pdf": supports_rag and is_rag_enabled_in_session,
        }
        
        # if location:
        #     state["location"] = location

        # Prevent RAG queries if graph does not support RAG (this check is still valid)
        if not supports_rag and not ("weather" in query.lower() or "temperature" in query.lower()):
            st.warning("RAG is not enabled. Please index a PDF before asking document questions.")
            return

        # Use invoke() to let the router_node in graph.py handle the weather/rag decision.
        # We only use run_direct as a potential optimization, not for routing.
        final_state = app.invoke(state)
        
        st.subheader("Answer")
        st.write(final_state.get("answer", "No answer generated."))

        # Debug information to help diagnose routing/indexing issues
        # st.markdown("---")
        # st.subheader("Debug Info")
        # st.write({
        #     "graph_supports_rag": supports_rag,
        #     "session_has_pdf": bool(st.session_state.get("has_pdf", False)),
        #     "state_sent": state,
        #     "chunk_count": st.session_state.get("chunk_count"),
        # })

        # route = final_state.get("route")
        # st.write("Resolved route:", route)
        # if route == "rag":
        #     docs = final_state.get("context_docs") or []
        #     st.write(f"Retrieved {len(docs)} documents:")
        #     for i, d in enumerate(docs[:4]):
        #         # show a short preview of each chunk
        #         st.write(f"Chunk {i+1}: ", getattr(d, "page_content", str(d))[:500])
        # else:
        #     st.write("Weather raw:", final_state.get("weather_raw"))


if __name__ == "__main__":
    main()