from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging
import time
import uuid
import requests


class HFEmbeddings(Embeddings):
    """Wrapper around a sentence-transformers model for embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # `SentenceTransformer.encode` already returns a list when
        # Request a numpy array and convert to plain Python lists so the
        # vectors are JSON-serializable for HTTP fallbacks.
        arr = self.model.encode(texts, convert_to_numpy=True)
        # arr may be a 2D numpy array; convert to list of lists
        try:
            return arr.tolist()
        except Exception:
            # Fallback: coerce each element
            return [list(map(float, a)) for a in arr]

    def embed_query(self, text: str) -> List[float]:
        arr = self.model.encode([text], convert_to_numpy=True)[0]
        try:
            return arr.tolist()
        except Exception:
            return list(map(float, arr))


class QdrantVectorStore:
    """Minimal Qdrant-backed vector store wrapper.

    This wrapper is optional and only used when `qdrant-client` and a running
    Qdrant instance are available. Use `from_documents` to create a collection
    and upsert document vectors.
    """

    def __init__(self, client, collection_name: str, embeddings: HFEmbeddings):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        # If the underlying client exposes a base URL we prefer that; otherwise
        # callers can provide it via the `from_documents` helper (we set it there).
        self.qdrant_url = getattr(client, "url", None) or getattr(client, "_base_url", None)

    @classmethod
    def from_documents(cls, docs: List[Document], collection_name: str = "pdf_collection", qdrant_url: str = "http://localhost:6333", embeddings: HFEmbeddings | None = None):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("qdrant-client is required for QdrantVectorStore: install 'qdrant-client'") from e

        if embeddings is None:
            embeddings = HFEmbeddings()

        client = QdrantClient(url=qdrant_url)

        # Create/recreate collection with an inferred vector size
        if docs:
            sample_text = docs[0].page_content
            vector = embeddings.embed_documents([sample_text])[0]
            vector_size = len(vector)
        else:
            vector_size = 384

        try:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
            )
        except Exception:
            # Older client versions may not support recreate_collection
            try:
                client.create_collection(collection_name=collection_name, vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE))
            except Exception:
                pass

        # Build upsert payloads as plain dicts to maximize compatibility
        points = []
        for idx, d in enumerate(docs):
            vec = embeddings.embed_documents([d.page_content])[0]
            # Use a UUID string for point IDs to avoid client/server
            # id-format mismatches. UUIDs are safe across versions.
            pid = str(uuid.uuid4())
            # 💡 Crucial: Ensure the payload key for the content is 'text'
            points.append({"id": pid, "vector": vec, "payload": {"text": d.page_content}})

        # Try upserting with a few retries/backoff to handle transient connection resets
        upsert_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                client.upsert(collection_name=collection_name, points=points)
                upsert_exc = None
                break
            except Exception as e:
                upsert_exc = e
                logging.warning("Qdrant upsert attempt %s failed: %s", attempt, e)
                # short exponential backoff
                time.sleep(2 ** (attempt - 1))

        if upsert_exc is not None:
            logging.exception("Qdrant upsert failed after retries")
            # Provide actionable information to the caller
            raise RuntimeError(f"Failed to upsert vectors to Qdrant collection after retries: {upsert_exc}")

        # Verify that points exist (best-effort). Some client versions expose `count`.
        try:
            count_resp = client.count(collection_name=collection_name, exact=True)
            # count_resp may be an object with 'count' or a dict
            count = getattr(count_resp, "count", None) or (count_resp.get("count") if isinstance(count_resp, dict) else None)
            logging.info("Qdrant collection '%s' contains %s points", collection_name, count)
        except Exception:
            # If count is unsupported, skip verification silently.
            pass

        store = cls(client=client, collection_name=collection_name, embeddings=embeddings)
        # Save the qdrant_url used to create the client so retriever can
        # fallback to the HTTP API when the client library lacks a search method.
        store.qdrant_url = qdrant_url
        return store

    def as_retriever(self, search_kwargs: dict | None = None):
        search_kwargs = search_kwargs or {}

        class Retriever:
            def __init__(self, store: QdrantVectorStore):
                self.store = store

            def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
                qvec = self.store.embeddings.embed_query(query)
                # Try multiple Qdrant client search APIs for compatibility.
                hits = None
                try:
                    # Prefer the official Qdrant search methods
                    if hasattr(self.store.client, "search"):
                        # Use models.SearchRequest for newer clients
                        try:
                            from qdrant_client.http.models import SearchRequest
                            hits = self.store.client.search(
                                collection_name=self.store.collection_name, 
                                query_vector=qvec, 
                                limit=k, 
                                search_params=SearchRequest(vector=qvec, limit=k, with_payload=True, with_vectors=False)
                            )
                        except Exception:
                            # Fallback for older search API signatures
                            hits = self.store.client.search(
                                collection_name=self.store.collection_name, 
                                query_vector=qvec, 
                                limit=k, 
                                with_payload=True
                            )
                    elif hasattr(self.store.client, "search_points"):
                        hits = self.store.client.search_points(collection_name=self.store.collection_name, query_vector=qvec, limit=k, with_payload=True)
                    else:
                        logging.warning("Qdrant client has no supported search method; attempting HTTP fallback")
                        # Try HTTP fallback against the Qdrant REST API.
                        try:
                            base = getattr(self.store, "qdrant_url", None) or "http://localhost:6333"
                            url = f"{base.rstrip('/')}/collections/{self.store.collection_name}/points/search"
                            payload = {"vector": qvec, "limit": k, "with_payload": True}
                            resp = requests.post(url, json=payload, timeout=10)
                            resp.raise_for_status()
                            j = resp.json()
                            hits = j.get("result", [])
                            # HTTP response shape is sometimes a list of hits directly under 'result'
                            if isinstance(hits, dict) and 'hits' in hits:
                                hits = hits['hits']
                        except Exception:
                            logging.exception("HTTP fallback to Qdrant search failed")
                            return []
                except Exception:
                    logging.exception("Qdrant search failed")
                    return []

                docs: List[Document] = []
                if not hits:
                    return docs

                for h in hits:
                    # Normalize payload extraction across different client versions
                    payload = None
                    # object with 'payload' attribute (newer client)
                    if hasattr(h, "payload") and h.payload is not None:
                        payload = getattr(h, "payload")
                    # some versions return {'payload': {...}, 'id': ...}
                    elif isinstance(h, dict) and "payload" in h and h["payload"] is not None:
                        payload = h.get("payload")
                    
                    if not payload:
                        payload = {}

                    text = None
                    if isinstance(payload, dict):
                        # 💡 CRITICAL FIX: Ensure extraction priority matches indexing key 'text'
                        text = payload.get("text") or payload.get("content") or payload.get("page_content")
                    
                    # Fallback to string conversion if no text key is found (less ideal)
                    if text is None:
                        text = str(payload)

                    # For now, we ignore metadata but can be added if needed
                    docs.append(Document(page_content=text or ""))

                return docs

        return Retriever(self)


def load_pdf(path: str | Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    return loader.load()


def build_qdrant_vectorstore_from_pdf(path: str | Path) -> QdrantVectorStore:
    """Load a PDF, split into chunks, and index in a Qdrant vector store."""
    docs = load_pdf(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HFEmbeddings()
    return QdrantVectorStore.from_documents(chunks, collection_name="pdf_collection", embeddings=embeddings)