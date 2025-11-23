from __future__ import annotations

from pathlib import Path

from src.rag import HFEmbeddings


def test_hf_embeddings_shapes(tmp_path: Path):
    emb = HFEmbeddings()
    docs = ["hello world", "test document"]
    vectors = emb.embed_documents(docs)
    assert len(vectors) == len(docs)
    assert all(len(v) == len(vectors[0]) for v in vectors)


