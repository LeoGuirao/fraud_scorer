
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import httpx
import openai
from langchain.schema import Document

from fraud_scorer.ai.config import RickAgentConfig
from fraud_scorer.ai.vector_store.manager import RickVectorStoreManager


class _StubEmbeddings:
    """Embeddings deterministas para pruebas offline."""

    def __init__(self, *, dimension: int = 6) -> None:
        self._dimension = dimension

    def embed_documents(self, texts):  # type: ignore[override]
        return [[float((idx + 1) % 7)] * self._dimension for idx, _ in enumerate(texts)]

    def embed_query(self, text):  # type: ignore[override]
        return [0.01] * self._dimension


class _RateLimitedEmbeddings(_StubEmbeddings):
    """Simula limitaciones de tasa reduciendo el tamaño de lote."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_history: List[int] = []
        self._raised = False

    def embed_documents(self, texts):  # type: ignore[override]
        self.batch_history.append(len(texts))
        if len(texts) > 1 and not self._raised:
            self._raised = True
            request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
            response = httpx.Response(status_code=429, headers={"retry-after": "0.1"}, request=request)
            raise openai.RateLimitError("rate limit", response=response, body=None)
        return super().embed_documents(texts)


def _make_config(tmp_path: Path, *, batch_size: int | None = None) -> RickAgentConfig:
    return RickAgentConfig(
        chroma_base_path=tmp_path,
        audit_log_path=tmp_path / "audit" / "agent_rick_audit.jsonl",
        embedding_batch_size=batch_size or RickAgentConfig.embedding_batch_size,
    )


def test_upsert_documents_persists_embeddings(tmp_path):
    config = _make_config(tmp_path)
    manager = RickVectorStoreManager(config=config, embeddings=_StubEmbeddings())

    docs = [
        Document(page_content="primer documento", metadata={"source_document": "doc1.json", "source": "ocr"}),
        Document(page_content="segundo documento", metadata={"source_document": "doc2.json", "source": "ocr"}),
    ]

    manager.upsert_documents("CASE-TEST-001", docs)

    case_dir = config.chroma_base_path / "CASE-TEST-001"
    assert case_dir.exists()

    manifest = case_dir / "index_manifest.json"
    assert manifest.exists()

    generated = [item for item in case_dir.iterdir() if item.name != "index_manifest.json"]
    assert generated, "Chroma no generó archivos persistidos"

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["vector_count"] == 2
    assert data["created_at"].endswith("Z")


def test_upsert_documents_recovers_from_rate_limit(tmp_path):
    config = _make_config(tmp_path, batch_size=8)
    embeddings = _RateLimitedEmbeddings()
    manager = RickVectorStoreManager(config=config, embeddings=embeddings)

    docs = [
        Document(page_content=f"documento {idx}", metadata={"source_document": f"doc{idx}.json", "source": "ocr"})
        for idx in range(3)
    ]

    manager.upsert_documents("CASE-RATE-001", docs)

    case_dir = config.chroma_base_path / "CASE-RATE-001"
    assert (case_dir / "index_manifest.json").exists()

    assert embeddings.batch_history[0] == 3
    assert embeddings.batch_history.count(1) >= 3


def test_upsert_documents_is_idempotent(tmp_path):
    config = _make_config(tmp_path)
    manager = RickVectorStoreManager(config=config, embeddings=_StubEmbeddings())

    docs = [
        Document(page_content="contenido", metadata={"source_document": "doc.json", "source": "ocr"}),
        Document(page_content="contenido extra", metadata={"source_document": "doc2.json", "source": "ocr"}),
    ]

    manager.upsert_documents("CASE-IDEMP-001", docs)
    manager.upsert_documents("CASE-IDEMP-001", docs)

    case_dir = config.chroma_base_path / "CASE-IDEMP-001"
    manifest = case_dir / "index_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["vector_count"] == 2
