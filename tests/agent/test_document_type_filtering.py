from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List

import pytest
from langchain.schema import Document

from fraud_scorer.ai.config import RickAgentConfig
from fraud_scorer.ai.retrieval.retriever import RickRetriever, RetrievedDocument
from fraud_scorer.ai.ingestion.document_loader import FraudCaseDocumentLoader


class _FakeCollection:
    def __init__(self, size: int) -> None:
        self._size = size

    def count(self) -> int:  # pragma: no cover - trivial getter
        return self._size


class _FakeChroma:
    def __init__(self, documents: List[Document]) -> None:
        self._documents = documents
        self._collection = _FakeCollection(len(documents))

    def similarity_search_with_score(self, query: str, k: int, filter: Dict[str, Any] | None = None) -> List[tuple[Document, float]]:
        # Devuelve todos los documentos sin respetar el filtro para verificar que el retriever lo aplique.
        scored: List[tuple[Document, float]] = []
        for idx, doc in enumerate(self._documents):
            # Distancia artificial: documentos más bajos en la lista son menos similares.
            distance = max(0.0, 1.0 - (0.2 * (len(self._documents) - idx)))
            scored.append((doc, distance))
        return scored[:k]

    def get(self, include: Iterable[str]):  # pragma: no cover - simple serialización
        payload: Dict[str, Any] = {}
        if "documents" in include:
            payload["documents"] = [doc.page_content for doc in self._documents]
        if "metadatas" in include:
            payload["metadatas"] = [dict(doc.metadata or {}) for doc in self._documents]
        if "ids" in include:
            payload["ids"] = [
                doc.metadata.get("vector_id", f"vec-{idx}") for idx, doc in enumerate(self._documents)
            ]
        if "embeddings" in include:
            payload["embeddings"] = [
                [float(idx + 1), float((idx + 1) * 2)] for idx in range(len(self._documents))
            ]
        return payload


class _FakeVectorStoreManager:
    def __init__(self, store: _FakeChroma) -> None:
        self._store = store

    def load_store(self, case_id: str) -> _FakeChroma:  # pragma: no cover - simple proxy
        return self._store


def _build_retriever(documents: List[Document]) -> RickRetriever:
    base_config = RickAgentConfig()
    config = replace(
        base_config,
        max_results=5,
        lexical_top_k=5,
        search_type="hybrid",
        similarity_threshold=0.0,
    )
    store = _FakeChroma(documents)
    manager = _FakeVectorStoreManager(store)
    return RickRetriever(config=config, vector_manager=manager)


def _document(content: str, *, doc_type: str, vector_id: str) -> Document:
    metadata = {
        "document_type": doc_type,
        "source_document": f"{doc_type}.pdf",
        "vector_id": vector_id,
    }
    return Document(page_content=content, metadata=metadata)


def test_retriever_filters_dense_and_lexical_results():
    doc_a = _document("El siniestro ocurrió en el kilómetro 57.", doc_type="denuncia_de_los_hechos", vector_id="a")
    # Documento B no coincide con el filtro pero contiene las mismas palabras clave para obligar al BM25 a considerar su puntaje.
    doc_b = _document("El siniestro ocurrió en el kilómetro 57, según el Oficio.", doc_type="oficio_denuncia", vector_id="b")
    retriever = _build_retriever([doc_a, doc_b])

    results = retriever.retrieve(
        "CASE-TEST",
        "¿Dónde ocurrió el siniestro?",
        metadata_filter={"document_type": "denuncia_de_los_hechos"},
    )

    assert len(results) == 1, "El retriever debe devolver solo documentos que coincidan con el filtro."
    assert results[0].document.metadata["document_type"] == "denuncia_de_los_hechos"

    # Sin filtro se deben recuperar ambos documentos.
    all_results = retriever.retrieve("CASE-TEST", "¿Dónde ocurrió el siniestro?", metadata_filter=None)
    assert {item.document.metadata["document_type"] for item in all_results} == {
        "denuncia_de_los_hechos",
        "oficio_denuncia",
    }


def test_retriever_returns_results_when_dense_candidates_are_filtered_out():
    # Diseñamos el resultado denso para que solo el documento B aparezca, pero el filtro debe impedir que llegue al ranking final.
    doc_a = _document("Denuncia: ubicación exacta del siniestro.", doc_type="denuncia_de_los_hechos", vector_id="a")
    doc_b = _document("Oficio de denuncia con ubicación.", doc_type="oficio_denuncia", vector_id="b")
    retriever = _build_retriever([doc_b, doc_a])  # Orden inverso para que la búsqueda densa priorice doc_b.

    filtered = retriever.retrieve(
        "CASE-TEST",
        "ubicación siniestro",
        metadata_filter={"document_type": "denuncia_de_los_hechos"},
    )

    assert filtered, "Incluso sin candidatos densos válidos, el BM25 debería aportar resultados filtrados."
    assert all(item.document.metadata["document_type"] == "denuncia_de_los_hechos" for item in filtered)


class _StubCacheManager:
    def __init__(self) -> None:
        self._folder = Path("/tmp/fake_case_folder")

    def _sanitize_filename(self, name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")

    def get_case_folder_path(self, case_id: str, case_index: Dict[str, Any]) -> Path:
        return self._folder

    def get_cache(self, document_path: Path, case_id: str | None = None) -> Dict[str, Any]:
        return {
            "text": (
                "Siendo las 19:30 horas del día 13 de febrero de 2024 en el kilómetro 57 de Matehuala, "
                "el operador reportó el asalto ocurrido momentos después de partir a las 19:00 horas."
            )
        }


def test_document_type_index_adds_ocr_aliases():
    loader = FraudCaseDocumentLoader(cache_manager=_StubCacheManager(), config=RickAgentConfig())
    case_index = {
        "classified_types": [
            {"filename": "3 DENUNCIA.pdf", "document_type": "denuncia_de_los_hechos"},
        ],
        "documents": [
            "data/ocr_cache/CASE/3_DENUNCIA/ocr_results_for_3_DENUNCIA.pdf.json",
        ],
        "consolidated_data": {"consolidated_fields": {}},
    }

    doc_type_map = loader._build_document_type_index(case_index)
    assert doc_type_map["ocr_results_for_3_DENUNCIA.pdf.json"] == "denuncia_de_los_hechos"

    documents = loader._build_ocr_documents(
        case_id="CASE-ALIAS",
        case_index=case_index,
        document_type_map=doc_type_map,
        document_hashes={case_index["documents"][0]: "hash"},
        base_metadata={"case_id": "CASE-ALIAS"},
    )

    assert documents
    metadata = documents[0].metadata
    assert metadata["document_type"] == "denuncia_de_los_hechos"
    assert metadata["source_document"] == "3 DENUNCIA.pdf"
    assert metadata["content_category"] == "denuncia_narrative"
    assert metadata["contains_temporal_info"] is True
    assert metadata["narrative_priority"] == 10
    highlights = metadata.get("structured_highlights") or {}
    assert "13 de febrero de 2024" in (", ".join(highlights.get("dates", [])) or "")
    assert "19:30" in (", ".join(highlights.get("time_ranges", [])) or "")

    summary_docs = loader._build_denuncia_highlights(case_index, {"case_id": "CASE-ALIAS"})
    assert summary_docs, "Se esperaba un resumen estructurado de la denuncia"
    summary_text = summary_docs[0].page_content
    assert "Horas mencionadas en narrativa:" in summary_text
    assert "19:30" in summary_text
    assert "Fecha mencionada en narrativa: 13 de febrero de 2024" in summary_text
