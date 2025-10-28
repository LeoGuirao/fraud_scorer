from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from langchain.schema import Document

from fraud_scorer.ai.retrieval.retriever import RetrievedDocument
from fraud_scorer.ai.tools import build_rag_search_tool, build_gps_query_tool
from fraud_scorer.ai.tools.rag_tool import _append_denuncia_highlights
from fraud_scorer.ai.orchestration.agente_rick import AgenteRickService
from fraud_scorer.services.gps_query_service import GPSDirectQueryService
from fraud_scorer.storage.gps_cache import GPSCacheManager


class _FakeRetriever:
    def __init__(self, documents: List[RetrievedDocument]) -> None:
        self._documents = documents
        self.calls = 0
        self.last_filter = None

    def retrieve(
        self,
        case_id: str,
        query: str,
        *,
        k: int | None = None,
        metadata_filter: Dict[str, Any] | None = None,
    ) -> List[RetrievedDocument]:  # pragma: no cover - simple mock
        self.calls += 1
        self.last_filter = metadata_filter
        filtered = self._documents
        if metadata_filter and "document_type" in metadata_filter:
            expected = metadata_filter["document_type"]
            filtered = [
                doc for doc in filtered if doc.document.metadata.get("document_type") == expected
            ]
        return filtered[: k or len(filtered)]


def _persist_sample(tmp_path: Path) -> tuple[GPSDirectQueryService, str, str]:
    manager = GPSCacheManager(base_dir=tmp_path / "gps")
    case_id = "CASE-AGENT"
    document_name = "gps_sample.csv"
    parsed_document = {
        "text": "timestamp,lat,long,speed,event\n",
        "tables": [
            {
                "headers": ["timestamp", "latitude", "longitude", "speed", "event_label"],
                "data_rows": [
                    ["2025-01-01T00:00:00Z", "19.1", "-99.1", "70", "OK"],
                    ["2025-01-01T01:30:00Z", "19.3", "-99.3", "95", "STOP"],
                ],
                "gps_plugin": "csv",
            }
        ],
        "metadata": {
            "file_name": document_name,
            "gps_direct": {"enabled": True, "schema_version": 1},
        },
        "gps_summary": {},
    }
    manager.persist_direct_output(case_id, tmp_path / document_name, parsed_document)
    service = GPSDirectQueryService(cache_manager=manager)
    return service, case_id, document_name


def test_rag_tool_returns_context_and_triggers_callback():
    doc = Document(page_content="Contenido relevante", metadata={"source_document": "denuncia.pdf", "document_type": "denuncia"})
    retrieved = RetrievedDocument(document=doc, similarity=0.92)
    retriever = _FakeRetriever([retrieved])

    captured: list[List[RetrievedDocument]] = []

    tool = build_rag_search_tool(retriever=retriever, max_results=5, on_results=lambda docs: captured.append(list(docs)))

    response = tool.func(case_id="CASE-TEST", query="ubicación del robo")

    assert "denuncia.pdf" in response
    assert "Contenido relevante" in response
    assert captured and captured[0][0].similarity == pytest.approx(0.92)
    assert retriever.calls == 1


def test_rag_tool_applies_document_type_filter():
    doc_a = Document(page_content="Texto A", metadata={"source_document": "doc_a.pdf", "document_type": "denuncia_de_los_hechos"})
    doc_b = Document(page_content="Texto B", metadata={"source_document": "doc_b.pdf", "document_type": "oficio_denuncia"})
    retrieved = [
        RetrievedDocument(document=doc_a, similarity=0.9),
        RetrievedDocument(document=doc_b, similarity=0.85),
    ]
    retriever = _FakeRetriever(retrieved)

    tool = build_rag_search_tool(retriever=retriever, max_results=5)
    response = tool.func(
        case_id="CASE-TEST",
        query="ubicación del siniestro",
        document_type="denuncia_de_los_hechos",
    )

    assert "doc_a.pdf" in response
    assert "doc_b.pdf" not in response
    assert retriever.last_filter == {"document_type": "denuncia_de_los_hechos"}


def test_rag_tool_validates_empty_inputs():
    retriever = _FakeRetriever([])
    tool = build_rag_search_tool(retriever=retriever, max_results=3)
    with pytest.raises(ValueError):
        tool.func(case_id="", query="algo")
    with pytest.raises(ValueError):
        tool.func(case_id="CASE", query="")
    with pytest.raises(ValueError):
        tool.func(case_id="CASE", query="x" * 601)


def test_gps_tool_returns_serialized_payload(tmp_path: Path):
    service, case_id, document_name = _persist_sample(tmp_path)
    captured: list[dict] = []

    tool = build_gps_query_tool(service=service, default_limit=5, on_result=lambda payload: captured.append(payload))
    response = tool.func(case_id=case_id, document_name=document_name, timestamp="2025-01-01T00:00:00Z")

    payload = json.loads(response)
    assert payload["row_count"] >= 1
    assert payload["preview"]
    assert captured and captured[0]["document_name"] == document_name


def test_gps_tool_rejects_conflicting_times(tmp_path: Path):
    service, case_id, document_name = _persist_sample(tmp_path)
    tool = build_gps_query_tool(service=service)
    with pytest.raises(ValueError):
        tool.func(
            case_id=case_id,
            document_name=document_name,
            timestamp="2025-01-01T00:00:00Z",
            start_time="2025-01-01T00:00:00Z",
        )


def test_gps_tool_distance_stats(tmp_path: Path):
    service, case_id, document_name = _persist_sample(tmp_path)
    tool = build_gps_query_tool(service=service, default_limit=5)

    response = tool.func(
        case_id=case_id,
        document_name=document_name,
        timestamp="2025-01-01T00:00:00Z",
        reference_point={"lat": 19.1, "lon": -99.1},
    )

    payload = json.loads(response)
    assert payload.get("reference_point") == {"lat": 19.1, "lon": -99.1}
    assert payload.get("distance_stats")
    assert "distance_km" in payload["preview"][0]


def test_gps_tool_validates_bounding_box(tmp_path: Path):
    service, case_id, document_name = _persist_sample(tmp_path)
    tool = build_gps_query_tool(service=service)
    with pytest.raises(ValueError):
        tool.func(
            case_id=case_id,
            document_name=document_name,
            bounding_box={"min_lat": 10.0, "max_lat": -5.0, "min_lon": -100.0, "max_lon": -99.0},
        )


def test_gps_hint_reminds_query_tool_use():
    service = object.__new__(AgenteRickService)
    service._gps_service = type(
        "StubGPSService",
        (),
        {"list_documents": staticmethod(lambda case_id: [{"document_name": "gps_sample.csv"}])},
    )()
    hint = AgenteRickService._get_gps_documents_hint(service, "CASE-123")
    assert "query_gps_location" in hint
    assert "gps_sample.csv" in hint


def test_append_highlights_handles_json_string():
    meta = {
        "document_type": "denuncia_de_los_hechos",
        "structured_highlights": json.dumps(
            {
                "time_ranges": ["19:30"],
                "vehicles": ["ABC123"],
                "reference_point": {"lat": 22.5, "lon": -100.6},
            }
        ),
    }
    text = "Texto base de la denuncia."
    enriched = _append_denuncia_highlights(text, meta)
    assert "Horas declaradas" in enriched
    assert "ABC123" in enriched
