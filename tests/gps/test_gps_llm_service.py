from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from fraud_scorer.services.gps_llm_service import GPSLLMQueryService
from fraud_scorer.services.gps_query_service import GPSDirectQueryService
from fraud_scorer.storage.gps_cache import GPSCacheManager


class _FakeLLM:
    def __init__(self, content: str = "Resumen sintetizado.") -> None:
        self.content = content

    def invoke(self, messages):  # pylint: disable=unused-argument
        class _Response:
            def __init__(self, content: str) -> None:
                self.content = content
                self.response_metadata = {"token_usage": {"prompt_tokens": 120, "completion_tokens": 80}}

        return _Response(self.content)


def _persist_sample(tmp_path: Path) -> tuple[GPSDirectQueryService, str, str]:
    manager = GPSCacheManager(base_dir=tmp_path / "gps")
    case_id = "CASE-LLM"
    document_name = "gps_llm.csv"
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


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("GPS_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("GPS_COST_THRESHOLD", "1.0")
    monkeypatch.setenv("GPS_LLM_PREVIEW_ROWS", "50")
    monkeypatch.setenv("GPS_LLM_MAX_CONTEXT_CHARS", "4000")
    yield
    monkeypatch.delenv("GPS_LLM_MODEL", raising=False)
    monkeypatch.delenv("GPS_COST_THRESHOLD", raising=False)
    monkeypatch.delenv("GPS_LLM_PREVIEW_ROWS", raising=False)
    monkeypatch.delenv("GPS_LLM_MAX_CONTEXT_CHARS", raising=False)


def test_gps_llm_service_generates_summary(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GPS_LLM_ENABLED", "true")
    query_service, case_id, document_name = _persist_sample(tmp_path)

    service = GPSLLMQueryService(query_service=query_service, llm=_FakeLLM())
    result = service.summarise_route(case_id=case_id, document_name=document_name)

    assert result["answer"].startswith("Resumen")
    assert result["model"] == os.getenv("GPS_LLM_MODEL")
    assert "estimated_cost" in result
    assert result["preview_rows"]


def test_gps_llm_service_respects_cost_threshold(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GPS_LLM_ENABLED", "true")
    monkeypatch.setenv("GPS_COST_THRESHOLD", "0.0001")
    query_service, case_id, document_name = _persist_sample(tmp_path)

    service = GPSLLMQueryService(query_service=query_service, llm=_FakeLLM())
    result = service.summarise_route(case_id=case_id, document_name=document_name)

    assert result.get("skipped") is True
    assert result.get("reason") == "estimated_cost_exceeds_threshold"


def test_gps_llm_service_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GPS_LLM_ENABLED", "false")
    service = GPSLLMQueryService(enabled=False)
    response = service.summarise_route(case_id="CASE", document_name="gps.csv")
    assert response["enabled"] is False
