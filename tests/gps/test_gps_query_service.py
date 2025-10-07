from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fraud_scorer.services.gps_query_service import GPSDirectQueryService
from fraud_scorer.storage.gps_cache import GPSCacheManager


def _persist_sample(manager: GPSCacheManager, case_id: str, document_name: str) -> None:
    parsed_document = {
        "text": "timestamp,lat,long,speed,event\n",
        "tables": [
            {
                "headers": ["timestamp", "lat", "long", "speed", "event"],
                "data_rows": [
                    ["2025-01-01T00:00:00Z", "19.30", "-99.10", "80", "OK"],
                    ["2025-01-01T01:30:00Z", "19.35", "-99.15", "10", "STOP"],
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
    manager.persist_direct_output(case_id, Path(document_name), parsed_document)


def test_gps_query_service_filters(tmp_path):
    base_dir = tmp_path / "gps"
    manager = GPSCacheManager(base_dir=base_dir)
    case_id = "CASE-GPS-001"
    document_name = "reporte_gps.csv"

    _persist_sample(manager, case_id, document_name)

    service = GPSDirectQueryService(cache_manager=manager)

    docs = service.list_documents(case_id)
    assert docs and docs[0]["dataset"]["row_count"] == 2

    result = service.query_dataset(
        case_id,
        document_name,
        start_time=datetime.fromisoformat("2025-01-01T01:00:00+00:00"),
        event_labels=["STOP"],
    )

    assert result["row_count"] == 1
    assert result["preview"][0]["event_label"] == "STOP"

    bbox_result = service.query_dataset(
        case_id,
        document_name,
        bounding_box={"min_lat": 19.0, "max_lat": 19.32, "min_lon": -99.2, "max_lon": -99.0},
    )
    assert bbox_result["row_count"] == 1


def test_gps_query_service_segmentation_and_anomalies(tmp_path):
    base_dir = tmp_path / "gps"
    manager = GPSCacheManager(base_dir=base_dir)
    case_id = "CASE-GPS-SEG"
    document_name = "gps_seg.csv"

    _persist_sample(manager, case_id, document_name)

    service = GPSDirectQueryService(cache_manager=manager)

    segments = service.segment_dataset(
        case_id,
        document_name,
        frequency="60min",
        metrics=("row_count", "avg_speed", "event_distribution"),
    )

    assert segments["frequency"] == "60min"
    assert segments["segments"]
    assert segments["segments"][0]["row_count"] >= 1

    anomalies = service.detect_anomalies(
        case_id,
        document_name,
        gap_threshold_minutes=60,
        speed_threshold=70,
    )

    anomaly_types = {item["type"] for item in anomalies["anomalies"]}
    assert "time_gap" in anomaly_types
    assert "high_speed" in anomaly_types
