from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_scorer.parsers.gps_direct_extractor import RAW_PAYLOAD_COLUMN
from fraud_scorer.storage.gps_cache import GPSCacheManager


def test_gps_cache_manager_persist_and_load(tmp_path):
    base_dir = tmp_path / "gps"
    manager = GPSCacheManager(base_dir=base_dir)

    document_path = tmp_path / "reporte_gps.csv"
    document_path.write_text("lat,long\n1,2\n", encoding="utf-8")

    parsed_document = {
        "text": "lat,long\\n1,2",
        "tables": [
            {
                "headers": ["lat", "long"],
                "data_rows": [["1", "2"]],
                "gps_plugin": "csv",
            }
        ],
        "metadata": {
            "file_name": "reporte_gps.csv",
            "gps_direct": {"enabled": True, "schema_version": 1},
        },
        "gps_summary": {"tables_extracted": 1},
    }

    manager.persist_direct_output("CASE-TEST", document_path, parsed_document)

    manifest = manager.get_manifest("CASE-TEST")
    assert "reporte_gps.csv" in manifest
    entry = manifest["reporte_gps.csv"]
    assert entry["dataset"]["row_count"] == 1
    assert entry.get("chunk_count") == 1

    df = manager.load_table("CASE-TEST", "reporte_gps.csv", 1)
    assert isinstance(df, pd.DataFrame)
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    assert float(df.iloc[0]["latitude"]) == 1.0
    assert float(df.iloc[0]["longitude"]) == 2.0

    full_df = manager.load_dataset("CASE-TEST", "reporte_gps.csv")
    assert full_df.shape[0] == 1
    assert RAW_PAYLOAD_COLUMN in full_df.columns

    text = manager.load_text("CASE-TEST", "reporte_gps.csv")
    assert "lat,long" in text

    metrics = manager.collect_global_metrics(top_documents=1, recent_limit=1)
    totals = metrics["totals"]
    assert totals["documents"] == 1
    assert totals["chunked_documents"] in {0, 1}
