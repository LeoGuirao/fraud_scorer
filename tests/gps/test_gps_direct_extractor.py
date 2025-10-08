from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_scorer.parsers.gps_direct_extractor import (
    GPSDirectExtractor,
    extract_tables_from_text_segments,
    normalize_gps_tables,
)


def _write_csv(path: Path, rows: int) -> None:
    header = "timestamp,lat,long,speed,event\n"
    lines = [header]
    for idx in range(rows):
        lines.append(f"2025-01-01T00:{idx:02d}:00Z,19.{idx},-99.{idx},50,OK\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_csv_extractor_chunking(tmp_path):
    csv_path = tmp_path / "gps_chunk.csv"
    _write_csv(csv_path, rows=5)

    extractor = GPSDirectExtractor(csv_chunk_rows=2, excel_chunk_rows=5)
    parsed = extractor.extract(csv_path)

    tables = parsed["tables"]
    assert len(tables) == 3

    metadata = parsed["metadata"].get("gps_direct", {})
    assert metadata.get("chunk_count") == 3
    assert metadata.get("normalized_row_count") == 5


def test_excel_extractor_chunking(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": [f"2025-01-01T0{i}:00:00Z" for i in range(5)],
            "latitude": [19.3 + i * 0.01 for i in range(5)],
            "longitude": [-99.1 - i * 0.01 for i in range(5)],
            "speed": [20 + i for i in range(5)],
        }
    )
    excel_path = tmp_path / "gps_chunk.xlsx"
    df.to_excel(excel_path, index=False)

    extractor = GPSDirectExtractor(csv_chunk_rows=5, excel_chunk_rows=2)
    parsed = extractor.extract(excel_path)

    tables = parsed["tables"]
    assert len(tables) == 3

    metadata = parsed["metadata"].get("gps_direct", {})
    assert metadata.get("chunk_count") == 3
    assert metadata.get("normalized_row_count") == 5


def test_regex_fallback_extracts_rows_from_text_segment():
    text_segment = """
    1ECO 010 351777096022718R83 3318346270 Encendido 2024-02-12 00:00:05 0Noreste(Dirección48) GPS 8Verdadero 22.463684/-97.850000
    1ECO 010 351777096022718R83 3318346270 Apagado 2024-02-12 00:05:10 12Sur(Dirección180) GPS 9Verdadero 22.463900/-97.851100
    """
    tables = extract_tables_from_text_segments([text_segment])
    assert tables
    assert tables[0]["row_count"] == 2

    df, warnings = normalize_gps_tables(tables)
    assert df.shape[0] == 2
    assert df["event_label"].iloc[0] == "Encendido"
    assert df["latitude"].notna().all()
    assert "missing_timestamps" not in warnings


def test_regex_fallback_returns_empty_when_no_coordinates():
    text_segment = "Este documento no contiene registros GPS válidos."
    tables = extract_tables_from_text_segments([text_segment])
    assert tables == []


def test_regex_fallback_reconstructs_multiline_record():
    text_segment = """
    1ECO 010 351777096022718R83 3318346270
    Encendido
    2024-02-12 00:00:05
    0
    Noreste(Dirección48)
    GPS 8
    Verdadero
    22.463684/-97.850000

    1ECO 010 351777096022718R83 3318346270
    Apagado
    2024-02-12 00:05:10
    12
    Sur(Dirección180)
    GPS 9
    Verdadero
    22.463900/-97.851100
    """
    tables = extract_tables_from_text_segments([text_segment])
    assert tables
    assert tables[0]["row_count"] == 2

    df, _ = normalize_gps_tables(tables)
    assert df.shape[0] == 2
    assert set(df["event_label"]) == {"Encendido", "Apagado"}
