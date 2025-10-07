from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_scorer.parsers.gps_direct_extractor import GPSDirectExtractor


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
