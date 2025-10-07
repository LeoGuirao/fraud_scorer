from __future__ import annotations

from pathlib import Path

from fraud_scorer.parsers.processing_hint import ProcessingHintBuilder


def test_processing_hint_with_gps_keyword(tmp_path):
    file_path = tmp_path / "reporte_monitoreo_gps.csv"
    file_path.write_text("lat,long\n0,0\n", encoding="utf-8")

    builder = ProcessingHintBuilder()
    hint = builder.build(file_path)

    assert hint.is_gps_candidate is True
    assert hint.confidence >= 0.45
    assert "keyword" in (hint.reason or "")


def test_processing_hint_for_generic_file(tmp_path):
    file_path = tmp_path / "documento_generico.txt"
    file_path.write_text("contenido", encoding="utf-8")

    builder = ProcessingHintBuilder()
    hint = builder.build(file_path)

    assert hint.is_gps_candidate is False
    assert hint.confidence < 0.45
