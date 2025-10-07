from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fraud_scorer.parsers.document_router import DocumentIntakeRouter
from fraud_scorer.parsers.processing_hint import ProcessingHint


@dataclass
class DummyExtractor:
    called: bool = False
    SUPPORTED_EXT = {".csv"}

    def extract(self, path: Path, hint: Optional[ProcessingHint] = None):
        self.called = True
        return {
            "text": "",
            "tables": [],
            "key_value_pairs": {},
            "metadata": {
                "gps_direct": {"enabled": True, "hint": hint.as_dict() if hint else None}
            },
        }


def test_document_router_invokes_direct_extractor_for_confident_hint(tmp_path, monkeypatch):
    dummy = DummyExtractor()
    router = DocumentIntakeRouter(gps_extractor=dummy)

    file_path = tmp_path / "telemetria_gps.csv"
    file_path.write_text("lat,long\n1,2\n", encoding="utf-8")

    hint = ProcessingHint(
        file_name=file_path.name,
        file_extension=".csv",
        mime_type="text/csv",
        file_size_bytes=file_path.stat().st_size,
        is_gps_candidate=True,
        confidence=0.9,
        manual_override=False,
        reason="test",
        detector_version="test",
    )

    fallback_called = {"value": False}

    def fallback(_: Path):  # pragma: no cover - only used when router fails
        fallback_called["value"] = True
        return None

    result = router.route(file_path, fallback, hint=hint)

    assert dummy.called is True
    assert fallback_called["value"] is False
    assert result is not None
    assert result["metadata"]["gps_direct"]["enabled"] is True
