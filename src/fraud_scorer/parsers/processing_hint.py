"""Utility helpers to build processing hints for the document intake router.

The GPS direct ingestion flow relies on simple heuristics that determine if a
document can bypass OCR.  The heuristics combine:

* metadata provided by the upload UI (manual toggle, mime type, size)
* cheap filename checks (keywords usually present on GPS reports)
* a lightweight inspection for vector PDFs (< 1 MB) to confirm selectable text

This module centralises that logic so it can be reused by the API layer,
background jobs or batch migrations.  The end result is a ``ProcessingHint``
structure that the ``DocumentIntakeRouter`` can use to pick the appropriate
pipeline.

All logic here must stay cheap and side-effect free; heavier work belongs to
the ``GPSDirectExtractor``.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


DEFAULT_DETECTOR_VERSION = "2025.10"
DEFAULT_VECTOR_SAMPLE_LIMIT_MB = float(os.getenv("GPS_VECTOR_SAMPLE_LIMIT_MB", "1.0"))
GPS_KEYWORDS = (
    "gps",
    "tracking",
    "rastreo",
    "monitoreo",
    "geocerca",
    "geofence",
    "bitacora",
    "recorrido",
)


@dataclass(slots=True)
class ProcessingHint:
    """Shared hint that guides routing decisions."""

    file_name: str
    file_extension: str
    mime_type: str
    file_size_bytes: int
    manual_override: bool = False
    is_gps_candidate: bool = False
    confidence: float = 0.0
    detector_version: str = DEFAULT_DETECTOR_VERSION
    vector_ratio: float = 0.0
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProcessingHintBuilder:
    """Factory that inspects a document and produces a ``ProcessingHint``."""

    def __init__(
        self,
        detector_version: str = DEFAULT_DETECTOR_VERSION,
        vector_sample_limit_mb: float = DEFAULT_VECTOR_SAMPLE_LIMIT_MB,
    ) -> None:
        self.detector_version = detector_version
        self.vector_sample_limit_bytes = int(vector_sample_limit_mb * 1024 * 1024)

    def build(
        self,
        document_path: Path,
        *,
        manual_override: bool = False,
        mime_type: Optional[str] = None,
    ) -> ProcessingHint:
        mime = mime_type or mimetypes.guess_type(document_path.name)[0] or "application/octet-stream"
        ext = document_path.suffix.lower()
        size_bytes = self._safe_file_size(document_path)

        hint = ProcessingHint(
            file_name=document_path.name,
            file_extension=ext,
            mime_type=mime,
            file_size_bytes=size_bytes,
            manual_override=manual_override,
            detector_version=self.detector_version,
        )

        if manual_override:
            hint.is_gps_candidate = True
            hint.confidence = 1.0
            hint.reason = "manual-override"
            return hint

        keyword_boost = self._keyword_boost(document_path.stem)
        ext_boost = 0.0
        if ext in {".pdf", ".csv", ".xlsx", ".xls"}:
            ext_boost = 0.25

        confidence = keyword_boost + ext_boost
        reason_parts = []
        if keyword_boost:
            reason_parts.append("keyword")
        if ext_boost:
            reason_parts.append("extension")

        vector_ratio = 0.0
        if ext == ".pdf" and size_bytes <= self.vector_sample_limit_bytes:
            vector_ratio = self._probe_pdf_vector_ratio(document_path)
            hint.vector_ratio = vector_ratio
            if vector_ratio >= 0.2:
                confidence += 0.35
                reason_parts.append("vector-text")

        hint.confidence = min(confidence, 1.0)
        hint.is_gps_candidate = hint.confidence >= 0.45
        hint.reason = ",".join(reason_parts) if reason_parts else ""
        return hint

    # --------------------
    # Helper functionality
    # --------------------

    def _keyword_boost(self, name: str) -> float:
        lowered = name.lower()
        hits = sum(1 for kw in GPS_KEYWORDS if kw in lowered)
        if hits == 0:
            return 0.0
        return min(0.5, 0.2 + hits * 0.05)

    def _probe_pdf_vector_ratio(self, path: Path) -> float:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(path)
            if not doc.page_count:
                return 0.0
            text_chars = 0
            vector_pages = 0
            for page in doc:
                extracted = page.get_text("text") or ""
                text_chars += len(extracted)
                if extracted.strip():
                    vector_pages += 1
            doc.close()
            if text_chars == 0:
                return 0.0
            return min(1.0, vector_pages / max(1, doc.page_count))
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("No se pudo sondear PDF vectorial %s: %s", path.name, exc)
            return 0.0

    def _safe_file_size(self, path: Path) -> int:
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0
        except Exception as exc:  # pragma: no cover - filesystem edge cases
            logger.debug("No se pudo leer tamaño de %s: %s", path, exc)
            return 0


def serialise_hint(hint: ProcessingHint) -> str:
    """Serialises a hint in a stable JSON form (useful for metadata)."""

    return json.dumps(hint.as_dict(), ensure_ascii=False, sort_keys=True)


def is_gps_direct_enabled() -> bool:
    """Reads environment flags to determine if GPS direct routing is active."""

    env_value = os.getenv("GPS_DIRECT_ENABLED", "true").strip().lower()
    return env_value in {"1", "true", "yes", "on"}


__all__ = [
    "ProcessingHint",
    "ProcessingHintBuilder",
    "DEFAULT_DETECTOR_VERSION",
    "serialise_hint",
    "is_gps_direct_enabled",
]

