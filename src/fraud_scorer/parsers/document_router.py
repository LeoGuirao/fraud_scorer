"""Routing layer that decides whether to use OCR or the GPS direct extractor."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from fraud_scorer.parsers.gps_direct_extractor import GPSDirectExtractor
from fraud_scorer.parsers.processing_hint import (
    ProcessingHint,
    ProcessingHintBuilder,
    is_gps_direct_enabled,
)
from fraud_scorer.parsers.types import ParsedDocument


logger = logging.getLogger(__name__)


class DocumentIntakeRouter:
    """Determines the best pipeline for a document before invoking the parser."""

    def __init__(
        self,
        *,
        gps_extractor: Optional[GPSDirectExtractor] = None,
        hint_builder: Optional[ProcessingHintBuilder] = None,
        confidence_threshold: float = 0.45,
    ) -> None:
        self.gps_extractor = gps_extractor or GPSDirectExtractor()
        self.hint_builder = hint_builder or ProcessingHintBuilder()
        self.confidence_threshold = confidence_threshold

    def route(
        self,
        document_path: Path,
        fallback: Callable[[Path], Optional[ParsedDocument]],
        *,
        hint: Optional[ProcessingHint] = None,
    ) -> Optional[ParsedDocument]:
        if not is_gps_direct_enabled():
            return fallback(document_path)

        effective_hint = hint or self.hint_builder.build(document_path)
        if not self._eligible_for_gps(document_path, effective_hint):
            return fallback(document_path)

        try:
            parsed = self.gps_extractor.extract(document_path, hint=effective_hint)
            logger.info(
                "GPSDirectExtractor utilizado para %s (confianza=%.2f, motivo=%s)",
                document_path.name,
                effective_hint.confidence,
                effective_hint.reason or "n/a",
            )
            return parsed
        except Exception as exc:
            logger.warning(
                "Ingesta directa falló para %s; regresando a OCR. Error: %s",
                document_path.name,
                exc,
            )
            return fallback(document_path)

    # ----------------------
    # Heurísticas de elegibilidad
    # ----------------------
    def _eligible_for_gps(self, path: Path, hint: ProcessingHint) -> bool:
        if hint.manual_override:
            return True
        if not hint.is_gps_candidate:
            return False
        if hint.confidence < self.confidence_threshold:
            return False

        ext = hint.file_extension or path.suffix.lower()
        if ext not in self.gps_extractor.SUPPORTED_EXT:
            return False

        max_file_mb = float(os.getenv("GPS_MAX_FILE_MB", "500"))
        if hint.file_size_bytes and hint.file_size_bytes > max_file_mb * 1024 * 1024:
            logger.debug(
                "Archivo %s excede límite de tamaño para GPS directo (%.2f MB)",
                path.name,
                hint.file_size_bytes / 1024 / 1024,
            )
            return False

        return True


__all__ = ["DocumentIntakeRouter"]

