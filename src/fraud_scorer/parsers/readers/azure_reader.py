# src/fraud_scorer/parsers/readers/azure_reader.py
"""
Adapter de Azure OCR a la salida unificada de parsers.

Este lector envuelve a `AzureOCRProcessor` y normaliza su `OCRResult`
al formato estándar definido en `fraud_scorer.parsers.types`:
{
  "text": str,
  "tables": List[Table],
  "key_value_pairs": Dict[str, Any],
  "metadata": DocumentMetadata
}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fraud_scorer.parsers.types import DocumentReader, ParsedDocument
from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor, OCRResult

logger = logging.getLogger(__name__)


class AzureOCRReader(DocumentReader):
    """
    Lector/adapter que usa Azure Document Intelligence para OCR
    y devuelve una estructura normalizada para el pipeline.
    """

    def __init__(self, ocr: AzureOCRProcessor):
        self.ocr = ocr

    def read(self, path: Path) -> ParsedDocument:
        """
        Ejecuta OCR sobre `path` y lo normaliza al formato unificado.
        """
        # Llamada síncrona (el processor ya gestiona errores y devuelve OCRResult consistente)
        logger.debug(f"AzureOCRReader: analizando documento con Azure OCR → {path.name}")
        res: Optional[OCRResult] = self.ocr.analyze_document(str(path))

        # Salvaguardas por si el proveedor retorna None (no debería, pero mejor blindar)
        if res is None:
            logger.warning(f"AzureOCRReader: resultado None recibido desde AzureOCRProcessor para {path.name}")
            return {
                "text": "",
                "tables": [],
                "key_value_pairs": {},
                "metadata": {
                    "source_type": "ocr",
                    "file_name": path.name,
                    "engine": "azure_document_intelligence",
                    "success": False,
                    "errors": [f"Proveedor OCR devolvió resultado vacío para {path.name}"],
                    "confidence": {"overall": 0.0},
                },
            }

        # Normalización a la salida unificada
        text = getattr(res, "text", "") or ""
        tables = getattr(res, "tables", None) or []
        key_values = getattr(res, "key_values", None) or {}
        confidence = getattr(res, "confidence", None) or {}
        errors = getattr(res, "errors", None) or []
        metadata_src = getattr(res, "metadata", None) or {}

        metadata = {
            **metadata_src,
            "source_type": "ocr",
            "engine": "azure_document_intelligence",
            "success": bool(getattr(res, "success", False)),
            "errors": errors,
            "confidence": confidence,
        }

        normalized: ParsedDocument = {
            "text": text,
            "tables": tables,
            "key_value_pairs": key_values,  # mapeo clave: key_values → key_value_pairs
            "metadata": metadata,
        }
        return normalized
