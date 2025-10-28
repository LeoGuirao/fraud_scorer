#!/usr/bin/env python3
"""
Reprocesa el OCR de un único documento y actualiza el caché/DB del caso.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reprocesa un documento usando Azure OCR y actualiza el caché del caso."
    )
    parser.add_argument(
        "--case",
        required=True,
        help="ID del caso destino (ej. CASE-2025-0001).",
    )
    parser.add_argument(
        "document",
        type=Path,
        help="Ruta al archivo PDF/imagen que reemplazará el OCR existente.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = args.document.expanduser()

    if not pdf_path.exists():
        print(f"[ERROR] No se encontró el archivo: {pdf_path}", file=sys.stderr)
        return 1

    _bootstrap_src_path()

    from fraud_scorer.parsers.document_parser import DocumentParser
    from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
    from fraud_scorer.storage.ocr_cache import OCRCacheManager

    ocr_processor = AzureOCRProcessor()
    parser = DocumentParser(ocr_processor)

    print(f"[INFO] Procesando documento con OCR: {pdf_path}")
    parsed = parser.parse_document(pdf_path)
    if not parsed:
        print("[ERROR] No se obtuvo resultado OCR; abortando.", file=sys.stderr)
        return 2

    print(f"[INFO] Guardando OCR en el cache del caso {args.case}…")
    cache = OCRCacheManager()
    cache.save_cache(pdf_path, parsed, case_id=args.case)

    print("[OK] OCR actualizado correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
