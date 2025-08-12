from __future__ import annotations
from typing import Optional, Dict, Any
from .db import (
    upsert_document, get_ocr_by_document_id, get_any_ocr_by_hash, copy_ocr_to_document,
    save_ocr_result, mark_ocr_success, sha256_of_file, get_extracted_by_document_id, save_extracted_data
)
from pathlib import Path
import json

def ensure_document_registered(case_id: str, filepath: str) -> tuple[str, str]:
    """
    Garantiza que el documento esté en la tabla documents y devuelve (document_id, file_hash).
    """
    p = Path(filepath)
    # upsert_document internamente calcula hash, pero queremos el hash explícito para reuso global
    file_hash = sha256_of_file(p)
    doc_id, _is_new = upsert_document(case_id, str(p), mime_type=None, page_count=None, language=None)
    return doc_id, file_hash

def try_get_cached_ocr(document_id: str, file_hash: str, allow_global: bool = True) -> Optional[Dict[str, Any]]:
    """
    Devuelve el OCR (dict) si ya existe para este document_id o por hash global.
    """
    row = get_ocr_by_document_id(document_id)
    if row:
        return _row_to_ocr_dict(row)

    if allow_global:
        any_ocr = get_any_ocr_by_hash(file_hash)
        if any_ocr:
            # copiar y devolver
            copy_ocr_to_document(any_ocr, document_id)
            return _row_to_ocr_dict(any_ocr)

    return None

def persist_ocr(document_id: str, ocr_dict: Dict[str, Any], engine: str, engine_version: Optional[str] = None) -> None:
    save_ocr_result(document_id, ocr_dict, engine, engine_version)
    mark_ocr_success(document_id, True)

def try_get_cached_extraction(document_id: str) -> Optional[Dict[str, Any]]:
    row = get_extracted_by_document_id(document_id)
    if not row:
        return None
    return {
        "document_type": row["document_type"],
        "entities": json.loads(row["entities"] or "{}"),
        "key_value_pairs": json.loads(row["key_value_pairs"] or "{}"),
        "extra": json.loads(row["extra"] or "{}"),
    }

def persist_extraction(document_id: str, extracted: Dict[str, Any], extractor_version: str = "v1") -> None:
    save_extracted_data(document_id, extracted, extractor_version)

def _row_to_ocr_dict(row) -> Dict[str, Any]:
    return {
        "text": row["raw_text"] or "",
        "key_value_pairs": json.loads(row["key_value_pairs"] or "{}"),
        "tables": json.loads(row["tables"] or "[]"),
        "entities": json.loads(row["entities"] or "[]"),
        "confidence_scores": json.loads(row["confidence"] or "{}"),
        "metadata": json.loads(row["metadata"] or "{}"),
        "errors": json.loads(row["errors"] or "[]"),
    }
