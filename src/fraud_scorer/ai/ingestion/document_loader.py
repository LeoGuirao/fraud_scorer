"""Carga y preparación de documentos para el índice RAG del Agente Rick."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import logging

from langchain.schema import Document

from fraud_scorer.storage.ocr_cache import OCRCacheManager

from ..config import RickAgentConfig, load_config
from .normalizer import deduplicate_documents
from .splitters import get_default_splitter, get_splitter_for_type
from .transformers import clean_text, combine_metadata, safe_json_dumps

logger = logging.getLogger(__name__)


class FraudCaseDocumentLoader:
    """Genera documentos listos para indexar a partir del case_index."""

    def __init__(
        self,
        *,
        cache_manager: OCRCacheManager | None = None,
        config: RickAgentConfig | None = None,
    ) -> None:
        self.cache_manager = cache_manager or OCRCacheManager()
        self.config = config or load_config()
        self.last_case_id: str | None = None

    def load_case_documents(self, case_id: str) -> List[Document]:
        """Carga y transforma toda la información relevante de un caso."""

        case_id, case_index = self._resolve_case_index(case_id)
        self.last_case_id = case_id

        base_metadata = {
            "case_id": case_id,
            "insured_name": case_index.get("insured_name"),
            "claim_number": case_index.get("claim_number"),
        }

        document_type_map = self._build_document_type_index(case_index)
        document_hashes = case_index.get("document_hashes") or {}

        raw_documents: List[Document] = []
        raw_documents.extend(
            self._build_ocr_documents(case_id, case_index, document_type_map, document_hashes, base_metadata)
        )
        raw_documents.extend(self._build_consolidated_fields(case_index, base_metadata))
        raw_documents.extend(self._build_extraction_results(case_index, base_metadata))
        raw_documents.extend(self._build_fraud_analyses(case_index, base_metadata))

        unique_documents = deduplicate_documents(raw_documents)
        return self._split_documents(unique_documents)

    # ------------------------------------------------------------------
    # Helpers de construcción

    def _build_document_type_index(self, case_index: Dict[str, Any]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}

        for item in case_index.get("classified_types") or []:
            if isinstance(item, dict):
                filename = (item.get("filename") or "").strip()
                doc_type = (item.get("document_type") or "").strip()
                if filename and doc_type:
                    mapping[filename] = doc_type

        manual = case_index.get("manual_classifications") or {}
        for filename, doc_type in manual.items():
            if doc_type:
                mapping[str(filename)] = str(doc_type)

        ai_map = case_index.get("ai_classifications") or {}
        for filename, doc_type in ai_map.items():
            mapping.setdefault(str(filename), str(doc_type))

        return mapping

    def _build_ocr_documents(
        self,
        case_id: str,
        case_index: Dict[str, Any],
        document_type_map: Dict[str, str],
        document_hashes: Dict[str, str],
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        documents: List[Document] = []
        case_folder = self.cache_manager.get_case_folder_path(case_id, case_index)

        for entry in case_index.get("documents") or []:
            path = Path(entry)
            filename = path.name
            doc_type = document_type_map.get(filename)

            try:
                ocr_payload = self.cache_manager.get_cache(path, case_id=case_id)
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("No se pudo cargar OCR desde %s: %s", path, exc)
                continue

            if not ocr_payload:
                logger.debug("Caso %s: sin OCR para %s", case_id, filename)
                continue

            text = clean_text(str(ocr_payload.get("text") or ""))
            if not text:
                continue

            metadata = combine_metadata(
                base_metadata,
                {
                    "source": "ocr",
                    "source_document": filename,
                    "document_type": doc_type,
                    "case_path": str(path),
                    "case_folder": str(case_folder),
                    "hash": document_hashes.get(entry),
                    "origin": "ocr_cache",
                },
            )

            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def _build_consolidated_fields(
        self,
        case_index: Dict[str, Any],
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        consolidated = (case_index.get("consolidated_data") or {}).get("consolidated_fields") or {}
        if not consolidated:
            return []

        payload = safe_json_dumps(consolidated)

        return [
            Document(
                page_content=payload,
                metadata=combine_metadata(
                    base_metadata,
                    {
                        "source": "consolidated_fields",
                        "document_type": "consolidated_fields",
                        "origin": "analysis_pipeline",
                    },
                ),
            )
        ]

    def _build_extraction_results(
        self,
        case_index: Dict[str, Any],
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        documents: List[Document] = []

        for item in case_index.get("extraction_results") or []:
            if isinstance(item, dict):
                source_doc = item.get("source_document")
                doc_type = item.get("document_type")
                extracted = item.get("extracted_fields") or item.get("fields") or item
            else:
                source_doc = getattr(item, "source_document", None)
                doc_type = getattr(item, "document_type", None)
                extracted = getattr(item, "extracted_fields", None) or {}

            if not extracted:
                continue

            content = safe_json_dumps(extracted)

            documents.append(
                Document(
                    page_content=content,
                    metadata=combine_metadata(
                        base_metadata,
                        {
                            "source": "extraction",
                            "origin": "analysis_pipeline",
                            "source_document": source_doc,
                            "document_type": doc_type,
                        },
                    ),
                )
            )

        return documents

    def _build_fraud_analyses(
        self,
        case_index: Dict[str, Any],
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        analyses = case_index.get("fraud_analyses") or []
        documents: List[Document] = []

        for item in analyses:
            if isinstance(item, dict):
                summary = item.get("summary") or item.get("analysis") or item
                source_doc = item.get("document_name") or item.get("document")
                doc_type = item.get("document_type") or "fraud_analysis"
            else:
                summary = getattr(item, "summary", None) or str(item)
                source_doc = getattr(item, "document_name", None)
                doc_type = getattr(item, "document_type", None) or "fraud_analysis"

            text = clean_text(str(summary or ""))
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata=combine_metadata(
                        base_metadata,
                        {
                            "source": "fraud_analysis",
                            "origin": "fraud_stage",
                            "document_type": doc_type,
                            "source_document": source_doc,
                        },
                    ),
                )
            )

        return documents

    # ------------------------------------------------------------------
    # Split y post-procesamiento

    def _split_documents(self, documents: Iterable[Document]) -> List[Document]:
        ready: List[Document] = []

        for doc in documents:
            doc_type = (doc.metadata.get("document_type") or "").lower()
            splitter = get_splitter_for_type(doc_type) if doc_type else get_default_splitter()
            parts = splitter.split_documents([doc])

            total = len(parts) or 1
            for idx, part in enumerate(parts, start=1):
                part.metadata = combine_metadata(
                    doc.metadata,
                    {
                        "chunk_index": idx,
                        "chunk_total": total,
                    },
                )
                ready.append(part)

        return ready

    # ------------------------------------------------------------------
    # Resolución de caso

    def _resolve_case_index(self, identifier: str) -> tuple[str, Dict[str, Any]]:
        case_index = self.cache_manager.get_case_index(identifier, auto_reconstruct=True)
        if case_index:
            return identifier, case_index

        normalized = _normalize(identifier)
        for candidate in self.cache_manager.list_cached_cases():
            for key in (candidate.get("case_id"), candidate.get("case_title"), candidate.get("folder_path")):
                if not key:
                    continue
                key_str = str(key)
                options = {key_str}
                try:
                    options.add(Path(key_str).name)
                except Exception:
                    pass
                if any(_normalize(opt) == normalized for opt in options):
                    resolved_id = candidate["case_id"]
                    index = self.cache_manager.get_case_index(resolved_id, auto_reconstruct=True)
                    if index:
                        logger.info("Identificador %s resuelto a case_id %s", identifier, resolved_id)
                        return resolved_id, index

        raise ValueError(f"No se encontró información del caso {identifier}")


def _normalize(value: str) -> str:
    simplified = value.lower().replace("-", " ").replace("_", " ")
    return "".join(ch for ch in simplified if ch.isalnum())


__all__ = ["FraudCaseDocumentLoader"]
