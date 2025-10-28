"""Capa de datos unificada para el análisis de fraude individual.

Esta capa consolida información estructurada del `case_index`, las
extracciones disponibles y, cuando es necesario, recurre al texto OCR
para reconstruir campos críticos siguiendo las recomendaciones de
*Better Practices*.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fraud_scorer.analyzers.correlation.utils.normalization import (
    is_missing,
    normalize_date,
    normalize_decimal_as_str,
)
from fraud_scorer.models.extraction import DocumentExtraction
from fraud_scorer.services.gps_query_service import GPSDirectQueryService


logger = logging.getLogger(__name__)


def _coerce_extraction(payload: Any) -> Optional[DocumentExtraction]:
    """Convierte payload heterogéneos a `DocumentExtraction`."""
    if isinstance(payload, DocumentExtraction):
        return payload
    if not isinstance(payload, dict):
        return None
    data = dict(payload)
    source_document = (
        data.get("source_document")
        or data.get("document_name")
        or data.get("file_name")
    )
    document_type = data.get("document_type") or "otro"
    if not source_document:
        return None
    try:
        return DocumentExtraction.model_validate(
            {
                "source_document": source_document,
                "document_type": document_type,
                "extracted_fields": data.get("extracted_fields")
                or data.get("fields")
                or {},
                "extraction_metadata": data.get("extraction_metadata") or {},
            }
        )
    except Exception:
        return None


@dataclass
class ResolvedField:
    value: Optional[str]
    source: str
    document: Optional[str] = None


class UnifiedDataLayer:
    """Acceso resiliente a información del caso y documentos."""

    DATE_FIELDS = {
        "fecha_ocurrencia",
        "fecha_reclamacion",
        "vigencia_inicio",
        "vigencia_fin",
        "fecha_emision",
        "fecha_timbrado",
        "fecha_apertura",
        "fecha_denuncia",
    }

    NUMERIC_FIELDS = {
        "monto_reclamacion",
        "monto_total",
        "valor_mercancia",
        "suma_asegurada",
        "deducible",
        "peso_bruto",
        "peso_total",
        "peso",
    }

    OCR_PATTERNS: Dict[str, re.Pattern[str]] = {
        "numero_siniestro": re.compile(
            r"(?:siniestro|expediente|folio)\s*[:#-]?\s*([A-Z0-9-]{6,})",
            re.IGNORECASE,
        ),
        "numero_poliza": re.compile(
            r"(?:p[óo]liza|policy)\s*(?:n[úu]m(?:ero)?|#)?\s*[:#-]?\s*([A-Z0-9-]{5,})",
            re.IGNORECASE,
        ),
        "monto_reclamacion": re.compile(
            r"(?:monto|importe)\s*(?:total\s*)?(?:reclamad[oa]|reclamaci[óo]n)\s*[:$]?\s*([$]?\s?[0-9.,]+)",
            re.IGNORECASE,
        ),
        "fecha_ocurrencia": re.compile(
            r"fecha\s*(?:de\s*)?(?:ocurrencia|siniestro|evento)\s*[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            re.IGNORECASE,
        ),
        "fecha_reclamacion": re.compile(
            r"fecha\s*(?:de\s*)?(?:reclamaci[óo]n|reporte)\s*[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            re.IGNORECASE,
        ),
    }

    KEY_CASE_FIELDS = (
        "numero_siniestro",
        "nombre_asegurado",
        "numero_poliza",
        "monto_reclamacion",
        "vigencia_inicio",
        "vigencia_fin",
    )

    KEY_DOCUMENT_FIELDS = (
        "numero_siniestro",
        "numero_poliza",
        "monto_reclamacion",
        "fecha_ocurrencia",
        "fecha_reclamacion",
        "origen",
        "destino",
        "operador_nombre",
        "placas",
        "vin",
    )

    def __init__(
        self,
        case_index: Dict[str, Any],
        *,
        extractions: Optional[Iterable[DocumentExtraction]] = None,
    ) -> None:
        self.case_index = case_index or {}
        raw_consolidated = (self.case_index.get("consolidated_data") or {}).get(
            "consolidated_fields",
            {},
        )
        self.consolidated_fields: Dict[str, Any] = dict(raw_consolidated)
        self.case_id: Optional[str] = self.case_index.get("case_id")
        self.claim_number: Optional[str] = self.case_index.get("claim_number")
        self.insured_name: Optional[str] = self.case_index.get("insured_name")

        self._extractions: Dict[str, DocumentExtraction] = {}
        candidates: List[DocumentExtraction] = []
        if extractions is not None:
            candidates.extend([ext for ext in extractions if isinstance(ext, DocumentExtraction)])
        else:
            for raw in self.case_index.get("extraction_results") or []:
                extraction = _coerce_extraction(raw)
                if extraction:
                    candidates.append(extraction)
        for extraction in candidates:
            self._extractions[extraction.source_document] = extraction

        self._field_cache: Dict[str, ResolvedField] = {}
        self._gps_service = GPSDirectQueryService()
        self.gps_documents: Dict[str, Any] = dict(self.case_index.get("gps_direct_documents") or {})

    @classmethod
    def from_case_index(
        cls,
        case_index: Dict[str, Any],
    ) -> "UnifiedDataLayer":
        return cls(case_index=case_index)

    # ------------------------------------------------------------------
    # GPS helpers
    # ------------------------------------------------------------------
    def has_gps_data(self) -> bool:
        return bool(self.gps_documents)

    def list_gps_documents(self) -> Dict[str, Any]:
        return dict(self.gps_documents)

    def get_gps_snapshot(
        self,
        document_name: str,
        *,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        event_labels: Optional[Iterable[str]] = None,
        bounding_box: Optional[Dict[str, float]] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        if document_name not in self.gps_documents:
            return {}

        start_dt = _coerce_datetime(start_time)
        end_dt = _coerce_datetime(end_time)

        return self._gps_service.query_dataset(
            case_id=self.case_index.get("case_id") or self.case_id or "",
            document_name=document_name,
            start_time=start_dt,
            end_time=end_dt,
            event_labels=list(event_labels) if event_labels else None,
            bounding_box=bounding_box,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Resolución de campos
    # ------------------------------------------------------------------
    def resolve_field(
        self,
        field: str,
        *,
        extraction: Optional[DocumentExtraction] = None,
        document_name: Optional[str] = None,
        ocr_text: Optional[str] = None,
    ) -> Optional[str]:
        return self.resolve_field_with_origin(
            field,
            extraction=extraction,
            document_name=document_name,
            ocr_text=ocr_text,
        ).value

    def resolve_field_with_origin(
        self,
        field: str,
        *,
        extraction: Optional[DocumentExtraction] = None,
        document_name: Optional[str] = None,
        ocr_text: Optional[str] = None,
    ) -> ResolvedField:
        cache_key = f"{field}|{getattr(extraction, 'source_document', '')}"
        if cache_key in self._field_cache:
            return self._field_cache[cache_key]

        # 1) Campos del documento analizado
        if extraction and field in extraction.extracted_fields:
            value = extraction.extracted_fields.get(field)
            normalized = self._normalize_field(field, value)
            if normalized is not None and not is_missing(normalized):
                resolved = ResolvedField(
                    value=normalized,
                    source="extraction",
                    document=document_name or extraction.source_document,
                )
                self._field_cache[cache_key] = resolved
                return resolved

        # 2) Consolidado del caso
        consolidated_value = self.consolidated_fields.get(field)
        normalized = self._normalize_field(field, consolidated_value)
        if normalized is not None and not is_missing(normalized):
            resolved = ResolvedField(value=normalized, source="consolidated", document=None)
            self._field_cache[cache_key] = resolved
            return resolved

        # 3) Campos top-level del case_index
        top_level = self.case_index.get(field)
        normalized = self._normalize_field(field, top_level)
        if normalized is not None and not is_missing(normalized):
            resolved = ResolvedField(value=normalized, source="case_index", document=None)
            self._field_cache[cache_key] = resolved
            return resolved

        # 4) Extracciones de otros documentos
        for other_name, other_ext in self._extractions.items():
            if extraction and other_name == extraction.source_document:
                continue
            value = other_ext.extracted_fields.get(field)
            normalized = self._normalize_field(field, value)
            if normalized is not None and not is_missing(normalized):
                resolved = ResolvedField(value=normalized, source="extraction", document=other_name)
                self._field_cache[cache_key] = resolved
                return resolved

        # 5) OCR fallback
        if ocr_text:
            regex = self.OCR_PATTERNS.get(field)
            if regex:
                match = regex.search(ocr_text)
                if match:
                    raw = match.group(1).strip()
                    normalized = self._normalize_field(field, raw)
                    if normalized is not None and not is_missing(normalized):
                        resolved = ResolvedField(value=normalized, source="ocr_regex", document=document_name)
                        self._field_cache[cache_key] = resolved
                        return resolved

        resolved = ResolvedField(value=None, source="missing", document=None)
        self._field_cache[cache_key] = resolved
        return resolved

    def _normalize_field(self, field: str, value: Any) -> Optional[str]:
        if is_missing(value):
            return None
        if field in self.DATE_FIELDS:
            normalized = normalize_date(value)
            if normalized:
                return normalized
        if field in self.NUMERIC_FIELDS:
            normalized = normalize_decimal_as_str(value)
            if normalized is not None:
                return normalized
        if isinstance(value, (list, tuple)):
            items = [str(v).strip() for v in value if not is_missing(v)]
            return ", ".join(items) if items else None
        return str(value).strip()

    # ------------------------------------------------------------------
    # Contextos
    # ------------------------------------------------------------------
    def build_case_context(self) -> Dict[str, Any]:
        core: Dict[str, Dict[str, Optional[str]]] = {}
        sources: Dict[str, Dict[str, Optional[str]]] = {}
        for field in self.KEY_CASE_FIELDS:
            resolved = self.resolve_field_with_origin(field)
            core[field] = {
                "value": resolved.value,
                "source": resolved.source,
            }
            if resolved.document:
                sources[field] = {
                    "document": resolved.document,
                    "document_type": self._extractions.get(resolved.document, None).document_type
                    if resolved.document in self._extractions
                    else None,
                }

        coverage = self._summarize_documents()

        gps_context: Dict[str, Any] = {}
        for name, entry in self.gps_documents.items():
            dataset_meta = entry.get("dataset") or {}
            summary = entry.get("summary") or {}
            gps_context[name] = {
                "row_count": dataset_meta.get("row_count"),
                "warnings": entry.get("normalization_warnings") or summary.get("warnings"),
                "time_span": summary.get("time_span"),
                "largest_gap_minutes": _largest_gap_minutes(summary),
                "h3_cells": summary.get("h3_cells"),
            }

        return {
            "case_id": self.case_id,
            "claim_number": core.get("numero_siniestro", {}).get("value") or self.claim_number,
            "insured_name": core.get("nombre_asegurado", {}).get("value") or self.insured_name,
            "core_fields": core,
            "field_sources": sources,
            "document_coverage": coverage,
            "last_updated_at": self.case_index.get("updated_at"),
            "gps_documents": gps_context,
            "gps_ingestion_audit": self.case_index.get("gps_ingestion_audit"),
        }

    def build_document_context(
        self,
        *,
        extraction: DocumentExtraction,
        ocr_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        resolved_fields: Dict[str, Optional[str]] = {}
        field_sources: Dict[str, Dict[str, Optional[str]]] = {}
        missing: List[str] = []
        ocr_text = str(ocr_result.get("text") or "")

        for field in self.KEY_DOCUMENT_FIELDS:
            resolved = self.resolve_field_with_origin(
                field,
                extraction=extraction,
                document_name=extraction.source_document,
                ocr_text=ocr_text,
            )
            resolved_fields[field] = resolved.value
            if resolved.document or resolved.source != "missing":
                field_sources[field] = {
                    "source": resolved.source,
                    "document": resolved.document,
                }
            if not self._is_present_in_document(resolved, extraction.source_document):
                if field not in missing:
                    missing.append(field)

        gps_info: Optional[Dict[str, Any]] = None
        if extraction.source_document in self.gps_documents:
            entry = self.gps_documents[extraction.source_document]
            dataset_meta = entry.get("dataset") or {}
            summary = entry.get("summary") or {}
            gps_info = {
                "row_count": dataset_meta.get("row_count"),
                "warnings": entry.get("normalization_warnings") or summary.get("warnings"),
                "time_span": summary.get("time_span"),
                "largest_gap_minutes": _largest_gap_minutes(summary),
                "preview_rows": entry.get("preview_rows"),
            }
            try:
                snapshot = self.get_gps_snapshot(extraction.source_document, limit=100)
                gps_info["preview"] = snapshot.get("preview")
                gps_info["query_summary"] = snapshot.get("summary")
            except Exception as exc:  # pragma: no cover - defensivo
                logger.debug("No se pudo construir snapshot GPS para %s: %s", extraction.source_document, exc)

        payload = {
            "document_type": extraction.document_type,
            "document_name": extraction.source_document,
            "resolved_fields": resolved_fields,
            "field_sources": field_sources,
            "missing_fields": missing,
            "ocr_word_count": len(ocr_text.split()),
        }
        if gps_info:
            payload["gps_summary"] = gps_info
        return payload

    @staticmethod
    def _is_present_in_document(resolved: ResolvedField, document_name: str) -> bool:
        if resolved.value is None:
            return False
        if resolved.source == "extraction":
            return not resolved.document or resolved.document == document_name
        if resolved.source == "ocr_regex":
            return not resolved.document or resolved.document == document_name
        return False

    def _summarize_documents(self) -> Dict[str, Any]:
        summary: Dict[str, int] = {}
        classified = self.case_index.get("classified_types") or []
        for item in classified:
            if not isinstance(item, dict):
                continue
            dtype = str(item.get("document_type") or "otro").strip()
            if not dtype:
                continue
            summary[dtype] = summary.get(dtype, 0) + 1
        # Incorporar extracciones si no existen en clasificados
        for extraction in self._extractions.values():
            dtype = extraction.document_type or "otro"
            summary.setdefault(dtype, 0)
        return {
            "by_type": summary,
            "total_documents": sum(summary.values()) if summary else 0,
        }


__all__ = ["UnifiedDataLayer", "ResolvedField"]


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _largest_gap_minutes(summary: Dict[str, Any]) -> Optional[int]:
    time_gaps = summary.get("time_gaps") or []
    values = [int(gap.get("gap_minutes") or 0) for gap in time_gaps if gap.get("gap_minutes")]
    return max(values) if values else None
