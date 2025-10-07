"""Carga y preparación de documentos para el índice RAG del Agente Rick."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from decimal import Decimal, InvalidOperation
import os
import re
import unicodedata
import json
import logging

from langchain.schema import Document

from fraud_scorer.storage.ocr_cache import OCRCacheManager

from ..config import RickAgentConfig, load_config
from .normalizer import deduplicate_documents
from .splitters import get_default_splitter, get_splitter_for_type
from .transformers import clean_text, combine_metadata, safe_json_dumps
from fraud_scorer.analyzers.correlation.utils.normalization import (
    normalize_date,
    normalize_decimal_as_str,
)

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

        consolidated_documents = self._build_consolidated_fields(case_index, base_metadata)
        raw_documents.extend(consolidated_documents)

        structured_card = self._build_structured_card(case_id, case_index, base_metadata)
        if structured_card:
            raw_documents.append(structured_card)

        raw_documents.extend(self._build_extraction_results(case_index, base_metadata))
        raw_documents.extend(self._build_fraud_analyses(case_index, base_metadata))
        raw_documents.extend(self._build_gps_documents(case_index, base_metadata))

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
            metadata = self._augment_source_metadata(metadata, filename)

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

        metadata = combine_metadata(
            base_metadata,
            {
                "source": "consolidated_fields",
                "document_type": "consolidated_fields",
                "origin": "analysis_pipeline",
            },
        )
        return [Document(page_content=payload, metadata=metadata)]

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

            metadata = combine_metadata(
                base_metadata,
                {
                    "source": "extraction",
                    "origin": "analysis_pipeline",
                    "source_document": source_doc,
                    "document_type": doc_type,
                },
            )
            metadata = self._augment_source_metadata(metadata, source_doc)

            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def _build_gps_documents(
        self,
        case_index: Dict[str, Any],
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        documents: List[Document] = []
        gps_entries = case_index.get("gps_direct_documents") or {}
        if not gps_entries:
            return documents

        for document_name, entry in gps_entries.items():
            summary = entry.get("summary") or {}
            dataset_meta = entry.get("dataset") or {}
            warnings = entry.get("normalization_warnings") or summary.get("warnings") or []
            ingestion_stats = entry.get("ingestion_stats") or {}

            summary_payload = {
                "document_name": document_name,
                "summary": summary,
                "dataset": {
                    "row_count": dataset_meta.get("row_count"),
                    "checksum": dataset_meta.get("checksum"),
                    "partitions": dataset_meta.get("partitions"),
                    "compression": dataset_meta.get("compression"),
                },
                "largest_gap_minutes": _largest_gap(summary),
                "warnings": warnings,
                "ingestion_stats": ingestion_stats,
            }

            summary_metadata = combine_metadata(
                base_metadata,
                {
                    "source": "gps_summary",
                    "origin": "gps_dal",
                    "source_document": document_name,
                    "document_type": "reporte_gps",
                    "gps_kind": "summary",
                },
            )
            documents.append(
                Document(page_content=safe_json_dumps(summary_payload), metadata=summary_metadata)
            )

            preview_rows = entry.get("preview_rows") or []
            if preview_rows:
                preview_metadata = combine_metadata(
                    base_metadata,
                    {
                        "source": "gps_preview",
                        "origin": "gps_dal",
                        "source_document": document_name,
                        "document_type": "reporte_gps",
                        "gps_kind": "preview",
                    },
                )
                documents.append(
                    Document(
                        page_content=safe_json_dumps(preview_rows[:100]),
                        metadata=preview_metadata,
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

            metadata = combine_metadata(
                base_metadata,
                {
                    "source": "fraud_analysis",
                    "origin": "fraud_stage",
                    "document_type": doc_type,
                    "source_document": source_doc,
                },
            )
            metadata = self._augment_source_metadata(metadata, source_doc)

            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def _build_structured_card(
        self,
        case_id: str,
        case_index: Dict[str, Any],
        base_metadata: Dict[str, Any],
    ) -> Document | None:
        entries = self._collect_structured_entries(case_id, case_index)
        if not any(entries.get(key) for key in _STRUCTURED_CARD_FIELDS):
            return None

        sections: list[tuple[str, list[str]]] = [
            ("DATOS GENERALES", [
                "case_id",
                "insured_name",
                "claim_number",
                "numero_siniestro",
                "numero_poliza",
                "tipo_siniestro",
                "bien_reclamado",
            ]),
            ("MONTOS Y COBERTURAS", [
                "monto_reclamacion",
                "suma_asegurada",
                "deducible",
                "coberturas",
            ]),
            ("CRONOLOGÍA", [
                "fecha_ocurrencia",
                "fecha_reclamacion",
                "fecha_timbrado",
            ]),
            ("LOGÍSTICA Y FISCALES", [
                "placas_vehiculo",
                "operador_nombre",
                "operador_identificacion",
                "peso_bruto_kg",
                "uuid_fiscal",
            ]),
        ]

        lines: list[str] = [f"TARJETA DE DATOS ESTRUCTURADOS - CASO {case_id}", ""]

        for title, keys in sections:
            section_lines = []
            for key in keys:
                config = _STRUCTURED_CARD_FIELDS.get(key)
                data = entries.get(key) or []
                if not config or not data:
                    continue
                formatted = self._format_structured_line(config, data)
                if formatted:
                    section_lines.append(f"- {formatted}")
            if section_lines:
                lines.append(f"{title}:")
                lines.extend(section_lines)
                lines.append("")

        payload = "\n".join(line for line in lines if line is not None).strip()
        if not payload:
            return None

        metadata = combine_metadata(
            base_metadata,
            {
                "source": "structured_card",
                "document_type": "structured_card",
                "origin": "analysis_pipeline",
                "source_document": f"structured_card_{case_id}.txt",
                "card_fields": [key for key, items in entries.items() if items],
            },
        )
        metadata = self._augment_source_metadata(metadata, metadata.get("source_document"))

        return Document(page_content=payload, metadata=metadata)

    def _collect_structured_entries(
        self,
        case_id: str,
        case_index: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        consolidated = (case_index.get("consolidated_data") or {}).get("consolidated_fields") or {}
        for field, value in consolidated.items():
            if value is None:
                continue
            values = _flatten_values(value)
            for item in values:
                entries[field].append({
                    "value": item,
                    "source": "consolidated_fields",
                    "document_type": None,
                })

        for result in case_index.get("extraction_results") or []:
            if not isinstance(result, dict):
                continue
            source_document = result.get("source_document")
            document_type = result.get("document_type")
            extracted = result.get("extracted_fields") or result.get("fields") or {}
            for field, value in extracted.items():
                if value is None:
                    continue
                for item in _flatten_values(value):
                    entries[field].append(
                        {
                            "value": item,
                            "source": source_document,
                            "document_type": document_type,
                        }
                    )

        if case_index.get("insured_name"):
            entries["insured_name"].append({
                "value": case_index["insured_name"],
                "source": "case_index",
                "document_type": None,
            })
        if case_index.get("claim_number"):
            entries["claim_number"].append({
                "value": case_index["claim_number"],
                "source": "case_index",
                "document_type": None,
            })

        entries["case_id"].append({
            "value": case_id,
            "source": "case_id",
            "document_type": None,
        })

        return entries

    def _format_structured_line(self, config: Dict[str, Any], items: List[Dict[str, Any]]) -> str:
        raw_values = [str(entry["value"]).strip() for entry in items if str(entry.get("value", "")).strip()]
        if not raw_values:
            return ""

        unique_values = list(dict.fromkeys(raw_values))
        main_value = "; ".join(unique_values)

        annotations: list[str] = []
        if config.get("normalize_date"):
            normalized_dates = [normalize_date(value) for value in unique_values]
            normalized_dates = [date for date in normalized_dates if date and date not in unique_values]
            if normalized_dates:
                annotations.append(" fechas normalizadas: " + ", ".join(dict.fromkeys(normalized_dates)))

        if config.get("normalize_amount"):
            normalized_amounts = [normalize_decimal_as_str(value) for value in unique_values]
            normalized_amounts = [amount for amount in normalized_amounts if amount]
            if normalized_amounts:
                annotations.append(" montos normalizados: " + ", ".join(dict.fromkeys(normalized_amounts)))

        if config.get("normalize_weight"):
            normalized = [normalize_decimal_as_str(value) for value in unique_values]
            normalized = [value for value in normalized if value]
            if normalized:
                tons: list[str] = []
                for amount in dict.fromkeys(normalized):
                    try:
                        kg = Decimal(amount)
                        tons_value = kg / Decimal("1000")
                        tons.append(format(tons_value.normalize(), "f").rstrip("0").rstrip("."))
                    except (InvalidOperation, ZeroDivisionError):
                        continue
                tons = [value for value in tons if value]
                if tons:
                    annotations.append(" toneladas aproximadas: " + ", ".join(tons))

        if config.get("aliases"):
            alias_values: list[str] = []
            for value in unique_values:
                alias_values.extend(self._generate_aliases(value))
            alias_values = [alias for alias in dict.fromkeys(alias_values) if alias]
            if alias_values:
                annotations.append(" alias: " + ", ".join(alias_values))

        sources = [entry.get("source") for entry in items if entry.get("source")]
        sources = [self.cache_manager._sanitize_filename(str(src)) for src in sources]
        sources = [src for src in dict.fromkeys(sources) if src]
        source_text = f" (fuentes: {', '.join(sources)})" if sources else ""

        annotation_text = "".join(annotations)
        label = config.get("label") or "Campo"
        return f"{label}: {main_value}{annotation_text}{source_text}".strip()

    def _augment_source_metadata(self, metadata: Dict[str, Any], source_document: str | None) -> Dict[str, Any]:
        if not source_document:
            return metadata

        result = dict(metadata)
        sanitized = self.cache_manager._sanitize_filename(str(source_document))
        result["source_document_normalized"] = sanitized

        base_name, ext = os.path.splitext(str(source_document))
        alias_candidates = [
            str(source_document),
            sanitized,
            sanitized.lower(),
            base_name,
            base_name.replace(" ", "_"),
            base_name.replace(" ", "-"),
            sanitized.replace("_", ""),
            sanitized.replace("_", "-"),
        ]
        alias_values = [alias.upper() for alias in alias_candidates if alias]
        alias_values = [alias for alias in dict.fromkeys(alias_values) if alias and alias != str(source_document).upper()]
        if alias_values:
            result["source_document_aliases"] = alias_values
        return result

    def _generate_aliases(self, value: str) -> List[str]:
        cleaned = unicodedata.normalize("NFKD", value)
        cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
        normalized = re.sub(r"[^A-Za-z0-9]", "", cleaned).upper()
        if not normalized:
            return []

        aliases = {
            normalized,
            normalized.replace("O", "0"),
        }
        if len(normalized) == 6:
            aliases.add(f"{normalized[:2]}-{normalized[2:4]}-{normalized[4:]}")
            aliases.add(f"{normalized[:3]}-{normalized[3:]}")
        if len(normalized) == 7:
            aliases.add(f"{normalized[:3]}-{normalized[3:5]}-{normalized[5:]}")

        aliases.discard(normalized)
        return list(dict.fromkeys(aliases))

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


def _largest_gap(summary: Dict[str, Any]) -> Optional[int]:
    gaps = summary.get("time_gaps") or []
    values = [int(item.get("gap_minutes") or 0) for item in gaps if item.get("gap_minutes")]
    return max(values) if values else None


def _normalize(value: str) -> str:
    simplified = value.lower().replace("-", " ").replace("_", " ")
    return "".join(ch for ch in simplified if ch.isalnum())


_STRUCTURED_CARD_FIELDS: Dict[str, Dict[str, Any]] = {
    "case_id": {"label": "Caso"},
    "insured_name": {"label": "Asegurado"},
    "claim_number": {"label": "Número de reclamación", "aliases": True},
    "numero_siniestro": {"label": "Número de siniestro", "aliases": True},
    "numero_poliza": {"label": "Número de póliza", "aliases": True},
    "tipo_siniestro": {"label": "Tipo de siniestro"},
    "bien_reclamado": {"label": "Bien reclamado"},
    "monto_reclamacion": {"label": "Monto reclamado", "normalize_amount": True},
    "suma_asegurada": {"label": "Suma asegurada", "normalize_amount": True},
    "deducible": {"label": "Deducible", "normalize_amount": True},
    "coberturas": {"label": "Coberturas"},
    "fecha_ocurrencia": {"label": "Fecha del siniestro", "normalize_date": True},
    "fecha_reclamacion": {"label": "Fecha de reclamación", "normalize_date": True},
    "fecha_timbrado": {"label": "Fecha de timbrado", "normalize_date": True},
    "placas_vehiculo": {"label": "Placas del vehículo", "aliases": True},
    "operador_nombre": {"label": "Operador"},
    "operador_identificacion": {"label": "Identificación del operador", "aliases": True},
    "peso_bruto_kg": {"label": "Peso bruto (kg)", "normalize_weight": True},
    "uuid_fiscal": {"label": "UUID fiscal", "aliases": True},
}


def _flatten_values(values: Any) -> Sequence[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        flattened: List[str] = []
        for value in values:
            flattened.extend(_flatten_values(value))
        return flattened
    return [str(values)]


__all__ = ["FraudCaseDocumentLoader"]
