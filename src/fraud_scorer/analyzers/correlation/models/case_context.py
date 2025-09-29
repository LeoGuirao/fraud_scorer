"""Modelos y utilidades para construir el contexto de correlación de un caso."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import Field, PrivateAttr

from fraud_scorer.analyzers.correlation.utils import EntityNormalizer
from fraud_scorer.models.extraction import (
    BaseModelCompat,
    ConsolidatedExtraction,
    ConsolidatedFields,
    DocumentExtraction,
)
from fraud_scorer.models.fraud_analysis import FraudAnalysisResult
from fraud_scorer.storage.db import get_conn

if TYPE_CHECKING:
    from fraud_scorer.storage.ocr_cache import OCRCacheManager


def _to_float(value: Any) -> Optional[float]:
    """Convierte strings o numéricos a float cuando sea razonable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _normalise_date(value: Any) -> Optional[str]:
    """Intenta normalizar fechas comunes a formato ISO (YYYY-MM-DD)."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(text, fmt)
                return dt.date().isoformat()
            except ValueError:
                continue
        # Último recurso: devolver la cadena original para trazabilidad
        return text
    return None


def _sanitize_filename(name: str) -> str:
    if not name:
        return "SIN_NOMBRE"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def _derive_document_name_from_cache_path(path: Path) -> str:
    name = path.name
    if name.startswith("ocr_results_for_"):
        name = name[len("ocr_results_for_") :]
    if name.endswith(".json"):
        name = name[:-5]
    return name


def _store_metadata_candidate(
    container: Dict[str, Dict[str, Any]],
    name: str,
    payload: Dict[str, Any],
) -> None:
    if not name:
        return
    entry = dict(container.get(name, {}))
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(entry.get(key), dict) and isinstance(value, dict):
            merged = dict(entry.get(key, {}))
            merged.update(value)
            entry[key] = merged
        else:
            entry.setdefault(key, value)
    container[name] = entry
    sanitized = _sanitize_filename(name)
    container.setdefault(sanitized, entry)


class DocumentSnapshot(BaseModelCompat):
    """Información consolidada de un documento dentro del contexto."""

    document_id: Optional[str] = None
    document_type: str
    raw_document_type: Optional[str] = None
    document_name: Optional[str] = None
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    field_aliases: Dict[str, str] = Field(default_factory=dict)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    fraud_result: Optional[FraudAnalysisResult] = None
    ocr_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class CaseContext(BaseModelCompat):
    """Contexto completo necesario para ejecutar el motor de correlación."""

    case_id: str
    consolidated: Optional[ConsolidatedExtraction] = None
    documents: List[DocumentSnapshot] = Field(default_factory=list)
    documents_by_type: Dict[str, List[DocumentSnapshot]] = Field(default_factory=dict)
    fraud_results: Dict[str, FraudAnalysisResult] = Field(default_factory=dict)
    aggregates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    entities: Dict[str, List[Any]] = Field(default_factory=dict)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    _data_tree: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _entity_normalizer: Optional[EntityNormalizer] = PrivateAttr(default=None)
    _document_alias_index: Dict[str, List[str]] = PrivateAttr(default_factory=dict)
    _field_alias_index: Dict[str, List[str]] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_case(
        cls,
        *,
        case_id: str,
        consolidated: Optional[ConsolidatedExtraction | Dict[str, Any]],
        extractions: Sequence[DocumentExtraction | Dict[str, Any]],
        fraud_results: Sequence[FraudAnalysisResult | Dict[str, Any]],
        case_index: Optional[Dict[str, Any]] = None,
        cache_manager: Optional["OCRCacheManager"] = None,
        entity_normalizer: Optional[EntityNormalizer] = None,
    ) -> "CaseContext":
        normalizer = entity_normalizer or EntityNormalizer.default()
        case_index = case_index or {}

        consolidated_obj = cls._ensure_consolidated(consolidated)
        extraction_objs = cls._ensure_extractions(extractions)
        fraud_objs = cls._ensure_fraud(fraud_results)

        doc_metadata_by_name, doc_metadata_by_id = cls._load_document_metadata(
            case_id,
            cache_manager=cache_manager,
            case_index=case_index,
        )

        documents: List[DocumentSnapshot] = []
        documents_by_type: Dict[str, List[DocumentSnapshot]] = defaultdict(list)
        fraud_index: Dict[str, FraudAnalysisResult] = {}

        fraud_by_name = {f.document_name: f for f in fraud_objs if f.document_name}
        fraud_by_id = {f.document_id: f for f in fraud_objs if f.document_id}

        for fraud in fraud_objs:
            if fraud.document_id:
                fraud_index[fraud.document_id] = fraud

        for extraction in extraction_objs:
            doc_name = extraction.source_document
            raw_doc_type = extraction.document_type or "desconocido"
            canonical_doc_type = normalizer.canonical_document_type(raw_doc_type) or raw_doc_type

            metadata = dict(extraction.extraction_metadata or {})
            doc_meta = cls._metadata_for_document(doc_name, doc_metadata_by_name)

            doc_id = (
                metadata.get("document_id")
                or doc_meta.get("document_id")
            )
            if not doc_id and doc_name:
                meta = doc_metadata_by_name.get(doc_name)
                if meta:
                    doc_id = meta.get("document_id")
            if not doc_id and doc_name:
                fraud_match = fraud_by_name.get(doc_name)
                if fraud_match:
                    doc_id = fraud_match.document_id

            fraud_result = None
            if doc_id and doc_id in fraud_by_id:
                fraud_result = fraud_by_id[doc_id]
            elif doc_name and doc_name in fraud_by_name:
                fraud_result = fraud_by_name[doc_name]

            original_name = (
                doc_meta.get("original_filename")
                or metadata.get("original_filename")
                or doc_name
            )

            fields = dict(extraction.extracted_fields or {})
            normalized_fields, field_aliases = normalizer.normalize_fields(fields)

            snapshot = DocumentSnapshot(
                document_id=doc_id,
                document_type=canonical_doc_type,
                raw_document_type=raw_doc_type,
                document_name=original_name,
                extracted_fields=normalized_fields,
                field_aliases=field_aliases,
                extraction_metadata=metadata,
                fraud_result=fraud_result,
                ocr_metadata=doc_meta.get("ocr_metadata", {}),
                created_at=doc_meta.get("created_at"),
            )
            documents.append(snapshot)
            documents_by_type[canonical_doc_type].append(snapshot)

        aggregates = cls._build_aggregates(documents_by_type)
        entities = cls._build_entities(consolidated_obj, documents, normalizer)
        timeline = cls._build_timeline(consolidated_obj, documents)

        metadata_payload = {
            "case_index": case_index,
            "entity_mappings_version": normalizer.version,
        }
        if cache_manager:
            metadata_payload["cache_base_dir"] = str(cache_manager.cache_dir)

        context = cls(
            case_id=case_id,
            consolidated=consolidated_obj,
            documents=documents,
            documents_by_type={k: v[:] for k, v in documents_by_type.items()},
            fraud_results=fraud_index,
            aggregates=aggregates,
            entities=entities,
            timeline=timeline,
            metadata=metadata_payload,
        )
        context._entity_normalizer = normalizer
        context._document_alias_index = normalizer.document_alias_index()
        context._field_alias_index = normalizer.field_alias_index()
        context._data_tree = context._build_data_tree()
        return context

    @staticmethod
    def _ensure_consolidated(
        consolidated: Optional[ConsolidatedExtraction | Dict[str, Any]]
    ) -> Optional[ConsolidatedExtraction]:
        if consolidated is None:
            return None
        if isinstance(consolidated, ConsolidatedExtraction):
            return consolidated
        return ConsolidatedExtraction.model_validate(consolidated)

    @staticmethod
    def _ensure_extractions(
        extractions: Sequence[DocumentExtraction | Dict[str, Any]]
    ) -> List[DocumentExtraction]:
        out: List[DocumentExtraction] = []
        for item in extractions or []:
            if isinstance(item, DocumentExtraction):
                out.append(item)
            else:
                out.append(DocumentExtraction.model_validate(item))
        return out

    @staticmethod
    def _ensure_fraud(
        fraud_results: Sequence[FraudAnalysisResult | Dict[str, Any]]
    ) -> List[FraudAnalysisResult]:
        out: List[FraudAnalysisResult] = []
        for item in fraud_results or []:
            if isinstance(item, FraudAnalysisResult):
                out.append(item)
            else:
                out.append(FraudAnalysisResult.model_validate(item))
        return out

    @classmethod
    def _load_document_metadata(
        cls,
        case_id: str,
        *,
        cache_manager: Optional["OCRCacheManager"],
        case_index: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Obtiene metadatos básicos de documentos desde la base de datos y el índice."""
        by_name: Dict[str, Dict[str, Any]] = {}
        by_id: Dict[str, Dict[str, Any]] = {}

        try:
            with get_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT d.id, d.filename, d.mime_type, d.filepath, d.page_count, d.language,
                           d.created_at
                    FROM documents d
                    WHERE d.case_id = ?
                    """,
                    (case_id,),
                ).fetchall()
        except Exception:
            rows = []

        for row in rows:
            doc_meta = {
                "document_id": row["id"],
                "filename": row["filename"],
                "mime_type": row["mime_type"],
                "filepath": row["filepath"],
                "page_count": row["page_count"],
                "language": row["language"],
                "created_at": row["created_at"],
                "ocr_metadata": {
                    "page_count": row["page_count"],
                    "language": row["language"],
                },
            }
            by_id[row["id"]] = doc_meta
            if row["filename"]:
                by_name[row["filename"]] = doc_meta
                sanitized = _sanitize_filename(row["filename"])
                by_name.setdefault(sanitized, doc_meta)

        case_data = case_index or {}
        cls._merge_case_index_documents(by_name, by_id, case_data)

        cache_files = list(dict.fromkeys((case_data.get("cache_files") or []) + (case_data.get("documents") or [])))
        cls._merge_cache_files_metadata(by_name, cache_files, cache_manager)

        # Asegurar que todas las claves tengan versión sanitizada
        for key, value in list(by_name.items()):
            sanitized = _sanitize_filename(key)
            by_name.setdefault(sanitized, value)

        return by_name, by_id

    @staticmethod
    def _merge_case_index_documents(
        by_name: Dict[str, Dict[str, Any]],
        by_id: Dict[str, Dict[str, Any]],
        case_data: Dict[str, Any],
    ) -> None:
        docs_meta = case_data.get("documents_metadata")
        if not isinstance(docs_meta, list):
            return
        for entry in docs_meta:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("filename")
            canonical_name = entry.get("canonical_name")
            doc_id = entry.get("document_id")
            payload = {
                "original_filename": entry.get("original_filename") or name or canonical_name,
                "document_type": entry.get("document_type") or entry.get("type"),
            }
            for candidate in filter(None, {name, canonical_name}):
                _store_metadata_candidate(by_name, candidate, payload)
            if doc_id:
                existing = dict(by_id.get(doc_id, {}))
                existing.update({k: v for k, v in payload.items() if v is not None})
                by_id[doc_id] = existing

    @staticmethod
    def _merge_cache_files_metadata(
        by_name: Dict[str, Dict[str, Any]],
        cache_files: Sequence[str],
        cache_manager: Optional["OCRCacheManager"],
    ) -> None:
        for path_str in cache_files or []:
            try:
                path = Path(path_str)
                if not path.exists() and cache_manager:
                    candidate = cache_manager.cache_dir / Path(path_str)
                    if candidate.exists():
                        path = candidate
                if not path.exists():
                    continue
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            metadata = data.get("metadata") or {}
            original_name = metadata.get("file_name")
            derived_name = _derive_document_name_from_cache_path(path)
            payload = {
                "original_filename": original_name or derived_name,
                "ocr_metadata": metadata,
                "page_count": metadata.get("page_count"),
                "language": metadata.get("language"),
            }
            for candidate in filter(None, {derived_name, original_name}):
                _store_metadata_candidate(by_name, candidate, payload)

    @staticmethod
    def _metadata_for_document(
        name: Optional[str],
        metadata_by_name: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not name:
            return {}
        if name in metadata_by_name:
            return metadata_by_name[name]
        sanitized = _sanitize_filename(name)
        if sanitized in metadata_by_name:
            return metadata_by_name[sanitized]
        lower = name.lower()
        for key, value in metadata_by_name.items():
            if key.lower() == lower:
                return value
        return {}

    @staticmethod
    def _build_aggregates(
        documents_by_type: Dict[str, List[DocumentSnapshot]]
    ) -> Dict[str, Dict[str, Any]]:
        aggregates: Dict[str, Dict[str, Any]] = {}
        for doc_type, docs in documents_by_type.items():
            field_totals: Dict[str, float] = defaultdict(float)
            field_values: Dict[str, List[Any]] = defaultdict(list)
            numeric_fields: Dict[str, bool] = defaultdict(bool)
            for doc in docs:
                for field, value in (doc.extracted_fields or {}).items():
                    if value is None:
                        continue
                    numeric = _to_float(value)
                    if numeric is not None:
                        field_totals[field] += numeric
                        numeric_fields[field] = True
                        field_values[field].append(numeric)
                    else:
                        field_values[field].append(value)
            aggregates[doc_type] = {}
            all_fields = set(list(field_totals.keys()) + list(field_values.keys()))
            for field in all_fields:
                if numeric_fields.get(field):
                    aggregates[doc_type][field] = field_totals[field]
                else:
                    aggregates[doc_type][field] = field_values[field]
            aggregates[doc_type]["_count"] = len(docs)
        return aggregates

    @staticmethod
    def _build_entities(
        consolidated: Optional[ConsolidatedExtraction],
        documents: Iterable[DocumentSnapshot],
        normalizer: Optional[EntityNormalizer],
    ) -> Dict[str, List[Any]]:
        entities: Dict[str, List[Any]] = defaultdict(list)

        if consolidated and consolidated.consolidated_fields:
            consolidated_fields: ConsolidatedFields = consolidated.consolidated_fields
            for field_name, value in (consolidated_fields.model_dump() or {}).items():
                if value is None:
                    continue
                canonical = normalizer.canonical_field_name(field_name) if normalizer else field_name
                entities[canonical].append(value)

        for doc in documents:
            for field_name, value in (doc.extracted_fields or {}).items():
                if value is None:
                    continue
                canonical = normalizer.canonical_field_name(field_name) if normalizer else field_name
                entities[canonical].append(value)

        return {k: entities[k] for k in entities}

    @staticmethod
    def _build_timeline(
        consolidated: Optional[ConsolidatedExtraction],
        documents: Iterable[DocumentSnapshot],
    ) -> List[Dict[str, Any]]:
        timeline: List[Dict[str, Any]] = []

        consolidated_map: Dict[str, str] = {
            "fecha_ocurrencia": "Fecha de ocurrencia",
            "fecha_reclamacion": "Fecha de reclamación",
            "vigencia_inicio": "Inicio de vigencia",
            "vigencia_fin": "Fin de vigencia",
        }

        if consolidated and consolidated.consolidated_fields:
            for field_name, label in consolidated_map.items():
                value = getattr(consolidated.consolidated_fields, field_name, None)
                norm = _normalise_date(value)
                if norm:
                    timeline.append(
                        {
                            "label": label,
                            "date": norm,
                            "source": "consolidated",
                            "field": field_name,
                        }
                    )

        for doc in documents:
            for field_name, value in (doc.extracted_fields or {}).items():
                if "fecha" not in field_name.lower():
                    continue
                norm = _normalise_date(value)
                if norm:
                    timeline.append(
                        {
                            "label": field_name,
                            "date": norm,
                            "source": doc.document_type,
                            "document_id": doc.document_id,
                        }
                    )

        timeline.sort(key=lambda item: item.get("date") or "")
        return timeline

    def _build_data_tree(self) -> Dict[str, Any]:
        doc_alias_index = getattr(self, "_document_alias_index", {})
        field_alias_index = getattr(self, "_field_alias_index", {})

        def _serialize_snapshot(doc: DocumentSnapshot) -> Dict[str, Any]:
            payload = doc.model_dump()
            fields = dict(payload.get("extracted_fields") or {})
            for alias, canonical in (doc.field_aliases or {}).items():
                if canonical in fields and alias not in fields:
                    fields[alias] = fields[canonical]
            payload["extracted_fields"] = fields
            payload.setdefault("document_aliases", doc_alias_index.get(doc.document_type, []))
            return payload

        documents_tree: Dict[str, List[Dict[str, Any]]] = {}
        for canonical, docs in self.documents_by_type.items():
            payloads = [_serialize_snapshot(doc) for doc in docs]
            documents_tree[canonical] = payloads
            for alias in doc_alias_index.get(canonical, []):
                if alias != canonical and alias not in documents_tree:
                    documents_tree[alias] = payloads

        aggregates_tree: Dict[str, Dict[str, Any]] = {}
        for doc_type, values in self.aggregates.items():
            enriched = dict(values)
            for canonical_field, aliases in field_alias_index.items():
                if canonical_field in enriched:
                    for alias in aliases:
                        if alias != canonical_field and alias not in enriched:
                            enriched[alias] = enriched[canonical_field]
            aggregates_tree[doc_type] = enriched
        for canonical, aliases in doc_alias_index.items():
            if canonical not in aggregates_tree:
                continue
            for alias in aliases:
                if alias != canonical and alias not in aggregates_tree:
                    aggregates_tree[alias] = aggregates_tree[canonical]

        entities_tree: Dict[str, List[Any]] = {k: list(v) for k, v in self.entities.items()}
        for canonical, aliases in field_alias_index.items():
            if canonical not in entities_tree:
                continue
            for alias in aliases:
                if alias != canonical and alias not in entities_tree:
                    entities_tree[alias] = entities_tree[canonical]

        documents_list = [_serialize_snapshot(doc) for doc in self.documents]

        tree: Dict[str, Any] = {
            "case_id": self.case_id,
            "consolidated": self.consolidated.model_dump() if self.consolidated else {},
            "documents": documents_tree,
            "documents_list": documents_list,
            "aggregates": aggregates_tree,
            "entities": entities_tree,
            "timeline": self.timeline,
            "fraud_results": {
                doc_id: fraud.model_dump() for doc_id, fraud in self.fraud_results.items()
            },
            "metadata": self.metadata,
        }
        return tree

    def resolve(self, path: str, default: Any = None) -> Any:
        """Resuelve una ruta tipo dotted access sobre la estructura del contexto."""
        if not path:
            return default

        current: Any = self._data_tree
        tokens = self._tokenise(path)
        for token in tokens:
            if isinstance(current, dict):
                current = current.get(token, default)
            elif isinstance(current, list):
                try:
                    index = int(token)
                except ValueError:
                    return default
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return default
            else:
                return default
            if current is default:
                return default
        return current

    @staticmethod
    def _tokenise(path: str) -> List[str]:
        tokens: List[str] = []
        for raw in path.split("."):
            if "[" in raw and raw.endswith("]"):
                base, index = raw[:-1].split("[", 1)
                if base:
                    tokens.append(base)
                tokens.append(index)
            else:
                tokens.append(raw)
        return [t for t in tokens if t]

    def as_data_tree(self) -> Dict[str, Any]:
        return dict(self._data_tree)
