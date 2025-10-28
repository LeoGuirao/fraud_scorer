"""Herramienta de búsqueda documental para el modo agentic del Agente Rick."""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Iterable, Optional

from langchain.tools import StructuredTool

from fraud_scorer.ai.retrieval import RickRetriever
from fraud_scorer.ai.retrieval.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

_DENUNCIA_TYPES = {
    "denuncia_de_los_hechos",
    "oficio_denuncia",
    "denuncia",
}

_TIME_RANGE_PATTERN = re.compile(
    r"\b\d{1,2}[:h]\d{2}(?:\s*(?:hrs?|horas?))?(?:\s*-\s*\d{1,2}[:h]\d{2}(?:\s*(?:hrs?|horas?))?)?",
    re.IGNORECASE,
)

_PLATE_PATTERN = re.compile(r"\b(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9-]{4,12}\b")


def _normalize_time_string(raw: str) -> str:
    value = raw.strip()
    value = re.sub(r"(?<=\d)[hH](?=\d)", ":", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _append_denuncia_highlights(text: str, metadata: dict) -> str:
    document_type = (metadata or {}).get("document_type")
    if document_type not in _DENUNCIA_TYPES:
        return text

    highlights: list[str] = []
    structured_raw = (metadata or {}).get("structured_highlights") or {}
    if isinstance(structured_raw, str):
        try:
            structured = json.loads(structured_raw)
        except (json.JSONDecodeError, TypeError):
            structured = {}
    else:
        structured = structured_raw if isinstance(structured_raw, dict) else {}

    time_matches = structured.get("time_ranges") or []
    if not time_matches:
        time_matches = sorted({_normalize_time_string(match.group(0)) for match in _TIME_RANGE_PATTERN.finditer(text)})
    if time_matches:
        highlights.append("Horas declaradas: " + ", ".join(dict.fromkeys(time_matches)))

    plate_matches = structured.get("vehicles") or []
    if not plate_matches:
        plate_matches = sorted({match.group(0) for match in _PLATE_PATTERN.finditer(text)})
    if plate_matches:
        highlights.append("Placas/números económicos detectados: " + ", ".join(dict.fromkeys(plate_matches)))

    reference_point = structured.get("reference_point")
    if reference_point:
        lat = reference_point.get("lat")
        lon = reference_point.get("lon")
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            lat_f = lon_f = None
        if lat_f is not None and lon_f is not None:
            highlights.append(f"Referencia geográfica sugerida: lat={lat_f:.6f}, lon={lon_f:.6f}")

    if not highlights:
        return text

    notes = "\n".join(f"- {item}" for item in highlights)
    return f"{text}\n[Claves detectadas]\n{notes}"

def _format_document(doc: RetrievedDocument) -> str:
    """Construye un bloque textual compacto con metadatos básicos."""
    meta = doc.document.metadata if isinstance(doc.document.metadata, dict) else {}
    source_document = meta.get("source_document") or meta.get("case_path") or "documento_sin_nombre"
    document_type = meta.get("document_type")
    phase = meta.get("source")
    header_parts: list[str] = [source_document]
    if document_type:
        header_parts.append(f"tipo: {document_type}")
    if phase:
        header_parts.append(f"fase: {phase}")
    header_parts.append(f"similitud: {doc.similarity:.2f}")
    header = " | ".join(header_parts)
    body = doc.document.page_content.strip()
    body = _append_denuncia_highlights(body, meta)
    return f"[{header}]\n{body}"


def build_rag_search_tool(
    *,
    retriever: RickRetriever,
    max_results: int,
    formatter: Optional[Callable[[RetrievedDocument], str]] = None,
    on_results: Optional[Callable[[list[RetrievedDocument]], None]] = None,
) -> StructuredTool:
    """Crea un StructuredTool que expone el retriever híbrido como herramienta LangChain."""

    if max_results <= 0:
        raise ValueError("max_results debe ser mayor a 0.")

    formatter = formatter or _format_document

    def _search(
        case_id: str,
        query: str,
        limit: Optional[int] = None,
        document_type: Optional[str] = None,
    ) -> str:
        cleaned_case_id = (case_id or "").strip()
        cleaned_query = (query or "").strip()
        if not cleaned_case_id:
            raise ValueError("case_id es obligatorio.")
        if not cleaned_query:
            raise ValueError("query es obligatoria.")
        if len(cleaned_query) > 600:
            raise ValueError("query excede la longitud máxima permitida (600 caracteres).")

        top_k = min(max_results, limit or max_results)
        metadata_filter = None
        if document_type:
            metadata_filter = {"document_type": document_type}

        results: Iterable[RetrievedDocument] = retriever.retrieve(
            cleaned_case_id,
            cleaned_query,
            k=top_k,
            metadata_filter=metadata_filter,
        )
        results_list = list(results)
        if on_results is not None:
            try:
                on_results(results_list)
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Callback on_results falló: %s", exc)
        if not results_list:
            logger.info("Tool search_case_documents sin resultados para %s", cleaned_case_id)
            return "NO_CONTEXT"

        snippets = [formatter(item) for item in results_list[:5]]
        return "\n\n".join(snippets)

    return StructuredTool.from_function(
        func=_search,
        name="search_case_documents",
        description=(
            "Busca información en los documentos OCR, pólizas y denuncias del caso. "
            "Úsalo para recuperar fechas, ubicaciones, declaraciones o cualquier dato textual. "
            "Incluye document_type para restringir la búsqueda a un tipo específico (ej. 'denuncia_de_los_hechos')."
        ),
    )


__all__ = ["build_rag_search_tool"]
