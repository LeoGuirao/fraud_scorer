#!/usr/bin/env python3
"""
CLI para ejecutar el análisis de fraude de un documento y mostrar la salida
resultante en consola. Reutiliza la misma cadena de preparación usada por
`FraudDocumentReprocessService`.

Ejemplo de uso:

    python scripts/preview_fraud_analysis.py --case-id CASE-2025-0001 \
        --document-name "1 1 Carta reclamacion gastos HDI.pdf" --no-save
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

try:  # pragma: no cover - carga opcional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - fallback silencioso
    pass

from fraud_scorer.analyzers.fraud_analyzer import FraudAnalyzer
from fraud_scorer.analyzers.unified_data_layer import UnifiedDataLayer
from fraud_scorer.storage.db import get_conn, save_extracted_data
from fraud_scorer.storage.ocr_cache import OCRCacheManager
from fraud_scorer.services.fraud_document_service import (
    FraudDocumentCatalog,
    FraudDocumentReprocessService,
)
from fraud_scorer.processors.ai.ai_field_extractor import AIFieldExtractor


def _resolve_document_id(case_id: str, document_id: Optional[str], document_name: Optional[str]) -> str:
    if document_id:
        return document_id
    if not document_name:
        raise ValueError("Debe especificar --document-id o --document-name.")

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM documents WHERE case_id = ? AND filename = ? ORDER BY created_at DESC LIMIT 1",
            (case_id, document_name),
        ).fetchone()
        if not row:
            raise LookupError(f"No se encontró documento '{document_name}' para el caso {case_id}")
        return row["id"]


SPANISH_MONTHS = {
    "ENERO": "01",
    "FEBRERO": "02",
    "MARZO": "03",
    "ABRIL": "04",
    "MAYO": "05",
    "JUNIO": "06",
    "JULIO": "07",
    "AGOSTO": "08",
    "SEPTIEMBRE": "09",
    "SETIEMBRE": "09",
    "OCTUBRE": "10",
    "NOVIEMBRE": "11",
    "DICIEMBRE": "12",
}


def _parse_spanish_date(text: str) -> Optional[str]:
    import re

    pattern = re.compile(
        r"(\d{1,2})\s+DE\s+([A-ZÁÉÍÓÚÑ]+)\s+DE\s+(\d{4})",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None
    day, month_text, year = match.groups()
    month_key = month_text.strip().upper()
    month = SPANISH_MONTHS.get(month_key)
    if not month:
        return None
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _fallback_letter_fields(fields: Dict[str, Any], raw_text: str) -> bool:
    import re

    updated = False
    upper_text = raw_text.upper()
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if not fields.get("nombre_asegurado") and lines:
        fields["nombre_asegurado"] = lines[0]
        updated = True

    if not fields.get("numero_poliza"):
        match = re.search(r"P[ÓO]LIZA[:\s]+([A-Z0-9\-\s]+)", upper_text)
        if match:
            fields["numero_poliza"] = match.group(1).strip().rstrip(".")
            updated = True

    if not fields.get("numero_siniestro"):
        match = re.search(r"SINIESTRO[:\s]+([A-Z0-9\-]+)", upper_text)
        if match:
            fields["numero_siniestro"] = match.group(1).strip()
            updated = True

    if not fields.get("fecha_reclamacion"):
        parsed = _parse_spanish_date(upper_text)
        if parsed:
            fields["fecha_reclamacion"] = parsed
            updated = True

    if not fields.get("fecha_ocurrencia"):
        match = re.search(r"D[ÍI]A\s+(\d{1,2})\s+DE\s+([A-ZÁÉÍÓÚÑ]+)\s+DE\s+(\d{4})", upper_text)
        if match:
            day, month_text, year = match.groups()
            month = SPANISH_MONTHS.get(month_text.upper(), "01")
            fields["fecha_ocurrencia"] = f"{year}-{int(month):02d}-{int(day):02d}"
            updated = True

    if not fields.get("monto_reclamacion"):
        pattern = re.compile(r"\$\s?([0-9,.]+)\s?\(TRES MILLONES", re.IGNORECASE)
        match = pattern.search(raw_text)
        if not match:
            match = re.search(r"\$\s?([0-9,.]+)", raw_text)
        if match:
            amt_str = match.group(1)
            fields["monto_reclamacion"] = f"${amt_str}"
            updated = True

    if not fields.get("lugar_hechos"):
        match = re.search(r"KIL[ÓO]METRO\s+\d+[^\n]*CARRETERA[^\n]+", raw_text, re.IGNORECASE)
        if match:
            lugar = match.group(0).strip().rstrip('. ')
            fields["lugar_hechos"] = lugar
            updated = True

    if not fields.get("bien_reclamado"):
        if "MATERIAL DE ACERO" in upper_text:
            fields["bien_reclamado"] = "material de acero"
            updated = True

    return updated


def _fallback_adjuster_fields(fields: Dict[str, Any], raw_text: str) -> bool:
    import re

    updated = False

    if not fields.get("monto_reclamacion"):
        match = re.search(
            r"Recibimos del Asegurado[^$]*\$\s?([0-9,.]+)",
            raw_text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            fields["monto_reclamacion"] = f"${match.group(1)}"
            updated = True
        else:
            fallback = re.search(r"\$\s?([0-9,.]+)", raw_text)
            if fallback:
                fields["monto_reclamacion"] = f"${fallback.group(1)}"
                updated = True

    if not fields.get("nombre_asegurado"):
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            if line.lower() == "asegurado" and idx + 2 < len(lines):
                fields["nombre_asegurado"] = lines[idx + 2]
                updated = True
                break

    return updated


async def _refresh_extraction(
    *,
    case_id: str,
    document_id: str,
    document_name: str,
    document_type: str,
    cache_manager: OCRCacheManager,
) -> None:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY requerido para refrescar la extracción.")

    case_index = cache_manager.get_case_index(case_id, auto_reconstruct=True) or {}
    documents = case_index.get("documents") or []
    sanitized_hint = document_name.replace(" ", "_")
    target_path: Optional[Path] = None
    for entry in documents:
        path_obj = Path(entry)
        name = path_obj.name
        if name.endswith(document_name):
            target_path = path_obj
            break
        if sanitized_hint and sanitized_hint in name:
            target_path = path_obj
            break
    if target_path is None:
        raise FileNotFoundError(f"No se encontró el documento {document_name} en el índice del caso {case_id}.")

    ocr_result = cache_manager.get_cache(target_path, case_id)
    if not ocr_result:
        raise RuntimeError(f"No existe OCR cacheado para {document_name}; ejecuta reprocess_single_ocr primero.")

    extractor = AIFieldExtractor(api_key=api_key)
    extraction = await extractor.extract_from_document_guided(
        content=ocr_result,
        document_name=document_name,
        document_type=document_type,
        route="ocr_text",
        use_cache=False,
    )

    raw_text = ocr_result.get("text")
    if document_type == "carta_de_reclamacion_formal_a_la_aseguradora" and raw_text:
        if _fallback_letter_fields(extraction.extracted_fields, raw_text):
            extraction.extraction_metadata = extraction.extraction_metadata or {}
            extraction.extraction_metadata.setdefault("fallback_letter", True)
    elif document_type == "informe_final_del_ajustador" and raw_text:
        if _fallback_adjuster_fields(extraction.extracted_fields, raw_text):
            extraction.extraction_metadata = extraction.extraction_metadata or {}
            extraction.extraction_metadata.setdefault("fallback_adjuster", True)

    save_extracted_data(
        document_id,
        {
            "document_type": extraction.document_type,
            "key_value_pairs": extraction.extracted_fields,
            "entities": {},
            "extraction_metadata": extraction.extraction_metadata,
        },
        extractor_version="ai-guided-refresh",
    )

    updated = False
    extraction_entry = {
        "source_document": document_name,
        "document_type": extraction.document_type,
        "extracted_fields": extraction.extracted_fields,
        "extraction_metadata": extraction.extraction_metadata,
    }
    results = case_index.setdefault("extraction_results", [])
    for idx, item in enumerate(results):
        if isinstance(item, dict) and item.get("source_document") == document_name:
            results[idx] = extraction_entry
            updated = True
            break
    if not updated:
        results.append(extraction_entry)

    index_path = Path("data/ocr_cache/case_index") / f"{case_id}.json"
    index_path.write_text(json.dumps(case_index, ensure_ascii=False, indent=2), encoding="utf-8")


async def preview_analysis(case_id: str, document_id: str, *, no_save: bool, refresh_extraction: bool) -> None:
    catalog = FraudDocumentCatalog()
    case_index = catalog.load_case_index(case_id)

    reports_dir = Path(os.getenv("FS_REPORTS_DIR", "data/reports"))
    templates_dir = Path(__file__).resolve().parents[1] / "src" / "fraud_scorer" / "templates"

    reprocess_service = FraudDocumentReprocessService(
        catalog=catalog,
        reports_dir=reports_dir,
        templates_dir=templates_dir,
    )

    context = reprocess_service._build_document_context(  # noqa: SLF001
        case_id=case_id,
        document_id=document_id,
        case_index=case_index,
        include_flag=True,
    )

    if refresh_extraction:
        await _refresh_extraction(
            case_id=case_id,
            document_id=document_id,
            document_name=context.document_name,
            document_type=context.document_type,
            cache_manager=catalog.cache_manager,
        )
        case_index = catalog.load_case_index(case_id)
        context = reprocess_service._build_document_context(  # noqa: SLF001
            case_id=case_id,
            document_id=document_id,
            case_index=case_index,
            include_flag=True,
        )

    data_layer = UnifiedDataLayer.from_case_index(case_index)
    analyzer = FraudAnalyzer()

    if no_save:
        async def _noop(*args, **kwargs):  # type: ignore[no-untyped-def]
            return None

        analyzer._save_analysis_to_db = _noop  # type: ignore[attr-defined]

    analysis = await analyzer.analyze_document(
        document_id=context.document_id,
        document_name=context.document_name,
        document_type=context.document_type,
        ocr_result=context.ocr_result,
        extraction=context.extraction,
        case_id=case_id,
        context=reprocess_service._build_analysis_context(case_index, data_layer=data_layer),  # noqa: SLF001
        data_layer=data_layer,
    )

    print("\n=== Análisis ===\n")
    print(analysis.analisis_completo)

    print("\n=== Indicadores ===\n")
    if analysis.indicators:
        for ind in analysis.indicators:
            print(f"- {ind.pattern} [{ind.severity}] ({ind.confidence:.2f}): {ind.description}")
    else:
        print("Sin indicadores registrados.")

    print("\n=== Verificaciones ===\n")
    for key, data in analysis.verificaciones.items():
        print(f"{key}: {data}")

    print("\n=== Validación Cruzada ===\n")
    for key, data in analysis.validacion_cruzada.items():
        print(f"{key}: {data}")

    print("\n=== Recomendaciones ===\n")
    if analysis.recommendations:
        for rec in analysis.recommendations:
            print(f"- {rec}")
    else:
        print("Sin recomendaciones.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Previsualiza el análisis de fraude de un documento concreto.")
    parser.add_argument("--case-id", required=True, help="ID del caso (ej. CASE-2025-0001)")
    parser.add_argument("--document-id", help="ID interno del documento (columna documents.id)")
    parser.add_argument("--document-name", help="Nombre del archivo (filename) para resolver el ID automáticamente")
    parser.add_argument("--no-save", action="store_true", help="No persistir el análisis en la base de datos")
    parser.add_argument("--refresh-extraction", action="store_true", help="Reejecutar la extracción guiada antes del análisis")
    args = parser.parse_args()

    document_id = _resolve_document_id(args.case_id, args.document_id, args.document_name)
    asyncio.run(preview_analysis(args.case_id, document_id, no_save=args.no_save, refresh_extraction=args.refresh_extraction))


if __name__ == "__main__":
    main()
