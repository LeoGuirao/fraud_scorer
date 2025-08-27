# src/fraud_scorer/pipelines/utils.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime
import os  # por si luego lees envs

logger = logging.getLogger(__name__)

# -------------------------------------------------
# DB access (para modo replay y consultas compartidas)
# -------------------------------------------------
try:
    from fraud_scorer.storage.db import get_conn  # type: ignore
except Exception:
    get_conn = None  # en entornos donde no haya DB (tests unitarios de utilidades)

# -------------------------------------------------
# Intentamos importar OCRResult (utils debe funcionar aunque no esté disponible)
# -------------------------------------------------
try:
    from fraud_scorer.processors.ocr.azure_ocr import OCRResult  # type: ignore
except Exception:
    OCRResult = None  # type: ignore[misc]

# -------------------------------------------------
# Mapeo canónico (YAML + semántico)
# apply_canonical_mapping fusiona en specific_fields los campos canónicos detectados.
# -------------------------------------------------
try:
    from fraud_scorer.pipelines.mapper import apply_canonical_mapping  # type: ignore
except Exception:
    apply_canonical_mapping = None  # fallback: si no está disponible, no rompe


# -------------------------------------------------
# Normalización de OCR (usado por run_report.py)
# -------------------------------------------------
def ocr_result_to_dict(ocr_obj: Any) -> Dict[str, Any]:
    """
    Normaliza un resultado de OCR (dict o dataclass) a un dict estable
    que entiende el extractor. Acepta:
      - dict parecido al OCR esperado
      - OCRResult (tu dataclass)
      - objetos con atributos similares
    """
    if isinstance(ocr_obj, dict):
        out: Dict[str, Any] = dict(ocr_obj)
        out.setdefault("text", out.get("text", ""))
        out.setdefault("tables", out.get("tables", []))
        if "key_value_pairs" not in out:
            out["key_value_pairs"] = out.get("key_values", {}) or {}
        if "confidence_scores" not in out:
            c = out.get("confidence", {})
            out["confidence_scores"] = c if isinstance(c, dict) else {}
        meta = out.get("metadata", {}) or {}
        # homogenizar metadatos básicos
        out.setdefault("page_count", out.get("page_count", meta.get("page_count", 1)))
        out.setdefault("language", out.get("language", meta.get("language", "es")))
        # ayudar a tener source_name si Azure guardó file_name
        if "source_name" not in meta and meta.get("file_name"):
            meta["source_name"] = meta["file_name"]
        out["metadata"] = meta
        return out

    if OCRResult and isinstance(ocr_obj, OCRResult):
        meta = ocr_obj.metadata or {}
        # ayudar a tener source_name si Azure guardó file_name
        if "source_name" not in meta and meta.get("file_name"):
            meta["source_name"] = meta["file_name"]
        return {
            "text": ocr_obj.text or "",
            "tables": ocr_obj.tables or [],
            "key_value_pairs": getattr(ocr_obj, "key_values", {}) or {},
            "entities": ocr_obj.entities or [],
            "confidence_scores": ocr_obj.confidence or {},
            "page_count": meta.get("page_count", 1),
            "language": meta.get("language", "es"),
            "metadata": meta,
            "errors": ocr_obj.errors or [],
            "success": bool(ocr_obj.success),
        }

    md = getattr(ocr_obj, "metadata", {}) or {}
    # ayudar a tener source_name si Azure guardó file_name
    if "source_name" not in md and md.get("file_name"):
        md["source_name"] = md["file_name"]
    return {
        "text": getattr(ocr_obj, "text", "") or "",
        "tables": getattr(ocr_obj, "tables", []) or [],
        "key_value_pairs": getattr(ocr_obj, "key_values", {}) or {},
        "entities": getattr(ocr_obj, "entities", []) or [],
        "confidence_scores": getattr(ocr_obj, "confidence", {}) or {},
        "page_count": md.get("page_count", 1),
        "language": md.get("language", "es"),
        "metadata": md,
        "errors": getattr(ocr_obj, "errors", []) or [],
        "success": bool(getattr(ocr_obj, "success", False)),
    }


# -------------------------------------------------
# Normalizador para TemplateProcessor
# -------------------------------------------------
def _normalize_for_template(
    extracted: Dict[str, Any],
    raw_ocr: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Unifica el formato que espera TemplateProcessor.extract_from_documents:
    {
      document_type, raw_text, entities(list), key_value_pairs, specific_fields, ocr_metadata
    }
    Aplica además:
      - Inyección de ocr_metadata.source_name (si solo viene file_name).
      - Mapeo canónico de campos (YAML + semántico) sobre specific_fields.
    """
    raw_ocr = raw_ocr or {}
    ext = extracted or {}

    # ENTITIES: si vienen como dict (del extractor), convertir a lista homogénea.
    entities = ext.get("entities")
    if isinstance(entities, dict):
        ent_list: List[Dict[str, Any]] = []
        for k, vals in entities.items():
            vals_iter = vals if isinstance(vals, list) else [vals]
            for v in vals_iter:
                ent_list.append({"type": k, "value": v})
        entities_out = ent_list
    elif isinstance(entities, list):
        entities_out = entities
    else:
        entities_out = raw_ocr.get("entities", [])

    # --- Asegurar metadatos de origen (source_name a partir de file_name) ---
    ocr_meta = ext.get("ocr_metadata", raw_ocr.get("metadata", {}) or {})
    if "source_name" not in ocr_meta:
        if ocr_meta.get("file_name"):
            ocr_meta["source_name"] = ocr_meta["file_name"]
        elif raw_ocr.get("metadata", {}) and raw_ocr["metadata"].get("file_name"):
            ocr_meta["source_name"] = raw_ocr["metadata"]["file_name"]

    # --- Aplicar mapeo canónico (YAML + semántico) ---
    extracted_aug = dict(ext)
    extracted_aug["ocr_metadata"] = ocr_meta
    if apply_canonical_mapping is not None:
        try:
            extracted_aug = apply_canonical_mapping(extracted_aug, raw_ocr)
        except Exception as e:
            # Blindaje: no cortar flujo si el mapeo (embeddings/fuzzy) falla
            logger.warning(f"apply_canonical_mapping falló; continuando sin mapeo canónico: {e}")

    return {
        "document_type": extracted_aug.get("document_type", "otro"),
        "raw_text": extracted_aug.get("raw_text") or raw_ocr.get("text", ""),
        "entities": entities_out,
        "key_value_pairs": extracted_aug.get("key_value_pairs", {}),
        "specific_fields": extracted_aug.get("specific_fields", {}),
        "ocr_metadata": ocr_meta,
    }


def build_docs_for_template_from_processed(
    processed_docs: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Recibe la lista de records producida por run_report tras OCR+extracción:
    [{
      file_name, file_path, ..., ocr_data: {...}, extracted_data: {...}
    }]
    y devuelve la lista normalizada para TemplateProcessor.
    """
    out: List[Dict[str, Any]] = []
    for d in processed_docs:
        ocr_data = d.get("ocr_data") or {}
        extracted = d.get("extracted_data") or {}
        if not extracted:
            continue  # saltamos documentos sin extracción útil
        out.append(_normalize_for_template(extracted, ocr_data))
    return out


# -------------------------------------------------
# Accesos a DB para modo replay
# -------------------------------------------------
def load_case_header(case_id: str) -> Dict[str, Any]:
    """
    Lee la fila de 'cases' para un case_id.
    """
    if not get_conn:
        raise RuntimeError("DB no disponible (get_conn no importable).")
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if not row:
            raise RuntimeError(f"case_id no encontrado: {case_id}")
        return dict(row)


def _jload(val: Any, default: Any) -> Any:
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return default


def build_docs_for_template_from_db(case_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Consulta la DB y devuelve:
      - docs_for_template: entrada para analyze_claim_documents() + TemplateProcessor
      - processed_docs: estructura útil para snapshots/json
    """
    if not get_conn:
        raise RuntimeError("DB no disponible (get_conn no importable).")

    docs_for_template: List[Dict[str, Any]] = []
    processed_docs: List[Dict[str, Any]] = []

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT d.id as document_id, d.filename, d.filepath, d.size_bytes, d.page_count, d.language,
                   o.raw_text, o.key_value_pairs, o.tables, o.entities, o.confidence, o.metadata, o.errors,
                   e.document_type, e.entities as e_entities, e.key_value_pairs as e_kv, e.extra
            FROM documents d
            LEFT JOIN ocr_results o ON o.document_id = d.id
            LEFT JOIN extracted_data e ON e.document_id = d.id
            WHERE d.case_id = ?
            ORDER BY d.created_at ASC
            """,
            (case_id,),
        ).fetchall()

    for r in rows:
        # parsear JSONs almacenados
        o_kv = _jload(r["key_value_pairs"], {}) if "key_value_pairs" in r.keys() else {}
        o_tables = _jload(r["tables"], []) if "tables" in r.keys() else []
        o_entities = _jload(r["entities"], []) if "entities" in r.keys() else []
        o_conf = _jload(r["confidence"], {}) if "confidence" in r.keys() else {}
        o_meta = _jload(r["metadata"], {}) if "metadata" in r.keys() else {}
        o_errs = _jload(r["errors"], []) if "errors" in r.keys() else []

        e_entities = _jload(r["e_entities"], {}) if "e_entities" in r.keys() else {}
        e_kv = _jload(r["e_kv"], {}) if "e_kv" in r.keys() else {}
        e_extra = _jload(r["extra"], {}) if "extra" in r.keys() else {}

        # Estructura "ocr_data" (similar a la usada en nuevas corridas)
        ocr_dict = {
            "text": r["raw_text"] or "",
            "tables": o_tables,
            "entities": o_entities,
            "key_value_pairs": o_kv,
            "confidence_scores": o_conf,
            "metadata": o_meta,
            "errors": o_errs,
            "page_count": r["page_count"],
            "language": r["language"],
        }

        # Estructura "extracted_data"
        extracted = {
            "document_type": r["document_type"] or "desconocido",
            "entities": e_entities,
            "key_value_pairs": e_kv,
            # extra trae campos varios según tu extractor:
            **(e_extra or {}),
        }

        processed_docs.append({
            "document_id": r["document_id"],
            "file_name": r["filename"],
            "file_path": r["filepath"],
            "file_size_kb": (r["size_bytes"] or 0) / 1024.0 if r["size_bytes"] else None,
            "ocr_data": ocr_dict,
            "extracted_data": extracted if extracted.get("document_type") else None,
        })

        # Aquí también pasa por el mismo normalizador -> mapeo canónico incluido
        docs_for_template.append(_normalize_for_template(extracted, ocr_dict))

    return docs_for_template, processed_docs


# -------------------------------------------------
# Riesgo / resumen
# -------------------------------------------------
def summarize_risk(fraud_score_0_1: float) -> str:
    """
    Convierte score [0,1] a etiqueta textual.
    """
    if fraud_score_0_1 < 0.3:
        return "bajo"
    if fraud_score_0_1 < 0.6:
        return "medio"
    return "alto"


# -------------------------------------------------
# Reporte (HTML/PDF) + JSON
# -------------------------------------------------
def render_report(
    template_processor: Any,
    docs_for_template: List[Dict[str, Any]],
    ai_analysis: Dict[str, Any],
    output_dir: Path,
    default_name: str
) -> Tuple[Path, Optional[Path]]:
    """
    Usa tu TemplateProcessor para generar HTML y PDF.
    Devuelve (html_path, pdf_path|None).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construir el InformeSiniestro desde docs+AI
    informe = template_processor.extract_from_documents(docs_for_template, ai_analysis)
    numero_siniestro = getattr(informe, "numero_siniestro", default_name)

    html_path = output_dir / f"INF-{numero_siniestro}.html"
    template_processor.generate_report(informe, str(html_path))

    pdf_path: Optional[Path] = None
    try:
        from weasyprint import HTML  # opcional
        pdf_path = html_path.with_suffix(".pdf")
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    except ImportError:
        logger.warning("WeasyPrint no instalado. PDF omitido.")
    except Exception as e:
        logger.warning(f"Error generando PDF: {e}")

    return html_path, pdf_path


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def save_json_snapshot(data: Dict[str, Any], path: Path) -> None:
    """
    Alias semántico para guardar snapshots técnicos (resultados, etc.)
    """
    save_json(path, data)
