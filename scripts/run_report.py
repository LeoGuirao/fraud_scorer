#!/usr/bin/env python3
import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fraud_scorer.run_report")

# ----------------------------
# Imports con fallback a src/
# ----------------------------
try:
    from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor, OCRResult
    from fraud_scorer.processors.ocr.document_extractor import UniversalDocumentExtractor
    from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
    from fraud_scorer.templates.template_processor import TemplateProcessor
    # ⬇️ create_case viene de storage.cases
    from fraud_scorer.storage.cases import create_case
    # ⬇️ el resto desde storage.db
    from fraud_scorer.storage.db import (
        get_conn,
        upsert_document,
        mark_ocr_success,
        save_ocr_result,
        save_extracted_data,
        create_run,
        save_ai_analysis,
    )
except ImportError:
    from pathlib import Path as _Path
    _REPO_ROOT = _Path(__file__).resolve().parents[1]
    _SRC_PATH = _REPO_ROOT / "src"
    if _SRC_PATH.exists():
        sys.path.insert(0, str(_SRC_PATH))
    from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor, OCRResult
    from fraud_scorer.processors.ocr.document_extractor import UniversalDocumentExtractor
    from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
    from fraud_scorer.templates.template_processor import TemplateProcessor
    from fraud_scorer.storage.cases import create_case
    from fraud_scorer.storage.db import (
        get_conn,
        upsert_document,
        mark_ocr_success,
        save_ocr_result,
        save_extracted_data,
        create_run,
        save_ai_analysis,
    )


# ----------------------------
# Utils
# ----------------------------
def ocr_result_to_dict(ocr_obj: Any) -> Dict[str, Any]:
    """
    Normaliza OCRResult/objeto a dict compatible con el extractor.
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
        out.setdefault("page_count", out.get("page_count", meta.get("page_count", 1)))
        out.setdefault("language", out.get("language", meta.get("language", "es")))
        return out

    if isinstance(ocr_obj, OCRResult):
        meta = ocr_obj.metadata or {}
        return {
            "text": ocr_obj.text or "",
            "tables": ocr_obj.tables or [],
            "key_value_pairs": ocr_obj.key_values or {},
            "entities": ocr_obj.entities or [],
            "confidence_scores": ocr_obj.confidence or {},
            "page_count": meta.get("page_count", 1),
            "language": meta.get("language", "es"),
            "metadata": meta,
            "errors": ocr_obj.errors or [],
            "success": bool(ocr_obj.success),
        }

    md = getattr(ocr_obj, "metadata", {}) or {}
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


def build_docs_for_template_from_processed(processed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    A partir de 'processed_docs' en memoria (con ocr_data + extracted_data) arma
    la estructura que consume TemplateProcessor y la IA del caso.
    """
    docs: List[Dict[str, Any]] = []
    for d in processed_docs:
        raw_ocr = d.get("ocr_data") or {}
        ext = d.get("extracted_data") or {}
        docs.append({
            "document_type": ext.get("document_type", "otro"),
            "raw_text": ext.get("raw_text", raw_ocr.get("text", "")),
            "entities": raw_ocr.get("entities", []),
            "key_value_pairs": ext.get("key_value_pairs", {}),
            "specific_fields": ext.get("specific_fields", {}),
            "ocr_metadata": ext.get("ocr_metadata", raw_ocr.get("metadata", {})),
        })
    return docs


def build_docs_for_template_from_db(case_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Lee de DB (documents + ocr_results + extracted_data) y construye:
      - docs_for_template (para IA y template)
      - processed_docs (estructura similar a la de la corrida en memoria)
    """
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
            # Cargar JSONs
            o_kv = json.loads(r["key_value_pairs"] or "{}") if "key_value_pairs" in r.keys() else {}
            o_tables = json.loads(r["tables"] or "[]") if "tables" in r.keys() else []
            o_entities = json.loads(r["entities"] or "[]") if "entities" in r.keys() else []
            o_conf = json.loads(r["confidence"] or "{}") if "confidence" in r.keys() else {}
            o_meta = json.loads(r["metadata"] or "{}") if "metadata" in r.keys() else {}
            o_errs = json.loads(r["errors"] or "[]") if "errors" in r.keys() else []

            e_entities = json.loads(r["e_entities"] or "{}") if "e_entities" in r.keys() else {}
            e_kv = json.loads(r["e_kv"] or "{}") if "e_kv" in r.keys() else {}
            e_extra = json.loads(r["extra"] or "{}") if "extra" in r.keys() else {}

            # processed_doc con lo mismo que produciríamos en nueva corrida
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
            extracted = {
                "document_type": r["document_type"] or "desconocido",
                "entities": e_entities,
                "key_value_pairs": e_kv,
                **e_extra,  # incluye raw_text, ocr_metadata, specific_fields si lo guardaste ahí
            }

            processed_docs.append({
                "file_name": r["filename"],
                "file_path": r["filepath"],
                "file_size_kb": (r["size_bytes"] or 0) / 1024.0 if r["size_bytes"] else None,
                "ocr_data": ocr_dict,
                "extracted_data": extracted if extracted.get("document_type") else None,
            })

            # entrada para template/IA
            docs_for_template.append({
                "document_type": extracted.get("document_type", "otro"),
                "raw_text": extracted.get("raw_text", ocr_dict.get("text", "")),
                "entities": ocr_dict.get("entities", []),
                "key_value_pairs": extracted.get("key_value_pairs", {}),
                "specific_fields": extracted.get("specific_fields", {}),
                "ocr_metadata": extracted.get("ocr_metadata", ocr_dict.get("metadata", {})),
            })

    return docs_for_template, processed_docs


# ----------------------------
# Sistema principal
# ----------------------------
class FraudAnalysisSystem:
    """Sistema completo de análisis de fraude con soporte DB/casos."""

    def __init__(self):
        print("Inicializando procesadores...")
        self.ocr_processor = AzureOCRProcessor()
        self.extractor = UniversalDocumentExtractor()
        self.ai_analyzer = AIDocumentAnalyzer()
        self.template_processor = TemplateProcessor()
        print("✓ Procesadores inicializados\n")

    async def process_new_folder_case(
        self,
        folder_path: Path,
        output_path: Path,
        case_title: Optional[str] = None,
        no_save: bool = False,
    ) -> Dict[str, Any]:
        """
        Nuevo caso: ingesta desde carpeta + OCR + extracción + IA + persistencia (opcional).
        """
        print("=" * 60)
        print(f"📁 Nuevo caso desde carpeta: {folder_path.name}")
        print("=" * 60)

        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
        documents = [p for p in folder_path.glob("*") if p.suffix.lower() in supported_extensions]

        print(f"\n✓ Encontrados {len(documents)} documentos\n")
        if not documents:
            raise RuntimeError("No se encontraron documentos para procesar")

        # Crear caso en DB
        title = case_title or folder_path.name
        case_id = create_case(title=title, base_path=str(folder_path))
        print(f"→ Case ID: {case_id}")

        processed_docs: List[Dict[str, Any]] = []
        ocr_errors: List[Dict[str, str]] = []

        print("🔍 Ejecutando OCR en documentos...")
        from tqdm import tqdm

        for doc_path in tqdm(documents, desc="OCR", unit="doc"):
            try:
                # Registra/evita duplicados por hash
                doc_id, created = upsert_document(
                    case_id=case_id,
                    filepath=str(doc_path),
                    mime_type=None,
                    page_count=None,
                    language=None,
                )

                # Hacer OCR
                raw_ocr = self.ocr_processor.analyze_document(str(doc_path))
                ocr_dict = ocr_result_to_dict(raw_ocr)

                # Persistir OCR
                save_ocr_result(
                    document_id=doc_id,
                    ocr_dict=ocr_dict,
                    engine="azure-di",
                    engine_version=(ocr_dict.get("metadata", {}) or {}).get("model_used", ""),
                )
                mark_ocr_success(doc_id, ok=True)

                # Extraer datos y persistir
                extracted = self.extractor.extract_structured_data(ocr_dict)
                save_extracted_data(doc_id, extracted)

                processed_docs.append({
                    "document_id": doc_id,  # ⬅️ guardamos id por si luego quieres per-doc IA
                    "file_name": doc_path.name,
                    "file_path": str(doc_path),
                    "file_size_kb": doc_path.stat().st_size / 1024.0,
                    "ocr_data": ocr_dict,
                    "extracted_data": extracted,
                })

            except Exception as e:
                logger.error(f"Error procesando {doc_path.name}: {e}", exc_info=False)
                ocr_errors.append({"file": doc_path.name, "error": str(e)})
                print(f"\n⚠️  Error procesando {doc_path.name}: {e}")

        usable = [d for d in processed_docs if d.get("extracted_data")]
        print(f"\n✓ OCR completado. Documentos con extracción útil: {len(usable)}")

        # IA del caso completo
        print("\n🤖 Analizando con AI (caso completo)...")
        docs_for_template = build_docs_for_template_from_processed(usable)
        ai_analysis = await self.ai_analyzer.analyze_claim_documents(docs_for_template)

        # Persistir corrida IA (opcional)
        run_id = None
        if not no_save:
            run_id = create_run(
                case_id=case_id,
                purpose="case_summary",
                llm_model="gpt-4-turbo-preview",
                params={"from": "new-folder"},
            )
            # (Opcional) si deseas guardar por-doc aquí, ya tienes document_id en processed_docs

        # Resumen simple
        fraud_score = float(ai_analysis.get("fraud_score", 0.0))
        risk_level = "bajo" if fraud_score < 0.3 else ("medio" if fraud_score < 0.6 else "alto")

        results: Dict[str, Any] = {
            "case_id": case_id,
            "folder_name": folder_path.name,
            "processing_date": datetime.now().isoformat(),
            "documents_processed": len(processed_docs),
            "documents": processed_docs,
            "errors": ocr_errors,
            "fraud_analysis": {
                "fraud_score": round(fraud_score * 100, 2),
                "risk_level": risk_level,
                "indicators": ai_analysis.get("fraud_indicators", []),
                "inconsistencies": ai_analysis.get("inconsistencies", []),
                "external_validations": ai_analysis.get("external_validations", []),
                "route_analysis": ai_analysis.get("route_analysis", {}),
            },
            "ai_analysis_raw": ai_analysis,
        }

        # Informe con plantilla
        print("\n📝 Generando informe con plantilla...")
        informe = self.template_processor.extract_from_documents(docs_for_template, ai_analysis)
        numero_siniestro = getattr(informe, "numero_siniestro", folder_path.name)
        pretty_html_path = output_path / f"INF-{numero_siniestro}.html"
        self.template_processor.generate_report(informe, str(pretty_html_path))
        print(f"✓ Informe HTML (plantilla) generado: {pretty_html_path}")

        # JSON técnico
        json_path = output_path / f"resultados_{folder_path.name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json.loads(json.dumps(results, default=str)), f, ensure_ascii=False, indent=2)
        print(f"✓ Resultados JSON guardados: {json_path}")

        # PDF
        try:
            from weasyprint import HTML
            pdf_path = pretty_html_path.with_suffix(".pdf")
            print("\n📄 Generando PDF...")
            HTML(filename=str(pretty_html_path)).write_pdf(str(pdf_path))
            print(f"✓ PDF generado: {pdf_path}")
        except ImportError:
            print("\n⚠️  WeasyPrint no instalado. Instala con: pip install weasyprint")
        except Exception as e:
            print(f"\n⚠️  Error generando PDF: {e}")

        return results

    async def reanalyze_case_from_db(
        self,
        case_id: str,
        output_path: Path,
        no_save: bool = False,
    ) -> Dict[str, Any]:
        """
        Reanaliza un caso YA OCR-eado, leyendo SOLO de la base local.
        No ejecuta OCR.
        """
        print("=" * 60)
        print(f"🔁 Re-analizando caso (sin OCR): {case_id}")
        print("=" * 60)

        docs_for_template, processed_docs = build_docs_for_template_from_db(case_id)
        if not docs_for_template:
            raise RuntimeError("No se encontraron documentos/ocr guardados para este case_id")

        # IA del caso completo
        print("\n🤖 Analizando con AI (caso completo)...")
        ai_analysis = await self.ai_analyzer.analyze_claim_documents(docs_for_template)

        # Persistir corrida IA (opcional)
        run_id = None
        if not no_save:
            run_id = create_run(
                case_id=case_id,
                purpose="case_summary",
                llm_model="gpt-4-turbo-preview",
                params={"from": "reanalyze"},
            )

        # Resumen simple
        fraud_score = float(ai_analysis.get("fraud_score", 0.0))
        risk_level = "bajo" if fraud_score < 0.3 else ("medio" if fraud_score < 0.6 else "alto")

        results: Dict[str, Any] = {
            "case_id": case_id,
            "processing_date": datetime.now().isoformat(),
            "documents_processed": len(processed_docs),
            "documents": processed_docs,
            "errors": [],
            "fraud_analysis": {
                "fraud_score": round(fraud_score * 100, 2),
                "risk_level": risk_level,
                "indicators": ai_analysis.get("fraud_indicators", []),
                "inconsistencies": ai_analysis.get("inconsistencies", []),
                "external_validations": ai_analysis.get("external_validations", []),
                "route_analysis": ai_analysis.get("route_analysis", {}),
            },
            "ai_analysis_raw": ai_analysis,
        }

        # Informe con plantilla
        print("\n📝 Generando informe con plantilla...")
        informe = self.template_processor.extract_from_documents(docs_for_template, ai_analysis)
        numero_siniestro = getattr(informe, "numero_siniestro", case_id)
        pretty_html_path = output_path / f"INF-{numero_siniestro}.html"
        self.template_processor.generate_report(informe, str(pretty_html_path))
        print(f"✓ Informe HTML (plantilla) generado: {pretty_html_path}")

        # JSON técnico
        json_path = output_path / f"resultados_{case_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json.loads(json.dumps(results, default=str)), f, ensure_ascii=False, indent=2)
        print(f"✓ Resultados JSON guardados: {json_path}")

        # PDF
        try:
            from weasyprint import HTML
            pdf_path = pretty_html_path.with_suffix(".pdf")
            print("\n📄 Generando PDF...")
            HTML(filename=str(pretty_html_path)).write_pdf(str(pdf_path))
            print(f"✓ PDF generado: {pdf_path}")
        except ImportError:
            print("\n⚠️  WeasyPrint no instalado. Instala con: pip install weasyprint")
        except Exception as e:
            print(f"\n⚠️  Error generando PDF: {e}")

        return results


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fraud Scorer - Nuevo caso (carpeta) o reanálisis por case_id."
    )
    p.add_argument("folder", nargs="?", help="Carpeta de documentos (modo nuevo caso).")
    p.add_argument("--case-id", dest="case_id", help="Reanaliza un caso existente (sin OCR).")
    p.add_argument("--out", dest="out", default="data/reports", help="Carpeta de salida de reportes.")
    p.add_argument("--title", dest="title", default=None, help="Título para el caso (modo carpeta).")
    p.add_argument("--no-save", dest="no_save", action="store_true", help="No persiste análisis de IA (pruebas).")
    args = p.parse_args(argv)

    # Validaciones simples: o folder o case_id
    if not args.folder and not args.case_id:
        p.error("Debes pasar una carpeta o --case-id")

    if args.folder and args.case_id:
        p.error("Usa solo carpeta o solo --case-id, no ambos.")

    return args


async def main(argv: List[str]) -> None:
    args = parse_args(argv)
    output_path = Path(args.out)
    output_path.mkdir(parents=True, exist_ok=True)

    system = FraudAnalysisSystem()

    if args.case_id:
        await system.reanalyze_case_from_db(
            case_id=args.case_id,
            output_path=output_path,
            no_save=args.no_save,
        )
    else:
        folder_path = Path(args.folder)
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"❌ Error: La carpeta {folder_path} no existe")
            sys.exit(1)

        await system.process_new_folder_case(
            folder_path=folder_path,
            output_path=output_path,
            case_title=args.title,
            no_save=args.no_save,
        )

    print("\n✅ Proceso completado exitosamente!")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
