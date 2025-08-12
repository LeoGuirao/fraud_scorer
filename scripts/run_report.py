#!/usr/bin/env python3
import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fraud_scorer.run_report")

# ------------------------------------------------------------------------------------
# Imports del paquete (con fallback para ejecución directa desde repo)
# ------------------------------------------------------------------------------------
try:
    # OCR / Extracción / IA
    from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
    from fraud_scorer.processors.ocr.document_extractor import UniversalDocumentExtractor
    from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer

    # Template procesador segmentado (forzado) + loader de config opcional
    from fraud_scorer.pipelines.segmented_processor import (
        SegmentedTemplateProcessor,
        load_pipeline_config,  # (no es obligatorio usarlo aquí, pero lo dejamos disponible)
    )

    # Storage
    from fraud_scorer.storage.cases import create_case
    from fraud_scorer.storage.db import (
        get_conn,
        upsert_document,
        mark_ocr_success,
        save_ocr_result,
        save_extracted_data,
        create_run,
    )

    # Utils con mapeo canónico integrado en los builders
    from fraud_scorer.pipelines.utils import (
        ocr_result_to_dict,
        build_docs_for_template_from_processed,
        build_docs_for_template_from_db,
    )
except ImportError:
    # Añadir src/ al sys.path si ejecutas el script directamente
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    _SRC_PATH = _REPO_ROOT / "src"
    if _SRC_PATH.exists():
        sys.path.insert(0, str(_SRC_PATH))

    from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
    from fraud_scorer.processors.ocr.document_extractor import UniversalDocumentExtractor
    from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer

    from fraud_scorer.pipelines.segmented_processor import (
        SegmentedTemplateProcessor,
        load_pipeline_config,
    )

    from fraud_scorer.storage.cases import create_case
    from fraud_scorer.storage.db import (
        get_conn,
        upsert_document,
        mark_ocr_success,
        save_ocr_result,
        save_extracted_data,
        create_run,
    )

    from fraud_scorer.pipelines.utils import (
        ocr_result_to_dict,
        build_docs_for_template_from_processed,
        build_docs_for_template_from_db,
    )


# ------------------------------------------------------------------------------------
# Helpers locales
# ------------------------------------------------------------------------------------
def _clean_outputs(out_dir: Path, case_id: Optional[str] = None) -> None:
    """
    Borra INF-*.html/pdf previos y, si se pasa case_id, el JSON técnico de ese caso.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = list(out_dir.glob("INF-*.html")) + list(out_dir.glob("INF-*.pdf"))
    if case_id:
        targets += list(out_dir.glob(f"resultados_{case_id}.json"))
    else:
        targets += list(out_dir.glob("resultados_*.json"))

    for pth in targets:
        try:
            pth.unlink()
            logger.debug(f"Eliminado: {pth}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"No se pudo eliminar {pth}: {e}")


# ------------------------------------------------------------------------------------
# Sistema principal
# ------------------------------------------------------------------------------------
class FraudAnalysisSystem:
    """
    Sistema completo de análisis de fraude con soporte DB/casos.
    **Segmentación forzada**: siempre usa SegmentedTemplateProcessor.
    """

    def __init__(self, config_path: Optional[str] = None):
        print("Inicializando procesadores...")
        self.ocr_processor = AzureOCRProcessor()
        self.extractor = UniversalDocumentExtractor()
        self.ai_analyzer = AIDocumentAnalyzer()
        # 👇 Siempre usamos el procesador segmentado, con config YAML opcional
        self.template_processor = SegmentedTemplateProcessor(config_path=config_path)
        print("✓ Procesadores inicializados\n")

    async def _build_ai_case_from_consolidated(self, consolidated: Dict[str, Any]) -> Dict[str, Any]:
        """
        En lugar de pasar TODOS los docs a la IA, pasamos un 'documento' sintético con el resumen consolidado.
        Minimiza el tamaño del prompt y evita mezcla de contexto.
        """
        synthetic_doc = {
            "document_type": "consolidated_summary",
            "raw_text": json.dumps(
                {
                    "case_info": consolidated.get("case_info", {}),
                    "validation_summary": consolidated.get("validation_summary", {}),
                    "processing_stats": consolidated.get("processing_stats", {}),
                },
                ensure_ascii=False,
            ),
            "entities": [],
            "key_value_pairs": consolidated.get("case_info", {}),
            "specific_fields": consolidated.get("case_info", {}),
            "ocr_metadata": {"source_name": "consolidated_summary.json"},
        }
        return await self.ai_analyzer.analyze_claim_documents([synthetic_doc])

    async def process_new_folder_case(
        self,
        folder_path: Path,
        output_path: Path,
        case_title: Optional[str] = None,
        no_save: bool = False,
        clobber: bool = False,
    ) -> Dict[str, Any]:
        """
        Nuevo caso: ingesta desde carpeta + OCR + extracción + segmentación/consolidación + IA liviana + persistencia opcional.
        """
        print("=" * 60)
        print(f"📁 Nuevo caso desde carpeta: {folder_path.name}")
        print("=" * 60)

        if clobber:
            _clean_outputs(output_path, case_id=None)

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
                # Registrar/evitar duplicados por hash
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

                # Inyectar metadatos de origen (para trazabilidad en el informe)
                meta = ocr_dict.setdefault("metadata", {}) or {}
                meta.setdefault("source_name", doc_path.name)   # nombre del archivo
                meta.setdefault("source_path", str(doc_path))   # ruta completa

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
                    "document_id": doc_id,
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

        # Normalización para el template (con mapeo canónico) y segmentación (aislado + consolidado)
        docs_for_template = build_docs_for_template_from_processed(usable)

        logger.info("Procesando documentos en modo segmentado/aislado…")
        consolidated = await self.template_processor.orchestrator.process_documents(docs_for_template)

        # IA del caso (liviana) sobre el consolidado
        print("\n🤖 Analizando con AI (resumen consolidado)…")
        ai_analysis = await self._build_ai_case_from_consolidated(consolidated)

        # Persistir corrida IA (opcional)
        run_id = None
        if not no_save:
            run_id = create_run(
                case_id=case_id,
                purpose="case_summary_segmented",
                llm_model="gpt-4-turbo-preview",
                params={"from": "new-folder-segmented"},
            )

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
            "segmented_consolidated": consolidated,  # 👈 guardamos el consolidado para auditoría
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

        # Informe con plantilla (segmentado) — construimos el Informe desde el consolidado
        print("\n📝 Generando informe (segmentado)…")
        informe = self.template_processor._build_informe_from_consolidated(consolidated, ai_analysis)

        # (Temporal) fija el número de siniestro para evitar inconsistencias
        informe.numero_siniestro = "1"

        pretty_html_path = output_path / f"INF-{informe.numero_siniestro}.html"
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
        clobber: bool = False,
    ) -> Dict[str, Any]:
        """
        Reanaliza un caso YA OCR-eado, leyendo SOLO de la base local (sin OCR),
        con pipeline segmentado y análisis IA consolidado.
        """
        print("=" * 60)
        print(f"🔁 Re-analizando caso (sin OCR): {case_id}")
        print("=" * 60)

        if clobber:
            _clean_outputs(output_path, case_id=case_id)

        # Usa builder de utils: aplica mapeo canónico y normalización
        docs_for_template, processed_docs = build_docs_for_template_from_db(case_id)
        if not docs_for_template:
            raise RuntimeError("No se encontraron documentos/ocr guardados para este case_id")

        # Segmentación + consolidación
        logger.info("Procesando documentos en modo segmentado/aislado…")
        consolidated = await self.template_processor.orchestrator.process_documents(docs_for_template)

        # IA del caso (liviana) sobre el consolidado
        print("\n🤖 Analizando con AI (resumen consolidado)…")
        ai_analysis = await self._build_ai_case_from_consolidated(consolidated)

        # Persistir corrida IA (opcional)
        run_id = None
        if not no_save:
            run_id = create_run(
                case_id=case_id,
                purpose="case_summary_segmented",
                llm_model="gpt-4-turbo-preview",
                params={"from": "reanalyze-segmented"},
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
            "segmented_consolidated": consolidated,  # 👈 guardamos el consolidado para auditoría
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

        # Informe con plantilla (segmentado)
        print("\n📝 Generando informe (segmentado)…")
        informe = self.template_processor._build_informe_from_consolidated(consolidated, ai_analysis)
        # (Temporal) fija el número de siniestro para evitar inconsistencias
        informe.numero_siniestro = "1"

        pretty_html_path = output_path / f"INF-{informe.numero_siniestro}.html"
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


# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fraud Scorer - Nuevo caso (carpeta) o reanálisis por case_id (pipeline segmentado)."
    )
    p.add_argument("folder", nargs="?", help="Carpeta de documentos (modo nuevo caso).")
    p.add_argument("--case-id", dest="case_id", help="Reanaliza un caso existente (sin OCR).")
    p.add_argument("--out", dest="out", default="data/reports", help="Carpeta de salida de reportes.")
    p.add_argument("--title", dest="title", default=None, help="Título para el caso (modo carpeta).")
    p.add_argument("--no-save", dest="no_save", action="store_true", help="No persiste análisis de IA (pruebas).")
    p.add_argument(
        "--clobber",
        action="store_true",
        help="Borra INF-*.html/pdf previos (y resultados_*.json) en --out antes de generar nuevos",
    )
    # 👇 NUEVO: config YAML opcional
    p.add_argument("--config", dest="config", default=None, help="Ruta a pipeline_config.yaml")
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

    # Pasamos la ruta del YAML (si viene) al sistema -> al SegmentedTemplateProcessor
    system = FraudAnalysisSystem(config_path=args.config)

    if args.case_id:
        await system.reanalyze_case_from_db(
            case_id=args.case_id,
            output_path=output_path,
            no_save=args.no_save,
            clobber=args.clobber,
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
            clobber=args.clobber,
        )

    print("\n✅ Proceso completado exitosamente!")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
