#!/usr/bin/env python3
import sys
import asyncio
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fraud_scorer")

# ----------------------------
# Imports con fallback a src/
# ----------------------------
try:
    from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor, OCRResult
    from fraud_scorer.processors.ocr.document_extractor import UniversalDocumentExtractor
    from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
    from fraud_scorer.templates.template_processor import TemplateProcessor
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


def ocr_result_to_dict(ocr_obj: Any) -> Dict[str, Any]:
    """
    Normaliza el OCRResult del módulo Azure a un dict que
    el extractor entiende (llaves compatibles).
    """
    # Si ya es dict, sólo garantizamos llaves esperadas
    if isinstance(ocr_obj, dict):
        out: Dict[str, Any] = dict(ocr_obj)
        out.setdefault("text", out.get("text", ""))
        out.setdefault("tables", out.get("tables", []))
        # aceptar key_values o key_value_pairs
        if "key_value_pairs" not in out:
            out["key_value_pairs"] = out.get("key_values", {}) or {}
        # confidence_scores
        if "confidence_scores" not in out:
            c = out.get("confidence", {})
            out["confidence_scores"] = c if isinstance(c, dict) else {}
        # page_count / language
        meta = out.get("metadata", {}) or {}
        out.setdefault("page_count", out.get("page_count", meta.get("page_count", 1)))
        out.setdefault("language", out.get("language", meta.get("language", "es")))
        return out

    # Si es tu dataclass OCRResult
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

    # Fallback genérico
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


class FraudAnalysisSystem:
    """Sistema completo de análisis de fraude"""

    def __init__(self):
        print("Inicializando procesadores...")
        self.ocr_processor = AzureOCRProcessor()
        self.extractor = UniversalDocumentExtractor()
        self.ai_analyzer = AIDocumentAnalyzer()
        self.template_processor = TemplateProcessor()
        print("✓ Procesadores inicializados\n")

    async def process_folder(self, folder_path: Path, output_path: Path) -> Dict[str, Any]:
        print("=" * 60)
        print(f"📁 Procesando carpeta: {folder_path.name}")
        print("=" * 60)

        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
        documents = [p for p in folder_path.glob("*") if p.suffix.lower() in supported_extensions]

        print(f"\n✓ Encontrados {len(documents)} documentos\n")
        if not documents:
            print("⚠️ No se encontraron documentos para procesar")
            return {}

        processed_docs: List[Dict[str, Any]] = []
        ocr_errors: List[Dict[str, str]] = []

        print("🔍 Ejecutando OCR en documentos...")
        from tqdm import tqdm

        for doc_path in tqdm(documents, desc="OCR", unit="doc"):
            try:
                raw_ocr = self.ocr_processor.analyze_document(str(doc_path))
                ocr_dict = ocr_result_to_dict(raw_ocr)

                record: Dict[str, Any] = {
                    "file_name": doc_path.name,
                    "file_path": str(doc_path),
                    "file_size_kb": doc_path.stat().st_size / 1024.0,
                    "ocr_data": ocr_dict,
                    "extracted_data": None,
                }

                # Extraer datos estructurados si hay algo de texto
                if (ocr_dict.get("text") or "").strip():
                    extracted = self.extractor.extract_structured_data(ocr_dict)
                    record["extracted_data"] = extracted

                processed_docs.append(record)

            except Exception as e:
                logger.error(f"Error procesando {doc_path.name}: {e}", exc_info=False)
                ocr_errors.append({"file": doc_path.name, "error": str(e)})
                print(f"\n⚠️  Error procesando {doc_path.name}: {e}")

        usable = [d for d in processed_docs if d.get("extracted_data")]
        print(f"\n✓ OCR completado. Documentos con extracción útil: {len(usable)}")

        # ================================
        # IA del CASO COMPLETO + Informe
        # ================================
        print("\n🤖 Analizando con AI (caso completo)...")

        # 1) Preparar documentos para TemplateProcessor (lo que espera tu plantilla)
        docs_for_template: List[Dict[str, Any]] = []
        for d in usable:
            raw_ocr = d.get("ocr_data") or {}
            ext = d.get("extracted_data") or {}
            docs_for_template.append({
                "document_type": ext.get("document_type", "otro"),
                "raw_text": ext.get("raw_text", raw_ocr.get("text", "")),
                # entidades como lista de dicts (del OCR crudo)
                "entities": raw_ocr.get("entities", []),
                # kv para campos clave (tu extractor usa 'key_value_pairs')
                "key_value_pairs": ext.get("key_value_pairs", {}),
                "specific_fields": ext.get("specific_fields", {}),
                "ocr_metadata": ext.get("ocr_metadata", raw_ocr.get("metadata", {})),
            })

        # 2) Análisis IA del caso completo (esto devuelve fraud_score, inconsistencies, etc.)
        ai_analysis = await self.ai_analyzer.analyze_claim_documents(docs_for_template)

        # 3) Resumen simple de riesgo para cabecera
        fraud_score = float(ai_analysis.get("fraud_score", 0.0))
        risk_level = "bajo" if fraud_score < 0.3 else ("medio" if fraud_score < 0.6 else "alto")

        # 4) Compilar resultados técnicos (para JSON)
        results: Dict[str, Any] = {
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
            "summary": self._generate_summary(processed_docs, {
                "fraud_score": round(fraud_score * 100, 2),
                "risk_level": risk_level
            }),
            "ai_analysis_raw": ai_analysis,  # por si quieres depurar
        }

        # 5) Generar INFORME “bonito” con tu plantilla
        print("\n📝 Generando informe con plantilla...")
        informe = self.template_processor.extract_from_documents(docs_for_template, ai_analysis)

        numero_siniestro = getattr(informe, "numero_siniestro", folder_path.name)
        pretty_html_path = output_path / f"INF-{numero_siniestro}.html"
        self.template_processor.generate_report(informe, str(pretty_html_path))
        print(f"✓ Informe HTML (plantilla) generado: {pretty_html_path}")

        # 6) Guardar también el JSON técnico
        json_path = output_path / f"resultados_{folder_path.name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json.loads(json.dumps(results, default=str)), f, ensure_ascii=False, indent=2)
        print(f"✓ Resultados JSON guardados: {json_path}")

        # 7) PDF (opcional)
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

    def _generate_summary(self, documents: List[Dict], fraud_analysis_like: Dict) -> Dict[str, Any]:
        doc_types: Dict[str, int] = {}
        for doc in documents:
            extracted = doc.get("extracted_data") or {}
            doc_type = extracted.get("document_type", "desconocido")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        return {
            "total_documents": len(documents),
            "document_types": doc_types,
            "fraud_score": fraud_analysis_like.get("fraud_score", 0),
            "risk_level": fraud_analysis_like.get("risk_level", "bajo"),
            "key_findings": fraud_analysis_like.get("indicators", []),
        }


async def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/run_report.py <carpeta_documentos> [carpeta_salida]")
        sys.exit(1)

    folder_path = Path(sys.argv[1])
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"❌ Error: La carpeta {folder_path} no existe")
        sys.exit(1)

    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/reports")
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        system = FraudAnalysisSystem()
        results = await system.process_folder(folder_path, output_path)

        # También guardamos un snapshot completo por si luego quieres revisar:
        snapshot_path = output_path / f"snapshot_{folder_path.name}.json"
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(json.loads(json.dumps(results, default=str)), f, ensure_ascii=False, indent=2)
        print(f"✓ Snapshot técnico guardado: {snapshot_path}")

        print("\n✅ Proceso completado exitosamente!")

    except Exception as e:
        logger.error(f"Error fatal: {e}", exc_info=True)
        print(f"\n❌ Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
