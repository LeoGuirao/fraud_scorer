#!/usr/bin/env python3
"""
Fraud Scorer v2.0 - Sistema de análisis con IA (solo v2, sin legacy)
"""

import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime

# Añadir la raíz del proyecto al path de Python
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fraud_scorer.run_report")

# ==== Componentes del sistema v2 ====
from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
from fraud_scorer.parsers.document_parser import DocumentParser
from fraud_scorer.storage.ocr_cache import OCRCacheManager
from fraud_scorer.storage.cases import create_case

from fraud_scorer.processors.ai.ai_field_extractor import AIFieldExtractor
from fraud_scorer.processors.ai.ai_consolidator import AIConsolidator
from fraud_scorer.templates.ai_report_generator import AIReportGenerator
from fraud_scorer.models.extraction import (
    DocumentExtraction,
    ConsolidatedExtraction,
)


class FraudAnalysisSystemV2:
    """
    Sistema de análisis de fraude v2.0 con IA y Cache OCR (sin legacy).
    """

    def __init__(self):
        # OCR + Parser
        self.ocr_processor = AzureOCRProcessor()
        self.document_parser = DocumentParser(self.ocr_processor)

        # Cache OCR
        self.cache_manager = OCRCacheManager()

        # IA v2
        self.extractor = AIFieldExtractor()
        self.consolidator = AIConsolidator()
        template_path = project_root / "src" / "fraud_scorer" / "templates"
        self.report_generator = AIReportGenerator(template_dir=template_path)

        logger.info("Sistema v2.0 inicializado con componentes de IA")

    async def process_case(
        self,
        folder_path: Path,
        output_path: Path,
        case_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Procesa un caso completo con el flujo v2 (solo IA).
        """
        logger.info("=" * 60)
        logger.info(f"📁 Procesando caso: {folder_path.name}")
        logger.info("🤖 Modo: IA Avanzada v2.0")
        logger.info("=" * 60)

        # Buscar documentos soportados (ignorar archivos '._' de macOS)
        supported_extensions = {
            ".pdf", ".png", ".jpg", ".jpeg", ".tiff",
            ".docx", ".xlsx", ".csv"
        }
        documents = [
            p for p in folder_path.glob("*")
            if p.suffix.lower() in supported_extensions and not p.name.startswith("._")
        ]
        if not documents:
            raise RuntimeError("No se encontraron documentos para procesar")

        logger.info(f"✓ Encontrados {len(documents)} documentos")

        # Crear caso en DB
        case_id = create_case(
            title=case_title or folder_path.name,
            base_path=str(folder_path)
        )
        logger.info(f"✓ Case ID: {case_id}")

        # Ejecutar pipeline v2
        return await self._process_with_ai(documents, case_id, output_path)

    async def _process_with_ai(
        self,
        documents: List[Path],
        case_id: str,
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Procesamiento con el sistema de IA y cache.
        """
        # ============================================
        # FASE 1: OCR/Parsing
        # ============================================
        logger.info("\n📖 FASE 1: Procesamiento de Documentos")
        logger.info("-" * 40)

        ocr_results: List[Dict[str, Any]] = []
        cache_files: List[str] = []

        for doc_path in documents:
            logger.info(f"  Procesando: {doc_path.name}")

            # Usar cache si existe
            if self.cache_manager and self.cache_manager.has_cache(doc_path):
                logger.info(f"  ⚡ Usando cache para: {doc_path.name}")
                ocr_result = self.cache_manager.get_cache(doc_path)
                if ocr_result:
                    cache_files.append(str(doc_path))
            else:
                # OCR/Parser tolerante a fallos
                logger.info(f"  🔄 Procesando con OCR/Parser: {doc_path.name}")
                try:
                    ocr_result = self.document_parser.parse_document(doc_path)
                    if self.cache_manager and ocr_result:
                        self.cache_manager.save_cache(doc_path, ocr_result)
                except Exception as e:
                    logger.error(f"  ❌ Error procesando {doc_path.name}: {e}", exc_info=True)
                    continue

            if ocr_result:
                ocr_results.append({
                    "filename": doc_path.name,
                    "ocr_result": ocr_result,
                    "document_type": None,  # se detectará dentro del extractor
                })

        logger.info(f"✓ Procesamiento completado: {len(ocr_results)}/{len(documents)} exitosos")

        # Guardar índice del caso para replay
        if self.cache_manager:
            case_data = {
                "case_title": case_id,
                "folder_path": str(documents[0].parent) if documents else "",
                "documents": [str(d) for d in documents],
                "cache_files": cache_files,
            }
            self.cache_manager.save_case_index(case_id, case_data)

        # ============================================
        # FASE 2: Extracción con IA
        # ============================================
        logger.info("\n🔍 FASE 2: Extracción de campos con IA")
        logger.info("-" * 40)

        extractions: List[DocumentExtraction] = await self.extractor.extract_from_documents_batch(
            documents=ocr_results,
            parallel_limit=3,
        )

        for extraction in extractions:
            fields_found = sum(1 for v in extraction.extracted_fields.values() if v is not None)
            logger.info(f"  ✓ {extraction.source_document}: {fields_found} campos extraídos")

        logger.info(f"✓ Extracción completada: {len(extractions)} documentos procesados")

        # ============================================
        # FASE 3: Consolidación con IA
        # ============================================
        logger.info("\n🧠 FASE 3: Consolidación inteligente")
        logger.info("-" * 40)

        consolidated: ConsolidatedExtraction = await self.consolidator.consolidate_extractions(
            extractions=extractions,
            case_id=case_id,
            use_advanced_reasoning=True,
        )

        # Conteo robusto (Pydantic v2/dict)
        fields_obj = getattr(consolidated, "consolidated_fields", {}) or {}
        if hasattr(fields_obj, "model_dump"):
            fields_dict = fields_obj.model_dump()
        elif hasattr(fields_obj, "dict"):
            fields_dict = fields_obj.dict()
        else:
            fields_dict = dict(fields_obj)

        fields_filled = sum(1 for v in fields_dict.values() if v is not None)
        total_fields = len(fields_dict)
        logger.info(f"✓ Campos consolidados: {fields_filled}/{total_fields}")

        if consolidated.conflicts_resolved:
            logger.info(f"✓ Conflictos resueltos: {len(consolidated.conflicts_resolved)}")
            for conflict in consolidated.conflicts_resolved[:3]:
                logger.info(
                    f"  - {conflict.get('field', 'N/A')}: {str(conflict.get('reasoning', ''))[:80]}..."
                )
        
        # --- OBTENER DATOS PARA NOMBRAR ARCHIVOS ---
        # Extraemos los datos del objeto `consolidated`. 
        # Asegúrate de que los nombres de los campos coincidan con los de tu modelo `ConsolidatedFields`
        insured_name_from_data = fields_dict.get("nombre_asegurado", "Desconocido")
        claim_number_from_data = fields_dict.get("numero_siniestro", f"SINIESTRO_{case_id}")
        logger.info(f"✓ Datos para organización: {insured_name_from_data} - {claim_number_from_data}")

        # ============================================
        # FASE 4: Análisis de fraude (IA)
        # ============================================
        logger.info("\n🔎 FASE 4: Análisis de fraude")
        logger.info("-" * 40)
        ai_analysis = await self._analyze_fraud(consolidated, extractions)
        fraud_score = ai_analysis.get("fraud_score", 0)
        risk_level = "BAJO" if fraud_score < 0.3 else ("MEDIO" if fraud_score < 0.6 else "ALTO")
        logger.info(f"✓ Fraud Score: {fraud_score:.2%}")
        logger.info(f"✓ Nivel de Riesgo: {risk_level}")

        # ============================================
        # FASE 5: Generación del reporte
        # ============================================
        logger.info("\n📝 FASE 5: Generación del reporte")
        logger.info("-" * 40)

        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generar nombres de archivo con nomenclatura dinámica
        def sanitize_filename(name: str) -> str:
            if not name: 
                return "SIN_NOMBRE"
            return re.sub(r'[^a-zA-Z0-9_.-]+', '_', name).strip('_')
        
        s_insured = sanitize_filename(insured_name_from_data)
        s_claim = sanitize_filename(claim_number_from_data)
        
        # HTML - con nomenclatura dinámica y reemplazo
        html_filename = f"{s_insured}_{s_claim}_INFORME.html"
        html_path = output_path / html_filename
        
        # Eliminar archivo existente si existe
        if html_path.exists():
            logger.info(f"  ⚠️ Reemplazando archivo existente: {html_filename}")
            html_path.unlink()
        
        html_content = self.report_generator.generate_report(
            consolidated_data=consolidated,
            ai_analysis=ai_analysis,
            output_path=html_path,
            insured_name=insured_name_from_data,
            claim_number=claim_number_from_data
        )
        logger.info(f"✓ HTML generado: {html_path}")

        # PDF - con nomenclatura dinámica y reemplazo
        pdf_filename = f"{s_insured}_{s_claim}_INFORME.pdf"
        pdf_path = output_path / pdf_filename
        
        # Eliminar archivo existente si existe
        if pdf_path.exists():
            logger.info(f"  ⚠️ Reemplazando archivo existente: {pdf_filename}")
            pdf_path.unlink()
            
        if self.report_generator.generate_pdf(html_content, pdf_path):
            logger.info(f"✓ PDF generado: {pdf_path}")

        # ============================================
        # FASE 6: Guardar resultados y Organizar archivos
        # ============================================
        logger.info("\n💾 FASE 6: Guardar resultados y Organizar archivos")
        logger.info("-" * 40)
        ocr_total = len(documents)
        ocr_success = len(ocr_results)
        extraction_total = ocr_success
        extraction_success = len(extractions)

        ocr_rate = (ocr_success / ocr_total) if ocr_total > 0 else 0
        extraction_rate = (extraction_success / extraction_total) if extraction_total > 0 else 0
        completion_rate = (fields_filled / total_fields) if total_fields > 0 else 0
        avg_confidence = (
            sum(consolidated.confidence_scores.values()) / len(consolidated.confidence_scores)
            if consolidated.confidence_scores else 0
        )

        # --- GUARDAR ARCHIVO CONSOLIDADO CON NOMENCLATURA DINÁMICA ---
        consolidated_filename = f"{s_insured}_{s_claim}_CONSOLIDADO.json"
        
        # GUARDAR el archivo consolidado en data/temp/pipeline_cache (usando ruta absoluta)
        pipeline_cache_dir = project_root / "data" / "temp" / "pipeline_cache"
        pipeline_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directorio pipeline_cache creado/verificado: {pipeline_cache_dir}")
        
        consolidated_json_path = pipeline_cache_dir / consolidated_filename
        
        # Eliminar archivo existente si existe
        if consolidated_json_path.exists():
            logger.info(f"  ⚠️ Reemplazando archivo consolidado existente: {consolidated_filename}")
            consolidated_json_path.unlink()
        
        logger.info(f"✓ Guardando archivo consolidado como: {consolidated_filename}")

        try:
            with open(consolidated_json_path, "w", encoding="utf-8") as f:
                # Guardamos solo los datos consolidados aquí
                json.dump(consolidated.model_dump(), f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"✓ JSON consolidado guardado exitosamente en: {consolidated_json_path}")
            logger.info(f"✓ Tamaño del archivo: {consolidated_json_path.stat().st_size} bytes")
        except Exception as e:
            logger.error(f"❌ Error guardando archivo consolidado: {e}")
            raise
        
        # --- LLAMAR A LA REORGANIZACIÓN DEL CACHÉ ---
        if self.cache_manager:
            self.cache_manager.reorganize_cache_for_case(
                case_id=case_id,
                insured_name=insured_name_from_data,
                claim_number=claim_number_from_data
            )

        results = {
            "case_id": case_id,
            "processing_date": datetime.now().isoformat(),
            "documents_processed": len(documents),
            "extraction_results": [e.model_dump() for e in extractions],
            "consolidated_data": consolidated.model_dump(),
            "fraud_analysis": ai_analysis,
            "processing_metrics": {
                "ocr_success_rate": f"{ocr_rate:.1%}",
                "extraction_success_rate": f"{extraction_rate:.1%}",
                "fields_completion_rate": f"{completion_rate:.1%}",
                "conflicts_resolved": len(consolidated.conflicts_resolved),
                "average_confidence": avg_confidence,
            },
        }

        # Guardamos el reporte completo de resultados (que incluye métricas, etc.) con nombre mejorado
        results_filename = f"{s_insured}_{s_claim}_RESULTADOS.json"
        json_path = output_path / results_filename
        
        # Eliminar archivo existente si existe
        if json_path.exists():
            logger.info(f"  ⚠️ Reemplazando archivo de resultados existente: {results_filename}")
            json_path.unlink()
            
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"✓ JSON de resultados guardado: {json_path}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)

        return results

    async def _analyze_fraud(
        self,
        consolidated: ConsolidatedExtraction,
        extractions: List[DocumentExtraction],
    ) -> Dict[str, Any]:
        """
        Análisis de fraude usando IA.
        """
        from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer

        analyzer = AIDocumentAnalyzer()

        # Payload ligero para IA
        docs_for_analysis = [
            {
                "document_type": extr.document_type,
                "key_value_pairs": extr.extracted_fields,
            }
            for extr in extractions
        ]

        analysis = await analyzer.analyze_claim_documents(docs_for_analysis)
        return analysis


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parser de argumentos (sin --use-legacy)."""
    p = argparse.ArgumentParser(description="Fraud Scorer v2.0 - Sistema de Análisis con IA")
    p.add_argument("folder", type=Path, help="Carpeta con documentos del caso")
    p.add_argument("--out", type=Path, default=Path("data/reports"), help="Carpeta de salida")
    p.add_argument("--title", help="Título del caso")
    p.add_argument("--debug", action="store_true", help="Modo debug con más logging")
    return p.parse_args(argv)


async def main(argv: List[str]) -> None:
    """Función principal."""
    args = parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.folder.is_dir():
        print(f"❌ Error: La carpeta {args.folder} no existe o no es un directorio.")
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)

    system = FraudAnalysisSystemV2()

    try:
        await system.process_case(
            folder_path=args.folder,
            output_path=args.out,
            case_title=args.title,
        )
    except Exception as e:
        logger.error(f"Error procesando caso: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Refuerzo: asegurar que 'src' esté en sys.path
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    asyncio.run(main(sys.argv[1:]))
