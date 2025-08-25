#!/usr/bin/env python3
"""
Script principal con el nuevo sistema de IA
"""
import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional  # ← Agregar Optional aquí
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fraud_scorer.run_report")

# Imports del sistema existente
from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
from fraud_scorer.storage.cases import create_case

# Imports del NUEVO sistema de IA
from fraud_scorer.ai_extractors.ai_field_extractor import AIFieldExtractor
from fraud_scorer.ai_extractors.ai_consolidator import AIConsolidator
from fraud_scorer.ai_extractors.ai_report_generator import AIReportGenerator
from fraud_scorer.ai_extractors.models.extraction_models import DocumentExtraction

# >>> NUEVOS IMPORTS (Cache + Parser)
from fraud_scorer.cache.ocr_cache_manager import OCRCacheManager
from fraud_scorer.parsers.document_parser import DocumentParser
# <<<


class FraudAnalysisSystemV2:
    """
    Sistema de análisis de fraude v2.0 con IA y Cache OCR
    """

    def __init__(self, use_ai: bool = True, use_cache: bool = True):
        """
        Inicializa el sistema

        Args:
            use_ai: Si True, usa el nuevo sistema de IA
            use_cache: Si True, usa cache para resultados OCR
        """
        self.use_ai = use_ai
        self.use_cache = use_cache

        # Componentes existentes
        self.ocr_processor = AzureOCRProcessor()

        # Nuevo: Document Parser para múltiples formatos
        self.document_parser = DocumentParser(self.ocr_processor)

        # Nuevo: Cache Manager
        self.cache_manager = OCRCacheManager() if use_cache else None

        if use_ai:
            # Nuevos componentes de IA
            self.extractor = AIFieldExtractor()
            self.consolidator = AIConsolidator()
            self.report_generator = AIReportGenerator()
            logger.info("Sistema inicializado con extractores de IA")
        else:
            # Sistema legacy
            from fraud_scorer.extractors.intelligent_extractor import IntelligentFieldExtractor

            self.extractor = IntelligentFieldExtractor()
            logger.info("Sistema inicializado con extractores legacy")

    async def process_case(
        self,
        folder_path: Path,
        output_path: Path,
        case_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Procesa un caso completo con el nuevo flujo de IA
        """
        logger.info("=" * 60)
        logger.info(f"📁 Procesando caso: {folder_path.name}")
        logger.info(f"🤖 Modo: {'IA Avanzada' if self.use_ai else 'Legacy'}")
        logger.info("=" * 60)

        # Obtener documentos
        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
        documents = [p for p in folder_path.glob("*") if p.suffix.lower() in supported_extensions]

        if not documents:
            raise RuntimeError("No se encontraron documentos para procesar")

        logger.info(f"✓ Encontrados {len(documents)} documentos")

        # Crear caso en DB
        case_id = create_case(title=case_title or folder_path.name, base_path=str(folder_path))
        logger.info(f"✓ Case ID: {case_id}")

        if self.use_ai:
            return await self._process_with_ai(documents, case_id, output_path)
        else:
            return await self._process_legacy(documents, case_id, output_path)

    async def _process_with_ai(
        self,
        documents: List[Path],
        case_id: str,
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Procesamiento con el nuevo sistema de IA y cache
        """

        # ============================================
        # FASE 1: OCR/Parsing de todos los documentos
        # ============================================
        logger.info("\n📖 FASE 1: Procesamiento de Documentos")
        logger.info("-" * 40)

        ocr_results = []
        cache_files = []  # Para guardar referencias al cache

        for doc_path in documents:
            logger.info(f"  Procesando: {doc_path.name}")

            # Verificar si hay cache disponible
            if self.cache_manager and self.cache_manager.has_cache(doc_path):
                logger.info(f"  ⚡ Usando cache para: {doc_path.name}")
                ocr_result = self.cache_manager.get_cache(doc_path)
                cache_files.append(str(doc_path))
            else:
                # Procesar con OCR o parser apropiado
                logger.info(f"  🔄 Procesando con OCR/Parser: {doc_path.name}")
                try:
                    # Usar el document parser que maneja múltiples formatos
                    ocr_result = self.document_parser.parse_document(doc_path)

                    # Guardar en cache si está habilitado
                    if self.cache_manager and ocr_result:
                        self.cache_manager.save_cache(doc_path, ocr_result)

                except Exception as e:
                    logger.error(f"  ❌ Error procesando {doc_path.name}: {e}")
                    continue

            if ocr_result:
                ocr_results.append({
                    "filename": doc_path.name,
                    "ocr_result": ocr_result,
                    "document_type": None,  # Se detectará después
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
        # FASE 2: Extracción con IA (documento por documento)
        # ============================================
        logger.info("\n🔍 FASE 2: Extracción de campos con IA")
        logger.info("-" * 40)

        extractions = await self.extractor.extract_from_documents_batch(
            documents=ocr_results,
            parallel_limit=3,  # Procesar hasta 3 documentos en paralelo
        )

        # Log de resultados de extracción
        for extraction in extractions:
            fields_found = sum(1 for v in extraction.extracted_fields.values() if v is not None)
            logger.info(f"  ✓ {extraction.source_document}: {fields_found} campos extraídos")

        logger.info(f"✓ Extracción completada: {len(extractions)} documentos procesados")

        # ============================================
        # FASE 3: Consolidación con IA
        # ============================================
        logger.info("\n🧠 FASE 3: Consolidación inteligente")
        logger.info("-" * 40)

        consolidated = await self.consolidator.consolidate_extractions(
            extractions=extractions,
            case_id=case_id,
            use_advanced_reasoning=True,
        )

        # Log de consolidación
        fields_filled = sum(1 for v in consolidated.consolidated_fields.values() if v is not None)
        logger.info(f"✓ Campos consolidados: {fields_filled}/{len(consolidated.consolidated_fields)}")

        if consolidated.conflicts_resolved:
            logger.info(f"✓ Conflictos resueltos: {len(consolidated.conflicts_resolved)}")
            for conflict in consolidated.conflicts_resolved[:3]:  # Mostrar primeros 3
                logger.info(f"  - {conflict['field']}: {conflict['reasoning'][:80]}...")

        # ============================================
        # FASE 4: Análisis de fraude (opcional)
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

        # Generar HTML
        html_path = output_path / f"INF-{case_id}.html"
        html_content = self.report_generator.generate_report(
            consolidated_data=consolidated,
            ai_analysis=ai_analysis,
            output_path=html_path,
        )
        logger.info(f"✓ HTML generado: {html_path}")

        # Generar PDF
        pdf_path = output_path / f"INF-{case_id}.pdf"
        if self.report_generator.generate_pdf(html_content, pdf_path):
            logger.info(f"✓ PDF generado: {pdf_path}")

        # ============================================
        # FASE 6: Guardar resultados JSON
        # ============================================
        results = {
            "case_id": case_id,
            "processing_date": datetime.now().isoformat(),
            "documents_processed": len(documents),
            "extraction_results": [e.dict() for e in extractions],
            "consolidated_data": consolidated.dict(),
            "fraud_analysis": ai_analysis,
            "processing_metrics": {
                "ocr_success_rate": f"{len(ocr_results)/len(documents):.1%}",
                "extraction_success_rate": f"{len(extractions)/len(ocr_results):.1%}",
                "fields_completion_rate": f"{fields_filled/len(consolidated.consolidated_fields):.1%}",
                "conflicts_resolved": len(consolidated.conflicts_resolved),
                "average_confidence": sum(consolidated.confidence_scores.values())
                / len(consolidated.confidence_scores)
                if consolidated.confidence_scores
                else 0,
            },
        }

        json_path = output_path / f"resultados_{case_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"✓ JSON guardado: {json_path}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)

        return results

    async def _analyze_fraud(
        self,
        consolidated: Any,
        extractions: List[DocumentExtraction],
    ) -> Dict[str, Any]:
        """
        Análisis de fraude usando GPT-4
        """
        from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer

        analyzer = AIDocumentAnalyzer()

        # Preparar documentos para análisis
        docs_for_analysis = []
        for extraction in extractions:
            docs_for_analysis.append(
                {
                    "document_type": extraction.document_type,
                    "key_value_pairs": extraction.extracted_fields,
                    "specific_fields": extraction.extracted_fields,
                    "raw_text": "",  # No enviamos texto para ahorrar tokens
                    "entities": [],
                }
            )

        # Analizar
        analysis = await analyzer.analyze_claim_documents(docs_for_analysis)

        return analysis

    def _ocr_to_dict(self, ocr_result) -> Dict[str, Any]:
        """Convierte resultado OCR a diccionario"""
        if hasattr(ocr_result, "__dict__"):
            return {
                "text": getattr(ocr_result, "text", ""),
                "tables": getattr(ocr_result, "tables", []),
                "key_value_pairs": getattr(ocr_result, "key_values", {}),
                "confidence": getattr(ocr_result, "confidence", {}),
                "metadata": getattr(ocr_result, "metadata", {}),
            }
        return ocr_result

    async def _process_legacy(
        self,
        documents: List[Path],
        case_id: str,
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Procesamiento con el sistema legacy (fallback)
        """
        logger.info("Usando sistema legacy...")
        # Aquí iría la lógica del sistema anterior
        return {"status": "legacy_not_implemented"}


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parser de argumentos"""
    p = argparse.ArgumentParser(description="Fraud Scorer v2.0 - Sistema de Análisis con IA")
    p.add_argument("folder", type=Path, help="Carpeta con documentos del caso")
    p.add_argument("--out", type=Path, default=Path("data/reports"), help="Carpeta de salida")
    p.add_argument("--title", help="Título del caso")
    p.add_argument("--use-legacy", action="store_true", help="Usar sistema legacy en lugar de IA")
    p.add_argument("--debug", action="store_true", help="Modo debug con más logging")

    return p.parse_args(argv)


async def main(argv: List[str]) -> None:
    """Función principal"""
    args = parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validar carpeta
    if not args.folder.exists() or not args.folder.is_dir():
        print(f"❌ Error: La carpeta {args.folder} no existe")
        sys.exit(1)

    # Crear carpeta de salida
    args.out.mkdir(parents=True, exist_ok=True)

    # Inicializar sistema
    system = FraudAnalysisSystemV2(use_ai=not args.use_legacy)

    # Procesar caso
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
    asyncio.run(main(sys.argv[1:]))
