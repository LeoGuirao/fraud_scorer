#!/usr/bin/env python3
"""
Sistema de Replay Interactivo con Cache OCR
"""
import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fraud_scorer.replay")

# Imports del sistema
from fraud_scorer.cache.ocr_cache_manager import OCRCacheManager
from fraud_scorer.ui.replay_ui import ReplayUI
from fraud_scorer.ai_extractors.ai_field_extractor import AIFieldExtractor
from fraud_scorer.ai_extractors.ai_consolidator import AIConsolidator
from fraud_scorer.ai_extractors.ai_report_generator import AIReportGenerator
from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer

class ReplaySystem:
    """
    Sistema de replay con cache OCR
    """
    
    def __init__(self):
        self.cache_manager = OCRCacheManager()
        self.ui = None  # Se inicializa después
        
    async def replay_case(
        self,
        case_id: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ejecuta el replay de un caso con las opciones especificadas
        """
        logger.info(f"Iniciando replay del caso {case_id}")
        
        # Obtener información del caso
        case_index = self.cache_manager.get_case_index(case_id)
        if not case_index:
            raise RuntimeError(f"No se encontró información del caso {case_id}")
        
        # Cargar resultados OCR del cache
        ocr_results = []
        for doc_path in case_index['documents']:
            doc_path = Path(doc_path)
            if self.cache_manager.has_cache(doc_path):
                ocr_result = self.cache_manager.get_cache(doc_path)
                ocr_results.append({
                    'filename': doc_path.name,
                    'ocr_result': ocr_result,
                    'document_type': None
                })
            else:
                logger.warning(f"No hay cache para {doc_path.name}")
        
        if not ocr_results:
            raise RuntimeError("No se encontraron resultados OCR en cache")
        
        # Procesar según las opciones
        if options.get('use_ai'):
            return await self._process_with_ai(
                ocr_results=ocr_results,
                case_id=case_id,
                options=options
            )
        else:
            return await self._process_legacy(
                ocr_results=ocr_results,
                case_id=case_id,
                options=options
            )
    
    async def _process_with_ai(
        self,
        ocr_results: List[Dict],
        case_id: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa con el sistema de IA usando los datos cacheados
        """
        output_path = Path(options.get('output_dir', 'data/reports'))
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes de IA
        extractor = AIFieldExtractor()
        consolidator = AIConsolidator()
        report_generator = AIReportGenerator()
        
        # Fase 1: Extracción (usando cache OCR)
        logger.info("Extrayendo campos con IA...")
        extractions = await extractor.extract_from_documents_batch(
            documents=ocr_results,
            parallel_limit=3
        )
        
        # Fase 2: Consolidación
        logger.info("Consolidando datos...")
        consolidated = await consolidator.consolidate_extractions(
            extractions=extractions,
            case_id=case_id,
            use_advanced_reasoning=True
        )
        
        # Fase 3: Análisis de fraude
        logger.info("Analizando fraude...")
        analyzer = AIDocumentAnalyzer()
        docs_for_analysis = []
        for extraction in extractions:
            docs_for_analysis.append({
                "document_type": extraction.document_type,
                "key_value_pairs": extraction.extracted_fields,
                "specific_fields": extraction.extracted_fields,
                "raw_text": "",
                "entities": []
            })
        
        ai_analysis = await analyzer.analyze_claim_documents(docs_for_analysis)
        
        # Fase 4: Generar reporte si se solicita
        if options.get('regenerate_report', True):
            logger.info("Generando reporte...")
            html_path = output_path / f"INF-{case_id}.html"
            html_content = report_generator.generate_report(
                consolidated_data=consolidated,
                ai_analysis=ai_analysis,
                output_path=html_path
            )
            
            # Intentar generar PDF
            pdf_path = output_path / f"INF-{case_id}.pdf"
            report_generator.generate_pdf(html_content, pdf_path)
        
        # Preparar resultados
        results = {
            "case_id": case_id,
            "replay_date": datetime.now().isoformat(),
            "options_used": options,
            "extraction_results": [e.dict() for e in extractions],
            "consolidated_data": consolidated.dict(),
            "fraud_analysis": ai_analysis,
            "output_path": str(output_path)
        }
        
        # Guardar JSON de resultados
        json_path = output_path / f"replay_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Replay completado. Resultados en: {output_path}")
        return results
    
    async def _process_legacy(
        self,
        ocr_results: List[Dict],
        case_id: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa con el sistema legacy
        """
        # Implementar procesamiento legacy si es necesario
        logger.info("Procesamiento legacy no implementado aún")
        return {"status": "legacy_not_implemented"}
    
    async def run_interactive(self):
        """
        Ejecuta la interfaz interactiva
        """
        self.ui = ReplayUI(self.cache_manager, self)
        await self.ui.run()
    
    async def run_cli(self, args):
        """
        Ejecuta desde línea de comandos
        """
        if args.list:
            # Listar casos disponibles
            cases = self.cache_manager.list_cached_cases()
            if not cases:
                print("No hay casos en cache")
            else:
                print(f"\n{'='*60}")
                print(f"{'Case ID':<20} {'Título':<30} {'Docs':<10}")
                print(f"{'='*60}")
                for case in cases:
                    print(f"{case['case_id']:<20} {case['case_title'][:30]:<30} {case['total_documents']:<10}")
                print(f"{'='*60}")
                print(f"Total: {len(cases)} casos\n")
            return
        
        if args.stats:
            # Mostrar estadísticas
            stats = self.cache_manager.get_cache_stats()
            print("\n📊 Estadísticas del Cache OCR:")
            print(f"  • Casos en cache: {stats['total_cases']}")
            print(f"  • Archivos cacheados: {stats['total_cached_files']}")
            print(f"  • Tamaño total: {stats['cache_size_mb']} MB")
            print(f"  • Directorio: {stats['cache_directory']}\n")
            return
        
        if args.case_id:
            # Replay de un caso específico
            options = {
                'use_ai': not args.use_legacy,
                'output_dir': args.out,
                'regenerate_report': not args.no_report,
                'model': args.model,
                'temperature': args.temperature,
                'per_doc': args.per_doc
            }
            
            results = await self.replay_case(args.case_id, options)
            print(f"\n✅ Replay completado para caso {args.case_id}")
            print(f"   Resultados en: {results['output_path']}\n")

def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parser de argumentos"""
    p = argparse.ArgumentParser(
        description="Sistema de Replay con Cache OCR - Fraud Scorer v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s                    # Modo interactivo
  %(prog)s --list            # Listar casos disponibles
  %(prog)s --stats           # Ver estadísticas del cache
  %(prog)s --case-id CASE-2025-0001  # Replay de un caso específico
        """
    )
    
    # Modos de operación
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        default=True,
        help='Modo interactivo (por defecto)'
    )
    mode_group.add_argument(
        '--case-id',
        help='Replay de un caso específico'
    )
    mode_group.add_argument(
        '--list', '-l',
        action='store_true',
        help='Listar casos disponibles'
    )
    mode_group.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Mostrar estadísticas del cache'
    )
    
    # Opciones de procesamiento
    p.add_argument(
        '--out', '-o',
        default='data/reports',
        help='Directorio de salida (default: data/reports)'
    )
    p.add_argument(
        '--use-legacy',
        action='store_true',
        help='Usar sistema legacy en lugar de IA'
    )
    p.add_argument(
        '--no-report',
        action='store_true',
        help='No generar reporte HTML/PDF'
    )
    p.add_argument(
        '--per-doc',
        action='store_true',
        help='Análisis por documento'
    )
    
    # Configuración de IA
    p.add_argument(
        '--model',
        default='gpt-4o-mini',
        choices=['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
        help='Modelo de IA a usar'
    )
    p.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperatura del modelo (0.0-1.0)'
    )
    
    return p.parse_args(argv)

async def main(argv: List[str]) -> None:
    """Función principal"""
    args = parse_args(argv)
    system = ReplaySystem()
    
    # Determinar modo de operación
    if args.list or args.stats or args.case_id:
        # Modo CLI
        await system.run_cli(args)
    else:
        # Modo interactivo
        try:
            # Verificar si tenemos rich instalado
            import rich
            await system.run_interactive()
        except ImportError:
            print("⚠️ Para usar el modo interactivo, instale: pip install rich")
            print("Puede usar el modo CLI con --help para ver las opciones")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))