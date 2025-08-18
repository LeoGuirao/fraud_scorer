# scripts/process_with_intelligent_extraction.py

#!/usr/bin/env python3
"""
Script principal para procesar documentos con extracción inteligente
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports del sistema
from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
from fraud_scorer.processors.ocr.document_extractor import UniversalDocumentExtractor
from fraud_scorer.extractors.intelligent_extractor import IntelligentFieldExtractor
from fraud_scorer.extractors.extraction_utils import ExtractionMonitor, DataValidator
from fraud_scorer.pipelines.segmented_processor import SegmentedTemplateProcessor
from fraud_scorer.storage.cases import create_case
from fraud_scorer.storage.db import upsert_document, save_ocr_result, save_extracted_data

class EnhancedFraudAnalysisSystem:
    """Sistema mejorado con extracción inteligente"""
    
    def __init__(self):
        logger.info("Inicializando sistema con extracción inteligente...")
        
        self.ocr_processor = AzureOCRProcessor()
        self.basic_extractor = UniversalDocumentExtractor()
        self.intelligent_extractor = IntelligentFieldExtractor()
        self.template_processor = SegmentedTemplateProcessor()
        self.monitor = ExtractionMonitor()
        self.validator = DataValidator()
        
        logger.info("✓ Sistema inicializado")
    
    async def process_case(self, folder_path: Path, output_path: Path) -> Dict[str, Any]:
        """Procesa un caso completo con extracción inteligente"""
        
        logger.info(f"Procesando caso: {folder_path.name}")
        
        # Crear caso en DB
        case_id = create_case(title=folder_path.name, base_path=str(folder_path))
        logger.info(f"Case ID: {case_id}")
        
        # Obtener documentos
        documents = list(folder_path.glob("*.pdf"))
        logger.info(f"Documentos encontrados: {len(documents)}")
        
        processed_docs = []
        extraction_summary = {
            'total_documents': len(documents),
            'successful_extractions': 0,
            'fields_extracted': {},
            'extraction_strategies_used': {}
        }
        
        # Procesar cada documento
        for doc_path in documents:
            logger.info(f"\nProcesando: {doc_path.name}")
            
            try:
                # 1. OCR
                logger.info("  → Ejecutando OCR...")
                ocr_result = self.ocr_processor.analyze_document(str(doc_path))
                
                # 2. Extracción básica (para tipo de documento)
                logger.info("  → Identificando tipo de documento...")
                basic_extraction = self.basic_extractor.extract_structured_data(ocr_result)
                doc_type = basic_extraction.get('document_type', 'otro')
                logger.info(f"  → Tipo: {doc_type}")
                
                # 3. Extracción inteligente
                logger.info("  → Aplicando extracción inteligente...")
                
                # Preparar documento para el extractor
                document = {
                    'document_type': doc_type,
                    'raw_text': ocr_result.get('text', ''),
                    'key_value_pairs': ocr_result.get('key_value_pairs', {}),
                    'tables': ocr_result.get('tables', []),
                }
                
                # Extraer campos inteligentemente
                extraction_results = self.intelligent_extractor.extract_all_fields(
                    document=document,
                    debug=True
                )
                
                # 4. Validar campos extraídos
                validated_fields = {}
                validation_report = {}
                
                for field_name, result in extraction_results.items():
                    if result.value is not None:
                        # Validar según tipo de campo
                        is_valid = True
                        error_msg = None
                        
                        if field_name == 'rfc':
                            is_valid, error_msg = self.validator.validate_rfc(result.value)
                        elif field_name == 'numero_poliza':
                            is_valid, error_msg = self.validator.validate_policy_number(result.value)
                        elif field_name == 'monto_reclamacion':
                            is_valid, error_msg = self.validator.validate_amount(result.value)
                        elif 'fecha' in field_name or 'vigencia' in field_name:
                            is_valid, error_msg = self.validator.validate_date(result.value)
                        
                        if is_valid:
                            validated_fields[field_name] = result.value
                            logger.info(f"    ✓ {field_name}: {result.value} (válido)")
                        else:
                            logger.warning(f"    ✗ {field_name}: {result.value} - {error_msg}")
                        
                        validation_report[field_name] = {
                            'valid': is_valid,
                            'error': error_msg,
                            'strategy': result.strategy,
                            'confidence': result.confidence
                        }
                        
                        # Actualizar estadísticas
                        if result.strategy not in extraction_summary['extraction_strategies_used']:
                            extraction_summary['extraction_strategies_used'][result.strategy] = 0
                        extraction_summary['extraction_strategies_used'][result.strategy] += 1
                
                # 5. Registrar en monitor
                for field_name, result in extraction_results.items():
                    self.monitor.log_extraction(doc_path.name, field_name, result)
                
                # 6. Guardar documento procesado
                processed_doc = {
                    'filename': doc_path.name,
                    'document_type': doc_type,
                    'fields_extracted': validated_fields,
                    'validation_report': validation_report,
                    'extraction_stats': {
                        'total_fields_attempted': len(extraction_results),
                        'fields_extracted': len(validated_fields),
                        'extraction_rate': f"{len(validated_fields)/len(extraction_results)*100:.1f}%" if extraction_results else "0%"
                    }
                }
                
                processed_docs.append(processed_doc)
                
                if validated_fields:
                    extraction_summary['successful_extractions'] += 1
                    for field in validated_fields.keys():
                        if field not in extraction_summary['fields_extracted']:
                            extraction_summary['fields_extracted'][field] = 0
                        extraction_summary['fields_extracted'][field] += 1
                
                # 7. Guardar en DB
                doc_id, _ = upsert_document(
                    case_id=case_id,
                    filepath=str(doc_path),
                    mime_type='application/pdf',
                    page_count=None,
                    language='es'
                )
                
                save_ocr_result(
                    document_id=doc_id,
                    ocr_dict=ocr_result,
                    engine='azure-di',
                    engine_version='latest'
                )
                
                # Combinar campos validados con la extracción básica
                enhanced_extraction = {**basic_extraction}
                enhanced_extraction['key_value_pairs'] = validated_fields
                
                save_extracted_data(doc_id, enhanced_extraction)
                
            except Exception as e:
                logger.error(f"Error procesando {doc_path.name}: {e}")
                continue
        
        # 8. Generar reporte de extracción
        monitor_report = self.monitor.generate_report()
        
        # 9. Generar informe final
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE EXTRACCIÓN")
        logger.info("="*60)
        logger.info(f"Documentos procesados: {extraction_summary['total_documents']}")
        logger.info(f"Extracciones exitosas: {extraction_summary['successful_extractions']}")
        logger.info(f"Tasa de éxito: {extraction_summary['successful_extractions']/extraction_summary['total_documents']*100:.1f}%")
        
        logger.info("\nCampos más extraídos:")
        for field, count in sorted(extraction_summary['fields_extracted'].items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  - {field}: {count} veces")
        
        logger.info("\nEstrategias más exitosas:")
        for strategy, count in sorted(extraction_summary['extraction_strategies_used'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {strategy}: {count} campos")
        
        # 10. Guardar resultados
        results = {
            'case_id': case_id,
            'processing_date': datetime.now().isoformat(),
            'documents_processed': processed_docs,
            'extraction_summary': extraction_summary,
            'monitor_report': monitor_report
        }
        
        # Guardar JSON
        output_file = output_path / f"extraction_report_{case_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ Reporte guardado en: {output_file}")
        
        # 11. Mostrar estadísticas del extractor
        extractor_stats = self.intelligent_extractor.get_stats_report()
        logger.info("\n" + "="*60)
        logger.info("ESTADÍSTICAS DEL EXTRACTOR INTELIGENTE")
        logger.info("="*60)
        
        for field, stats in extractor_stats.items():
            logger.info(f"\n{field}:")
            if 'overall' in stats:
                logger.info(f"  Tasa de éxito general: {stats['overall']['success_rate']}")
            if 'strategies' in stats:
                for strategy, strategy_stats in stats['strategies'].items():
                    logger.info(f"  - {strategy}: {strategy_stats['success_rate']}")
        
        return results

async def main():
    parser = argparse.ArgumentParser(description='Procesar caso con extracción inteligente')
    parser.add_argument('folder', type=Path, help='Carpeta con documentos')
    parser.add_argument('--output', type=Path, default=Path('data/reports'), help='Carpeta de salida')
    args = parser.parse_args()
    
    if not args.folder.exists():
        logger.error(f"La carpeta {args.folder} no existe")
        return
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    system = EnhancedFraudAnalysisSystem()
    await system.process_case(args.folder, args.output)

if __name__ == "__main__":
    asyncio.run(main())