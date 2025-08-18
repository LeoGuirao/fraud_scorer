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
        load_pipeline_config,
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
    
    # 🔄 IMPORT DEL INTELLIGENT EXTRACTOR (REEMPLAZO TOTAL)
    from fraud_scorer.extractors.intelligent_extractor import IntelligentFieldExtractor
    INTELLIGENT_EXTRACTOR_AVAILABLE = True
    
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
    
    # 🔄 IMPORT DEL INTELLIGENT EXTRACTOR
    try:
        from fraud_scorer.extractors.intelligent_extractor import IntelligentFieldExtractor
        INTELLIGENT_EXTRACTOR_AVAILABLE = True
    except ImportError:
        logger.error("❌ IntelligentFieldExtractor NO DISPONIBLE - Este módulo es REQUERIDO")
        logger.error("   Instala el extractor en: src/fraud_scorer/extractors/intelligent_extractor.py")
        sys.exit(1)


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


def _summarize_docs_for_results(processed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Crea un resumen compacto por documento para el JSON final (evita guardar OCR completo).
    """
    summary = []
    for d in processed_docs:
        ex = d.get("extracted_data") or {}
        kv_sf = ex.get("specific_fields") or {}
        # document_type puede venir en differentes lugares; hacemos fallback defensivo
        doc_type = (
            d.get("document_type")
            or ex.get("document_type")
            or ((d.get("ocr_data") or {}).get("metadata") or {}).get("document_type")
            or "unknown"
        )
        file_name = d.get("file_name") or ((d.get("ocr_data") or {}).get("metadata") or {}).get("source_name")
        summary.append(
            {
                "file_name": file_name,
                "document_type": doc_type,
                "extracted_fields_count": len(kv_sf),
            }
        )
    return summary


def _consolidated_summary(consolidated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce el tamaño del consolidado a lo útil para auditoría.
    """
    return {
        "case_info": consolidated.get("case_info", {}),
        "validation_summary": consolidated.get("validation_summary", {}),
        "processing_stats": consolidated.get("processing_stats", {}),
    }


# 🔄 FUNCIÓN COMPLETAMENTE NUEVA - REEMPLAZA LA EXTRACCIÓN ANTIGUA
def _extract_with_intelligent(ocr_dict: Dict[str, Any], file_name: str) -> Dict[str, Any]:
    """
    REEMPLAZA COMPLETAMENTE la extracción antigua con IntelligentFieldExtractor.
    
    Args:
        ocr_dict: Resultado del OCR de Azure
        file_name: Nombre del archivo para logging
        
    Returns:
        Diccionario con la estructura esperada por el sistema
    """
    logger.info(f"  → Extrayendo con IntelligentFieldExtractor: {file_name}")
    
    # Inicializar el extractor inteligente
    extractor = IntelligentFieldExtractor()
    
    # Primero, determinar el tipo de documento basándose en el contenido
    raw_text = ocr_dict.get('text', '').lower()
    
    # Detección simple de tipo de documento
    document_type = 'otro'
    if any(word in raw_text for word in ['póliza', 'poliza', 'vigencia', 'cobertura', 'asegurado', 'prima']):
        document_type = 'poliza'
    elif any(word in raw_text for word in ['factura', 'invoice', 'cfdi', 'comprobante fiscal', 'subtotal', 'iva']):
        document_type = 'factura'
    elif any(word in raw_text for word in ['denuncia', 'ministerio', 'delito', 'querella']):
        document_type = 'denuncia'
    elif any(word in raw_text for word in ['carta porte', 'guía', 'remisión', 'transporte']):
        document_type = 'carta_porte'
    elif any(word in raw_text for word in ['gps', 'coordenadas', 'tracking', 'rastreo', 'ubicación']):
        document_type = 'bitacora_gps'
    
    logger.debug(f"    Tipo detectado: {document_type}")
    
    # Preparar documento para el extractor
    document = {
        'document_type': document_type,
        'raw_text': ocr_dict.get('text', ''),
        'key_value_pairs': {},  # El OCR de Azure no siempre da key-value pairs limpios
        'tables': ocr_dict.get('tables', []),
    }
    
    # Si el OCR tiene key_value_pairs, intentar extraerlos
    if 'key_value_pairs' in ocr_dict:
        document['key_value_pairs'] = ocr_dict['key_value_pairs']
    
    # Mapeo completo de campos por tipo de documento
    fields_mapping = {
        'poliza': [
            'numero_poliza', 'nombre_asegurado', 'rfc', 
            'vigencia_inicio', 'vigencia_fin', 'domicilio_poliza',
            'tipo_siniestro', 'monto_reclamacion'
        ],
        'factura': [
            'rfc', 'fecha_siniestro', 'monto_reclamacion', 
            'total', 'numero_factura', 'emisor', 'receptor'
        ],
        'denuncia': [
            'fecha_siniestro', 'lugar_hechos', 'tipo_siniestro',
            'numero_denuncia', 'autoridad', 'descripcion_hechos'
        ],
        'carta_porte': [
            'origen', 'destino', 'fecha_transporte', 'transportista',
            'tipo_mercancia', 'peso', 'lugar_hechos'
        ],
        'bitacora_gps': [
            'coordenadas', 'fecha_evento', 'velocidad', 
            'direccion', 'lugar_hechos'
        ],
        'otro': [
            'numero_poliza', 'nombre_asegurado', 'rfc',
            'monto_reclamacion', 'fecha_siniestro'
        ]
    }
    
    # Obtener campos a extraer
    fields_to_extract = fields_mapping.get(document_type, fields_mapping['otro'])
    
    # EXTRAER TODOS LOS CAMPOS POSIBLES
    extraction_results = extractor.extract_all_fields(
        document=document,
        fields_to_extract=fields_to_extract,
        debug=False  # Cambiar a True para ver detalles
    )
    
    # Construir la estructura esperada por el sistema
    extracted_data = {
        'document_type': document_type,
        'confidence': 0.0,
        'entities': [],
        'key_value_pairs': {},
        'specific_fields': {},
        'tables': ocr_dict.get('tables', []),
        'extraction_metadata': {
            'extractor': 'IntelligentFieldExtractor',
            'timestamp': datetime.now().isoformat(),
            'fields_attempted': len(fields_to_extract),
            'fields_extracted': 0
        }
    }
    
    # Llenar los campos extraídos
    total_confidence = 0.0
    fields_extracted = 0
    
    for field_name, result in extraction_results.items():
        if result.value is not None:
            # Agregar a specific_fields
            extracted_data['specific_fields'][field_name] = result.value
            
            # También agregar a key_value_pairs para compatibilidad
            extracted_data['key_value_pairs'][field_name] = result.value
            
            # Acumular confianza
            total_confidence += result.confidence
            fields_extracted += 1
            
            logger.debug(f"    ✓ {field_name}: {result.value} (confianza: {result.confidence:.2f})")
    
    # Calcular confianza promedio
    if fields_extracted > 0:
        extracted_data['confidence'] = total_confidence / fields_extracted
        extracted_data['extraction_metadata']['fields_extracted'] = fields_extracted
        logger.info(f"    → Extraídos {fields_extracted}/{len(fields_to_extract)} campos (confianza promedio: {extracted_data['confidence']:.2f})")
    else:
        logger.warning(f"    ⚠ No se extrajeron campos del documento")
    
    # Agregar entidades si las hay (compatibilidad)
    if 'entities' in ocr_dict:
        extracted_data['entities'] = ocr_dict['entities']
    
    return extracted_data


# ------------------------------------------------------------------------------------
# Sistema principal
# ------------------------------------------------------------------------------------
class FraudAnalysisSystem:
    """
    Sistema completo de análisis de fraude con soporte DB/casos.
    🔄 AHORA USA SOLO IntelligentFieldExtractor para TODA la extracción.
    """

    def __init__(self, config_path: Optional[str] = None, use_legacy: bool = False):
        print("Inicializando procesadores...")
        self.ocr_processor = AzureOCRProcessor()
        
        # 🔄 DECISIÓN: Usar legacy extractor o IntelligentFieldExtractor
        self.use_legacy = use_legacy
        if use_legacy:
            print("⚠️  Usando extractor LEGACY (UniversalDocumentExtractor)")
            self.extractor = UniversalDocumentExtractor()
        else:
            if not INTELLIGENT_EXTRACTOR_AVAILABLE:
                print("❌ IntelligentFieldExtractor no disponible pero es requerido")
                sys.exit(1)
            print("✓ Usando IntelligentFieldExtractor (REEMPLAZO COMPLETO)")
            self.extractor = None  # No necesitamos el UniversalDocumentExtractor
        
        self.ai_analyzer = AIDocumentAnalyzer()
        self.template_processor = SegmentedTemplateProcessor(config_path=config_path)
        
        print("✓ Procesadores inicializados\n")

    async def _build_ai_case_from_consolidated(self, consolidated: Dict[str, Any]) -> Dict[str, Any]:
        """
        En lugar de pasar TODOS los docs a la IA, pasamos un 'documento' sintético con el resumen consolidado.
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
        Nuevo caso con IntelligentFieldExtractor como REEMPLAZO TOTAL.
        """
        print("=" * 60)
        print(f"📁 Nuevo caso desde carpeta: {folder_path.name}")
        if not self.use_legacy:
            print("🧠 Extracción: IntelligentFieldExtractor (COMPLETO)")
        else:
            print("⚠️  Extracción: Legacy (UniversalDocumentExtractor)")
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
        
        # Estadísticas de extracción
        extraction_stats = {
            'total_documents': len(documents),
            'documents_with_extraction': 0,
            'total_fields_extracted': 0,
            'fields_by_type': {},
            'extraction_times': []
        }

        print("🔍 Ejecutando OCR y extracción inteligente...")
        from tqdm import tqdm

        for doc_path in tqdm(documents, desc="Procesando", unit="doc"):
            try:
                start_time = datetime.now()
                
                # Registrar documento en DB
                doc_id, created = upsert_document(
                    case_id=case_id,
                    filepath=str(doc_path),
                    mime_type=None,
                    page_count=None,
                    language=None,
                )

                # OCR con Azure
                raw_ocr = self.ocr_processor.analyze_document(str(doc_path))
                ocr_dict = ocr_result_to_dict(raw_ocr)

                # Inyectar metadatos
                meta = ocr_dict.setdefault("metadata", {}) or {}
                meta.setdefault("source_name", doc_path.name)
                meta.setdefault("source_path", str(doc_path))

                # Persistir OCR
                save_ocr_result(
                    document_id=doc_id,
                    ocr_dict=ocr_dict,
                    engine="azure-di",
                    engine_version=(ocr_dict.get("metadata", {}) or {}).get("model_used", ""),
                )
                mark_ocr_success(doc_id, ok=True)

                # 🔄 EXTRACCIÓN: Usar IntelligentFieldExtractor o legacy
                if self.use_legacy:
                    # Método antiguo
                    extracted = self.extractor.extract_structured_data(ocr_dict)
                else:
                    # 🔄 MÉTODO NUEVO - REEMPLAZO COMPLETO
                    extracted = _extract_with_intelligent(ocr_dict, doc_path.name)
                
                # Actualizar estadísticas
                if extracted.get('specific_fields'):
                    extraction_stats['documents_with_extraction'] += 1
                    extraction_stats['total_fields_extracted'] += len(extracted['specific_fields'])
                    
                    doc_type = extracted.get('document_type', 'unknown')
                    if doc_type not in extraction_stats['fields_by_type']:
                        extraction_stats['fields_by_type'][doc_type] = 0
                    extraction_stats['fields_by_type'][doc_type] += len(extracted['specific_fields'])
                
                # Tiempo de procesamiento
                processing_time = (datetime.now() - start_time).total_seconds()
                extraction_stats['extraction_times'].append(processing_time)
                
                # Persistir extracción
                save_extracted_data(doc_id, extracted)

                processed_docs.append({
                    "document_id": doc_id,
                    "file_name": doc_path.name,
                    "file_path": str(doc_path),
                    "file_size_kb": doc_path.stat().st_size / 1024.0,
                    "ocr_data": ocr_dict,
                    "extracted_data": extracted,
                    "processing_time_seconds": processing_time
                })

            except Exception as e:
                logger.error(f"Error procesando {doc_path.name}: {e}", exc_info=True)
                ocr_errors.append({"file": doc_path.name, "error": str(e)})
                print(f"\n⚠️  Error procesando {doc_path.name}: {e}")

        # Resumen de extracción
        usable = [d for d in processed_docs if d.get("extracted_data")]
        print(f"\n✓ Procesamiento completado:")
        print(f"  • Documentos procesados: {len(processed_docs)}")
        print(f"  • Documentos con campos extraídos: {extraction_stats['documents_with_extraction']}")
        print(f"  • Total de campos extraídos: {extraction_stats['total_fields_extracted']}")
        
        if extraction_stats['fields_by_type']:
            print(f"  • Campos por tipo de documento:")
            for doc_type, count in extraction_stats['fields_by_type'].items():
                print(f"    - {doc_type}: {count} campos")
        
        if extraction_stats['extraction_times']:
            avg_time = sum(extraction_stats['extraction_times']) / len(extraction_stats['extraction_times'])
            print(f"  • Tiempo promedio por documento: {avg_time:.2f} segundos")

        # Normalización para el template
        docs_for_template = build_docs_for_template_from_processed(usable)

        logger.info("Procesando documentos en modo segmentado/aislado…")
        consolidated = await self.template_processor.orchestrator.process_documents(docs_for_template)

        # IA del caso
        print("\n🤖 Analizando con AI (resumen consolidado)…")
        ai_analysis = await self._build_ai_case_from_consolidated(consolidated)

        # Persistir corrida IA
        run_id = None
        if not no_save:
            run_id = create_run(
                case_id=case_id,
                purpose="case_summary_segmented",
                llm_model="gpt-4-turbo-preview",
                params={
                    "from": "new-folder-segmented",
                    "extractor": "IntelligentFieldExtractor" if not self.use_legacy else "UniversalDocumentExtractor"
                },
            )

        # Resumen
        fraud_score = float(ai_analysis.get("fraud_score", 0.0))
        risk_level = "bajo" if fraud_score < 0.3 else ("medio" if fraud_score < 0.6 else "alto")

        results: Dict[str, Any] = {
            "case_id": case_id,
            "folder_name": folder_path.name,
            "processing_date": datetime.now().isoformat(),
            "documents_processed": len(processed_docs),
            "documents_summary": _summarize_docs_for_results(processed_docs),
            "consolidated_summary": _consolidated_summary(consolidated),
            "extraction_stats": extraction_stats,
            "errors": ocr_errors,
            "fraud_analysis": {
                "fraud_score": round(fraud_score * 100, 2),
                "risk_level": risk_level,
                "indicators": (ai_analysis.get("fraud_indicators", []) or [])[:5],
                "inconsistencies": (ai_analysis.get("inconsistencies", []) or [])[:5],
                "external_validations": ai_analysis.get("external_validations", []),
                "route_analysis": ai_analysis.get("route_analysis", {}),
            },
            "ai_analysis_raw": ai_analysis,
        }

        # Generar informe HTML
        print("\n📝 Generando informe (segmentado)…")
        informe = self.template_processor._build_informe_from_consolidated(consolidated, ai_analysis)
        informe.numero_siniestro = case_id

        pretty_html_path = output_path / f"INF-{informe.numero_siniestro}.html"
        self.template_processor.generate_report(informe, str(pretty_html_path))
        print(f"✓ Informe HTML generado: {pretty_html_path}")

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

    async def reanalyze_case_from_db(
        self,
        case_id: str,
        output_path: Path,
        no_save: bool = False,
        clobber: bool = False,
    ) -> Dict[str, Any]:
        """
        Reanaliza un caso YA OCR-eado.
        NOTA: Esto usa los datos ya extraídos en DB, no re-extrae.
        """
        print("=" * 60)
        print(f"🔁 Re-analizando caso (sin OCR): {case_id}")
        print("=" * 60)

        if clobber:
            _clean_outputs(output_path, case_id=case_id)

        # Cargar desde DB
        docs_for_template, processed_docs = build_docs_for_template_from_db(case_id)
        if not docs_for_template:
            raise RuntimeError("No se encontraron documentos/ocr guardados para este case_id")

        # Segmentación + consolidación
        logger.info("Procesando documentos en modo segmentado/aislado…")
        consolidated = await self.template_processor.orchestrator.process_documents(docs_for_template)

        # IA
        print("\n🤖 Analizando con AI (resumen consolidado)…")
        ai_analysis = await self._build_ai_case_from_consolidated(consolidated)

        # Persistir corrida IA
        run_id = None
        if not no_save:
            run_id = create_run(
                case_id=case_id,
                purpose="case_summary_segmented",
                llm_model="gpt-4-turbo-preview",
                params={"from": "reanalyze-segmented"},
            )

        # Resumen
        fraud_score = float(ai_analysis.get("fraud_score", 0.0))
        risk_level = "bajo" if fraud_score < 0.3 else ("medio" if fraud_score < 0.6 else "alto")

        results: Dict[str, Any] = {
            "case_id": case_id,
            "processing_date": datetime.now().isoformat(),
            "documents_processed": len(processed_docs),
            "documents_summary": _summarize_docs_for_results(processed_docs),
            "consolidated_summary": _consolidated_summary(consolidated),
            "errors": [],
            "fraud_analysis": {
                "fraud_score": round(fraud_score * 100, 2),
                "risk_level": risk_level,
                "indicators": (ai_analysis.get("fraud_indicators", []) or [])[:5],
                "inconsistencies": (ai_analysis.get("inconsistencies", []) or [])[:5],
                "external_validations": ai_analysis.get("external_validations", []),
                "route_analysis": ai_analysis.get("route_analysis", {}),
            },
            "ai_analysis_raw": ai_analysis,
        }

        # Informe
        print("\n📝 Generando informe (segmentado)…")
        informe = self.template_processor._build_informe_from_consolidated(consolidated, ai_analysis)
        informe.numero_siniestro = case_id

        pretty_html_path = output_path / f"INF-{informe.numero_siniestro}.html"
        self.template_processor.generate_report(informe, str(pretty_html_path))
        print(f"✓ Informe HTML generado: {pretty_html_path}")

        # JSON
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
        description="Fraud Scorer - Procesamiento con IntelligentFieldExtractor"
    )
    p.add_argument("folder", nargs="?", help="Carpeta de documentos (modo nuevo caso).")
    p.add_argument("--case-id", dest="case_id", help="Reanaliza un caso existente (sin OCR).")
    p.add_argument("--out", dest="out", default="data/reports", help="Carpeta de salida de reportes.")
    p.add_argument("--title", dest="title", default=None, help="Título para el caso (modo carpeta).")
    p.add_argument("--no-save", dest="no_save", action="store_true", help="No persiste análisis de IA.")
    p.add_argument("--clobber", action="store_true", help="Borra reportes previos antes de generar nuevos")
    p.add_argument("--config", dest="config", default=None, help="Ruta a pipeline_config.yaml")
    
    # 🔄 Opción para usar el extractor legacy (solo para comparación)
    p.add_argument(
        "--use-legacy",
        dest="use_legacy",
        action="store_true",
        help="Usar UniversalDocumentExtractor en lugar de IntelligentFieldExtractor"
    )
    p.add_argument("--debug", action="store_true", help="Activa modo debug con más logging")
    
    args = p.parse_args(argv)

    if not args.folder and not args.case_id:
        p.error("Debes pasar una carpeta o --case-id")
    if args.folder and args.case_id:
        p.error("Usa solo carpeta o solo --case-id, no ambos.")

    return args


async def main(argv: List[str]) -> None:
    args = parse_args(argv)
    
    # Configurar logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Modo DEBUG activado")
    
    output_path = Path(args.out)
    output_path.mkdir(parents=True, exist_ok=True)

    # 🔄 Sistema usa IntelligentFieldExtractor por defecto
    system = FraudAnalysisSystem(
        config_path=args.config,
        use_legacy=args.use_legacy
    )

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