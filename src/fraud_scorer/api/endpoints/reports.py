"""
API Endpoints para generación automática de informes de siniestros
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, FileResponse
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import shutil
import uuid
from datetime import datetime
import logging

from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
from fraud_scorer.templates.ai_report_generator import AIReportGenerator
from fraud_scorer.models.feedback import FeedbackPayload
from fraud_scorer.storage.feedback import save_feedback_from_json, validate_feedback_data
from fraud_scorer.storage.cases import get_case_by_id
from fraud_scorer.storage.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

# Directorio para informes generados
REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Cache de informes en proceso
processing_status = {}


@router.post("/generate")
async def generate_report(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    claim_number: Optional[str] = None,
    expedited: bool = False
):
    """
    Genera un informe de siniestro a partir de los documentos proporcionados
    
    Args:
        files: Lista de archivos de documentos del siniestro
        claim_number: Número de siniestro (opcional, se extraerá si no se proporciona)
        expedited: Si es true, procesa de forma síncrona (más rápido pero bloquea)
    
    Returns:
        ID del proceso y estado
    """
    # Validar archivos
    if not files:
        raise HTTPException(status_code=400, detail="No se proporcionaron documentos")
    
    # Generar ID de proceso
    process_id = str(uuid.uuid4())
    
    # Crear carpeta temporal para los documentos
    temp_folder = Path(f"data/temp/{process_id}")
    temp_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        # Guardar archivos temporalmente
        saved_files = []
        for file in files:
            file_path = temp_folder / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_files.append(str(file_path))
            logger.info(f"Archivo guardado: {file.filename}")
        
        # Inicializar estado
        processing_status[process_id] = {
            "status": "processing",
            "started_at": datetime.now().isoformat(),
            "files_count": len(files),
            "claim_number": claim_number,
            "progress": 0,
            "message": "Iniciando procesamiento..."
        }
        
        if expedited:
            # Procesamiento síncrono (bloquea pero es más rápido)
            result = await _process_documents_and_generate_report(
                process_id, 
                saved_files, 
                claim_number
            )
            return {
                "process_id": process_id,
                "status": "completed",
                "report_url": f"/api/v1/reports/{process_id}/download",
                "preview_url": f"/api/v1/reports/{process_id}/preview",
                "claim_number": result.get("claim_number")
            }
        else:
            # Procesamiento en background
            background_tasks.add_task(
                _process_documents_and_generate_report,
                process_id,
                saved_files,
                claim_number
            )
            
            return {
                "process_id": process_id,
                "status": "processing",
                "message": "El informe está siendo generado. Use el endpoint de estado para verificar el progreso.",
                "status_url": f"/api/v1/reports/{process_id}/status"
            }
    
    except Exception as e:
        logger.error(f"Error iniciando procesamiento: {e}")
        # Limpiar archivos temporales
        shutil.rmtree(temp_folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{process_id}/status")
async def get_report_status(process_id: str):
    """
    Obtiene el estado del procesamiento de un informe
    """
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Proceso no encontrado")
    
    status = processing_status[process_id]
    
    if status["status"] == "completed":
        status["report_url"] = f"/api/v1/reports/{process_id}/download"
        status["preview_url"] = f"/api/v1/reports/{process_id}/preview"
    
    return status


@router.get("/{process_id}/preview", response_class=HTMLResponse)
async def preview_report(process_id: str):
    """
    Muestra una vista previa HTML del informe generado
    """
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Proceso no encontrado")
    
    status = processing_status[process_id]
    
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="El informe aún no está listo")
    
    report_path = status.get("report_path")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Archivo de informe no encontrado")
    
    with open(report_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


@router.get("/{process_id}/download")
async def download_report(process_id: str):
    """
    Descarga el informe generado como archivo HTML
    """
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Proceso no encontrado")
    
    status = processing_status[process_id]
    
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="El informe aún no está listo")
    
    report_path = status.get("report_path")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Archivo de informe no encontrado")
    
    # Obtener nombre del archivo desde el path
    import os
    filename = os.path.basename(report_path)
    
    return FileResponse(
        path=report_path,
        filename=filename,
        media_type="text/html"
    )


@router.get("/{process_id}/data")
async def get_report_data(process_id: str):
    """
    Obtiene los datos estructurados del informe (JSON)
    """
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Proceso no encontrado")
    
    status = processing_status[process_id]
    
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="El informe aún no está listo")
    
    report_data = status.get("report_data")
    if not report_data:
        raise HTTPException(status_code=404, detail="Datos del informe no encontrados")
    
    return report_data


@router.delete("/{process_id}")
async def delete_report(process_id: str):
    """
    Elimina un informe y sus archivos temporales
    """
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Proceso no encontrado")
    
    try:
        # Eliminar archivo de informe
        status = processing_status[process_id]
        report_path = status.get("report_path")
        if report_path and Path(report_path).exists():
            Path(report_path).unlink()
        
        # Eliminar carpeta temporal
        temp_folder = Path(f"data/temp/{process_id}")
        if temp_folder.exists():
            shutil.rmtree(temp_folder)
        
        # Eliminar del cache
        del processing_status[process_id]
        
        return {"message": "Informe eliminado exitosamente"}
    
    except Exception as e:
        logger.error(f"Error eliminando informe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-document")
async def analyze_single_document(
    file: UploadFile = File(...),
    extract_entities: bool = True,
    detect_fraud: bool = True
):
    """
    Analiza un documento individual y retorna información estructurada
    """
    try:
        # Guardar archivo temporalmente
        temp_file = Path(f"data/temp/{uuid.uuid4()}_{file.filename}")
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_file, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Procesar con OCR
        ocr_processor = AzureOCRProcessor()
        ocr_result = ocr_processor.analyze_document(str(temp_file))
        
        response = {
            "filename": file.filename,
            "document_type": _detect_document_type(file.filename, ocr_result),
            "text_length": len(ocr_result.get("text", "")),
            "confidence": ocr_result.get("confidence", {}),
            "tables_found": len(ocr_result.get("tables", [])),
            "key_values": ocr_result.get("key_values", {})
        }
        
        if extract_entities:
            response["entities"] = ocr_result.get("entities", [])
        
        if detect_fraud:
            # Análisis rápido de fraude
            ai_analyzer = AIDocumentAnalyzer()
            fraud_analysis = await ai_analyzer.analyze_document(
                ocr_result,
                str(temp_file)
            )
            response["fraud_indicators"] = fraud_analysis.get("alerts", [])
            response["risk_level"] = fraud_analysis.get("risk_level", "low")
        
        # Limpiar archivo temporal
        temp_file.unlink()
        
        return response
    
    except Exception as e:
        logger.error(f"Error analizando documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_available_templates():
    """
    Obtiene la lista de plantillas disponibles para informes
    """
    return {
        "templates": [
            {
                "id": "standard",
                "name": "Informe Estándar",
                "description": "Plantilla completa para análisis de siniestros",
                "sections": [
                    "Información General",
                    "Análisis del Turno",
                    "Planteamiento del Problema",
                    "Métodos de Investigación",
                    "Análisis Documental",
                    "Consideraciones",
                    "Conclusión"
                ]
            },
            {
                "id": "executive",
                "name": "Resumen Ejecutivo",
                "description": "Versión condensada para decisiones rápidas",
                "sections": [
                    "Información General",
                    "Hallazgos Principales",
                    "Conclusión y Recomendación"
                ]
            },
            {
                "id": "fraud_focus",
                "name": "Enfoque en Fraude",
                "description": "Plantilla especializada en detección de fraude",
                "sections": [
                    "Información General",
                    "Indicadores de Fraude",
                    "Evidencia Documental",
                    "Validaciones Externas",
                    "Conclusión"
                ]
            }
        ]
    }


# Función auxiliar para procesar documentos y generar informe
async def _process_documents_and_generate_report(
    process_id: str,
    file_paths: List[str],
    claim_number: Optional[str] = None
) -> Dict[str, Any]:
    """
    Procesa los documentos y genera el informe completo
    """
    try:
        # Actualizar estado
        processing_status[process_id]["status"] = "processing"
        processing_status[process_id]["progress"] = 10
        processing_status[process_id]["message"] = "Iniciando OCR de documentos..."
        
        # Inicializar procesadores
        ocr_processor = AzureOCRProcessor()
        ai_analyzer = AIDocumentAnalyzer()
        template_processor = AIReportGenerator()
        
        # Procesar cada documento con OCR
        ocr_results = []
        total_files = len(file_paths)
        
        for idx, file_path in enumerate(file_paths):
            logger.info(f"Procesando documento {idx + 1}/{total_files}: {Path(file_path).name}")
            
            # Actualizar progreso
            progress = 10 + (40 * (idx + 1) / total_files)
            processing_status[process_id]["progress"] = progress
            processing_status[process_id]["message"] = f"Procesando documento {idx + 1} de {total_files}"
            
            # OCR
            ocr_result = ocr_processor.analyze_document(file_path)
            ocr_result['file_name'] = Path(file_path).name
            ocr_result['document_type'] = _detect_document_type(Path(file_path).name, ocr_result)
            ocr_results.append(ocr_result)
        
        # Análisis con AI
        processing_status[process_id]["progress"] = 60
        processing_status[process_id]["message"] = "Analizando documentos con AI..."
        
        ai_analysis = await ai_analyzer.analyze_claim_documents(ocr_results)
        
        # Extraer información y generar informe
        processing_status[process_id]["progress"] = 80
        processing_status[process_id]["message"] = "Generando informe..."
        
        # Extraer campos de los resultados OCR y análisis AI
        from fraud_scorer.processors.ai.ai_field_extractor import AIFieldExtractor
        from fraud_scorer.processors.ai.ai_consolidator import AIConsolidator
        from fraud_scorer.models.extraction import DocumentExtraction, ConsolidatedExtraction
        
        extractor = AIFieldExtractor()
        consolidator = AIConsolidator()
        
        # Extraer campos de cada documento
        extractions = []
        for ocr_result in ocr_results:
            extraction = await extractor.extract_fields(ocr_result)
            extractions.append(extraction)
        
        # Consolidar extracciones
        consolidated = await consolidator.consolidate_extractions(
            extractions=extractions,
            case_id=process_id,
            use_advanced_reasoning=True
        )
        
        # Obtener datos del asegurado y siniestro
        insured_name = consolidated.fields.nombre_asegurado or "DESCONOCIDO"
        if not claim_number:
            claim_number = consolidated.fields.numero_siniestro or process_id
        
        # Sanitizar nombres para el sistema de archivos
        from fraud_scorer.services.replay_service import sanitize_filename
        s_insured = sanitize_filename(insured_name)
        s_claim = sanitize_filename(claim_number)
        
        # Generar HTML
        report_filename = f"INF-{s_insured}-{s_claim}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = REPORTS_DIR / report_filename
        
        # Generar el reporte con los datos consolidados
        template_processor.generate_report(
            consolidated_data=consolidated,
            ai_analysis=ai_analysis,
            output_path=report_path
        )
        
        # Actualizar estado final
        processing_status[process_id]["status"] = "completed"
        processing_status[process_id]["progress"] = 100
        processing_status[process_id]["message"] = "Informe generado exitosamente"
        processing_status[process_id]["completed_at"] = datetime.now().isoformat()
        processing_status[process_id]["report_path"] = str(report_path)
        processing_status[process_id]["claim_number"] = claim_number
        processing_status[process_id]["report_data"] = template_processor._dataclass_to_dict(consolidated)
        
        # Limpiar archivos temporales
        temp_folder = Path(f"data/temp/{process_id}")
        if temp_folder.exists():
            shutil.rmtree(temp_folder)
        
        logger.info(f"Informe generado exitosamente: {report_path}")
        
        return {
            "claim_number": claim_number,
            "report_path": str(report_path)
        }
    
    except Exception as e:
        logger.error(f"Error generando informe: {e}")
        processing_status[process_id]["status"] = "error"
        processing_status[process_id]["error"] = str(e)
        processing_status[process_id]["completed_at"] = datetime.now().isoformat()
        raise


def _detect_document_type(filename: str, ocr_result: Dict) -> str:
    """
    Detecta el tipo de documento basado en el nombre y contenido
    """
    filename_lower = filename.lower()
    text_lower = ocr_result.get('text', '').lower()
    
    # Mapeo de palabras clave a tipos de documento
    type_mappings = {
        'carta_reclamacion': ['reclamacion', 'reclamación', 'reclamo'],
        'carta_respuesta': ['respuesta', 'contestación'],
        'carpeta_investigacion': ['carpeta', 'investigacion', 'investigación', 'ministerio'],
        'tarjeta_circulacion': ['circulacion', 'circulación', 'vehicular'],
        'factura_compra': ['factura', 'cfdi', 'comprobante fiscal'],
        'bitacora_viaje': ['bitacora', 'bitácora', 'viaje'],
        'reporte_gps': ['gps', 'telemetria', 'telemetría', 'rastreo'],
        'denuncia': ['denuncia', 'querella'],
        'carta_porte': ['carta porte', 'complemento carta porte'],
        'validacion_mp': ['validacion', 'validación', 'ministerio publico'],
        'consulta_repuve': ['repuve', 'registro vehicular'],
        'candados_seguridad': ['candados', 'seguridad', 'sellos'],
        'peritaje': ['peritaje', 'dictamen', 'pericial']
    }
    
    for doc_type, keywords in type_mappings.items():
        if any(keyword in filename_lower or keyword in text_lower for keyword in keywords):
            return doc_type
    
    return 'otro'


def get_case_data_for_report(case_id: str) -> Dict[str, Any]:
    """
    Obtiene los datos completos de un caso para generar el reporte de feedback.
    """
    case = get_case_by_id(case_id)
    if not case:
        return None
    
    with get_conn() as conn:
        # Obtener documentos del caso
        documents = conn.execute(
            "SELECT * FROM documents WHERE case_id = ? ORDER BY created_at",
            (case_id,)
        ).fetchall()
        
        # Obtener análisis de AI más reciente
        ai_analyses = conn.execute(
            """
            SELECT ai.* FROM ai_analyses ai
            JOIN documents d ON d.id = ai.document_id
            WHERE d.case_id = ?
            ORDER BY ai.processed_at DESC
            """,
            (case_id,)
        ).fetchall()
        
        # Buscar datos extraídos consolidados en el cache
        import json
        from pathlib import Path
        
        pipeline_cache_dir = Path("data/temp/pipeline_cache")
        consolidated_files = list(pipeline_cache_dir.glob("*ARCHIVO CONSOLIDADO.json"))
        
        consolidated_data = None
        for file in consolidated_files:
            if case_id in file.name:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        consolidated_data = json.load(f)
                    break
                except Exception as e:
                    logger.warning(f"Error leyendo archivo consolidado {file}: {e}")
        
        return {
            "case": dict(case),
            "documents": [dict(doc) for doc in documents],
            "ai_analyses": [dict(analysis) for analysis in ai_analyses],
            "consolidated_data": consolidated_data
        }


@router.get("/report/{case_id}/feedback", response_class=HTMLResponse)
async def get_interactive_report(case_id: str):
    """
    Genera y devuelve un reporte HTML interactivo para validación y feedback.
    """
    try:
        # Obtener datos del caso
        report_data = get_case_data_for_report(case_id)
        if not report_data:
            raise HTTPException(status_code=404, detail="Caso no encontrado o sin datos procesados.")
        
        # Generar reporte usando el template de feedback
        generator = AIReportGenerator()
        
        # Renderizar usando el template de feedback
        html_content = generator.render_html_template(
            template_name="report_template_feedback.html",
            data=report_data
        )
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error generando reporte interactivo: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar el reporte interactivo: {e}")


@router.post("/report/{case_id}/submit_feedback")
async def submit_feedback(case_id: str, payload: FeedbackPayload):
    """
    Recibe el JSON de feedback desde la interfaz y lo guarda en la base de datos.
    """
    try:
        # Validar que el caso exista
        case = get_case_by_id(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Caso no encontrado")
        
        # Convertir payload a lista de diccionarios
        feedback_data = [item.model_dump() for item in payload.feedback]
        
        # Validar datos
        if not validate_feedback_data(feedback_data):
            raise HTTPException(status_code=400, detail="Datos de feedback inválidos")
        
        # Guardar feedback
        save_feedback_from_json(case_id, feedback_data)
        
        logger.info(f"Feedback guardado para caso {case_id}: {len(feedback_data)} elementos")
        
        return {
            "status": "success", 
            "message": "Feedback recibido y guardado correctamente.",
            "items_saved": len(feedback_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error guardando feedback: {e}")
        raise HTTPException(status_code=500, detail=f"No se pudo guardar el feedback: {e}")


# Endpoint para obtener estadísticas
@router.get("/stats")
async def get_report_statistics():
    """
    Obtiene estadísticas de los informes generados
    """
    total_reports = len([f for f in REPORTS_DIR.glob("*.html")])
    processing = len([s for s in processing_status.values() if s["status"] == "processing"])
    completed = len([s for s in processing_status.values() if s["status"] == "completed"])
    errors = len([s for s in processing_status.values() if s["status"] == "error"])
    
    return {
        "total_reports_generated": total_reports,
        "currently_processing": processing,
        "completed_today": completed,
        "errors_today": errors,
        "storage_used_mb": sum(f.stat().st_size for f in REPORTS_DIR.glob("*.html")) / (1024 * 1024)
    }