# src/fraud_scorer/api/web_interface.py

"""
Interfaz Web para Fraud Scorer
Permite a usuarios cargar documentos y recibir reportes procesados
"""

import os
import sys
import asyncio
import tempfile
import shutil
import logging
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Añadir el directorio del proyecto al path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))

# Importar tu sistema
from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
from fraud_scorer.processors.ai.ai_field_extractor import AIFieldExtractor
from fraud_scorer.processors.ai.ai_consolidator import AIConsolidator
from fraud_scorer.templates.ai_report_generator import AIReportGenerator
from fraud_scorer.storage.cases import create_case
from fraud_scorer.storage.ocr_cache import OCRCacheManager
from fraud_scorer.parsers.document_parser import DocumentParser
from fraud_scorer.api.endpoints.replay import router as replay_router

# Inicializar FastAPI
app = FastAPI(
    title="Fraud Scorer Web Interface",
    description="Sistema de Análisis de Siniestros con Detección de Fraude",
    version="2.0.0"
)

# Configurar CORS para permitir acceso desde la red local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar directorios
UPLOAD_DIR = project_root / "data" / "uploads"
REPORTS_DIR = project_root / "data" / "reports"
TEMP_DIR = project_root / "data" / "temp"
TEMPLATES_DIR = project_root / "src" / "fraud_scorer" / "api" / "templates"
STATIC_DIR = project_root / "src" / "fraud_scorer" / "api" / "static"

# Crear directorios si no existen
for dir_path in [UPLOAD_DIR, REPORTS_DIR, TEMP_DIR, TEMPLATES_DIR, STATIC_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configurar plantillas y archivos estáticos
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Incluir router de replay con prefijo
app.include_router(replay_router, prefix="/replay", tags=["replay"])

# Log de todas las rutas registradas (para debugging)
@app.on_event("startup")
async def log_routes():
    """Log todas las rutas disponibles al iniciar"""
    logger.info("=" * 60)
    logger.info("RUTAS DISPONIBLES EN LA API:")
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(route.methods) if route.methods else "N/A"
            logger.info(f"  {methods:8} -> {route.path}")
    logger.info("=" * 60)

# Estado de procesamiento (en memoria para simplicidad)
processing_status = {}

class FraudScorerProcessor:
    """Clase principal para procesar documentos"""
    
    def __init__(self):
        self.ocr_processor = AzureOCRProcessor()
        self.document_parser = DocumentParser(self.ocr_processor)
        self.cache_manager = OCRCacheManager()
        self.extractor = AIFieldExtractor()
        self.consolidator = AIConsolidator()
        self.report_generator = AIReportGenerator()
        logger.info("FraudScorerProcessor inicializado")
    
    async def process_case(self, files: List[Path], case_title: str) -> dict:
        """Procesa un caso completo"""
        try:
            # Crear caso en DB
            case_id = create_case(title=case_title)
            logger.info(f"Procesando caso: {case_id}")
            
            # Procesar cada documento
            ocr_results = []
            for file_path in files:
                logger.info(f"Procesando: {file_path.name}")
                
                # Verificar cache
                if self.cache_manager.has_cache(file_path):
                    logger.info(f"Usando cache para: {file_path.name}")
                    ocr_result = self.cache_manager.get_cache(file_path)
                else:
                    # Procesar con OCR/Parser
                    ocr_result = self.document_parser.parse_document(file_path)
                    if ocr_result:
                        self.cache_manager.save_cache(file_path, ocr_result)
                
                if ocr_result:
                    ocr_results.append({
                        "filename": file_path.name,
                        "ocr_result": ocr_result,
                        "document_type": self._detect_document_type(file_path.name)
                    })
            
            # Extraer campos con IA
            logger.info("Extrayendo campos con IA...")
            extractions = await self.extractor.extract_from_documents_batch(
                documents=ocr_results,
                parallel_limit=3
            )
            
            # Consolidar resultados
            logger.info("Consolidando resultados...")
            consolidated = await self.consolidator.consolidate_extractions(
                extractions=extractions,
                case_id=case_id,
                use_advanced_reasoning=True
            )
            
            # Análisis de fraude
            logger.info("Analizando fraude...")
            from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
            analyzer = AIDocumentAnalyzer()
            
            docs_for_analysis = []
            for doc in ocr_results:
                docs_for_analysis.append({
                    "document_type": doc.get("document_type", "otro"),
                    "key_value_pairs": doc.get("ocr_result", {}).get("key_value_pairs", {}),
                    "specific_fields": {}
                })
            
            ai_analysis = await analyzer.analyze_claim_documents(docs_for_analysis)
            
            # Generar reporte
            logger.info("Generando reporte HTML...")
            html_path = REPORTS_DIR / f"INF-{case_id}.html"
            self.report_generator.generate_report(
                consolidated_data=consolidated,
                ai_analysis=ai_analysis,
                output_path=html_path
            )
            
            return {
                "success": True,
                "case_id": case_id,
                "html_path": str(html_path),
                "fraud_score": ai_analysis.get("fraud_score", 0),
                "risk_level": self._get_risk_level(ai_analysis.get("fraud_score", 0))
            }
            
        except Exception as e:
            logger.error(f"Error procesando caso: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _detect_document_type(self, filename: str) -> str:
        """Detecta el tipo de documento basado en el nombre"""
        filename_lower = filename.lower()
        
        if "factura" in filename_lower or "invoice" in filename_lower:
            return "factura"
        elif "poliza" in filename_lower or "policy" in filename_lower:
            return "poliza"
        elif "denuncia" in filename_lower:
            return "denuncia"
        elif "carta" in filename_lower:
            return "carta_reclamacion"
        else:
            return "otro"
    
    def _get_risk_level(self, fraud_score: float) -> str:
        """Determina el nivel de riesgo"""
        if fraud_score < 0.3:
            return "BAJO"
        elif fraud_score < 0.6:
            return "MEDIO"
        else:
            return "ALTO"

# Instancia global del procesador
processor = FraudScorerProcessor()

async def load_consolidated_data(case_id: str) -> Optional[dict]:
    """Carga los datos consolidados de un caso"""
    try:
        pipeline_cache_dir = project_root / "data" / "temp" / "pipeline_cache"
        logger.info(f"Buscando datos consolidados para caso {case_id} en {pipeline_cache_dir}")
        
        # Método 1: Buscar por nombre de archivo (método antiguo - por si acaso)
        for file_path in pipeline_cache_dir.glob(f"*{case_id}*.json"):
            if "ARCHIVO CONSOLIDADO" in file_path.name:
                logger.info(f"Encontrado archivo por nombre: {file_path.name}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return extract_fields_from_data(data, case_id)
        
        # Método 2: Buscar en TODOS los archivos consolidados por contenido
        logger.info("Buscando por contenido en todos los archivos consolidados...")
        consolidated_files = list(pipeline_cache_dir.glob("*ARCHIVO CONSOLIDADO*.json"))
        logger.info(f"Archivos consolidados encontrados: {[f.name for f in consolidated_files]}")
        
        for file_path in consolidated_files:
            try:
                logger.info(f"Revisando archivo: {file_path.name}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Buscar el case_id en diferentes lugares del JSON
                if (data.get("case_id") == case_id or 
                    data.get("consolidated_fields", {}).get("numero_siniestro") == case_id or
                    case_id in str(data)):
                    logger.info(f"¡Encontrado caso {case_id} en archivo: {file_path.name}!")
                    return extract_fields_from_data(data, case_id)
                    
            except Exception as file_error:
                logger.warning(f"Error leyendo archivo {file_path.name}: {file_error}")
                continue
        
        # Método 3: Buscar en archivos de resultados en reports/
        logger.info("Buscando en archivos de resultados...")
        reports_dir = project_root / "data" / "reports"
        for file_path in reports_dir.glob("resultados_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if data.get("case_id") == case_id:
                    logger.info(f"Encontrado en archivo de resultados: {file_path.name}")
                    if "consolidated_data" in data:
                        fields = data["consolidated_data"].get("consolidated_fields", {})
                        return {
                            "case_id": case_id,
                            **extract_fields_dict(fields)
                        }
            except Exception as file_error:
                logger.warning(f"Error leyendo archivo de resultados {file_path.name}: {file_error}")
                continue
        
        logger.warning(f"No se encontraron datos consolidados para caso {case_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error cargando datos consolidados: {e}", exc_info=True)
        return None

def extract_fields_from_data(data: dict, case_id: str) -> dict:
    """Extrae campos de los datos consolidados"""
    fields = data.get("consolidated_fields", {})
    return {
        "case_id": case_id,
        **extract_fields_dict(fields)
    }

def extract_fields_dict(fields: dict) -> dict:
    """Extrae y mapea los campos a un diccionario plano"""
    return {
        "numero_siniestro": fields.get("numero_siniestro"),
        "nombre_asegurado": fields.get("nombre_asegurado"),
        "monto_reclamacion": fields.get("monto_reclamacion"),
        "numero_poliza": fields.get("numero_poliza"),
        "vigencia": fields.get("vigencia"),
        "domicilio_poliza": fields.get("domicilio_poliza"),
        "tipo_siniestro": fields.get("tipo_siniestro"),
        "fecha_ocurrencia": fields.get("fecha_ocurrencia"),
        "fecha_reclamacion": fields.get("fecha_reclamacion"),
        "lugar_hechos": fields.get("lugar_hechos"),
        "bien_reclamado": fields.get("bien_reclamado"),
        "ajuste": fields.get("ajuste"),
    }

def apply_feedback_corrections(original_data: dict, feedback_data: dict) -> dict:
    """Aplica las correcciones del feedback a los datos originales"""
    corrected_data = original_data.copy()
    
    # Mapeo de etiquetas a campos
    field_mapping = {
        "NÚMERO DE SINIESTRO": "numero_siniestro",
        "NOMBRE DEL ASEGURADO": "nombre_asegurado", 
        "MONTO DE RECLAMACIÓN": "monto_reclamacion",
        "NÚMERO DE PÓLIZA": "numero_poliza",
        "VIGENCIA": "vigencia",
        "DOMICILIO DE LA PÓLIZA": "domicilio_poliza",
        "TIPO DE SINIESTRO": "tipo_siniestro",
        "FECHA DE OCURRENCIA": "fecha_ocurrencia",
        "FECHA DE RECLAMACIÓN": "fecha_reclamacion",
        "LUGAR DE LOS HECHOS": "lugar_hechos",
        "BIEN RECLAMADO": "bien_reclamado",
        "AJUSTE": "ajuste"
    }
    
    # Aplicar correcciones
    fields_feedback = feedback_data.get("fields", {})
    for label, field_data in fields_feedback.items():
        # Buscar el campo correspondiente
        field_key = None
        for map_label, map_key in field_mapping.items():
            if map_label.lower() in label.lower() or label.lower() in map_label.lower():
                field_key = map_key
                break
        
        if field_key and field_data.get("final_value"):
            corrected_data[field_key] = field_data["final_value"]
            logger.info(f"Campo corregido: {field_key} = {field_data['final_value']}")
    
    return corrected_data

def create_mock_consolidated_extraction(corrected_data: dict, case_id: str):
    """Crea un objeto mock de ConsolidatedExtraction con los datos corregidos"""
    from fraud_scorer.models.extraction import ConsolidatedExtraction, ConsolidatedFields
    
    # Crear campos consolidados
    consolidated_fields = ConsolidatedFields(
        numero_siniestro=corrected_data.get("numero_siniestro"),
        nombre_asegurado=corrected_data.get("nombre_asegurado"),
        monto_reclamacion=corrected_data.get("monto_reclamacion"),
        numero_poliza=corrected_data.get("numero_poliza"),
        vigencia=corrected_data.get("vigencia"),
        domicilio_poliza=corrected_data.get("domicilio_poliza"),
        tipo_siniestro=corrected_data.get("tipo_siniestro"),
        fecha_ocurrencia=corrected_data.get("fecha_ocurrencia"),
        fecha_reclamacion=corrected_data.get("fecha_reclamacion"),
        lugar_hechos=corrected_data.get("lugar_hechos"),
        bien_reclamado=corrected_data.get("bien_reclamado"),
        ajuste=corrected_data.get("ajuste")
    )
    
    # Crear consolidación mock
    mock_consolidated = ConsolidatedExtraction(
        case_id=case_id,
        consolidated_fields=consolidated_fields,
        source_documents=[],
        confidence_scores={},
        conflicts_resolved=[],
        processing_notes=["Datos corregidos por validación manual del usuario"]
    )
    
    return mock_consolidated

@app.get("/", response_class=HTMLResponse)
async def get_main_portal(request: Request):
    """
    Sirve la nueva página principal (el portal) que permite al usuario
    elegir entre 'Upload' y 'Replay'.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    """
    Sirve la página original de carga de documentos.
    Esta es la nueva ruta para la funcionalidad de 'run_report'.
    """
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Endpoint para subir archivos y procesarlos"""
    if not files:
        raise HTTPException(status_code=400, detail="No se subieron archivos")
    
    # Generar ID único para este proceso
    process_id = str(uuid.uuid4())
    
    # Crear directorio temporal para este proceso
    process_dir = TEMP_DIR / process_id
    process_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar archivos
    saved_files = []
    for file in files:
        file_path = process_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_files.append(file_path)
        logger.info(f"Archivo guardado: {file.filename}")
    
    # Inicializar estado
    processing_status[process_id] = {
        "status": "processing",
        "message": "Procesando documentos...",
        "progress": 0,
        "started_at": datetime.now().isoformat()
    }
    
    # Procesar en background
    background_tasks.add_task(
        process_documents_background,
        process_id,
        saved_files
    )
    
    return {
        "process_id": process_id,
        "message": "Procesamiento iniciado",
        "status_url": f"/status/{process_id}"
    }

async def process_documents_background(process_id: str, files: List[Path]):
    """Procesa documentos en background usando el sistema completo de run_report.py"""
    try:
        # Actualizar estado
        processing_status[process_id]["message"] = "Iniciando procesamiento..."
        processing_status[process_id]["progress"] = 10
        
        # Usar el sistema completo de FraudAnalysisSystemV2 de run_report.py
        # Agregar la ruta de scripts al path
        scripts_path = project_root / "scripts"
        if str(scripts_path) not in sys.path:
            sys.path.insert(0, str(scripts_path))
        
        from run_report import FraudAnalysisSystemV2
        
        # Los archivos ya están en el directorio temporal correcto
        # No necesitamos copiarlos, solo usar la ruta del directorio padre
        temp_case_dir = files[0].parent if files else TEMP_DIR / process_id
        
        processing_status[process_id]["message"] = "Procesando documentos con IA..."
        processing_status[process_id]["progress"] = 30
        
        # Inicializar el sistema v2
        system = FraudAnalysisSystemV2()
        
        # Procesar el caso usando el sistema completo
        case_title = f"Caso_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        processing_status[process_id]["message"] = "Analizando con IA..."
        processing_status[process_id]["progress"] = 60
        
        # Ejecutar el procesamiento completo
        result = await system.process_case(
            folder_path=temp_case_dir,
            output_path=REPORTS_DIR,
            case_title=case_title
        )
        
        processing_status[process_id]["message"] = "Finalizando..."
        processing_status[process_id]["progress"] = 90
        
        # Extraer información del resultado
        case_id = result.get("case_id", "UNKNOWN")
        fraud_analysis = result.get("fraud_analysis", {})
        fraud_score = fraud_analysis.get("fraud_score", 0)
        risk_level = "BAJO" if fraud_score < 0.3 else ("MEDIO" if fraud_score < 0.6 else "ALTO")
        
        processing_status[process_id] = {
            "status": "completed",
            "message": "Procesamiento completado - Listo para validación",
            "progress": 100,
            "case_id": case_id,
            "report_url": f"/feedback/{case_id}",  # Redirigir primero al feedback
            "fraud_score": fraud_score,
            "risk_level": risk_level,
            "completed_at": datetime.now().isoformat()
        }
        
        logger.info(f"Procesamiento completo exitoso para caso {case_id}")
            
    except Exception as e:
        logger.error(f"Error en proceso {process_id}: {e}", exc_info=True)
        processing_status[process_id] = {
            "status": "error",
            "message": f"Error procesando documentos: {str(e)}",
            "progress": 0
        }
    finally:
        # Limpiar archivos temporales
        try:
            shutil.rmtree(TEMP_DIR / process_id)
        except Exception as cleanup_error:
            logger.warning(f"No se pudo limpiar directorio temporal {process_id}: {cleanup_error}")

@app.get("/status/{process_id}")
async def get_status(process_id: str):
    """Obtiene el estado del procesamiento"""
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Proceso no encontrado")
    
    return processing_status[process_id]

@app.get("/feedback/{case_id}", response_class=HTMLResponse)
async def get_feedback_page(case_id: str, request: Request):
    """Sirve la página de feedback para validación de datos"""
    # Cargar datos consolidados del caso
    consolidated_data = await load_consolidated_data(case_id)
    
    if not consolidated_data:
        raise HTTPException(status_code=404, detail="Datos del caso no encontrados")
    
    # Renderizar template de feedback con los datos
    template_path = project_root / "src" / "fraud_scorer" / "templates" / "report_template_feedback.html"
    
    # Leer el template y renderizarlo con los datos usando Jinja2
    from jinja2 import Environment, FileSystemLoader
    
    template_dir = project_root / "src" / "fraud_scorer" / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("report_template_feedback.html")
    
    # Renderizar con los datos consolidados
    rendered_html = template.render(**consolidated_data)
    
    return HTMLResponse(content=rendered_html)

@app.post("/process-feedback/{case_id}")
async def process_feedback(case_id: str, feedback_data: dict):
    """Procesa el feedback del usuario y regenera el reporte final con correcciones"""
    try:
        # Cargar datos consolidados originales
        original_data = await load_consolidated_data(case_id)
        if not original_data:
            raise HTTPException(status_code=404, detail="Datos del caso no encontrados")
        
        # Aplicar correcciones del feedback
        corrected_data = apply_feedback_corrections(original_data, feedback_data)
        
        # Regenerar reporte con datos corregidos
        html_path = REPORTS_DIR / f"INF-{case_id}.html"
        
        # Usar el generador de reportes con los datos corregidos
        from fraud_scorer.templates.ai_report_generator import AIReportGenerator
        from fraud_scorer.models.extraction import ConsolidatedExtraction
        
        report_generator = AIReportGenerator()
        
        # Crear un objeto mock de ConsolidatedExtraction con los datos corregidos
        mock_consolidated = create_mock_consolidated_extraction(corrected_data, case_id)
        mock_ai_analysis = {
            "fraud_score": 0.3,  # Valor por defecto
            "risk_indicators": [],
            "confidence_score": 0.8,
            "recommendations": ["Revisar documentación adicional"]
        }
        
        # Generar el reporte final
        html_content = report_generator.generate_report(
            consolidated_data=mock_consolidated,
            ai_analysis=mock_ai_analysis,
            output_path=html_path
        )
        
        # Guardar feedback para mejorar el sistema
        feedback_path = TEMP_DIR / f"feedback_{case_id}.json"
        with open(feedback_path, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Feedback procesado y reporte regenerado para caso {case_id}")
        
        return {
            "success": True,
            "message": "Reporte final generado exitosamente",
            "report_url": f"/report/{case_id}"
        }
        
    except Exception as e:
        logger.error(f"Error procesando feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error procesando feedback: {str(e)}")

@app.get("/report/{case_id}", response_class=HTMLResponse)
async def get_report(case_id: str):
    """Devuelve el reporte HTML final"""
    report_path = REPORTS_DIR / f"INF-{case_id}.html"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return HTMLResponse(content=content)

@app.get("/health")
async def health_check():
    """Endpoint de salud"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/debug/routes")
async def debug_routes():
    """Endpoint de debugging - muestra todas las rutas disponibles"""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else [],
                "name": route.name if hasattr(route, "name") else "N/A"
            })
    
    return {
        "total_routes": len(routes),
        "routes": sorted(routes, key=lambda x: x["path"]),
        "replay_endpoints": [r for r in routes if "/replay" in r["path"]]
    }

# ============================================================================
# REPLAY SYSTEM WEB ROUTES - Now handled by replay router
# ============================================================================
# Las rutas de replay están ahora manejadas por el router en endpoints/replay.py

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Inicia el servidor"""
    logger.info(f"Iniciando servidor en http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)

if __name__ == "__main__":
    start_server()