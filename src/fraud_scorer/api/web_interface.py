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
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
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
from fraud_scorer.api.auth_simple import (
    authenticate as simple_authenticate,
    mark_authenticated,
    clear_authentication,
    is_authenticated,
    current_user,
    last_login,
)

# Inicializar FastAPI
app = FastAPI(
    title="Fraud Scorer Web Interface",
    description="Sistema de Análisis de Siniestros con OCR y generación de reportes",
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
FRAUD_SCORER_TEMPLATES = project_root / "src" / "fraud_scorer" / "templates"
FRAUD_SCORER_STATIC = project_root / "src" / "fraud_scorer" / "static"

# Crear directorios si no existen
for dir_path in [UPLOAD_DIR, REPORTS_DIR, TEMP_DIR, TEMPLATES_DIR, STATIC_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configurar plantillas y archivos estáticos
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Montar también los archivos estáticos de fraud_scorer si existen
if FRAUD_SCORER_STATIC.exists():
    app.mount("/fs-static", StaticFiles(directory=str(FRAUD_SCORER_STATIC)), name="fs_static")

# Configurar templates globales de fraud_scorer para feedback
fraud_scorer_templates = Jinja2Templates(directory=str(FRAUD_SCORER_TEMPLATES)) if FRAUD_SCORER_TEMPLATES.exists() else None

PUBLIC_PATHS = {
    "/login",
    "/api/login",
    "/favicon.ico",
}
PUBLIC_PREFIXES = (
    "/static",
    "/fs-static",
)


def _is_public_path(path: str) -> bool:
    """Return True when the path is allowed without authentication."""
    if path in PUBLIC_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)


@app.middleware("http")
async def simple_auth_guard(request: Request, call_next):
    if _is_public_path(request.url.path):
        return await call_next(request)
    if not is_authenticated():
        if request.method.upper() in {"GET", "HEAD"}:
            return RedirectResponse(url="/login", status_code=302)
        return JSONResponse(
            {"detail": "Credenciales requeridas. Inicie sesión desde /login."},
            status_code=401,
        )
    return await call_next(request)


# Middleware para manejar errores
@app.middleware("http")
async def error_guard(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.exception("Error no manejado en la aplicación")
        return HTMLResponse(
            content="<h1>Error interno del servidor</h1><p>Ha ocurrido un error inesperado.</p>",
            status_code=500
        )

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
# Mapa interno para tareas en ejecución (no se expone en respuestas)
active_jobs: dict[str, dict] = {}


def _build_case_summary(case_id: str, case_index: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a normalized payload with case metadata for UI consumption."""
    case_data: Dict[str, Any] = case_index or {}
    consolidated_fields = (
        (case_data.get("consolidated_data") or {}).get("consolidated_fields")
        or {}
    )

    insured_name = (
        consolidated_fields.get("nombre_asegurado")
        or case_data.get("insured_name")
        or ""
    )
    claim_number = (
        consolidated_fields.get("numero_siniestro")
        or case_data.get("claim_number")
        or ""
    )

    total_documents = case_data.get("total_documents")
    if total_documents is None:
        documents = case_data.get("documents") or []
        total_documents = len(documents)

    return {
        "case_id": case_id,
        "case_info": case_data,
        "insured_name": insured_name,
        "claim_number": claim_number,
        "has_ocr": bool(case_data.get("documents")),
        "has_classifications": bool(case_data.get("classified_types")),
        "has_extraction": bool(case_data.get("extraction_results")),
        "has_consolidation": bool(case_data.get("consolidated_data")),
        "has_fraud_analysis": bool(case_data.get("fraud_analyses")),
        "total_documents": total_documents,
        "processed_at": case_data.get("processed_at"),
    }

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
                if self.cache_manager.has_cache(file_path, case_id=None):
                    logger.info(f"Usando cache para: {file_path.name}")
                    ocr_result = self.cache_manager.get_cache(file_path, case_id)
                else:
                    # Procesar con OCR/Parser
                    ocr_result = self.document_parser.parse_document(file_path)
                    if ocr_result:
                        # Guardar directo en vista humana usando case_id
                        self.cache_manager.save_cache(file_path, ocr_result, case_id)
                
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
            
            # Generar reporte
            logger.info("Generando reporte HTML...")
            html_path = REPORTS_DIR / f"INF-{case_id}.html"
            self.report_generator.generate_report(
                consolidated_data=consolidated,
                output_path=html_path
            )
            
            # Limpieza global de shards antiguos tras finalizar
            try:
                self.cache_manager.cleanup_shards()
            except Exception as e:
                logger.warning(f"cleanup_shards falló: {e}")
            
            return {
                "success": True,
                "case_id": case_id,
                "html_path": str(html_path)
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
            return "denuncia_de_los_hechos"
        elif "carta" in filename_lower:
            return "carta_reclamacion"
        else:
            return "otro"
    
    # (Cálculo de nivel de riesgo eliminado)

# Instancia global del procesador
processor = FraudScorerProcessor()

async def load_consolidated_data(case_id: str) -> Optional[dict]:
    """Carga los datos consolidados de un caso"""
    try:
        pipeline_cache_dir = project_root / "data" / "temp" / "pipeline_cache"
        logger.info(f"Buscando datos consolidados para caso {case_id} en {pipeline_cache_dir}")
        
        # Buscar en TODOS los archivos consolidados por contenido
        logger.info("Buscando archivos consolidados...")
        # Buscar archivos que terminen en _CONSOLIDADO.json
        consolidated_files = list(pipeline_cache_dir.glob("*_CONSOLIDADO.json"))
        logger.info(f"Archivos consolidados encontrados: {[f.name for f in consolidated_files]}")
        
        for file_path in consolidated_files:
            try:
                logger.info(f"Revisando archivo: {file_path.name}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Buscar el case_id en el JSON
                if data.get("case_id") == case_id:
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
        "numero_siniestro": fields.get("numero_siniestro") or "",
        "nombre_asegurado": fields.get("nombre_asegurado") or "",
        "monto_reclamacion": fields.get("monto_reclamacion") or "",
        "numero_poliza": fields.get("numero_poliza") or "",
        "vigencia": fields.get("vigencia") or "",
        "domicilio_poliza": fields.get("domicilio_poliza") or "",
        "tipo_siniestro": fields.get("tipo_siniestro") or "",
        "fecha_ocurrencia": fields.get("fecha_ocurrencia") or "",
        "fecha_reclamacion": fields.get("fecha_reclamacion") or "",
        "lugar_hechos": fields.get("lugar_hechos") or "",
        "bien_reclamado": fields.get("bien_reclamado") or "",
        "ajuste": fields.get("ajuste") or "",
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

@app.get("/login", response_class=HTMLResponse)
async def get_login_page(request: Request):
    if is_authenticated():
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/api/login")
async def login(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"detail": "Formato de datos inválido."},
            status_code=400,
        )

    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not simple_authenticate(username, password):
        return JSONResponse(
            {"detail": "Credenciales incorrectas. Intente nuevamente."},
            status_code=401,
        )

    mark_authenticated(username)
    return JSONResponse(
        {
            "success": True,
            "message": "Inicio de sesión exitoso. Redirigiendo...",
            "redirect": "/",
        }
    )


@app.post("/api/logout")
async def logout():
    if is_authenticated():
        clear_authentication()
    return JSONResponse({"success": True, "message": "Sesión finalizada."})


@app.get("/", response_class=HTMLResponse)
async def get_main_portal(request: Request):
    """
    Sirve la nueva página principal (el portal) que permite al usuario
    elegir entre 'Upload' y 'Replay'.
    """
    last_login_at = last_login()
    context = {
        "request": request,
        "user": current_user(),
        "last_login_iso": last_login_at.isoformat() if last_login_at else None,
        "last_login_display": last_login_at.strftime("%d/%m/%Y %H:%M:%S") if last_login_at else None,
    }
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/upload", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    """
    Sirve la página original de carga de documentos.
    Esta es la nueva ruta para la funcionalidad de 'run_report'.
    """
    return templates.TemplateResponse("upload.html", {"request": request, "user": current_user()})

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
    
    # Habilitar checkpoint 1.4.1 solo para este proceso
    os.environ["ENABLE_CLASSIFICATION_REVIEW"] = "true"

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
        
        # Importar dinámicamente usando importlib para evitar problemas de resolución
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_report", scripts_path / "run_report.py")
        run_report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_report)
        FraudAnalysisSystemV2 = run_report.FraudAnalysisSystemV2
        
        # Los archivos ya están en el directorio temporal correcto
        # No necesitamos copiarlos, solo usar la ruta del directorio padre
        temp_case_dir = files[0].parent if files else TEMP_DIR / process_id
        
        processing_status[process_id]["message"] = "Procesando documentos con IA..."
        processing_status[process_id]["progress"] = 30
        
        # Inicializar el sistema v2
        system = FraudAnalysisSystemV2()
        
        # Limpiar marcadores de revisiones previas para evitar confusión
        try:
            for marker in pipeline_cache_dir.glob("*.awaiting_review"):
                marker.unlink(missing_ok=True)  # type: ignore[arg-type]
            for marker in pipeline_cache_dir.glob("*.resume"):
                marker.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

        # Procesar el caso usando el sistema completo, con pausa 1.4.1
        case_title = f"Caso_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Lanzar como tarea para poder monitorear marcador de revisión
        task = asyncio.create_task(system.process_case(
            folder_path=temp_case_dir,
            output_path=REPORTS_DIR,
            case_title=case_title
        ))
        active_jobs[process_id] = {"task": task, "system": system}

        # Monitorear pausa 1.4.1 mediante marcador de archivo
        base = os.getenv("FS_DATA_DIR", "data")
        pipeline_cache_dir = Path(base) / "temp" / "pipeline_cache"
        pipeline_cache_dir.mkdir(parents=True, exist_ok=True)
        awaiting_set = False
        detected_case_id = None

        while not task.done():
            # Descubrir marcador *.awaiting_review
            try:
                markers = list(pipeline_cache_dir.glob("*.awaiting_review"))
                if markers and not awaiting_set:
                    # Tomar el primero (entorno de un solo proceso)
                    m = markers[0]
                    detected_case_id = m.stem  # 'CASEID' si el archivo es 'CASEID.awaiting_review'
                    processing_status[process_id] = {
                        **processing_status[process_id],
                        "status": "awaiting_review",
                        "message": "Esperando revisión de clasificación",
                        "progress": 35,
                        "case_id": detected_case_id,
                    }
                    awaiting_set = True
            except Exception:
                pass

            await asyncio.sleep(0.5)

        # Obtener resultado final si la tarea terminó sin excepción
        result = None
        try:
            result = task.result()
        except asyncio.CancelledError:
            processing_status[process_id] = {
                "status": "cancelled",
                "message": "Proceso cancelado por el usuario",
                "progress": 0,
            }
            return
        except Exception as e:
            raise e

        processing_status[process_id]["message"] = "Finalizando..."
        processing_status[process_id]["progress"] = 90

        # Extraer información del resultado
        case_id = (result or {}).get("case_id", detected_case_id or "UNKNOWN")
        processing_status[process_id] = {
            "status": "completed",
            "message": "Procesamiento completado - Listo para validación",
            "progress": 100,
            "case_id": case_id,
            "report_url": f"/report/{case_id}",
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

# ===============================
# Endpoints mínimos de checkpoint
# ===============================
from fraud_scorer.processors.document_classifier import DocumentType

@app.get("/api/case/{case_id}/classifications")
async def get_classifications(case_id: str):
    try:
        cm = OCRCacheManager()
        case = cm.get_case_index(case_id, auto_reconstruct=True)

        if not case:
            logger.warning(f"Caso {case_id} no encontrado en índice de cache")

            # Verificar si existe en la BD
            from fraud_scorer.storage.cases import get_conn
            try:
                with get_conn() as conn:
                    case_row = conn.execute(
                        "SELECT case_id, name FROM cases WHERE case_id = ?",
                        (case_id,)
                    ).fetchone()

                    if not case_row:
                        logger.error(f"Caso {case_id} no existe en BD")
                        raise HTTPException(status_code=404, detail=f"Caso {case_id} no encontrado")
                    else:
                        logger.info(f"Caso {case_id} existe en BD pero no tiene índice de cache")
                        # Crear índice básico
                        case = {
                            "case_id": case_id,
                            "case_title": case_row["name"],
                            "classified_types": [],
                            "status": "no_cache_index"
                        }
            except Exception as db_e:
                logger.error(f"Error consultando BD para caso {case_id}: {db_e}")
                raise HTTPException(status_code=404, detail=f"Caso {case_id} no encontrado")
    except Exception as e:
        logger.error(f"Error obteniendo clasificaciones para {case_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

    classifications = case.get("classified_types", []) or []

    # Buscar clasificaciones manuales previas
    manual_classifications = cm.get_manual_classifications(case_id)
    ai_classifications = case.get("ai_classifications", {})
    ai_prediction_details = case.get("ai_prediction_details", {})
    ai_predictions_history = case.get("ai_predictions_history", [])

    # Enriquecer cada documento con la información manual y de IA
    for doc in classifications:
        filename = doc.get("filename")
        if not filename:
            continue

        detail = ai_prediction_details.get(filename, {}) if isinstance(ai_prediction_details, dict) else {}

        if "ai_document_type" not in doc and isinstance(ai_classifications, dict):
            baseline_type = ai_classifications.get(filename)
            if baseline_type:
                doc["ai_document_type"] = baseline_type

        if detail:
            doc.setdefault("ai_document_type", detail.get("document_type"))
            doc.setdefault("ai_confidence", detail.get("confidence"))
            doc.setdefault("ai_reasons", detail.get("reasons"))

        if manual_classifications and filename in manual_classifications:
            doc["document_type"] = manual_classifications[filename]
            doc["has_manual_classification"] = True

    # Construir catálogo de tipos agrupados por categoría
    doc_types = _get_grouped_document_types()

    return {
        "case_id": case_id,
        "classifications": classifications,
        "document_types": doc_types,
        "has_previous_classifications": bool(manual_classifications),
        "manual_classifications": manual_classifications,
        "ai_classifications": ai_classifications,
        "ai_prediction_details": ai_prediction_details,
        "ai_predictions_history": ai_predictions_history,
    }

@lru_cache(maxsize=1)
def _get_grouped_document_types():
    """Retorna los tipos de documento agrupados por categorías"""
    return {
        "categories": [
            {
                "label": "Cartas de Reclamación",
                "options": [
                    {"value": DocumentType.CARTA_RECLAMACION_ASEGURADORA.value,
                     "label": "Carta De Reclamación Formal A La Aseguradora"},
                    {"value": DocumentType.CARTA_RECLAMACION_TRANSPORTISTA.value,
                     "label": "Carta De Reclamación Formal Al Transportista"},
                ]
            },
            {
                "label": "Documentos de Transporte",
                "options": [
                    {"value": DocumentType.GUIAS_Y_FACTURAS.value,
                     "label": "Guías Y Facturas"},
                    {"value": DocumentType.GUIAS_Y_FACTURAS_CONSOLIDADAS.value,
                     "label": "Guías Y Facturas Consolidadas"},
                    {"value": DocumentType.SALIDA_DE_ALMACEN.value,
                     "label": "Salida De Almacén"},
                    {"value": DocumentType.CFDI_CARTA_PORTE.value,
                     "label": "Cfdi Carta Porte"},
                    {"value": DocumentType.CARTA_PORTE_SIMPLE.value,
                     "label": "Carta Porte Simple"},
                ]
            },
            {
                "label": "Documentos Vehiculares",
                "options": [
                    {"value": DocumentType.TARJETA_CIRCULACION.value,
                     "label": "Tarjeta De Circulación Vehículo"},
                    {"value": DocumentType.LICENCIA_OPERADOR.value,
                     "label": "Licencia Del Operador"},
                    {"value": DocumentType.FICHA_TECNICA_VEHICULO.value,
                     "label": "Ficha Técnica De Vehículo"},
                ]
            },
            {
                "label": "Documentos de Siniestro",
                "options": [
                    {"value": DocumentType.AVISO_SINIESTRO_TRANSPORTISTA.value,
                     "label": "Aviso De Siniestro Transportista"},
                    {"value": DocumentType.REPORTE_GPS.value,
                     "label": "Reporte GPS"},
                    {"value": DocumentType.NARRACION_DE_HECHOS.value,
                     "label": "Narración De Hechos"},
                    {"value": DocumentType.DECLARACION_UNIVERSAL_ACCIDENTE.value,
                     "label": "Declaración Universal De Accidente"},
                ]
            },
            {
                "label": "Documentos Legales",
                "options": [
                    {"value": DocumentType.CARPETA_INVESTIGACION.value,
                     "label": "Carpeta De Investigación"},
                    {"value": DocumentType.ACREDITACION_PROPIEDAD.value,
                     "label": "Acreditación De Propiedad Y Representación"},
                    {"value": DocumentType.DENUNCIA_DE_LOS_HECHOS.value,
                     "label": "Denuncia De Los Hechos"},
                    {"value": DocumentType.OFICIO_DENUNCIA.value,
                     "label": "Oficio De Denuncia"},
                ]
            },
            {
                "label": "Documentos de Seguro",
                "options": [
                    {"value": DocumentType.POLIZA_ASEGURADORA.value,
                     "label": "Póliza De La Aseguradora"},
                    {"value": DocumentType.INFORME_PRELIMINAR_AJUSTADOR.value,
                     "label": "Informe Preliminar Del Ajustador"},
                    {"value": DocumentType.INFORME_FINAL_AJUSTADOR.value,
                     "label": "Informe Final Del Ajustador"},
                    {"value": DocumentType.EXPEDIENTE_COBRANZA.value,
                     "label": "Expediente De Cobranza"},
                    {"value": DocumentType.CHECKLIST_ANTIFRAUDE.value,
                     "label": "Checklist Antifraude"},
                    {"value": DocumentType.DETERMINACION_DE_PERDIDA.value,
                     "label": "Determinación De Pérdida"},
                ]
            },
            {
                "label": "Facturas y Documentos Comerciales",
                "options": [
                    {"value": DocumentType.FACTURA_COMERCIAL_CFDI.value,
                     "label": "Factura Comercial CFDI"},
                    {"value": DocumentType.FACTURAS_COMERCIALES_INTERNACIONALES.value,
                     "label": "Facturas Comerciales Internacionales"},
                    {"value": DocumentType.PEDIDO_POR_CORREO.value,
                     "label": "Pedido Por Correo"},
                    {"value": DocumentType.REPORTE_COSTOS_RENDIMIENTOS.value,
                     "label": "Reporte De Costos Y Rendimientos"},
                ]
            },
            {
                "label": "Documentos Generales",
                "options": [
                    {"value": DocumentType.IDENTIFICACION_OFICIAL.value,
                     "label": "Identificación Oficial"},
                    {"value": DocumentType.NOTAS_DE_REPARACION.value,
                     "label": "Notas De Reparación"},
                    {"value": DocumentType.DICTAMEN_TECNICO.value,
                     "label": "Dictamen Técnico"},
                    {"value": DocumentType.COMPROBANTE_DOMICILIO.value,
                     "label": "Comprobante De Domicilio"},
                    {"value": DocumentType.CONSTANCIA_IMSS_OPERADOR.value,
                     "label": "Constancia IMSS Del Operador"},
                ]
            },
            {
                "label": "Comercio Exterior Y Aduanas",
                "options": [
                    {"value": DocumentType.PEDIMENTO_IMPORTACION.value,
                     "label": "Pedimento De Importación"},
                    {"value": DocumentType.OFICIO_DESADUANADO.value,
                     "label": "Oficio De Desaduanado"},
                    {"value": DocumentType.CONOCIMIENTO_EMBARQUE.value,
                     "label": "Conocimiento De Embarque"},
                ]
            },
            {
                "label": "Contratos Y Protocolos Operativos",
                "options": [
                    {"value": DocumentType.CONTRATO_PRESTACION_SERVICIO_TRANSPORTISTA.value,
                     "label": "Contrato De Prestación De Servicio Al Transportista"},
                    {"value": DocumentType.PROTOCOLO_ACCION_REACCION.value,
                     "label": "Protocolo De Acción Y Reacción"},
                ]
            },
            {
                "label": "Soporte Operativo",
                "options": [
                    {"value": DocumentType.CARTA_ACLARATORIA_PEAJE.value,
                     "label": "Carta Aclaratoria De Peaje"},
                ]
            },
            {
                "label": "Otros",
                "options": [
                    {"value": DocumentType.OTRO.value,
                     "label": "Otro"},
                ]
            }
        ]
    }


def _audit_document_type_groups():
    grouped = _get_grouped_document_types()
    enum_values = {dt.value for dt in DocumentType}
    category_by_type = {}
    duplicates: defaultdict[str, List[str]] = defaultdict(list)

    for category in grouped.get("categories", []):
        cat_label = category.get("label") or ""
        for option in category.get("options", []):
            value = option.get("value")
            if not value:
                continue
            if value in category_by_type:
                duplicates[value].append(cat_label)
            else:
                category_by_type[value] = cat_label

    grouped_values = set(category_by_type.keys())
    missing = sorted(enum_values - grouped_values)
    unknown = sorted(grouped_values - enum_values)

    duplicate_result = {
        value: sorted({category_by_type[value], *labels})
        for value, labels in duplicates.items()
    }

    return {
        "missing_document_types": missing,
        "unknown_group_entries": unknown,
        "duplicate_assignments": duplicate_result,
        "total_document_types": len(enum_values),
        "categorized_document_types": len(grouped_values),
    }


_DOCUMENT_TYPE_AUDIT_SNAPSHOT = _audit_document_type_groups()

if _DOCUMENT_TYPE_AUDIT_SNAPSHOT["missing_document_types"]:
    logger.warning(
        "DocumentType sin categoría asignada: %s",
        ", ".join(_DOCUMENT_TYPE_AUDIT_SNAPSHOT["missing_document_types"]),
    )

if _DOCUMENT_TYPE_AUDIT_SNAPSHOT["unknown_group_entries"]:
    logger.warning(
        "Valores en categorías sin DocumentType asociado: %s",
        ", ".join(_DOCUMENT_TYPE_AUDIT_SNAPSHOT["unknown_group_entries"]),
    )

if _DOCUMENT_TYPE_AUDIT_SNAPSHOT["duplicate_assignments"]:
    logger.warning(
        "DocumentType con múltiples categorías: %s",
        _DOCUMENT_TYPE_AUDIT_SNAPSHOT["duplicate_assignments"],
    )

from fastapi import Body

@app.post("/api/case/{case_id}/update-classifications")
async def update_classifications(case_id: str, payload: dict = Body(...)):
    try:
        cm = OCRCacheManager()
        case = cm.get_case_index(case_id, auto_reconstruct=True)

        if not case:
            logger.warning(f"Intentando actualizar clasificaciones para caso {case_id} que no tiene índice")

            # Verificar si existe en BD
            from fraud_scorer.storage.cases import get_conn
            try:
                with get_conn() as conn:
                    case_row = conn.execute(
                        "SELECT case_id, name FROM cases WHERE case_id = ?",
                        (case_id,)
                    ).fetchone()

                    if not case_row:
                        logger.error(f"Caso {case_id} no existe en BD para actualizar clasificaciones")
                        raise HTTPException(status_code=404, detail=f"Caso {case_id} no encontrado")
                    else:
                        logger.info(f"Creando índice básico para {case_id}")
                        # Crear índice básico
                        case = {
                            "case_id": case_id,
                            "case_title": case_row["name"],
                            "classified_types": [],
                            "status": "reconstructed_for_update"
                        }
            except Exception as db_e:
                logger.error(f"Error consultando BD para actualizar {case_id}: {db_e}")
                raise HTTPException(status_code=404, detail=f"Caso {case_id} no encontrado")
    except Exception as e:
        logger.error(f"Error actualizando clasificaciones para {case_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

    updates = (payload or {}).get("classifications", {})
    ai_originals = (payload or {}).get("ai_classifications", {})

    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="Payload inválido")

    valid_values = {dt.value for dt in DocumentType}
    current = case.get("classified_types", []) or []
    by_name = {item.get("filename"): item for item in current if item.get("filename")}
    previous_types = {item.get("filename"): item.get("document_type") for item in current if item.get("filename")}

    manual_existing = case.get("manual_classifications") or {}
    final_manual = dict(manual_existing)

    ai_baseline = dict(case.get("ai_classifications") or {})
    if not isinstance(ai_originals, dict):
        ai_originals = {}

    if ai_originals:
        ai_baseline = dict(ai_originals)
        logger.info(
            "Actualizando baseline de IA para caso %s: %d clasificaciones",
            case_id,
            len(ai_baseline),
        )
        case["ai_classifications"] = dict(ai_baseline)
        details_payload = (payload or {}).get("ai_prediction_details")
        if isinstance(details_payload, dict):
            case["ai_prediction_details"] = dict(details_payload)

    for filename, new_type in updates.items():
        if new_type not in valid_values:
            raise HTTPException(status_code=400, detail=f"Tipo inválido para {filename}: {new_type}")

        item = by_name.get(filename)
        if item is None:
            item = {"filename": filename, "document_type": new_type}
            current.append(item)
            by_name[filename] = item
        else:
            item["document_type"] = new_type

        baseline_type = ai_baseline.get(filename)
        if baseline_type is None and ai_originals:
            baseline_type = ai_originals.get(filename)
        if baseline_type is None:
            baseline_type = previous_types.get(filename)

        if baseline_type is None:
            baseline_type = new_type

        item["manually_reviewed"] = new_type != baseline_type

        if new_type != baseline_type:
            final_manual[filename] = new_type
        else:
            final_manual.pop(filename, None)

    case["classified_types"] = current
    if final_manual:
        case["manual_classifications"] = final_manual
    else:
        case.pop("manual_classifications", None)

    try:
        cm.save_case_index(case_id, case)
        cm.save_manual_classifications(
            case_id,
            final_manual,
            ai_classifications=ai_baseline if ai_baseline else None,
            replace=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo guardar: {e}")

    return {"success": True}


@app.get("/api/system/document-type-audit")
async def get_document_type_audit():
    snapshot = _audit_document_type_groups()
    grouped = _get_grouped_document_types()
    return {
        "audit": snapshot,
        "document_types": sorted(dt.value for dt in DocumentType),
        "categories": grouped["categories"],
    }


@app.post("/api/case/{case_id}/continue-processing")
async def continue_processing(case_id: str):
    # Señal de reanudación para run_report
    base = os.getenv("FS_DATA_DIR", "data")
    pc_dir = Path(base) / "temp" / "pipeline_cache"
    pc_dir.mkdir(parents=True, exist_ok=True)
    resume_marker = pc_dir / f"{case_id}.resume"
    try:
        with open(resume_marker, "w", encoding="utf-8") as f:
            f.write(json.dumps({"case_id": case_id, "ts": datetime.now().isoformat()}))
    except Exception:
        try:
            resume_marker.touch(exist_ok=True)
        except Exception:
            raise HTTPException(status_code=500, detail="No se pudo crear señal de continuación")
    # Actualizar estado a 'processing' para que la UI salga de la revisión
    try:
        for pid, st in list(processing_status.items()):
            if st.get("case_id") == case_id and st.get("status") == "awaiting_review":
                st["status"] = "processing"
                st["message"] = "Reanudando procesamiento tras revisión..."
                st["progress"] = max(40, st.get("progress", 35))
                processing_status[pid] = st
                break
    except Exception:
        pass
    return {"success": True}

@app.post("/api/case/{case_id}/check-existing")
async def check_existing_case(case_id: str = None, payload: dict = Body(...)):
    """Detecta si los archivos subidos pertenecen a un caso ya procesado.
    Estrategia:
      1) Si llegan hashes (sha256) de hasta N archivos (cliente), buscar en DB por coincidencias y elegir case_id con más matches.
      2) Si no hay hashes o no hay matches, caer al heurístico por nombre (clasificaciones o deducir desde JSONs del índice).
    """
    # 1) Extraer archivos y hashes del payload
    files: list[str] = []
    hashes: list[str] = []
    if isinstance(payload, dict):
        raw_files = payload.get("files")
        if isinstance(raw_files, list):
            files = [str(item) for item in raw_files if item]
        raw_hashes = payload.get("hashes")
        if isinstance(raw_hashes, list):
            hashes = [str(h).lower() for h in raw_hashes if h]
    elif isinstance(payload, list):
        files = [str(item) for item in payload if item]

    filenames = {Path(name).name for name in files}

    # 2) Intento por hash (preferente)
    if hashes:
        try:
            from fraud_scorer.storage.db import get_conn
            with get_conn() as conn:
                qmarks = ",".join(["?"] * len(hashes))
                sql = f"""
                    SELECT case_id, COUNT(*) as cnt
                    FROM documents
                    WHERE file_hash IN ({qmarks})
                    GROUP BY case_id
                    ORDER BY cnt DESC
                    LIMIT 1
                """
                row = conn.execute(sql, tuple(hashes)).fetchone()
                if row and row["case_id"]:
                    candidate_id = row["case_id"]
                    cm = OCRCacheManager()
                    case_index = cm.get_case_index(candidate_id, auto_reconstruct=True)
                    summary = _build_case_summary(candidate_id, case_index)
                    summary["existing_case"] = True
                    return summary
        except Exception as e:
            logger.warning(f"check-existing por hash falló: {e}")

    # 3) Fallback por nombre (compat)
    if not filenames:
        return {"existing_case": False}

    cm = OCRCacheManager()
    cases = cm.list_cases_from_cache() or []

    for case_info in cases:
        candidate_id = case_info.get('case_id') or case_info.get('case_id')
        if not candidate_id:
            continue
        case_index = cm.get_case_index(candidate_id)
        if not case_index:
            continue
        doc_names = set()
        # Preferir nombres originales desde clasificaciones previas
        try:
            for item in (case_index.get('classified_types') or []):
                fn = item.get('filename')
                if fn:
                    doc_names.add(Path(fn).name)
        except Exception:
            pass
        # Fallback: intentar deducir desde rutas de JSON en 'documents'
        if not doc_names:
            try:
                for doc in (case_index.get('documents') or []):
                    name = Path(str(doc)).name
                    if name.startswith('ocr_results_for_') and name.endswith('.json'):
                        original = name[len('ocr_results_for_'):-len('.json')]
                        if original:
                            doc_names.add(original)
                    else:
                        doc_names.add(name)
            except Exception:
                pass
        if not doc_names.intersection(filenames):
            continue

        summary = _build_case_summary(candidate_id, case_index)
        summary["existing_case"] = True
        return summary

    return {"existing_case": False}


@app.get("/api/case/{case_id}/summary")
async def get_case_summary(case_id: str):
    """Return consolidated metadata for a stored case."""
    cm = OCRCacheManager()
    case_index = cm.get_case_index(case_id, auto_reconstruct=True)

    if not case_index:
        raise HTTPException(status_code=404, detail="Caso no encontrado")

    summary = _build_case_summary(case_id, case_index)
    return {"success": True, "data": summary}


@app.post("/api/case/{case_id}/reprocess")
async def reprocess_case(
    case_id: str,
    background_tasks: BackgroundTasks,
    options_payload: dict = Body(...)
):
    """
    Reprocesa fases selectivas de un caso existente.

    Options:
        - reprocess_ocr: bool - Reprocesar OCR
        - reprocess_classification: bool - Reclasificar documentos
        - reprocess_policy_detection: bool - Redetectar tipo de póliza
        - reprocess_extraction: bool - Reextraer campos (Fase 2)
        - reprocess_consolidation: bool - Reconsolidar datos (Fase 3)
        - reprocess_fraud: bool - Reanalizar fraude (Fase 3.5)
    """
    cm = OCRCacheManager()
    case_index = cm.get_case_index(case_id, auto_reconstruct=True)

    if not case_index:
        raise HTTPException(status_code=404, detail="Caso no encontrado")

    raw_options = options_payload or {}
    if isinstance(raw_options.get("options"), dict):
        raw_options = raw_options["options"]

    option_keys = [
        "reprocess_ocr",
        "reprocess_classification",
        "reprocess_policy_detection",
        "reprocess_extraction",
        "reprocess_consolidation",
        "reprocess_fraud",
    ]
    options = {key: bool(raw_options.get(key, False)) for key in option_keys}

    # Generar nuevo process_id para monitoreo
    process_id = str(uuid.uuid4())

    # Preparar archivos desde el caso existente
    case_folder = cm.get_case_folder_path(case_id, case_index)
    if not case_folder.exists():
        docs = case_index.get('documents') or []
        if docs:
            existing_docs = [Path(p) for p in docs if Path(p).exists()]
            if existing_docs:
                case_folder = existing_docs[0].parent
        if not case_folder.exists():
            try:
                insured = case_index.get('insured_name') or ""
                claim = case_index.get('claim_number') or case_id
                title = case_index.get('case_title', case_id)
                ref_name = cm._sanitize_filename(insured) if insured else cm._sanitize_filename(title)
                ref_claim = cm._sanitize_filename(claim) if claim else case_id
                cm.reorganize_cache_for_case(case_id, ref_name, ref_claim)
                case_folder = cm.get_case_folder_path(case_id, case_index)
            except Exception as exc:
                logger.error(f"No se pudo reconstruir carpeta de caso {case_id}: {exc}")

    if not case_folder.exists():
        raise HTTPException(status_code=404, detail="Carpeta del caso no encontrada")

    # Inicializar estado
    processing_status[process_id] = {
        "status": "processing",
        "message": "Preparando reprocesamiento selectivo...",
        "progress": 0,
        "started_at": datetime.now().isoformat(),
        "case_id": case_id,
        "reprocess_options": options
    }

    # Procesar en background con opciones selectivas
    background_tasks.add_task(
        reprocess_case_background,
        process_id,
        case_id,
        options
    )

    return {
        "process_id": process_id,
        "message": "Reprocesamiento iniciado",
        "status_url": f"/status/{process_id}"
    }

async def reprocess_case_background(process_id: str, case_id: str, options: dict):
    """
    Ejecuta el reprocesamiento selectivo en background.
    """
    try:
        processing_status[process_id]["message"] = "Iniciando reprocesamiento selectivo..."
        processing_status[process_id]["progress"] = 10

        # Importar sistema de análisis
        scripts_path = project_root / "scripts"
        if str(scripts_path) not in sys.path:
            sys.path.insert(0, str(scripts_path))

        import importlib.util
        spec = importlib.util.spec_from_file_location("run_report", scripts_path / "run_report.py")
        run_report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_report)
        FraudAnalysisSystemV2 = run_report.FraudAnalysisSystemV2

        # Crear instancia del sistema
        system = FraudAnalysisSystemV2()

        # Configurar opciones de reprocesamiento
        system.reprocess_options = options or {}

        # Asegurar que el checkpoint de clasificación esté habilitado
        os.environ["ENABLE_CLASSIFICATION_REVIEW"] = "true"

        # Obtener carpeta del caso
        cm = OCRCacheManager()
        case_index = cm.get_case_index(case_id)
        case_folder = cm.get_case_folder_path(case_id, case_index)

        processing_status[process_id]["message"] = "Ejecutando fases seleccionadas..."
        processing_status[process_id]["progress"] = 30

        base = os.getenv("FS_DATA_DIR", "data")
        pipeline_cache_dir = Path(base) / "temp" / "pipeline_cache"
        pipeline_cache_dir.mkdir(parents=True, exist_ok=True)

        # Limpiar marcadores obsoletos únicamente para este caso
        for suffix in ("awaiting_review", "resume"):
            marker = pipeline_cache_dir / f"{case_id}.{suffix}"
            try:
                marker.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

        # Ejecutar reprocesamiento
        task = asyncio.create_task(system.process_case(
            folder_path=case_folder,
            output_path=REPORTS_DIR,
            case_title=f"Reproceso_{case_id}",
            reprocess_mode=True,
            reprocess_options=options,
            existing_case_id=case_id
        ))

        active_jobs[process_id] = {"task": task, "system": system}

        awaiting_reported = False

        # Monitorear marcador de checkpoint igual que en el flujo principal
        while not task.done():
            if not awaiting_reported:
                marker = pipeline_cache_dir / f"{case_id}.awaiting_review"
                if marker.exists():
                    current_status = processing_status.get(process_id, {})
                    processing_status[process_id] = {
                        **current_status,
                        "status": "awaiting_review",
                        "message": "Esperando revisión de clasificación",
                        "progress": max(35, current_status.get("progress", 30)),
                        "case_id": case_id,
                    }
                    awaiting_reported = True
            await asyncio.sleep(0.5)

        # Esperar resultado
        result = await task

        processing_status[process_id] = {
            "status": "completed",
            "message": "Reprocesamiento completado exitosamente",
            "progress": 100,
            "case_id": case_id,
            "report_url": f"/report/{case_id}",
            "completed_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error en reprocesamiento {process_id}: {e}", exc_info=True)
        processing_status[process_id] = {
            "status": "error",
            "message": f"Error en reprocesamiento: {str(e)}",
            "progress": 0
        }

@app.post("/cancel/{process_id}")
async def cancel_process_api(process_id: str):
    job = active_jobs.get(process_id)
    if not job:
        return {"success": False, "message": "Proceso no encontrado o ya finalizado"}
    try:
        task = job.get("task")
        if task and not task.done():
            task.cancel()
        processing_status[process_id] = {
            "status": "cancelled",
            "message": "Proceso cancelado por el usuario",
            "progress": 0,
        }
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/status/stream/{case_id}")
async def status_stream(case_id: str):
    """Endpoint SSE para streaming de eventos de progreso en tiempo real"""
    async def generate():
        base = os.getenv("FS_DATA_DIR", "data")
        path = os.path.join(base, "temp", "pipeline_cache", f"{case_id}.status.jsonl")
        offset = 0
        
        while True:
            try:
                # Leer nuevas líneas del archivo JSONL
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        f.seek(offset)
                        chunk = f.read()
                        offset = f.tell()
                    
                    if chunk:
                        # Procesar cada línea nueva
                        for line in chunk.splitlines():
                            if line.strip():
                                yield f"data: {line.decode('utf-8')}\n\n"
                
                # Esperar un poco antes de verificar nuevos eventos
                await asyncio.sleep(0.5)
                
            except FileNotFoundError:
                # El archivo aún no existe, esperar
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error en SSE stream: {e}")
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                break
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Para nginx
        }
    )

@app.get("/feedback/{case_id}", response_class=HTMLResponse)
async def get_feedback_page(case_id: str, request: Request):
    """Sirve la página de feedback para validación de datos"""
    # Cargar datos consolidados del caso
    consolidated_data = await load_consolidated_data(case_id)
    
    if not consolidated_data:
        raise HTTPException(status_code=404, detail="Datos del caso no encontrados")
    
    # Log para debug
    logger.info(f"Datos consolidados para renderizar: {consolidated_data}")
    
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
        # Generar el reporte final
        html_content = report_generator.generate_report(
            consolidated_data=mock_consolidated,
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
    """Devuelve el reporte HTML final resolviendo nomenclatura v2 dinámicamente.

    Intenta primero la nomenclatura v2: <ASEGURADO>_<SINIESTRO>_INFORME.html
    Si no existe, cae a la nomenclatura legacy: INF-<CASE_ID>.html
    """
    # Intentar con nomenclatura v2 usando datos del índice del caso
    try:
        cm = OCRCacheManager()
        case = cm.get_case_index(case_id)
        if case:
            insured = case.get("insured_name") or ""
            claim = case.get("claim_number") or ""
            if insured and claim:
                s_insured = cm._sanitize_filename(insured)  # reuse internal sanitizer
                s_claim = cm._sanitize_filename(claim)
                v2_path = REPORTS_DIR / f"{s_insured}_{s_claim}_INFORME.html"
                if v2_path.exists():
                    with open(v2_path, "r", encoding="utf-8") as f:
                        return HTMLResponse(content=f.read())
    except Exception:
        pass

    # Fallback legacy por case_id
    legacy_path = REPORTS_DIR / f"INF-{case_id}.html"
    if legacy_path.exists():
        with open(legacy_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    raise HTTPException(status_code=404, detail="Reporte no encontrado")

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
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
