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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Página principal con formulario de carga"""
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
    """Procesa documentos en background"""
    try:
        # Actualizar estado
        processing_status[process_id]["message"] = "Ejecutando OCR..."
        processing_status[process_id]["progress"] = 25
        
        # Procesar
        case_title = f"Caso_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = await processor.process_case(files, case_title)
        
        if result["success"]:
            processing_status[process_id] = {
                "status": "completed",
                "message": "Procesamiento completado",
                "progress": 100,
                "case_id": result["case_id"],
                "report_url": f"/report/{result['case_id']}",
                "fraud_score": result["fraud_score"],
                "risk_level": result["risk_level"],
                "completed_at": datetime.now().isoformat()
            }
        else:
            processing_status[process_id] = {
                "status": "error",
                "message": f"Error: {result['error']}",
                "progress": 0
            }
            
    except Exception as e:
        logger.error(f"Error en proceso {process_id}: {e}")
        processing_status[process_id] = {
            "status": "error",
            "message": str(e),
            "progress": 0
        }
    finally:
        # Limpiar archivos temporales
        try:
            shutil.rmtree(TEMP_DIR / process_id)
        except:
            pass

@app.get("/status/{process_id}")
async def get_status(process_id: str):
    """Obtiene el estado del procesamiento"""
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Proceso no encontrado")
    
    return processing_status[process_id]

@app.get("/report/{case_id}", response_class=HTMLResponse)
async def get_report(case_id: str):
    """Devuelve el reporte HTML"""
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

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Inicia el servidor"""
    logger.info(f"Iniciando servidor en http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)

if __name__ == "__main__":
    start_server()