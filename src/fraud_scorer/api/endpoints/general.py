from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import os
from datetime import datetime
from pathlib import Path

router = APIRouter(tags=["General"])

# NOTA: UPLOAD_DIR se define aquí o se importa de un archivo de configuración central.
UPLOAD_DIR = Path("data/raw")

@router.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Sistema de Análisis de Siniestros - API v2.0", # Actualizado a v2.0
        "status": "active",
        "endpoints": {
            "health": "/health",
            "docs": "/docs"
        }
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        upload_exists = UPLOAD_DIR.exists()
        azure_configured = bool(os.getenv('AZURE_ENDPOINT') and os.getenv('AZURE_OCR_KEY'))
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "api": "running",
                "upload_directory": "available" if upload_exists else "missing",
                "azure_ocr": "configured" if azure_configured else "not_configured"
            },
            "version": "2.0.0" # Actualizado a v2.0
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )