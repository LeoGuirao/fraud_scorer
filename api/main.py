"""
FastAPI main application con integración OCR
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any
import uuid
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear directorio para uploads si no existe
UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Sistema de Análisis de Siniestros",
    description="API para análisis automático de documentos de siniestros con OCR y detección de fraude",
    version="1.0.0"
)

# CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501", "http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Sistema de Análisis de Siniestros - API v1.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Verificar que el directorio de uploads existe
        upload_exists = UPLOAD_DIR.exists()
        
        # Verificar credenciales de Azure (opcional)
        azure_configured = bool(os.getenv('AZURE_ENDPOINT') and os.getenv('AZURE_OCR_KEY'))
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "api": "running",
                "upload_directory": "available" if upload_exists else "missing",
                "azure_ocr": "configured" if azure_configured else "not_configured"
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/upload")
async def upload_and_analyze_document(file: UploadFile = File(...)):
    """
    Endpoint principal: Sube un documento y ejecuta OCR
    """
    try:
        # 1. Validar archivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo requerido")
        
        # Validar tipo de archivo
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.txt'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Tipo de archivo no soportado. Permitidos: {', '.join(allowed_extensions)}"
            )
        
        # 2. Generar nombre único y guardar archivo
        file_id = str(uuid.uuid4())
        file_name = f"{file_id}_{file.filename}"
        file_path = UPLOAD_DIR / file_name
        
        # Leer contenido del archivo
        content = await file.read()
        
        # Guardar archivo
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"Archivo guardado: {file_path}")
        
        # 3. Ejecutar OCR con Azure
        try:
            from processors.ocr.azure_ocr import analyze_document
            logger.info("Iniciando análisis OCR...")
            
            ocr_data = analyze_document(str(file_path))
            logger.info("OCR completado exitosamente")
            
            # 4. Preparar respuesta con datos extraídos
            response = {
                "success": True,
                "file_info": {
                    "id": file_id,
                    "original_name": file.filename,
                    "saved_name": file_name,
                    "size_bytes": len(content),
                    "content_type": file.content_type,
                    "extension": file_extension,
                    "path": str(file_path)
                },
                "ocr_results": {
                    "text_preview": ocr_data["text"][:200] + ("..." if len(ocr_data["text"]) > 200 else ""),
                    "text_length": len(ocr_data["text"]),
                    "key_values": ocr_data["key_values"],
                    "tables_count": len(ocr_data["tables"]),
                    "entities_count": len(ocr_data["entities"]),
                    "confidence": ocr_data["confidence"],
                    "metadata": ocr_data["metadata"]
                },
                "processing": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "completed",
                    "model_used": ocr_data["metadata"]["model_used"]
                },
                "message": "Documento procesado exitosamente con OCR"
            }
            
            # Opcional: incluir tablas si son pocas
            if len(ocr_data["tables"]) <= 3:
                response["ocr_results"]["tables"] = ocr_data["tables"]
            
            return response
            
        except ImportError as e:
            logger.error(f"Error importando módulo OCR: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Módulo OCR no disponible. Verifica la instalación de azure-ai-formrecognizer"
            )
        
        except ValueError as e:
            logger.error(f"Error de configuración OCR: {e}")
            raise HTTPException(
                status_code=500,
                detail="OCR no configurado. Verifica las credenciales de Azure en .env"
            )
        
        except Exception as e:
            logger.error(f"Error en OCR: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error procesando documento con OCR: {str(e)}"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error general en upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.get("/files")
async def list_uploaded_files():
    """
    Lista archivos subidos (útil para debugging)
    """
    try:
        files = []
        if UPLOAD_DIR.exists():
            for file_path in UPLOAD_DIR.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "name": file_path.name,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return {
            "files": files,
            "count": len(files),
            "directory": str(UPLOAD_DIR)
        }
    
    except Exception as e:
        logger.error(f"Error listando archivos: {e}")
        raise HTTPException(status_code=500, detail="Error listando archivos")

@app.delete("/files/{file_name}")
async def delete_file(file_name: str):
    """
    Elimina un archivo específico (útil para limpieza)
    """
    try:
        file_path = UPLOAD_DIR / file_name
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        file_path.unlink()
        
        return {
            "success": True,
            "message": f"Archivo {file_name} eliminado exitosamente"
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error eliminando archivo: {e}")
        raise HTTPException(status_code=500, detail="Error eliminando archivo")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)