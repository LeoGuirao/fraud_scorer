from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
import os
import shutil
from pathlib import Path
import uuid
from datetime import datetime

# Se crea un APIRouter. Todas las rutas en este archivo usarán esta variable.
router = APIRouter(prefix="/documents", tags=["Documents"])

logger = logging.getLogger(__name__)

# Es una buena práctica definir las rutas de directorios en un archivo de configuración central,
# pero por ahora lo mantenemos aquí para que funcione.
UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Se reemplaza @app.post por @router.post
@router.post("/upload")
async def upload_and_analyze_document(file: UploadFile = File(...)):
    """
    Endpoint principal: Sube un documento y ejecuta OCR.
    NOTA: Esta es la lógica de la v1. Debería actualizarse para usar el pipeline de la v2.
    """
    try:
        # 1. Validar archivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo requerido")
        
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
        
        content = await file.read()
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"Archivo guardado: {file_path}")
        
        # 3. Ejecutar OCR con Azure (Lógica de ejemplo de la v1)
        try:
            from fraud_scorer.processors.ocr.azure_ocr import analyze_document
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
        raise
    
    except Exception as e:
        logger.error(f"Error general en upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

# Se reemplaza @app.get por @router.get
@router.get("/files")
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

# Se reemplaza @app.delete por @router.delete
@router.delete("/files/{file_name}")
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

# Se elimina el bloque if __name__ == "__main__".
# Este bloque solo debe existir en el archivo principal que inicia la aplicación (main.py).