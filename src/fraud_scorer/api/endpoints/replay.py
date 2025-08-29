# src/fraud_scorer/api/endpoints/replay.py

from fastapi import APIRouter, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from ...services.replay_service import ReplayService

logger = logging.getLogger(__name__)

router = APIRouter()
# Asegúrate de que el directorio de templates es correcto
templates = Jinja2Templates(directory="src/fraud_scorer/api/templates")
replay_service = ReplayService()

# --- Modelos Pydantic para validar las peticiones de la API ---
class ReplayConfigPayload(BaseModel):
    case_id: str
    use_ai: bool = Field(True, alias='useAi')
    detailed_analysis: bool = Field(False, alias='detailedAnalysis')
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    output_dir: Optional[str] = Field(None, alias='outputDir')
    regenerate_report: bool = Field(True, alias='regenerateReport')

class CacheClearPayload(BaseModel):
    cases: List[str]

# --- Endpoints para servir las páginas HTML (Vistas) ---

@router.get("/", response_class=HTMLResponse)
async def get_replay_dashboard(request: Request):
    return templates.TemplateResponse("replay_dashboard.html", {"request": request})

@router.get("/cases", response_class=HTMLResponse)
async def get_cases_page(request: Request):
    return templates.TemplateResponse("replay_cases.html", {"request": request})

@router.get("/specific", response_class=HTMLResponse)
async def get_specific_page(request: Request):
    """Página para replay de caso específico por ID"""
    return templates.TemplateResponse("replay_specific.html", {"request": request})

@router.get("/config/{case_id}", response_class=HTMLResponse)
async def get_config_page(request: Request, case_id: str):
    # Aquí puedes pasar información del caso a la plantilla si es necesario
    return templates.TemplateResponse("replay_config.html", {"request": request, "case_id": case_id})

@router.get("/cache", response_class=HTMLResponse)
async def get_cache_page(request: Request):
    return templates.TemplateResponse("replay_cache.html", {"request": request})

@router.get("/detailed-stats", response_class=HTMLResponse)
async def get_stats_page(request: Request):
    return templates.TemplateResponse("replay_stats.html", {"request": request})


# --- Endpoints de API para la lógica (devuelven JSON) ---

@router.get("/api/stats")
async def get_stats_data():
    """
    Obtiene las estadísticas del cache OCR para mostrar en el dashboard.
    """
    try:
        stats = replay_service.get_cache_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas del cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )

@router.get("/api/cases")
async def get_cases_data():
    """
    Lista todos los casos disponibles para replay desde el cache.
    """
    try:
        cases = replay_service.list_available_cases()
        return {
            "success": True,
            "data": cases
        }
    except Exception as e:
        logger.error(f"Error listando casos disponibles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listando casos: {str(e)}"
        )

@router.get("/api/case/{case_id}/info")
async def get_case_info_data(case_id: str):
    """
    Obtiene información detallada de un caso específico.
    """
    try:
        case_info = replay_service.cache_manager.get_case_index(case_id)
        if not case_info:
            raise HTTPException(
                status_code=404,
                detail=f"Caso {case_id} no encontrado"
            )
        
        return {
            "success": True,
            "data": case_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo información del caso {case_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo información del caso: {str(e)}"
        )

@router.post("/api/process")
async def process_replay_api(payload: ReplayConfigPayload):
    """
    Ejecuta el replay de un caso con la configuración especificada.
    """
    try:
        # Validar que el caso existe
        case_info = replay_service.cache_manager.get_case_index(payload.case_id)
        if not case_info:
            raise HTTPException(
                status_code=404,
                detail=f"Caso {payload.case_id} no encontrado"
            )

        # Convertir payload a dict usando el alias mapping
        config_dict = payload.model_dump(by_alias=False)
        
        logger.info(f"Iniciando replay para caso {payload.case_id}")
        
        # Ejecutar el replay
        result = await replay_service.process_replay(config_dict)
        
        return {
            "success": True,
            "message": f"Replay completado para caso {payload.case_id}",
            "data": {
                "case_id": result["case_id"],
                "output_path": result["output_path"],
                "replay_date": result["replay_date"],
                "fraud_analysis": result.get("fraud_analysis"),
                "document_count": len(result.get("extraction_results", []))
            }
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error procesando replay para caso {payload.case_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno en el servidor durante el replay.")

@router.post("/api/cache/delete")  # Usamos POST para un cuerpo de solicitud con lista
async def clear_cache_api(payload: CacheClearPayload):
    """
    Limpia el cache de múltiples casos o todo el cache.
    """
    try:
        result = replay_service.clear_cache(payload.cases)
        return {
            "success": True,
            "message": result["message"],
            "data": {
                "cleared": result.get("cleared", []),
                "errors": result.get("errors", [])
            }
        }
    except Exception as e:
        logger.error(f"Error limpiando caché: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error al limpiar el caché.")

@router.delete("/api/cache/{case_id}")
async def clear_single_case_cache(case_id: str):
    """
    Limpia el cache de un caso específico.
    """
    try:
        result = replay_service.clear_cache([case_id])
        
        if result["status"] == "success":
            return {
                "success": True,
                "message": result["message"],
                "data": result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Error limpiando cache")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error limpiando cache del caso {case_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error limpiando cache: {str(e)}"
        )