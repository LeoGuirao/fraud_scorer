# src/fraud_scorer/storage/feedback.py

import logging
from typing import List, Dict, Any, Optional
from .db import save_feedback, get_feedback_by_case, get_feedback_stats, get_conn

logger = logging.getLogger(__name__)

def save_feedback_from_json(case_id: str, feedback_data: List[Dict[str, Any]]) -> None:
    """
    Guarda una lista de registros de feedback en la base de datos para un caso específico.
    """
    try:
        with get_conn() as conn:
            # Primero, borramos cualquier feedback antiguo para este caso para evitar duplicados
            conn.execute("DELETE FROM feedback WHERE case_id = ?", (case_id,))
            
            # Guardamos los nuevos registros de feedback
            feedback_count = 0
            for item in feedback_data:
                field_name = item.get("field", "")
                original_value = item.get("originalValue")
                corrected_value = item.get("newValue")
                status = item.get("status", "")
                
                # Convertir a string si no es None
                original_str = str(original_value) if original_value is not None else None
                corrected_str = str(corrected_value) if corrected_value is not None else None
                
                if field_name and status:
                    save_feedback(case_id, field_name, original_str, corrected_str, status)
                    feedback_count += 1
                    
        logger.info(f"Feedback para el caso {case_id} guardado exitosamente. {feedback_count} registros.")
        
    except Exception as e:
        logger.error(f"Error al guardar el feedback para el caso {case_id}: {e}")
        raise

def get_case_feedback(case_id: str) -> List[Dict[str, Any]]:
    """
    Obtiene todo el feedback para un caso específico en formato JSON.
    """
    try:
        rows = get_feedback_by_case(case_id)
        feedback_list = []
        
        for row in rows:
            feedback_item = {
                "id": row["id"],
                "field": row["field_name"],
                "originalValue": row["original_value"],
                "correctedValue": row["corrected_value"], 
                "status": row["status"],
                "createdAt": row["created_at"]
            }
            feedback_list.append(feedback_item)
            
        return feedback_list
        
    except Exception as e:
        logger.error(f"Error al obtener feedback para el caso {case_id}: {e}")
        return []

def get_feedback_summary(case_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Obtiene un resumen estadístico del feedback.
    """
    try:
        stats = get_feedback_stats(case_id)
        total = sum(stats.values())
        
        summary = {
            "total": total,
            "confirmed": stats.get("confirmed", 0),
            "corrected": stats.get("corrected", 0),
            "accuracy_rate": round((stats.get("confirmed", 0) / total * 100), 2) if total > 0 else 0
        }
        
        if case_id:
            summary["case_id"] = case_id
            
        return summary
        
    except Exception as e:
        logger.error(f"Error al obtener resumen de feedback: {e}")
        return {"total": 0, "confirmed": 0, "corrected": 0, "accuracy_rate": 0}

def validate_feedback_data(feedback_data: List[Dict[str, Any]]) -> bool:
    """
    Valida que los datos de feedback tengan el formato correcto.
    """
    if not isinstance(feedback_data, list):
        logger.error("Los datos de feedback deben ser una lista")
        return False
        
    for item in feedback_data:
        if not isinstance(item, dict):
            logger.error("Cada elemento de feedback debe ser un diccionario")
            return False
            
        if "field" not in item or "status" not in item:
            logger.error("Cada elemento debe tener 'field' y 'status'")
            return False
            
        if item["status"] not in ["confirmed", "corrected"]:
            logger.error(f"Status inválido: {item['status']}. Debe ser 'confirmed' o 'corrected'")
            return False
            
    return True