# src/fraud_scorer/ai_extractors/config.py

"""
Configuración para el sistema de extracción con IA
"""
import os
from enum import Enum
from typing import Dict, Any
from pathlib import Path

class ModelType(Enum):
    """Modelos disponibles para cada tarea"""
    EXTRACTOR = "gpt-4o-mini"  # Más barato para extracción simple
    CONSOLIDATOR = "gpt-4o"     # Más potente para razonamiento
    GENERATOR = "gpt-4o-mini"   # Para generación de reportes

class FieldPriority(Enum):
    """Prioridad de fuentes por tipo de documento"""
    POLIZA = 1
    DENUNCIA = 2
    FACTURA = 3
    PERITAJE = 4
    CARTA_PORTE = 5
    OTRO = 99

class ExtractionConfig:
    """Configuración del sistema de extracción"""
    
    # Campos a extraer
    REQUIRED_FIELDS = [
        "numero_siniestro",
        "nombre_asegurado", 
        "numero_poliza",
        "vigencia_inicio",
        "vigencia_fin",
        "domicilio_poliza",
        "bien_reclamado",
        "monto_reclamacion",
        "tipo_siniestro",
        "fecha_ocurrencia",
        "fecha_reclamacion",
        "lugar_hechos",
        "ajuste",
        "conclusiones"
    ]
    
    # Mapeo de tipos de documento a prioridades
    DOCUMENT_PRIORITIES = {
        "poliza": FieldPriority.POLIZA,
        "poliza_seguro": FieldPriority.POLIZA,
        "denuncia": FieldPriority.DENUNCIA,
        "factura": FieldPriority.FACTURA,
        "factura_compra": FieldPriority.FACTURA,
        "peritaje": FieldPriority.PERITAJE,
        "carta_porte": FieldPriority.CARTA_PORTE,
    }
    
    # Reglas de prioridad por campo
    FIELD_SOURCE_RULES = {
        "numero_poliza": ["poliza", "poliza_seguro"],
        "nombre_asegurado": ["poliza", "poliza_seguro", "denuncia"],
        "vigencia_inicio": ["poliza", "poliza_seguro"],
        "vigencia_fin": ["poliza", "poliza_seguro"],
        "domicilio_poliza": ["poliza", "poliza_seguro"],
        "fecha_ocurrencia": ["denuncia", "peritaje", "carta_porte"],
        "lugar_hechos": ["denuncia", "peritaje", "bitacora_gps"],
        "monto_reclamacion": ["factura", "factura_compra", "peritaje"],
        "bien_reclamado": ["factura", "factura_compra", "carta_porte"],
    }
    
    # Configuración de OpenAI
    OPENAI_CONFIG = {
        "temperature": 0.1,  # Muy bajo para consistencia
        "max_tokens": 2000,
        "timeout": 30,
        "max_retries": 3
    }
    
    # Rutas
    PROMPTS_DIR = Path(__file__).parent / "prompts"
    EXAMPLES_DIR = Path("data/training_examples")
    
    @classmethod
    def get_model_for_task(cls, task: str) -> str:
        """Obtiene el modelo apropiado para cada tarea"""
        if task == "extraction":
            return ModelType.EXTRACTOR.value
        elif task == "consolidation":
            return ModelType.CONSOLIDATOR.value
        elif task == "generation":
            return ModelType.GENERATOR.value
        return ModelType.EXTRACTOR.value