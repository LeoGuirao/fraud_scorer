# src/fraud_scorer/ai_extractors/models/extraction_models.py
"""
Modelos Pydantic para validación y consolidación de datos extraídos.
Incluye compatibilidad Pydantic v1/v2 y la clase ConsolidatedFields.
"""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import re
import json


# =========================
#   Compatibilidad Pydantic
# =========================

class BaseModelCompat(BaseModel):
    """
    Proporciona .model_dump() tanto en Pydantic v1 como en v2.
    - En v2: delega a BaseModel.model_dump().
    - En v1: usa BaseModel.dict() por debajo.
    """
    def model_dump(self, *args, **kwargs):
        try:
            # Pydantic v2
            return super().model_dump(*args, **kwargs)  # type: ignore[attr-defined]
        except Exception:
            # Pydantic v1
            return super().dict(*args, **kwargs)

    @classmethod
    def model_validate(cls, obj):
        try:
            # Pydantic v2
            return super().model_validate(obj)  # type: ignore[attr-defined]
        except Exception:
            # Pydantic v1
            return cls.parse_obj(obj)


# =========================
#     Modelos de Extracción
# =========================

class ExtractedField(BaseModelCompat):
    """Modelo para un campo extraído individual (no siempre usado por el pipeline actual)."""
    value: Optional[Any] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_page: Optional[int] = None
    source_section: Optional[str] = None


class DocumentExtraction(BaseModelCompat):
    """Resultado de extracción de un documento."""
    source_document: str
    document_type: str
    extracted_fields: Dict[str, Optional[Any]] = Field(default_factory=dict)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('extracted_fields')
    def _validate_dates(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida y normaliza campos de fecha conocidos si vienen como string.
        (Deja la implementación real de parsing pendiente para tu formato específico).
        """
        date_fields = ['fecha_ocurrencia', 'fecha_reclamacion', 'vigencia_inicio', 'vigencia_fin']
        for field in date_fields:
            if field in v and v[field]:
                if isinstance(v[field], str):
                    # Aquí podrías normalizar a 'YYYY-MM-DD' si lo necesitas.
                    # Por ahora, lo dejamos tal cual para no introducir errores.
                    v[field] = v[field].strip()
        return v

    @validator('extracted_fields')
    def _validate_amounts(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida y normaliza el monto de la reclamación si llega como string con símbolos.
        Nota: ConsolidatedFields lo almacena como string para el template; la conversión a float
        aquí solo intenta limpiar el input. En consolidación se convertirá a str sin inventar.
        """
        key = 'monto_reclamacion'
        if key in v and v[key]:
            if isinstance(v[key], str):
                # Mantén sólo dígitos, puntos y comas, luego quita separadores de miles simples.
                clean = re.sub(r'[^\d.,-]', '', v[key]).strip()
                # Intenta parsear como número para validar (sin forzar)
                try:
                    num = float(clean.replace(',', ''))
                    # Conserva como string "limpia" para el template posteriormente
                    # (evitamos perder formato monetario esperado).
                    v[key] = f"{num}"
                except Exception:
                    # Si no se puede parsear, deja el string limpio
                    v[key] = clean or v[key]
        return v


# =========================
#     Consolidación final
# =========================

class ConsolidatedFields(BaseModelCompat):
    """
    Estructura final consolidada para el reporte HTML/PDF.
    IMPORTANTE: estos nombres deben coincidir con los esperados por tus plantillas.
    Todos son opcionales; si faltan, el reporte debe mostrar guiones/“NO ESPECIFICADO”.
    """
    numero_siniestro: Optional[str] = None
    nombre_asegurado: Optional[str] = None
    monto_reclamacion: Optional[str] = None   # string para facilitar formateo en template
    numero_poliza: Optional[str] = None
    vigencia: Optional[str] = None
    domicilio_poliza: Optional[str] = None
    tipo_siniestro: Optional[str] = None
    fecha_ocurrencia: Optional[str] = None
    fecha_reclamacion: Optional[str] = None
    lugar_hechos: Optional[str] = None
    bien_reclamado: Optional[str] = None
    ajuste: Optional[str] = None
    conclusiones: Optional[str] = None  # aparece en JSON aunque no siempre en HTML

    class Config:
        anystr_strip_whitespace = True  # Pydantic v1
    # En Pydantic v2 podrías usar:
    # from pydantic import ConfigDict
    # model_config = ConfigDict(str_strip_whitespace=True)


class ConsolidatedExtraction(BaseModelCompat):
    """Resultado consolidado de todos los documentos para un caso."""
    case_id: str
    consolidated_fields: ConsolidatedFields
    consolidation_sources: Dict[str, str] = Field(default_factory=dict)
    conflicts_resolved: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)


class ExtractionBatch(BaseModelCompat):
    """
    Batch de extracciones para consolidar.
    AIConsolidator sólo necesita la lista `extractions`.
    """
    extractions: List[DocumentExtraction]


# ==============
#  Utilidades
# ==============

def safe_json(obj: Any) -> str:
    """Pequeña ayuda para depurar estructuras sin romper caracteres."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)
