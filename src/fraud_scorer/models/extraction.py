# src/fraud_scorer/models/extraction.py
"""
Modelos Pydantic para validación y consolidación de datos extraídos.
Incluye compatibilidad Pydantic v1/v2 y la clase ConsolidatedFields.
"""

from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime
import re
import json

from fraud_scorer.analyzers.correlation.utils.normalization import (
    normalize_date,
    normalize_decimal_as_str,
)


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

    @field_validator('extracted_fields', mode='after')
    def _normalize_extraction_fields(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            return value or {}
        normalized = dict(value)

        date_fields = {
            'fecha_ocurrencia',
            'fecha_reclamacion',
            'vigencia_inicio',
            'vigencia_fin',
            'fecha_timbrado',
            'fecha_emision',
            'fecha_denuncia',
            'fecha_apertura',
        }
        numeric_fields = {
            'monto_reclamacion',
            'monto_total',
            'valor_mercancia',
            'suma_asegurada',
            'deducible',
            'peso_bruto',
            'peso_total',
            'peso',
        }

        for field in date_fields:
            if field in normalized and normalized[field] is not None:
                normalised_value = normalize_date(normalized[field])
                if normalised_value:
                    normalized[field] = normalised_value
                elif isinstance(normalized[field], str):
                    normalized[field] = normalized[field].strip()

        for field in numeric_fields:
            if field in normalized and normalized[field] is not None:
                normalised_value = normalize_decimal_as_str(normalized[field])
                if normalised_value is not None:
                    normalized[field] = normalised_value
                elif isinstance(normalized[field], str):
                    cleaned = re.sub(r'[^\d.,-]', '', normalized[field]).strip()
                    normalized[field] = cleaned or normalized[field]

        return normalized


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
    monto_reclamacion: Optional[Union[str, float]] = None   # aceptar numérico o string
    suma_asegurada: Optional[Union[str, float]] = None
    numero_poliza: Optional[str] = None
    # Compatibilidad: mantener campos separados y cadena agregada
    vigencia_inicio: Optional[str] = None
    vigencia_fin: Optional[str] = None
    vigencia: Optional[str] = None
    domicilio_poliza: Optional[str] = None
    tipo_siniestro: Optional[str] = None
    fecha_ocurrencia: Optional[str] = None
    fecha_reclamacion: Optional[str] = None
    lugar_hechos: Optional[str] = None
    bien_reclamado: Optional[str] = None
    ajuste: Optional[str] = None
    conclusiones: Optional[str] = None  # aparece en JSON aunque no siempre en HTML

    # Pydantic v2 configuration
    model_config = ConfigDict(str_strip_whitespace=True)


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
#  Progress Events
# ==============

class ProgressEvent(BaseModelCompat):
    """Event model for real-time progress tracking via JSONL files."""
    timestamp: float
    case_id: str
    stage: str               # "upload" | "ocr" | "extract" | "consolidate" | "analyze" | "report"
    doc_index: Optional[int] = None
    doc_total: Optional[int] = None
    status: str              # "started" | "running" | "done" | "error"
    elapsed_ms: Optional[int] = None
    avg_stage_ms: Optional[int] = None
    eta_ms: Optional[int] = None
    message: Optional[str] = None


# ==============
#  Utilidades
# ==============

def safe_json(obj: Any) -> str:
    """Pequeña ayuda para depurar estructuras sin romper caracteres."""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)
