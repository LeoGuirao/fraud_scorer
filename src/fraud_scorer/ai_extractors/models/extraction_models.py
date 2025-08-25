# src/fraud_scorer/ai_extractors/models/extraction_models.py

"""
Modelos Pydantic para validación de datos extraídos
"""
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import re

class ExtractedField(BaseModel):
    """Modelo para un campo extraído individual"""
    value: Optional[Any] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_page: Optional[int] = None
    source_section: Optional[str] = None
    
class DocumentExtraction(BaseModel):
    """Resultado de extracción de un documento"""
    source_document: str
    document_type: str
    extracted_fields: Dict[str, Optional[Any]]
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('extracted_fields')
    def validate_dates(cls, v):
        """Valida y formatea fechas"""
        date_fields = ['fecha_ocurrencia', 'fecha_reclamacion', 'vigencia_inicio', 'vigencia_fin']
        for field in date_fields:
            if field in v and v[field]:
                # Intentar parsear y reformatear la fecha
                if isinstance(v[field], str):
                    # Aquí agregarías lógica de parsing de fechas
                    pass
        return v
    
    @validator('extracted_fields')
    def validate_amounts(cls, v):
        """Valida y formatea montos"""
        if 'monto_reclamacion' in v and v['monto_reclamacion']:
            # Convertir a float si es string
            if isinstance(v['monto_reclamacion'], str):
                # Limpiar el string de símbolos
                clean = re.sub(r'[^\d.,]', '', v['monto_reclamacion'])
                clean = clean.replace(',', '')
                try:
                    v['monto_reclamacion'] = float(clean)
                except:
                    v['monto_reclamacion'] = 0.0
        return v

class ConsolidatedExtraction(BaseModel):
    """Resultado consolidado de todos los documentos"""
    case_id: str
    consolidated_fields: Dict[str, Optional[Any]]
    consolidation_sources: Dict[str, str]
    conflicts_resolved: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    
class ExtractionBatch(BaseModel):
    """Batch de extracciones para consolidar"""
    case_id: str
    documents: List[DocumentExtraction]
    total_documents: int
    processing_timestamp: datetime = Field(default_factory=datetime.now)