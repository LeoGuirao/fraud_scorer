# src/fraud_scorer/pipelines/canonical.py
from typing import Optional, Literal
from pydantic import BaseModel

Risk = Literal["bajo", "medio", "alto"]

class CanonicalFields(BaseModel):
    # Lo que consume TemplateProcessor
    numero_siniestro: Optional[str] = None
    nombre_asegurado: Optional[str] = None
    numero_poliza: Optional[str] = None
    vigencia_inicio: Optional[str] = None  # dd/mm/yyyy o yyyy-mm-dd, lo normalizamos luego
    vigencia_fin: Optional[str] = None
    domicilio_poliza: Optional[str] = None
    bien_reclamado: Optional[str] = None
    claim_amount: Optional[str] = None
    claim_type: Optional[str] = None
    fecha_siniestro: Optional[str] = None
    fecha_reclamacion: Optional[str] = None
    incident_location: Optional[str] = None

    class Config:
        extra = "allow"
