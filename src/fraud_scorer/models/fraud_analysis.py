"""
Modelos Pydantic para el Sistema de Análisis de Fraudes por Documento
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from enum import Enum


class RiskLevel(str, Enum):
    BAJO = "bajo"
    MEDIO = "medio"
    ALTO = "alto"
    CRITICO = "critico"


class FraudIndicator(BaseModel):
    pattern: str
    description: str
    severity: str  # bajo | medio | alto | critico
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: Optional[str] = None
    location: Optional[Dict[str, Any]] = None  # página, bbox, sección, etc.


class EvidenceGap(BaseModel):
    gap: str
    impact: Optional[str] = None
    priority: Optional[str] = Field(default=None, description="alta | media | baja")
    suggested_action: Optional[str] = None
    missing_documents: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)


class FraudAnalysisResult(BaseModel):
    # Identificación
    document_id: str
    document_name: str
    document_type: str
    case_id: str
    analysis_id: Optional[str] = None
    prompt_hash: Optional[str] = None

    # Métricas principales
    risk_level: RiskLevel
    fraud_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    # Análisis narrativo completo
    analisis_completo: str = Field(
        default="",
        description="Análisis narrativo exhaustivo del documento (3-5 párrafos)"
    )

    # Detalles estructurados
    indicators: List[FraudIndicator] = Field(default_factory=list)
    evidence_gaps: List[EvidenceGap] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    verificaciones: Dict[str, Any] = Field(default_factory=dict)
    validacion_cruzada: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    analysis_model: str
    guide_version: str
    processing_time_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    include_in_report: bool = Field(
        default=True,
        description="Permite ocultar el análisis en el reporte final sin perder trazabilidad",
    )

    @model_validator(mode="after")
    def _check_score_risk_coherence(self) -> "FraudAnalysisResult":
        score = self.fraud_score
        risk: RiskLevel = self.risk_level
        if risk == RiskLevel.BAJO and score > 0.30:
            raise ValueError(f"Score {score} muy alto para riesgo bajo (<= 0.30)")
        if risk == RiskLevel.MEDIO and not (0.30 <= score < 0.60):
            raise ValueError(f"Score {score} fuera de rango para riesgo medio (0.30–0.59)")
        if risk == RiskLevel.ALTO and not (0.60 <= score < 0.85):
            raise ValueError(f"Score {score} fuera de rango para riesgo alto (0.60–0.84)")
        if risk == RiskLevel.CRITICO and score < 0.85:
            raise ValueError(f"Score {score} muy bajo para riesgo crítico (>= 0.85)")

        # Normalización defensiva de evidence_gaps
        normalized_gaps: List[EvidenceGap] = []
        for item in self.evidence_gaps:
            if isinstance(item, EvidenceGap):
                normalized_gaps.append(item)
            elif isinstance(item, dict):
                normalized_gaps.append(EvidenceGap(**item))
            elif isinstance(item, str):
                normalized_gaps.append(EvidenceGap(gap=item))
        self.evidence_gaps = normalized_gaps

        # Normalización defensiva de estructuras dict
        if not isinstance(self.verificaciones, dict):
            self.verificaciones = dict(self.verificaciones or {})
        if not isinstance(self.validacion_cruzada, dict):
            self.validacion_cruzada = dict(self.validacion_cruzada or {})
        return self


class DocumentAnalysisSection(BaseModel):
    tipo: str
    titulo: str
    icono: str
    nombre_archivo: str
    risk_level: str
    risk_color: str
    fraud_score: str  # porcentaje string
    analisis: Dict[str, Any]
    presente: bool = True  # compatibilidad histórica
    include_in_report: bool = True


class FraudMetrics(BaseModel):
    """Resumen operativo sin consolidación automática"""
    documentos_totales: int
    documentos_criticos: int
    documentos_alto_riesgo: int
    confianza_promedio: float  # porcentaje 0-100
    indicadores_totales: int
