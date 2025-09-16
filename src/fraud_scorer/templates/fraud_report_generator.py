"""
Generador de reportes con sección de Análisis de Fraude por Documento
"""
from __future__ import annotations
from typing import Dict, Any, List
from datetime import datetime

from fraud_scorer.templates.ai_report_generator import AIReportGenerator
from fraud_scorer.models.extraction import ConsolidatedExtraction
from fraud_scorer.models.fraud_analysis import FraudAnalysisResult, FraudMetrics


class FraudReportGenerator(AIReportGenerator):
    DOCUMENT_METADATA: Dict[str, Dict[str, Any]] = {
        'carta_de_reclamacion_formal_a_la_aseguradora': {
            'titulo': 'Carta de Reclamación a la Aseguradora',
            'icono': 'ri-file-list-3-line',
            'orden': 1,
        },
        'carpeta_de_investigacion': {
            'titulo': 'Carpeta de Investigación MP',
            'icono': 'ri-folder-lock-line',
            'orden': 2,
        },
        'denuncia_de_los_hechos': {
            'titulo': 'Denuncia de los Hechos',
            'icono': 'ri-article-line',
            'orden': 3,
        },
        'poliza_de_la_aseguradora': {
            'titulo': 'Póliza de Seguro',
            'icono': 'ri-shield-check-line',
            'orden': 4,
        },
        'cfdi_carta_porte': {
            'titulo': 'CFDI Carta Porte',
            'icono': 'ri-truck-line',
            'orden': 5,
        },
        'facturas_comerciales_internacionales': {
            'titulo': 'Factura Comercial Internacional',
            'icono': 'ri-bill-line',
            'orden': 6,
        },
        'guias_y_facturas': {
            'titulo': 'Facturas/Guías',
            'icono': 'ri-bill-line',
            'orden': 7,
        },
    }

    def prepare_fraud_report_data(
        self,
        consolidated_data: ConsolidatedExtraction,
        fraud_analyses: List[FraudAnalysisResult],
        documents_metadata: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        base = self._extract_base_data(consolidated_data)

        docs: List[Dict[str, Any]] = []
        for a in fraud_analyses:
            meta = self.DOCUMENT_METADATA.get(
                a.document_type,
                {"titulo": self._format_document_title(a.document_type), "icono": "ri-file-line", "orden": 99},
            )
            docs.append(
                {
                    "tipo": a.document_type,
                    "titulo": meta["titulo"],
                    "icono": meta["icono"],
                    "nombre_archivo": a.document_name,
                    "risk_level": a.risk_level.value,
                    "risk_color": self._get_risk_color(a.risk_level.value),
                    "fraud_score": f"{a.fraud_score * 100:.1f}%",
                    "confidence": f"{a.confidence * 100:.0f}%",
                    "analisis_completo": a.analisis_completo,
                    "analisis": {
                        "indicadores": self._format_indicators(a.indicators),
                        "evidencia": a.evidence,
                        "recomendaciones": a.recommendations,
                        "total_indicadores": len(a.indicators),
                    },
                }
            )

        docs.sort(key=lambda x: self.DOCUMENT_METADATA.get(x["tipo"], {}).get("orden", 99))

        metrics = self._calculate_metrics(fraud_analyses)

        return {
            **base,
            "documentos_analizados": docs,
            "total_documentos": len(docs),
            "metricas_fraude": metrics.__dict__,
            "timestamp_analisis": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "tiene_analisis_fraude": len(docs) > 0,
            "mostrar_seccion_fraude": len(docs) > 0,
        }

    def _extract_base_data(self, consolidated_data: ConsolidatedExtraction) -> Dict[str, Any]:
        """
        Extrae los datos base del consolidado reutilizando la lógica del generador estándar
        para alimentar la plantilla (número de siniestro, asegurado, póliza, fechas, etc.).
        """
        try:
            # Reutiliza el mapeo y formateos del generador base
            return self._prepare_template_data(consolidated_data)
        except Exception:
            # Fallback defensivo: construir estructura mínima
            def _to_dict_safe(obj: Any) -> Dict[str, Any]:
                if obj is None:
                    return {}
                if isinstance(obj, dict):
                    return obj
                for attr in ("model_dump", "dict"):
                    fn = getattr(obj, attr, None)
                    if callable(fn):
                        try:
                            return fn()  # type: ignore[misc]
                        except Exception:
                            pass
                try:
                    return dict(obj)  # type: ignore[arg-type]
                except Exception:
                    return {}

            cd = _to_dict_safe(consolidated_data)
            fields = _to_dict_safe(cd.get("consolidated_fields"))

            base: Dict[str, Any] = {
                "numero_siniestro": fields.get("numero_siniestro"),
                "nombre_asegurado": fields.get("nombre_asegurado"),
                "numero_poliza": fields.get("numero_poliza"),
                "vigencia": None,
                "vigencia_desde": fields.get("vigencia_inicio"),
                "vigencia_hasta": fields.get("vigencia_fin"),
                "domicilio_poliza": fields.get("domicilio_poliza"),
                "bien_reclamado": fields.get("bien_reclamado"),
                "monto_reclamacion": fields.get("monto_reclamacion"),
                "tipo_siniestro": fields.get("tipo_siniestro"),
                "fecha_ocurrencia": fields.get("fecha_ocurrencia"),
                "fecha_reclamacion": fields.get("fecha_reclamacion"),
                "lugar_hechos": fields.get("lugar_hechos"),
                "ajuste": fields.get("ajuste"),
                "confidence_scores": cd.get("confidence_scores", {}),
                "consolidation_sources": cd.get("consolidation_sources", {}),
            }
            return base

    def _format_document_title(self, doc_type: str) -> str:
        return doc_type.replace("_", " ").title()

    def _get_risk_color(self, risk: str) -> str:
        colors = {"bajo": "#16a34a", "medio": "#f59e0b", "alto": "#ef4444", "critico": "#991b1b"}
        return colors.get(risk, "#6b7280")

    def _format_indicators(self, indicators: List[Any]) -> List[Dict[str, Any]]:
        out = []
        for ind in indicators:
            try:
                # ind puede ser FraudIndicator o dict
                d = ind.dict() if hasattr(ind, "dict") else dict(ind)
                out.append(
                    {
                        "tipo": d.get("pattern"),
                        "descripcion": d.get("description"),
                        "severidad": d.get("severity"),
                        "color": self._get_risk_color(d.get("severity", "medio")),
                    }
                )
            except Exception:
                pass
        return out

    def _calculate_metrics(self, analyses: List[FraudAnalysisResult]) -> FraudMetrics:
        if not analyses:
            return FraudMetrics(
                documentos_totales=0,
                documentos_criticos=0,
                documentos_alto_riesgo=0,
                confianza_promedio=0.0,
                indicadores_totales=0,
            )
        crit = sum(1 for a in analyses if a.risk_level.value == "critico")
        alto = sum(1 for a in analyses if a.risk_level.value == "alto")
        conf = sum(a.confidence for a in analyses) / len(analyses)
        total_ind = sum(len(a.indicators) for a in analyses)
        return FraudMetrics(
            documentos_totales=len(analyses),
            documentos_criticos=crit,
            documentos_alto_riesgo=alto,
            confianza_promedio=round(conf * 100, 1),
            indicadores_totales=total_ind,
        )
