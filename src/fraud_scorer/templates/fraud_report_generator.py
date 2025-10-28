"""
Generador de reportes con sección de Análisis de Fraude por Documento
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime

from fraud_scorer.analyzers.correlation.models import CorrelationReport, CorrelationFinding
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
        'oficio_denuncia': {
            'titulo': 'Oficio de Denuncia',
            'icono': 'ri-file-text-line',
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
        'pedimento_importacion': {
            'titulo': 'Pedimento de Importación',
            'icono': 'ri-file-paper-2-line',
            'orden': 6,
        },
        'conocimiento_de_embarque': {
            'titulo': 'Conocimiento de Embarque',
            'icono': 'ri-ship-line',
            'orden': 7,
        },
        'contrato_prestacion_servicio_transportista': {
            'titulo': 'Contrato Servicio Transportista',
            'icono': 'ri-briefcase-4-line',
            'orden': 8,
        },
        'oficio_de_desaduanado': {
            'titulo': 'Oficio de Desaduanado',
            'icono': 'ri-file-check-line',
            'orden': 9,
        },
        'carta_aclatoria_comprobantes_peaje': {
            'titulo': 'Carta Aclaratoria Peaje',
            'icono': 'ri-roadster-line',
            'orden': 10,
        },
        'carta_porte_simple': {
            'titulo': 'Carta Porte Simple',
            'icono': 'ri-truck-line',
            'orden': 11,
        },
        'protocolo_de_accion_y_reaccion': {
            'titulo': 'Protocolo de Acción y Reacción',
            'icono': 'ri-shield-keyhole-line',
            'orden': 12,
        },
        'facturas_comerciales_internacionales': {
            'titulo': 'Factura Comercial Internacional',
            'icono': 'ri-bill-line',
            'orden': 13,
        },
        'guias_y_facturas': {
            'titulo': 'Facturas/Guías',
            'icono': 'ri-bill-line',
            'orden': 14,
        },
    }

    def prepare_fraud_report_data(
        self,
        consolidated_data: ConsolidatedExtraction,
        fraud_analyses: List[FraudAnalysisResult],
        documents_metadata: List[Dict[str, Any]],
        correlation_report: Optional[CorrelationReport] = None,
    ) -> Dict[str, Any]:
        base = self._extract_base_data(consolidated_data)

        visible_analyses = [
            analysis
            for analysis in fraud_analyses
            if getattr(analysis, "include_in_report", True)
        ]

        grouped: Dict[str, List[FraudAnalysisResult]] = {}
        for analysis in visible_analyses:
            grouped.setdefault(analysis.document_type, []).append(analysis)

        docs: List[Dict[str, Any]] = []
        for doc_type, group in grouped.items():
            meta = self.DOCUMENT_METADATA.get(
                doc_type,
                {"titulo": self._format_document_title(doc_type), "icono": "ri-file-line", "orden": 99},
            )
            if doc_type == "carta_porte_simple" and len(group) > 1:
                docs.append(self._build_carta_porte_multi(meta, group))
                continue
            for analysis in group:
                titulo = meta["titulo"]
                if doc_type == "carta_porte_simple":
                    titulo = meta["titulo"]
                docs.append(self._serialize_analysis(analysis, titulo, meta["icono"]))

        docs.sort(key=lambda x: self.DOCUMENT_METADATA.get(x["tipo"], {}).get("orden", 99))

        metrics_all = self._calculate_metrics(fraud_analyses)
        metrics_visible = self._calculate_metrics(visible_analyses)
        metrics_payload = metrics_all.__dict__.copy()
        metrics_payload.update(
            {
                "documentos_publicables": len(visible_analyses),
                "documentos_publicables_criticos": metrics_visible.documentos_criticos,
                "documentos_publicables_alto_riesgo": metrics_visible.documentos_alto_riesgo,
                "confianza_promedio_publicables": metrics_visible.confianza_promedio,
                "indicadores_publicables": metrics_visible.indicadores_totales,
            }
        )
        correlation_data = self._prepare_correlation_section(correlation_report)

        return {
            **base,
            "documentos_analizados": docs,
            "total_documentos": len(docs),
            "metricas_fraude": metrics_payload,
            "timestamp_analisis": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "tiene_analisis_fraude": len(docs) > 0,
            "mostrar_seccion_fraude": len(docs) > 0,
            "correlacion_inter_documentos": correlation_data,
            "mostrar_seccion_correlacion": correlation_data.get("has_findings", False),
        }

    def _serialize_analysis(
        self,
        analysis: FraudAnalysisResult,
        titulo: str,
        icono: str,
    ) -> Dict[str, Any]:
        return {
            "tipo": analysis.document_type,
            "titulo": titulo,
            "icono": icono,
            "nombre_archivo": analysis.document_name,
            "risk_level": analysis.risk_level.value,
            "risk_color": self._get_risk_color(analysis.risk_level.value),
            "fraud_score": f"{analysis.fraud_score * 100:.1f}%",
            "confidence": f"{analysis.confidence * 100:.0f}%",
            "analisis_completo": analysis.analisis_completo,
            "analisis": {
                "indicadores": self._format_indicators(analysis.indicators),
                "recomendaciones": analysis.recommendations,
                "verificaciones": analysis.verificaciones,
                "validacion_cruzada": analysis.validacion_cruzada,
                "total_indicadores": len(analysis.indicators),
            },
        }

    def _build_carta_porte_multi(
        self,
        meta: Dict[str, Any],
        analyses: List[FraudAnalysisResult],
    ) -> Dict[str, Any]:
        count = len(analyses)
        sorted_group = sorted(analyses, key=lambda item: item.document_name or "")
        categories: List[str] = []
        for element in sorted_group:
            carta_layer = (element.validacion_cruzada or {}).get("carta_porte") or {}
            categoria = carta_layer.get("categoria_bienes")
            if categoria:
                categories.append(str(categoria))
        category_phrase = self._format_category_phrase(categories)
        intro = (
            f"Se presentan {self._number_to_spanish_word(count)} cartas de porte para la justificación del traslado de {category_phrase}."
        )

        risk_priority = {"bajo": 1, "medio": 2, "alto": 3, "critico": 4}
        highest = max(sorted_group, key=lambda item: risk_priority.get(item.risk_level.value, 1))
        max_score = max(element.fraud_score for element in sorted_group)
        min_confidence = min(element.confidence for element in sorted_group)

        aggregated = {
            "tipo": "carta_porte_simple",
            "titulo": meta["titulo"],
            "icono": meta["icono"],
            "nombre_archivo": f"Múltiples ({count} documentos)",
            "risk_level": highest.risk_level.value,
            "risk_color": self._get_risk_color(highest.risk_level.value),
            "fraud_score": f"{max_score * 100:.1f}%",
            "confidence": f"{min_confidence * 100:.0f}%",
            "analisis_completo": intro,
            "analisis": {
                "indicadores": [],
                "recomendaciones": [],
                "verificaciones": {},
                "validacion_cruzada": {},
                "total_indicadores": 0,
            },
            "subanalyses": [],
            "is_multi": True,
        }

        for idx, item in enumerate(sorted_group, start=1):
            sub_payload = self._serialize_analysis(item, f"Carta Porte Simple {idx}", meta["icono"])
            aggregated["subanalyses"].append(sub_payload)
        return aggregated

    def _number_to_spanish_word(self, value: int) -> str:
        mapping = {
            1: "una",
            2: "dos",
            3: "tres",
            4: "cuatro",
            5: "cinco",
            6: "seis",
            7: "siete",
            8: "ocho",
            9: "nueve",
            10: "diez",
        }
        return mapping.get(value, str(value))

    def _format_category_phrase(self, categories: List[str]) -> str:
        unique = [item for item in dict.fromkeys(categories) if item]
        if not unique:
            return "mercancía"
        if len(unique) == 1:
            return unique[0]
        if len(unique) == 2:
            return f"{unique[0]} y {unique[1]}"
        return ", ".join(unique[:-1]) + f" y {unique[-1]}"

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

    def _prepare_correlation_section(
        self,
        report: Optional[CorrelationReport],
    ) -> Dict[str, Any]:
        if not report:
            return {"summary": {}, "findings": [], "has_findings": False}

        findings = [self._format_finding(f) for f in report.findings]
        summary = report.as_summary()
        return {
            "summary": summary,
            "findings": findings,
            "has_findings": bool(findings),
        }

    def _format_finding(self, finding: CorrelationFinding) -> Dict[str, Any]:
        return {
            "id": finding.id,
            "rule_id": finding.rule_id,
            "rule_version": finding.rule_version,
            "status": finding.status.value,
            "severity": finding.severity.value,
            "summary": finding.summary,
            "description": finding.description,
            "documents": finding.documents_involved,
            "entities": finding.entities_involved,
            "recommendation": finding.recommendation,
            "tags": finding.tags,
            "evidence": finding.evidence,
            "metadata": finding.metadata,
        }
